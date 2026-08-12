// This hook is the answer to "how does the UI receive the response in
// batches while the LLM keeps generating?"
//
// The backend (see ../../backend/chat.py) sends the answer as an SSE
// (Server-Sent Events) stream: a normal HTTP response that STAYS OPEN and
// writes small pieces of text to it over time, instead of sending one
// complete response at the end.
//
// We can't use the browser's built-in `EventSource` API for this, because
// EventSource can only make GET requests — we need to POST the question.
// So instead we do it manually with `fetch()` + a ReadableStream: fetch()
// gives us access to the response body as a stream of raw bytes, arriving
// in chunks as the network delivers them (NOT the whole file/response at
// once). We read that stream in a loop, and every time a new piece
// arrives, we append it to React state — which triggers a re-render,
// which is what you SEE as "the text growing on screen".
//
// So: the LLM generates one token at a time -> the backend forwards each
// token the instant it arrives -> the browser's network stack delivers
// bytes in small bursts as they arrive over the wire -> each burst here
// becomes a `read()` result -> each result becomes a state update -> each
// state update becomes a re-render. There's no single place that
// "batches" anything — it's the same continuous stream, visible at every layer.

import { useState, useCallback } from "react";
import { API_BASE } from "./api";

export function useChatStream() {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const sendMessage = useCallback(async (question) => {
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", sources: [] },
    ]);
    setIsStreaming(true);

    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    // response.body is a ReadableStream of raw bytes. getReader() gives us
    // a handle to pull chunks from it one at a time, as they arrive.
    const reader = response.body.getReader();
    const decoder = new TextDecoder(); // bytes -> text; a chunk can split a UTF-8 char mid-byte, decoder handles that across calls
    let buffer = ""; // holds any partial SSE event that arrived without its trailing blank line yet

    while (true) {
      const { done, value } = await reader.read();
      if (done) break; // stream closed by the server (after chat.py's "done" event)

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by a blank line ("\n\n"). A single
      // reader.read() call might deliver half an event, multiple whole
      // events, or anything in between — network chunking doesn't respect
      // our message boundaries. So we split on "\n\n" and keep any
      // trailing incomplete piece in `buffer` for next time.
      const events = buffer.split("\n\n");
      buffer = events.pop(); // last piece may be incomplete — keep it for the next read()

      for (const rawEvent of events) {
        const eventTypeMatch = rawEvent.match(/^event: (.+)$/m);
        const dataMatch = rawEvent.match(/^data: (.+)$/m);
        if (!eventTypeMatch || !dataMatch) continue;

        const eventType = eventTypeMatch[1];
        const data = JSON.parse(dataMatch[1]);

        if (eventType === "sources") {
          setMessages((prev) => updateLastAssistant(prev, (m) => ({ ...m, sources: data })));
        } else if (eventType === "token") {
          // This is the line that makes text "stream in": append this
          // piece to whatever the assistant message already has.
          setMessages((prev) =>
            updateLastAssistant(prev, (m) => ({ ...m, content: m.content + data }))
          );
        } else if (eventType === "done") {
          setIsStreaming(false);
        }
      }
    }

    setIsStreaming(false);
  }, []);

  return { messages, isStreaming, sendMessage };
}

function updateLastAssistant(messages, updateFn) {
  const last = messages[messages.length - 1];
  return [...messages.slice(0, -1), updateFn(last)];
}
