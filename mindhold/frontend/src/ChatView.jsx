import { useState } from "react";
import { useChatStream } from "./useChatStream";

export default function ChatView() {
  const { messages, isStreaming, sendMessage } = useChatStream();
  const [input, setInput] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;
    sendMessage(input.trim());
    setInput("");
  };

  return (
    <div className="chat-view">
      <h1>Chat</h1>

      <div className="messages">
        {messages.length === 0 && (
          <p className="empty-hint">Ask a question about your notes.</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <div className="role-label">{m.role === "user" ? "You" : "Assistant"}</div>
            <div className="content">
              {m.content || (m.role === "assistant" && isStreaming && i === messages.length - 1 ? "…" : "")}
            </div>
            {m.sources && m.sources.length > 0 && (
              <div className="sources">Sources: {m.sources.join(", ")}</div>
            )}
          </div>
        ))}
      </div>

      <form className="input-row" onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={isStreaming}
        />
        <button type="submit" disabled={isStreaming || !input.trim()}>
          {isStreaming ? "Streaming..." : "Send"}
        </button>
      </form>
    </div>
  );
}
