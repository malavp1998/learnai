"""
STEP 5: Tool calling + ToolMessage, done as a proper LOOP.

So far the LLM could only respond with text. Real agents need to DO
things — run a calculation, look something up, call an API. LangChain
lets you hand the model a list of Python functions ("tools"). The
model can then say "call this function with these arguments" instead
of just replying in text.

The 4th message type, ToolMessage, is how we send the tool's RESULT
back to the model so it can use it to write the final answer.

IMPORTANT LESSON (learned by testing this exact script):
A single request -> tool_call -> ToolMessage -> response round trip is
NOT enough for multi-step problems. If you ask "multiply 12 by 7, then
add 5 to that", the model can return BOTH tool calls in one go —
[multiply(12, 7), add(84, 5)] — because it silently pre-computed 84
itself instead of waiting for your tool's real answer. That means the
"add" call's arguments came from the MODEL'S guess, not from actually
running multiply(). If the model's mental math were wrong, you'd get a
wrong final answer and never know it.

The fix: use a question the model CANNOT pre-solve on its own (a
lookup it has no way to know), and process tool calls ONE AT A TIME in
a loop:
  1. Send messages -> model responds.
  2. If the response has tool_calls: run ONLY the first one, append
     its ToolMessage, and go back to step 1 -- this forces the model
     to see each real result before planning the next step.
  3. If the response has NO tool_calls: it's the final text answer,
     stop the loop.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()


# --- Step A: define tools ---
# @tool turns a normal Python function into something the model can call.
# The docstring matters — the model reads it to decide WHEN to use the tool.
#
# lookup_price is deliberately something the model CANNOT know or guess —
# there's no way to "pre-compute" a made-up product price in its head.
# This forces it to actually wait for the tool result before planning
# the next step, unlike plain arithmetic it can do mentally.

PRICES = {"apple": 40, "mango": 90, "laptop": 55000}


@tool
def lookup_price(item: str) -> int:
    """Look up the price (in rupees) of a single item by name."""
    return PRICES.get(item.lower(), -1)


@tool
def add(a: int, b: int) -> int:
    """Add two integers together and return the result."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together and return the result."""
    return a * b


tools = [lookup_price, add, multiply]
tools_by_name = {t.name: t for t in tools}

# --- Step B: bind tools to the model ---
# This tells the model "here are functions you're allowed to call."
# It does NOT execute anything by itself.
# Note: llama-3.x models on Groq are flaky at formatting tool calls
# (malformed output, or silently skipping the tool). "openai/gpt-oss-20b"
# has proven reliable for tool calling in testing.
llm = ChatGroq(model="openai/gpt-oss-20b")
llm_with_tools = llm.bind_tools(tools)

# --- Step C: the question requires TWO sequential, dependent tool calls ---
# The model must: (1) look up mango's price, THEN (2) multiply that
# real result by 3. It cannot skip straight to step 2 because it has
# no way to know mango's price without calling the tool.
messages = [
    HumanMessage(content="Multiplye 12 by 7, then add 5 to that?")
]

# --- Step D: the agent loop ---
# Keep invoking the model and executing whatever tool call comes back,
# ONE STEP AT A TIME, until the model responds with no tool_calls left
# (meaning it's ready to give the final text answer).
max_steps = 5  # safety limit so a bug can't loop forever

for step in range(max_steps):
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    if not ai_response.tool_calls:
        # No more tool calls requested -> this is the final answer.
        print(f"\n[step {step}] Model gave final answer (no more tool calls).")
        break

    print(f"\n[step {step}] Model requested tool call(s): {ai_response.tool_calls}")

    # Execute EVERY tool call the model asked for in this response
    # (usually just one, since we're forcing single-step-at-a-time
    # reasoning by feeding results back immediately).

    print("ai_response.tool_calls: ", ai_response.tool_calls)

    for call in ai_response.tool_calls:
        tool_fn = tools_by_name[call["name"]]
        result = tool_fn.invoke(call["args"])  # runs the REAL Python function
        print(f"    -> ran {call['name']}({call['args']}) = {result}")

        # ToolMessage carries the REAL result back, tagged with
        # tool_call_id so the model knows which request this answers.
        messages.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )
else:
    print("\nHit max_steps without a final answer — check for a loop bug.")

print("\nFull conversation:")
for m in messages:
    detail = m.content or getattr(m, "tool_calls", "")
    print(" -", type(m).__name__, "->", detail)

print("\nFinal answer:", messages[-1].content)
