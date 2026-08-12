"""
STEP 7: LangGraph — a multi-agent system (researcher -> writer),
routed by a supervisor, with conditional edges and a loop.

This is the thing plain LangChain chains can't express: a SUPERVISOR
node looks at shared STATE and decides, at runtime, which agent runs
next. That routing decision is a conditional edge. The graph can send
control back to the supervisor after each agent runs -- that's the
loop -- until the supervisor decides the work is done.

Flow:

    START -> supervisor --(route: "researcher")--> researcher --+
                 ^                                               |
                 |                                               |
                 +---------------(back to supervisor)------------+
                 |
                 +--(route: "writer")--> writer --+
                 |                                 |
                 +---------------------------------+
                 |
                 +--(route: "done")--> END

Every node reads and writes the SAME shared state dict. That's how
the two agents "talk" to each other -- not via direct function calls,
but by both reading/writing a common state object that flows through
the graph.
"""

from dotenv import load_dotenv
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")


# --- Step A: shared state ---
# Every node receives this whole dict and returns a PARTIAL update.
# LangGraph merges the update into state itself (you don't do it).
#
# - messages: full conversation trail, using add_messages so new
#   messages APPEND instead of overwriting (like a reducer).
# - research_notes / draft: plain fields that get OVERWRITTEN each
#   time a node returns them (no reducer = last write wins).
# - next_step: what the supervisor decided, read by the conditional edge.
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    research_notes: str
    draft: str
    next_step: str


# --- Step B: the researcher agent ---
# Reads state["topic"], writes state["research_notes"].
# It does NOT know or care who reads its output afterward.
def researcher_node(state: AgentState) -> dict:
    prompt = (
        f"You are a research agent. Gather 3-4 concise, factual bullet "
        f"points about: {state['topic']}. No fluff, just facts."
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    return {
        "research_notes": response.content,
        "messages": [response],
        "next_step": "supervisor",
    }


# --- Step C: the writer agent ---
# Reads state["research_notes"] (written by the researcher) and
# state["topic"] -- this is the "hand-off": the writer never talked
# to the researcher directly, it just reads what landed in shared state.
def writer_node(state: AgentState) -> dict:
    prompt = (
        f"You are a writer agent. Using ONLY these research notes, write "
        f"a short, engaging 3-sentence summary about {state['topic']} "
        f"for a general audience.\n\nResearch notes:\n{state['research_notes']}"
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    return {
        "draft": response.content,
        "messages": [response],
        "next_step": "supervisor",
    }


# --- Step D: the supervisor agent ---
# Decides routing based on what's already in state. This is plain
# Python logic here (cheap + deterministic); it could equally be an
# LLM call that outputs a routing decision as structured output.
def supervisor_node(state: AgentState) -> dict:
    if not state.get("research_notes"):
        decision = "researcher"
    elif not state.get("draft"):
        decision = "writer"
    else:
        decision = "done"
    return {"next_step": decision}


# --- Step E: the conditional edge function ---
# LangGraph calls this AFTER supervisor_node runs. Its return value
# must match one of the keys in the routing map passed to
# add_conditional_edges below.
def route_from_supervisor(state: AgentState) -> Literal["researcher", "writer", "done"]:
    return state["next_step"]


# --- Step F: build the graph ---
graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)

graph.add_edge(START, "supervisor")

# Conditional edge: supervisor's decision picks the next node.
graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "writer": "writer",
        "done": END,
    },
)

# Fixed edges: both agents ALWAYS report back to the supervisor.
# This is the loop -- control returns upstream instead of ending.
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")

app = graph.compile()


# --- Step G: run it ---
if __name__ == "__main__":
    initial_state = {
        "messages": [HumanMessage(content="Write about the Indian Space Research Organisation (ISRO)")],
        "topic": "the Indian Space Research Organisation (ISRO)",
        "research_notes": "",
        "draft": "",
        "next_step": "",
    }

    final_state = app.invoke(initial_state)

    print("--- Research notes (from researcher) ---")
    print(final_state["research_notes"])
    print("\n--- Final draft (from writer) ---")
    print(final_state["draft"])
