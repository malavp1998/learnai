"""
STEP 14: A minimal AGENT HARNESS — the loop + permission gate + tool
executor that turns a plain LLM into something that can actually ACT on
your machine, built from scratch to show what tools like Claude Code,
Cursor, and Aider actually are under the hood.

WHAT "HARNESS" MEANS: an LLM by itself is text-in, text-out — it cannot
read a file, run a command, or do anything in the real world. A harness
is the surrounding program that (1) loops the model against tools,
(2) actually EXECUTES what the model asks for, (3) decides what's safe
to run automatically vs. what needs a human's okay, and (4) feeds real
results back so the model can plan its next step. Every AI coding agent
you've heard of is this loop, industrialized.

WHY THIS ISN'T JUST 05_tool_calling.py AGAIN:
05_tool_calling.py's loop calls tools blindly the instant the model
requests them (multiply, add, a price lookup — all harmless, read-only,
side-effect-free). A real harness can't do that, because real tools
have real consequences: reading an arbitrary file could leak secrets,
running a shell command could delete data. This file adds the piece
05 skips entirely: a PERMISSION GATE between "the model requested this"
and "this actually runs" — the same boundary Claude Code itself has
(that's why you get asked to approve certain tool calls and not others).

THE THREE TOOLS, deliberately chosen to represent three trust tiers:
  - list_directory : read-only, low-risk        -> AUTO-APPROVED
  - read_file       : read-only, but path-sensitive
                       (could read secrets)      -> AUTO-APPROVED if
                                                     inside this project,
                                                     BLOCKED otherwise
  - run_shell_command : can mutate/delete state  -> ALWAYS asks for
                                                     human confirmation,
                                                     and REFUSES a
                                                     denylist of
                                                     destructive patterns
                                                     even if approved

THE LOOP (same shape as 05_tool_calling.py's fix, generalized):
  1. Send messages -> model responds.
  2. If tool_calls: for EACH one, run it through the permission gate,
     execute (or refuse), append a REAL ToolMessage with the REAL
     result (or refusal reason), go back to step 1.
  3. If no tool_calls: final answer, stop.
  4. max_steps guards against runaway loops -- same safety net as 05.

WHAT THIS FILE IS NOT: a production harness. Real ones (Claude Code
included) add sandboxing, structured permission modes (auto-accept
vs. ask vs. deny per tool), audit logs, cost/token tracking, and far
more tools. This is the minimal skeleton that makes the core mechanism
visible -- the loop, the executor, and the gate -- without the scale.
"""

import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.resolve()

# A destructive-command denylist -- this is the harness saying "even if
# a human approves, some things are never allowed to run as-is." Real
# harnesses do this at multiple layers (this is the simplest one:
# substring matching on the command text before execution).
DENYLISTED_COMMAND_PATTERNS = ["rm -rf", "sudo", ">:", "mkfs", ":(){ :|:& };:"]


# --- Step A: define the tools ---
# Same @tool decorator as 05_tool_calling.py -- the model only ever sees
# the name + docstring + argument schema. It NEVER executes these
# functions itself; that stays entirely in the harness's control (Step D).

@tool
def list_directory(path: str = ".") -> str:
    """List files and folders inside a directory relative to the project root."""
    target = (PROJECT_ROOT / path).resolve()
    if not str(target).startswith(str(PROJECT_ROOT)):
        return "ERROR: path escapes the project root, refused."
    if not target.exists():
        return f"ERROR: {path} does not exist."
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return "\n".join(entries) if entries else "(empty directory)"


@tool
def read_file(path: str) -> str:
    """Read and return the text contents of a file, given a path relative to the project root."""
    target = (PROJECT_ROOT / path).resolve()
    if not str(target).startswith(str(PROJECT_ROOT)):
        return "ERROR: path escapes the project root, refused."
    if not target.exists() or not target.is_file():
        return f"ERROR: {path} is not a file that exists."
    return target.read_text(errors="replace")[:2000]  # capped so one huge file can't blow the context


@tool
def run_shell_command(command: str) -> str:
    """Run a read-only shell command (e.g. 'ls', 'wc -l file.txt', 'git log --oneline -5') inside the project root and return its output."""
    for bad_pattern in DENYLISTED_COMMAND_PATTERNS:
        if bad_pattern in command:
            return f"REFUSED: command matches denylisted pattern {bad_pattern!r}."
    result = subprocess.run(
        command, shell=True, cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=10,
    )
    output = (result.stdout + result.stderr).strip()
    return output[:1500] if output else "(no output)"


TOOLS = [list_directory, read_file, run_shell_command]
tools_by_name = {t.name: t for t in TOOLS}

# Which tools the harness will run WITHOUT asking a human -- the actual
# permission POLICY. This is the piece 05_tool_calling.py has no
# equivalent of: not every tool call is equally safe to auto-execute.
AUTO_APPROVED_TOOLS = {"list_directory", "read_file"}


llm = ChatGroq(model="openai/gpt-oss-20b").bind_tools(TOOLS)


# --- Step B: the permission gate ---
# Sits BETWEEN "the model requested this" and "this actually runs." This
# function is the harness's real job -- everything else here is plumbing
# around it. A production harness makes this configurable (auto-accept
# mode, ask-every-time mode, project-level allow/deny lists) -- this is
# the simplest version: a fixed set of auto-approved tool NAMES, and a
# human `input()` prompt for everything else.
def request_permission(tool_name: str, args: dict) -> bool:
    if tool_name in AUTO_APPROVED_TOOLS:
        print(f"  [auto-approved] {tool_name}({args})")
        return True

    print(f"  [permission needed] {tool_name}({args})")
    answer = input("  Allow this? [y/N]: ").strip().lower()
    return answer == "y"


# --- Step C: the tool executor ---
# The ONLY place any tool actually runs. Separated from the loop below so
# the gate-then-execute sequence is explicit and impossible to skip.
def execute_tool_call(call: dict) -> str:
    name, args = call["name"], call["args"]

    if not request_permission(name, args):
        return "REFUSED: user denied permission for this tool call."

    try:
        return str(tools_by_name[name].invoke(args))
    except Exception as e:
        return f"ERROR running {name}: {e}"


# --- Step D: the agent loop itself ---
# Structurally identical to 05_tool_calling.py's fixed loop -- invoke,
# check tool_calls, execute for real, feed the real result back, repeat
# until no more tool_calls or max_steps is hit. The only new piece is
# that execute_tool_call() now runs through the permission gate instead
# of calling the tool unconditionally.
def run_agent(task: str, max_steps: int = 6):
    messages = [HumanMessage(content=task)]
    print(f"TASK: {task}\n")

    for step in range(1, max_steps + 1):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"\nFINAL ANSWER:\n{response.content}")
            return response.content

        print(f"--- step {step}: model requested {len(response.tool_calls)} tool call(s) ---")
        for call in response.tool_calls:
            result = execute_tool_call(call)
            preview = result if len(result) <= 200 else result[:200] + "...(truncated)"
            print(f"  -> {preview}\n")
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    print("\n[stopped: max_steps reached without a final answer]")
    return None


if __name__ == "__main__":
    # A task that requires CHAINING tools: list files to discover what
    # exists, then read one, then summarize -- the model can't answer
    # this from its own knowledge, it has to actually use the tools and
    # see real results, same "can't pre-solve it mentally" principle as
    # 05_tool_calling.py's lookup_price example.
    run_agent(
        "Look at the docs/ directory, then read whichever file's name "
        "sounds most related to space or planets, and tell me one fact "
        "from it."
    )
