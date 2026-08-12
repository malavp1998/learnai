"""
STEP 4: Conversation memory.

By default an LLM call is stateless — each .invoke() has no idea what
you asked before. To build a chatbot, WE keep the history and resend
it every time. LangChain's message types (HumanMessage/AIMessage) give
us a standard way to represent that history.

Run this and chat in your terminal. Type 'quit' to exit.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# The running conversation history, sent in full on every turn.
history = [
    SystemMessage(content="You are a friendly, concise AI tutor for a coding beginner."),
]

print("Chat with the AI tutor (type 'quit' to exit)\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in {"quit", "exit"}:
        break

    history.append(HumanMessage(content=user_input))

    # We pass the WHOLE history, not just the latest message —
    # that's what gives the model "memory" of earlier turns.
    response = llm.invoke(history)

    print(f"AI: {response.content}\n")

    # Save the AI's reply too, so the next turn includes it.
    history.append(AIMessage(content=response.content))
    # print(history)
