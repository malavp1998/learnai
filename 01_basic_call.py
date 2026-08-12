"""
STEP 1: The simplest possible LangChain program.

All LangChain does here is give us a standard `.invoke()` interface
to talk to an LLM. No chains, no prompts, no memory yet.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from .env

# The model wrapper. "llama-3.3-70b-versatile" is a solid free Groq model.
llm = ChatGroq(model="llama-3.3-70b-versatile")





# .invoke() sends a message and blocks until the full reply comes back.
response = llm.invoke("Who is Piyush Malav?")

print(response.content)
