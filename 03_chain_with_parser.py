"""
STEP 3: Output parsers + a multi-step chain.

response.content (from step 2) is fine, but LangChain gives you parsers
so the chain itself returns clean data instead of a message object.

We also chain THREE steps together: prompt -> llm -> parser.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# --- Example A: plain string output ---
# prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
# str_chain = prompt | llm | StrOutputParser()

# result = str_chain.invoke({"topic": "vector databases"})
# print(type(result), "->", result)  # now a plain str, not an AIMessage



# print("---")

# leere

# --- Example B: structured output (a Python list) ---
list_parser = CommaSeparatedListOutputParser()

list_prompt = ChatPromptTemplate.from_template(
    "List 5 popular Python web frameworks. {format_instructions}"
)

list_chain = list_prompt | llm | list_parser

frameworks = list_chain.invoke(
    {"format_instructions": list_parser.get_format_instructions()}
)
print(type(frameworks), "->", frameworks)  # a real Python list
print("First framework:", frameworks[0])
