"""
STEP 2: Prompt templates + chaining.

Instead of hardcoding a string, we define a reusable template with
placeholders. Then we "chain" the template into the model using the
`|` (pipe) operator: output of the left side becomes input to the right.

This pipe syntax is called LCEL (LangChain Expression Language).
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# {topic} and {audience} are placeholders filled in later.
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one simple sentence for a {audience}."
)

# prompt | llm builds a chain: fill the template -> send it to the model.
chain = prompt | llm

response = chain.invoke({"topic": "recursion", "audience": "5 year old"})
print(response.content)

print("---")

# Same chain, reused with different inputs — this is the whole point
# of templates: define the shape once, run it many times.
response2 = chain.invoke({"topic": "neural networks", "audience": "college student"})
print(response2.content)
