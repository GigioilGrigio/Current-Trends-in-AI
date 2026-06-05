from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# BASIC API CALL

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
)

response = llm.invoke("Write a short poem about AI and Python")
print(response.content)

# LLM CHAIN CALLS

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template(
    "What is a good name for a company that makes {product}?"
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"product": "AI tools for doctors"})
print(result)

topic_prompt = PromptTemplate.from_template(
    "Write an email subject line for this topic: {topic}"
)

email_chain = topic_prompt | llm | StrOutputParser()

print(email_chain.invoke({"topic": "AI automation in healthcare"}))

# MEMORY

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}


def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chat = RunnableWithMessageHistory(
    llm,
    get_history,
)

response = chat.invoke(
    "Hi! My name is Alex.", config={"configurable": {"session_id": "user-1"}}
)

print(response.content)

# TOOLS
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


tools = [multiply]
llm_with_tools = llm.bind_tools(tools)


response = llm_with_tools.invoke("What is 23 multiplied by 17?")

messages = [HumanMessage(content="What is 23 multiplied by 17?"), response]

if response.tool_calls:
    for tool_call in response.tool_calls:
        if tool_call["name"] == "multiply":
            result = multiply.invoke(tool_call["args"])

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    final_response = llm_with_tools.invoke(messages)
    print(final_response.content)
else:
    print(response.content)


# AGENTS


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage


tools = [multiply, add]
literary_agent = create_agent(
    model="gemini-2.5-flash-lite",
    tools=tools,
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with solving basic math problems.",
            },
        ]
    ),
)

result = literary_agent.invoke({"messages": [HumanMessage("What is (23 * 17) + 10?")]})
result
