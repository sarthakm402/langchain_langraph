
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import FAISS
"""react agent using react wrapper"""
# @tool
# def calculator(expression: str) -> str:
#     return str(eval(expression))
# vector_store=FAISS.load_local("the local folder path yaha",allow_dangerous_deserialization=True)
# retriver=vector_store.as_retriever(search_kwargs={"k":3})
# @tool
# def retrieve(query):
#     docs=retriver.invoke(query)
#     return "\n\n".join([doc.page_content for doc in docs])
# llm = "model here"

# agent = create_react_agent(
#     model=llm,
#     tools=[calculator,retrieve]
# )

# query = {"messages": [("user", "What is 12 * 18?")]}

# try:
#     response = agent.invoke(query,config={"recursion_limit": 5})
#     print(response["messages"][-1].content)
# except Exception as e:
#     print("Agent error:", e)

"""manual implementation"""
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class AgentState(TypedDict):
    messages: List

@tool
def calculator(expression: str) -> str:
    return str(eval(expression))

vector_store = FAISS.load_local(
    "your_local_folder_path",
    embeddings=None,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

@tool
def retrieve(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

llm = "llm here"
model = llm.bind_tools([calculator, retrieve])

def reason(state: AgentState):
    response = model.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def run_tool(state: AgentState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    args = tool_call["args"]

    if tool_name == "calculator":
        result = calculator.invoke(args)
    elif tool_name == "retrieve":
        result = retrieve.invoke(args)

    return {"messages": state["messages"] + [result]}

def route_tools(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return last_message.tool_calls[0]["name"]
    return "finish"

graph = StateGraph(AgentState)

graph.add_node("reason", reason)
graph.add_node("tool_node", run_tool)

graph.set_entry_point("reason")

graph.add_conditional_edges(
    "reason",
    route_tools,
    {
        "calculator": "tool_node",
        "retrieve": "tool_node",
        "finish": END
    }
)

graph.add_edge("tool_node", "reason")

agent = graph.compile()

agent.invoke({
    "messages": [("user", "what is 12 * 18")]
})
   
   


