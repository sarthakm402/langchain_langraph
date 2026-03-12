
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
