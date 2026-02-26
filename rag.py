from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda
llm=RunnableLambda(lambda x:"Dummy model")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context to answer also keep in mind the history so u could answer any further question."),
    ("placeholder", "{history}"),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])
data="ur data here"
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunks=splitter.create_documents([data])
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "cpu"},
)
vectorstore=FAISS.from_documents(chunks,embeddings)
retriver=vectorstore.as_retriever(search_kwargs={"k":3})
session_hist={}
max_hist=5
def get_history(session_id):
    if session_id not in session_hist:
        session_hist[session_id]=ChatMessageHistory()
    return session_hist[session_id]
def prune_hist(session_id):
    history=session_hist[session_id]
    if len(history.messages)>max_hist:
        history.messages=history.messages[-max_hist:]
def getretriever_docs(docs):
    print("docs included")
    for i, d in  enumerate(docs):
        print(f"chunk{i+1}")
        print("content :"f"{d.page_content[:100]}")
    return docs
def log_prompt(prompt_value):
    print("\n--- FINAL PROMPT SENT TO LLM ---")
    print(prompt_value.to_string())
    print("\nPrompt length (chars):", len(prompt_value.to_string()))
    return prompt_value
rag_chain = (
    RunnableParallel(
        context=(RunnableLambda(lambda x: x["question"])|retriver|RunnableLambda(getretriever_docs)),
        question=RunnableLambda(lambda x: x["question"]),
        history=RunnableLambda(lambda x: x["history"]))

    | prompt
    |RunnableLambda(log_prompt)
    | llm
)

chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history"
)
response = chain_with_memory.invoke(
   {"question":"What is this document about?"},#this is beacuse our starting is runnable parallel so we need dict as input 
    config={"configurable": {"session_id": "user1"}}
)

# print(response.content) this for when we have actual model
print(response)

