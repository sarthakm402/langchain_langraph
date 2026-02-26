from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough 
llm="model here"
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context to answer also keep in mind the history so u could answer any further question."),
    ("placeholder", "{history}"),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])
data="ur data here"
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunks=splitter.create_documents([data])
chunks=splitter.create_documents(data)
embeddings=HuggingFaceEmbeddings(
    # model_name="sentence-transformers/all-mpnet-base-v2",
    # model_kwargs={"device": "cuda"},
)
vectorstore=FAISS.from_documents(chunks,embeddings)
retriver=vectorstore.as_retriever(search_kwargs={"k":3})
session_hist={}
def get_history(session_id):
    if session_id not in session_hist:
        session_hist[session_id]=ChatMessageHistory()
    return session_hist[session_id]
rag_chain = (
    {
        "context": retriver,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history"
)
response = chain_with_memory.invoke(
    "What is this document about?",
    config={"configurable": {"session_id": "user1"}}
)

print(response.content)

