from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
model="model here"
prompt=ChatPromptTemplate.from_messages(
    [
        ("human","Summarise the {topic}"),
        ("system","the summary of the topic is {summary}")  
    ]
)
session_hist={}
def get_history(session_id):
    if session_id not in session_hist:
        session_hist[session_id]=ChatMessageHistory
    return session_hist[session_id]
data="ur data here"
splitter=RecursiveCharacterTextSplitter(chunk_size=500 chunk_overlap=50)
chunks=splitter.create_documents(data)
embeddings=HuggingFaceEmbeddings(
    # model_name="sentence-transformers/all-mpnet-base-v2",
    # model_kwargs={"device": "cuda"},
)
vectorstore=FAISS(chunks,embeddings)
retriver=vectorstore.as_retriever(search_kwargs={"k":3})



