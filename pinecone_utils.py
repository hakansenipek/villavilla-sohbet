from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os, logging, streamlit as st

def init_pinecone():
    api_key = os.environ.get("PINECONE_API_KEY")
    environment = os.environ.get("PINECONE_ENVIRONMENT")
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not api_key or not environment or not index_name:
        st.error("Pinecone API bilgileri eksik.")
        return None, None

    pc = PineconeClient(api_key=api_key)

    indexes = pc.list_indexes()
    if index_name not in indexes:
        st.info(f"Pinecone indeksi '{index_name}' oluşturuluyor...")
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
        st.success(f"Indeks oluşturuldu: {index_name}")
    
    return pc.Index(index_name), index_name

def create_or_update_vector_db(documents, namespace="villa_villa"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index, _ = init_pinecone()
    if not index: return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=400)
    chunks = splitter.split_documents(documents)

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_field="text", namespace=namespace)
    vectorstore.add_documents(chunks)

    return vectorstore

def load_vector_db_from_pinecone(namespace="villa_villa"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index, _ = init_pinecone()
    if not index: return None

    stats = index.describe_index_stats()
    if namespace not in stats.get("namespaces", {}) or stats["namespaces"][namespace]["vector_count"] == 0:
        return None

    return PineconeVectorStore(index=index, embedding=embeddings, text_field="text", namespace=namespace)
