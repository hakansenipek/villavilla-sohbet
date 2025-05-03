import os
import logging
import streamlit as st
from pinecone import Pinecone as Pinecone, ServerlessSpec
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

def init_pinecone():
    """Pinecone bağlantısını başlatır ve index nesnesi döner."""
    api_key = os.environ.get("PINECONE_API_KEY")
    environment = os.environ.get("PINECONE_ENVIRONMENT")
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not api_key or not environment or not index_name:
        st.error("Pinecone API bilgileri eksik. secrets.toml dosyasını kontrol edin.")
        logging.error("Eksik Pinecone API bilgisi.")
        return None, None

    try:
        pc = Pinecone(api_key=api_key)

        # Var olan indeksler arasında arama
        indexes = pc.list_indexes().names()
        if index_name not in indexes:
            st.info(f"Pinecone indeksi '{index_name}' oluşturuluyor...")
            pc.create_index(
                name=index_name,
                dimension=3072,  # text-embedding-3-large için
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
            st.success(f"Indeks oluşturuldu: {index_name}")
        
        return pc.Index(index_name), index_name

    except Exception as e:
        st.error(f"Pinecone başlatma hatası: {str(e)}")
        logging.exception("Pinecone init hatası")
        return None, None

def create_or_update_vector_db(documents, namespace="villa_villa"):
    """Belgeleri Pinecone'a ekler veya günceller."""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        index, _ = init_pinecone()
        if not index:
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=400)
        chunks = splitter.split_documents(documents)

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_field="text",
            namespace=namespace
        )

        vectorstore.add_documents(chunks)
        st.success(f"{len(chunks)} belge vektöre dönüştürülüp Pinecone'a yüklendi.")
        return vectorstore

    except Exception as e:
        st.error(f"Vektör veritabanı oluşturulamadı: {str(e)}")
        logging.exception("Vektör oluşturma hatası")
        return None

def load_vector_db_from_pinecone(namespace="villa_villa"):
    """Pinecone'dan daha önce kaydedilmiş veritabanını yükler."""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        index, _ = init_pinecone()
        if not index:
            return None

        stats = index.describe_index_stats()
        if namespace not in stats.get("namespaces", {}) or stats["namespaces"][namespace]["vector_count"] == 0:
            st.warning(f"Pinecone namespace '{namespace}' boş. Önce veri yükleyin.")
            return None

        return PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_field="text",
            namespace=namespace
        )

    except Exception as e:
        st.error(f"Pinecone'dan veritabanı yüklenemedi: {str(e)}")
        logging.exception("Veritabanı yükleme hatası")
        return None
