# streamlit_app.py

import os
import sys
import logging
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from docx import Document as DocxDocument

# ---------------------------------------
# 1. Loglama Ayarları
# ---------------------------------------
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------
# 2. Streamlit Sayfa Yapısı - Sadeleştirilmiş
# ---------------------------------------
st.set_page_config(page_title="Villa Villa Yapay Zeka", layout="wide")

# Başlık bölümü
col1, col2 = st.columns([1, 5])
with col1:
    try:
        st.image("assets/villa_villa_logo.jpg", width=100)
    except Exception:
        pass
with col2:
    st.title("Villa Villa Yapay Zeka ile Sohbet")

st.markdown("---")

# ---------------------------------------
# 3. OpenAI API Anahtarı Yönetimi (Sadece Secrets)
# ---------------------------------------
def set_openai_api_key():
    # Secrets'dan API anahtarını al
    openai_api_key = st.secrets.get("openai", {}).get("api_key", None)
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        return True
    
    # API anahtarı bulunamadı
    st.error("API anahtarı bulunamadı. Lütfen Streamlit Secrets ayarlarını kontrol edin.")
    return False

# ---------------------------------------
# 4. Belgeleri Yükleme Fonksiyonu (.docx)
# ---------------------------------------
def load_documents_from_folder(folder_path="data"):
    documents = []
    if not os.path.exists(folder_path):
        st.error(f"Belge klasörü bulunamadı: {folder_path}")
        return documents

    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            full_path = os.path.join(folder_path, filename)
            try:
                docx = DocxDocument(full_path)
                text = "\n".join([p.text for p in docx.paragraphs if p.text.strip() != ""])
                documents.append(Document(page_content=text, metadata={"source": filename}))
                print(f"Belge yüklendi: {filename}, İçerik uzunluğu: {len(text)} karakter")
            except Exception as e:
                logging.error(f"{filename} yüklenirken hata: {str(e)}")
    
    return documents

# ---------------------------------------
# 5. Vektör Veritabanı Oluşturma
# ---------------------------------------
def create_vector_db(documents):
    try:
        # Belgeleri parçalara böl
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        print(f"Belgeler {len(chunks)} parçaya bölündü")
        
        # Embeddings oluştur
        try:
            # API anahtarını kontrol et
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                print("API anahtarı bulunamadı!")
                return None
                
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
            print("OpenAIEmbeddings başarıyla oluşturuldu")
            
            # Vektör veritabanı oluştur
            vector_db = DocArrayInMemorySearch.from_documents(
                documents=chunks,
                embedding=embeddings,
            )
            print("Vektör veritabanı başarıyla oluşturuldu")
            return vector_db
            
        except Exception as e:
            import traceback
            error_msg = f"Embedding hatası: {str(e)}"
            print(error_msg)
            print(f"Hata detayı: {traceback.format_exc()}")
            return None
            
    except Exception as e:
        logging.error(f"Vektör veritabanı hatası: {str(e)}")
        return None

# ---------------------------------------
# 6. Özel Prompt Şablonu
# ---------------------------------------
def create_qa_prompt():
    template = """
    Sen Villa Villa şirketinin finans ve operasyon asistanısın. Aşağıdaki belgelerden alınan bilgilere dayanarak soruları yanıtla:

    Belgelerden Bilgiler:
    {context}

    Sohbet Geçmişi:
    {chat_history}

    Soru:
    {question}

    Yanıtın sadece belgelerden aldığın bilgilere dayanmalı. Eğer belgede yanıt yoksa, "Bu konuda mevcut belgelerimde bilgi bulamadım" şeklinde belirt. Tahmin yürütme, bilmediğin konularda uydurma.

    Yanıt:
    """
    return PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)

# ---------------------------------------
# 7. Chat Zinciri Kurulumu
# ---------------------------------------
def create_chat_chain(vector_db):
    try:
        llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4-turbo"
        )
        
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 15}
        )
        qa_prompt = create_qa_prompt()
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        return chain
    except Exception as e:
        logging.error(f"Chat zinciri hatası: {str(e)}")
        return None

# ---------------------------------------
# 8. Ana Uygulama
# ---------------------------------------
def main():
    # Global API anahtarı ayarla
    if not set_openai_api_key():
        st.stop()

    # Session state değişkenleri
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "documents" not in st.session_state or "vector_db" not in st.session_state or "chat_chain" not in st.session_state:
        # Belgeleri yükle
        with st.spinner("Belgeler yükleniyor..."):
            documents = load_documents_from_folder("data")
            if not documents:
                st.error("Hiç belge bulunamadı! Lütfen 'data' klasörüne .docx belgelerinizi ekleyin.")
                st.stop()
            
            st.session_state.documents = documents
        
        # Vektör veritabanı oluştur
        with st.spinner("Vektör veritabanı oluşturuluyor..."):
            vector_db = create_vector_db(documents)
            if not vector_db:
                st.error("Vektör veritabanı oluşturulamadı!")
                st.stop()
            
            st.session_state.vector_db = vector_db
        
        # Chat zinciri oluştur
        with st.spinner("Sohbet sistemi hazırlanıyor..."):
            chat_chain = create_chat_chain(vector_db)
            if not chat_chain:
                st.error("Sohbet sistemi oluşturulamadı!")
                st.stop()
            
            st.session_state.chat_chain = chat_chain
    
    # Sohbet geçmişini görüntüle - Merkezi konumlandırma
    chat_container = st.container()
    with chat_container:
        for i in range(0, len(st.session_state.chat_history), 2):
            if i < len(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.markdown(st.session_state.chat_history[i][1])
            
            if i+1 < len(st.session_state.chat_history):
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.chat_history[i+1][1])
    
    # Kullanıcı girişi
    user_input = st.chat_input("Sorunuzu yazınız...")
    
    # Temizleme butonları - Chat altında
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Sohbeti Temizle", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🔄 Önbelleği Temizle", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.chat_history.append(("user", user_input))
        
        try:
            # Sohbet geçmişini uygun formata dönüştür
            chat_formatted = []
            for i in range(0, len(st.session_state.chat_history)-1, 2):
                if i+1 < len(st.session_state.chat_history):
                    chat_formatted.append((st.session_state.chat_history[i][1], 
                                        st.session_state.chat_history[i+1][1]))
            
            # Yanıt oluştur
            with st.spinner("Düşünüyor..."):
                response = st.session_state.chat_chain({
                    "question": user_input,
                    "chat_history": chat_formatted
                })
            
            # Yanıtı göster
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            
            # Yanıtı geçmişe ekle
            st.session_state.chat_history.append(("assistant", response["answer"]))
            
        except Exception as e:
            logging.error(f"Yanıt hatası: {str(e)}")
            with st.chat_message("assistant"):
                st.error("Üzgünüm, yanıt oluşturulurken bir hata oluştu.")
            st.session_state.chat_history.append(("assistant", "Üzgünüm, bir hata oluştu."))

# ---------------------------------------
# 9. Uygulama Başlatılıyor
# ---------------------------------------
if __name__ == "__main__":
    main()