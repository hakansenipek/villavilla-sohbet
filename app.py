# streamlit_app.py

import os
import sys
import tempfile
import logging
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

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
# 2. Streamlit Sayfa Yapısı
# ---------------------------------------
st.set_page_config(page_title="Villa Villa Yapay Zeka", layout="centered")

col1, col2 = st.columns([1, 3])
with col1:
    try:
        st.image("assets/villa_villa_logo.jpg", width=100)
    except Exception:
        pass
with col2:
    st.title("Villa Villa Yapay Zeka ile Sohbet")

st.markdown("---")

# ---------------------------------------
# 3. API Anahtarı Yönetimi
# ---------------------------------------
openai_api_key = st.secrets.get("openai", {}).get("api_key", None)
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.error("API anahtarı girilmedi. Lütfen API anahtarınızı girin.")
        st.stop()

# ---------------------------------------
# 4. Belgeleri Yükleme Fonksiyonu
# ---------------------------------------
def load_documents():
    yapilan_isler = """
    Villa Villa Organizasyon - Yapılan İşler

    Tarih: 15 Nisan 2025
    Müşteri: ABC Şirketi
    Etkinlik Türü: Kurumsal Yıl Dönümü
    Kişi Sayısı: 150
    Menü: Ana yemek, 5 çeşit meze, 2 çeşit tatlı
    Toplam Maliyet: 75,000 TL
    """
    
    genel_gider = """
    Villa Villa Organizasyon - Genel Giderler

    Nisan 2025:
    Kira: 15,000 TL
    Elektrik: 2,500 TL
    Su: 800 TL
    İnternet: 600 TL
    Personel Maaşları: 45,000 TL
    Toplam: 63,900 TL
    """
    
    docs = [
        Document(page_content=yapilan_isler, metadata={"source": "yapilan_isler"}),
        Document(page_content=genel_gider, metadata={"source": "genel_gider"})
    ]
    return docs

# ---------------------------------------
# 5. Vektör Veritabanı (Chroma)
# ---------------------------------------
def create_vector_db(documents):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        persist_directory = tempfile.mkdtemp()
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        return vector_db
    
    except Exception as e:
        logging.error(str(e))
        st.error("Vektör veritabanı oluşturulamadı.")
        return None

# ---------------------------------------
# 6. Prompt Şablonu
# ---------------------------------------
def create_qa_prompt():
    template = """
    Sen Villa Villa şirketinin yapay zekâ destekli bir asistanısın. Aşağıdaki bilgiler ışığında soruları yanıtla:

    Belgeler:
    {context}

    Sohbet Geçmişi:
    {chat_history}

    Soru:
    {question}

    Talimatlar:
    - Yalnızca belgelerde verilen bilgiye dayan.
    - Bilgi yoksa 'Bu konuda elimde veri bulunmamaktadır.' de.
    - Her zaman nazik ve profesyonel bir dil kullan.

    Cevabın:
    """
    return PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)

# ---------------------------------------
# 7. Chat Chain Kurulumu
# ---------------------------------------
def create_chat_chain(vector_db):
    llm = ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4-0125-preview"
    )
    
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    qa_prompt = create_qa_prompt()
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    
    return chain

# ---------------------------------------
# 8. Ana Uygulama
# ---------------------------------------
def main():
    # Giriş yapıldı mı kontrol
    if "authenticated" not in st.session_state:
        password = st.sidebar.text_input("Yönetici Girişi (şifre)", type="password")
        if password == "villavilla123":  # Basit yönetici şifresi
            st.session_state.authenticated = True
        else:
            st.warning("Yönetici girişini yapınız.")
            st.stop()
    
    # Belgeleri yükle ve veritabanı oluştur
    with st.spinner("Belgeler yükleniyor..."):
        documents = load_documents()
        vector_db = create_vector_db(documents)
    
    if not vector_db:
        st.error("Veritabanı oluşturulamadı.")
        st.stop()
    
    chat_chain = create_chat_chain(vector_db)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.success("Sisteme başarıyla giriş yaptınız. Şimdi sorularınızı sorabilirsiniz.")
    
    # Sohbet geçmişini göster
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)
    
    # Sohbet kutusu
    user_input = st.chat_input("Sormak istediğiniz bir şey var mı?")
    
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.chat_history.append(("user", user_input))
        
        try:
            chat_formatted = []
            for i in range(0, len(st.session_state.chat_history)-1, 2):
                if i+1 < len(st.session_state.chat_history):
                    chat_formatted.append((st.session_state.chat_history[i][1], st.session_state.chat_history[i+1][1]))
            
            response = chat_chain({
                "question": user_input,
                "chat_history": chat_formatted
            })
            
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            
            st.session_state.chat_history.append(("assistant", response["answer"]))
        
        except Exception as e:
            logging.error(str(e))
            with st.chat_message("assistant"):
                st.error("Yanıt üretilemedi. Lütfen tekrar deneyiniz.")

# ---------------------------------------
# 9. Uygulamayı Başlat
# ---------------------------------------
if __name__ == "__main__":
    main()

