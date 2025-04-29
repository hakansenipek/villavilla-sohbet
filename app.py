# app.py

import os
import sys
import logging
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from docx import Document as DocxDocument
import pandas as pd
import time
import io
import requests
import re

# ---------------------------------------
# 1. Loglama Ayarları
# ---------------------------------------
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Hata loglaması için özel handler
error_handler = logging.FileHandler("logs/error.log")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(error_handler)

# ---------------------------------------
# 2. Streamlit Sayfa Yapısı
# ---------------------------------------
st.set_page_config(page_title="Villa Villa Yapay Zeka", layout="wide", 
                   initial_sidebar_state="expanded")

# CSS ile özelleştirme
st.markdown("""
<style>
    .main-header {
        display: flex;
        align-items: center;
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .chat-message-user {
        background-color: #e6f7ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .chat-message-assistant {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Başlık bölümü
with st.container():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 5])
    with col1:
        try:
            st.image("assets/villa_villa_logo.jpg", width=100)
        except Exception as e:
            logging.error(f"Logo yüklenirken hata: {str(e)}")
    with col2:
        st.title("Villa Villa Yapay Zeka ile Sohbet")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------
# 3. OpenAI API Anahtarı Yönetimi
# ---------------------------------------
def load_api_keys():
    # OpenAI API anahtarını al
    openai_api_key = st.secrets.get("openai", {}).get("api_key", None)
    if not openai_api_key:
        st.error("OpenAI API anahtarı bulunamadı. Lütfen Streamlit Secrets ayarlarını kontrol edin.")
        logging.error("OpenAI API anahtarı bulunamadı")
        return False
    
    os.environ["OPENAI_API_KEY"] = openai_api_key
    return True

# ---------------------------------------
# 4. Google Drive Doküman İndirme
# ---------------------------------------
def extract_document_id(url):
    """Google Doküman URL'sinden doküman ID'sini çıkarır."""
    pattern = r"/d/([a-zA-Z0-9-_]+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

def download_google_doc_as_text(doc_url):
    """Google Dokümanı metin olarak indirir."""
    try:
        doc_id = extract_document_id(doc_url)
        if not doc_id:
            logging.error(f"Geçersiz Google Doküman URL'si: {doc_url}")
            return None, None
        
        # Google Docs'un export URL'si
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        
        # Dokümanı indir
        response = requests.get(export_url)
        if response.status_code != 200:
            logging.error(f"Doküman indirilemedi. Durum kodu: {response.status_code}")
            return None, None
        
        # Dosya adını al (URL'den tahmin et)
        file_name = f"{doc_id}.txt"
        
        return response.text, file_name
        
    except Exception as e:
        logging.error(f"Doküman indirme hatası: {str(e)}")
        return None, None

def load_documents_from_urls(doc_urls):
    """Verilen URL listesinden Google Dokümanları yükler."""
    documents = []
    
    # Yükleme durumu göstergesi
    progress_bar = st.progress(0)
    st.write(f"Google Drive'dan {len(doc_urls)} doküman yükleniyor...")
    
    for idx, (doc_name, doc_url) in enumerate(doc_urls.items()):
        try:
            # Dokümanı indir
            doc_content, file_name = download_google_doc_as_text(doc_url)
            if not doc_content:
                st.warning(f"{doc_name} dokümanı indirilemedi. URL'yi kontrol edin.")
                continue
            
            # Dokümanı LangChain formatına dönüştür
            documents.append(Document(
                page_content=doc_content, 
                metadata={"source": doc_name, "url": doc_url}
            ))
            
            logging.info(f"Doküman yüklendi: {doc_name}, İçerik uzunluğu: {len(doc_content)} karakter")
            
            # İlerleme durumunu güncelle
            progress_bar.progress((idx + 1) / len(doc_urls))
            
        except Exception as e:
            logging.error(f"{doc_name} işlenirken hata: {str(e)}")
    
    return documents

# ---------------------------------------
# 5. Vektör Veritabanı Oluşturma
# ---------------------------------------
def create_vector_db(documents):
    try:
        # Belgeleri parçalara böl
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500, 
            chunk_overlap=400,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        logging.info(f"Belgeler {len(chunks)} parçaya bölündü")
        
        # Chunk örneklerini ve veri dağılımını logla
        if chunks:
            logging.info(f"Örnek chunk içeriği (ilk 200 karakter): {chunks[0].page_content[:200]}")
            chunk_lengths = [len(chunk.page_content) for chunk in chunks]
            logging.info(f"Chunk uzunluk istatistikleri - Min: {min(chunk_lengths)}, Max: {max(chunk_lengths)}, Ortalama: {sum(chunk_lengths)/len(chunk_lengths)}")
        
        # Embeddings oluştur
        try:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logging.error("API anahtarı bulunamadı!")
                return None
                
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
            logging.info("OpenAIEmbeddings başarıyla oluşturuldu")
            
            # Vektör veritabanı oluştur
            vector_db = DocArrayInMemorySearch.from_documents(
                documents=chunks,
                embedding=embeddings,
            )
            logging.info(f"Vektör veritabanı başarıyla oluşturuldu, {len(chunks)} adet vektör içeriyor")
            return vector_db
            
        except Exception as e:
            import traceback
            error_msg = f"Embedding hatası: {str(e)}"
            logging.error(error_msg)
            logging.error(f"Hata detayı: {traceback.format_exc()}")
            return None
            
    except Exception as e:
        logging.error(f"Vektör veritabanı hatası: {str(e)}")
        return None

# ---------------------------------------
# 6. Özel Prompt Şablonu
# ---------------------------------------
def create_qa_prompt():
    template = """
    Sen Villa Villa şirketinin finans ve operasyon asistanısın. Verilen bilgilere dayanarak kapsamlı ve doğru yanıtlar vermelisin.

    # Görevin
    - İşletme finansları, tedarikçi bilgileri, giderler, personel bilgileri ve geçmiş işlerle ilgili sorulara cevap ver
    - Tüm cevaplarını aşağıdaki belgelerden aldığın bilgilere dayandır
    - Emin olmadığın konularda tahmin yürütme
    - Bilginin eksik olduğu durumlarda dürüstçe bilmediğini söyle

    # Belgelerden Alınan Bilgiler:
    {context}

    # Sohbet Geçmişi:
    {chat_history}

    # Mevcut Soru:
    {question}

    # Yanıt Kuralları:
    1. Bilgiyi net ve düzgün bir şekilde organize et
    2. Sayısal veriler için tablolar kullanmayı tercih et
    3. Detaylı finansal analizlerde bilgileri kategorilere ayır
    4. Tahmini veriler sunacaksan, bunun tahmini olduğunu açıkça belirt
    5. Bilgi yoksa "Bu konuda mevcut belgelerimde bilgi bulamadım. Villa Villa yönetimine danışmanızı öneririm." de

    # Yanıtını aşağıya yaz:
    """
    return PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)

# ---------------------------------------
# 7. Chat Zinciri Kurulumu
# ---------------------------------------
def create_chat_chain(vector_db):
    try:
        llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4-turbo",
            verbose=True,
            streaming=True
        )
        
        # Retriever - Similarity arama ve daha fazla belge getirme
        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}
        )
        
        qa_prompt = create_qa_prompt()
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=True
        )
        return chain
    except Exception as e:
        logging.error(f"Chat zinciri hatası: {str(e)}")
        return None

# ---------------------------------------
# 8. Ana Uygulama
# ---------------------------------------
def main():
    # Sidebar bilgileri
    with st.sidebar:
        st.subheader("Villa Villa Asistan Hakkında")
        st.info("""Bu yapay zeka asistanı, Villa Villa şirketinin finansal ve 
        operasyonel bilgilerine dayanarak sorularınızı yanıtlamak için tasarlanmıştır.
        Tedarikçi bilgileri, gider analizleri, personel bilgileri ve daha fazlası 
        hakkında sorular sorabilirsiniz.""")
        
        st.subheader("Örnek Sorular")
        st.markdown("""
        - Kasım 2024'teki toplam giderler nelerdir?
        - En büyük tedarikçimiz hangisidir?
        - Personel maaşları ne kadardır?
        - Araç giderleri nelerdir?
        - Metro'dan yapılan alışverişlerin toplam tutarı nedir?
        """)
        
        # Veri yenileme butonu
        refresh_data = st.button("🔄 Verileri Yenile", use_container_width=True)
        
    # API anahtarlarını yükle
    if not load_api_keys():
        st.stop()

    # Session state değişkenleri
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Google Drive doküman URL'leri
    doc_urls = {
	"folder_id": "1E0a7mgsFnCqAjSrDifXfTqFWdgX0O9I-" 
        "gelen_faturalar": "1TfyGyepmojRdD6xd7WD73I8BEKLypwtzGOa9tvVVtPE",
        "genel_gider": "1TkQsG2f9BBIiSiIE_sI-Q2k9cNoD8PzxcSf1qq-izow",
        "personel_giderleri": "1F9xxY5VztoBi7lqH95jQ-TzOTBGH00_5y-ZbcHGZTHI",
        "villa_villa_tanitim": "16rXwlBEkjbH2pEcUgtseuYMhvZONZkitWGhgVtNDJsY",
        "yapilan_isler": "1D6jDry4yEeWEWpluDMqOTmqNuBc449Oc84hcVIEqf1w"
    }
    
    # Veriler yeniden yüklensin mi?
    if refresh_data or "documents" not in st.session_state:
        try:
            # Belgeleri URL'lerden yükle
            with st.spinner("Google Drive'dan belgeler yükleniyor..."):
                documents = load_documents_from_urls(doc_urls)
                if not documents:
                    st.error("Hiçbir doküman yüklenemedi! URL'leri kontrol edin.")
                    st.stop()
                
                st.session_state.documents = documents
                logging.info(f"Toplam {len(documents)} belge Drive'dan yüklendi")
            
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
                
            st.success("Sistem başarıyla yüklendi! Sorunuzu sorabilirsiniz.")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Uygulama başlatılırken bir hata oluştu: {str(e)}")
            logging.error(f"Uygulama başlatılırken bir hata oluştu: {str(e)}")
            st.stop()
    
    # Sohbet geçmişini görüntüle
    chat_container = st.container()
    with chat_container:
        for i in range(0, len(st.session_state.chat_history), 2):
            if i < len(st.session_state.chat_history):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(st.session_state.chat_history[i][1])
            
            if i+1 < len(st.session_state.chat_history):
                with st.chat_message("assistant", avatar="🏛️"):
                    st.markdown(st.session_state.chat_history[i+1][1])
    
    # Kullanıcı girişi
    user_input = st.chat_input("Villa Villa hakkında bir soru sorun...")
    
    # Temizleme butonları
    cols = st.columns(2)
    with cols[0]:
        if st.button("🧹 Sohbeti Temizle", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with cols[1]:
        if st.button("🔄 Önbelleği Yenile", use_container_width=True):
            # Sadece sohbet geçmişini koruyarak sistemi yenile
            chat_history = st.session_state.chat_history
            for key in list(st.session_state.keys()):
                if key != "chat_history":
                    del st.session_state[key]
            st.session_state.chat_history = chat_history
            st.rerun()
    
    if user_input:
        logging.info(f"Kullanıcı sorusu: {user_input}")
        
        with st.chat_message("user", avatar="👤"):
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
            message_placeholder = st.empty()
            with st.chat_message("assistant", avatar="🏛️"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Düşünme animasyonu
                with st.spinner("Villa Villa Asistanı düşünüyor..."):
                    response = st.session_state.chat_chain({
                        "question": user_input,
                        "chat_history": chat_formatted
                    })
                    full_response = response["answer"]
                    
                    # Kaynakları logla
                    if "source_documents" in response:
                        sources = [doc.metadata.get("source", "Bilinmeyen Kaynak") 
                                  for doc in response["source_documents"]]
                        logging.info(f"Yanıt kaynakları: {set(sources)}")
                
                # Yanıtı göster
                message_placeholder.markdown(full_response)
            
            # Yanıtı geçmişe ekle
            st.session_state.chat_history.append(("assistant", full_response))
            
        except Exception as e:
            logging.error(f"Yanıt hatası: {str(e)}")
            with st.chat_message("assistant", avatar="🏛️"):
                st.error("Üzgünüm, yanıt oluşturulurken bir hata oluştu. Lütfen tekrar deneyin veya sorunuzu farklı bir şekilde sorun.")
            st.session_state.chat_history.append(("assistant", "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."))

# ---------------------------------------
# 9. Uygulama Başlatılıyor
# ---------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Kritik uygulama hatası: {str(e)}")
        st.error("Beklenmeyen bir hata oluştu. Lütfen logs klasörünü kontrol edin veya uygulamayı yeniden başlatın.")