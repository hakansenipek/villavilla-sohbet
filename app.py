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
import pandas as pd
import time

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
# 2. Streamlit Sayfa Yapısı - Geliştirilmiş
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
# 3. OpenAI API Anahtarı Yönetimi (Secrets ve opsiyonel girdi)
# ---------------------------------------
def set_openai_api_key():
    # Secrets'dan API anahtarını al
    openai_api_key = st.secrets.get("openai", {}).get("api_key", None)
    
    # Eğer sidebar'da API anahtarı giriş alanı istiyorsanız
    # with st.sidebar:
    #     user_api_key = st.text_input("OpenAI API Key (İsteğe bağlı)", 
    #                                  type="password", 
    #                                  help="Halihazırda bir API anahtarı tanımlanmış, bu alanı boş bırakabilirsiniz.")
    #     if user_api_key:
    #         openai_api_key = user_api_key
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        return True
    
    # API anahtarı bulunamadı
    st.error("API anahtarı bulunamadı. Lütfen Streamlit Secrets ayarlarını kontrol edin.")
    logging.error("API anahtarı bulunamadı")
    return False

# ---------------------------------------
# 4. Belgeleri Yükleme Fonksiyonu (Genişletilmiş - docx, txt, csv desteği)
# ---------------------------------------
def load_documents_from_folder(folder_path="data"):
    documents = []
    if not os.path.exists(folder_path):
        st.error(f"Belge klasörü bulunamadı: {folder_path}")
        logging.error(f"Belge klasörü bulunamadı: {folder_path}")
        return documents

    # Yükleme durumu göstergesi
    total_files = len([f for f in os.listdir(folder_path) 
                      if f.endswith(('.docx', '.txt', '.csv'))])
    progress_bar = st.progress(0)
    
    for idx, filename in enumerate(os.listdir(folder_path)):
        try:
            full_path = os.path.join(folder_path, filename)
            
            # DOCX Belgesi
            if filename.endswith(".docx"):
                docx = DocxDocument(full_path)
                text = "\n".join([p.text for p in docx.paragraphs if p.text.strip() != ""])
                documents.append(Document(page_content=text, metadata={"source": filename}))
                logging.info(f"Belge yüklendi: {filename}, İçerik uzunluğu: {len(text)} karakter")
            
            # TXT Belgesi
            elif filename.endswith(".txt"):
                with open(full_path, "r", encoding="utf-8") as file:
                    text = file.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
                logging.info(f"Belge yüklendi: {filename}, İçerik uzunluğu: {len(text)} karakter")
            
            # CSV Belgesi - Tablo verisini metin formatına dönüştürme
            elif filename.endswith(".csv"):
                try:
                    df = pd.read_csv(full_path)
                    # CSV yapısını koruyarak metin formatına dönüştürme
                    text = f"# {filename} içeriği:\n\n"
                    text += df.to_string(index=False) + "\n\n"
                    # Sütun bilgilerini ekle
                    text += f"Bu tabloda şu sütunlar bulunmaktadır: {', '.join(df.columns)}\n"
                    documents.append(Document(page_content=text, metadata={"source": filename}))
                    logging.info(f"CSV belgesi yüklendi: {filename}, Satır sayısı: {len(df)}")
                except Exception as e:
                    logging.error(f"CSV belgesi {filename} işlenirken hata: {str(e)}")
            
            # İlerleme durumunu güncelle
            progress_bar.progress((idx + 1) / total_files)
            
        except Exception as e:
            logging.error(f"{filename} yüklenirken hata: {str(e)}")
    
    return documents

# ---------------------------------------
# 5. Vektör Veritabanı Oluşturma (Geliştirilmiş hata yakalama)
# ---------------------------------------
def create_vector_db(documents):
    try:
        # Belgeleri parçalara böl
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=300,
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
# 6. Özel Prompt Şablonu (Türkçe ve Bağlam Odaklı)
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
# 7. Chat Zinciri Kurulumu (Geliştirilmiş Parametreler)
# ---------------------------------------
def create_chat_chain(vector_db):
    try:
        llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4-turbo",
            verbose=True,
            streaming=True
        )
        
        retriever = vector_db.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": 8,  # Geri döndürülecek döküman sayısı
                "fetch_k": 15,  # İlk sorgu için alınacak döküman sayısı
                "lambda_mult": 0.7  # Çeşitlilik faktörü (0-1) - 0.7 dengeli
            }
        )
        qa_prompt = create_qa_prompt()
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=True,
            rephrase_question=True  # Soruyu yeniden formüle et
        )
        return chain
    except Exception as e:
        logging.error(f"Chat zinciri hatası: {str(e)}")
        return None

# ---------------------------------------
# 8. Ana Uygulama (Geliştirilmiş Kullanıcı Arayüzü)
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
    
    # Global API anahtarı ayarla
    if not set_openai_api_key():
        st.stop()

    # Session state değişkenleri
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "documents" not in st.session_state or "vector_db" not in st.session_state or "chat_chain" not in st.session_state:
        try:
            # Belgeleri yükle
            with st.spinner("Belgeler yükleniyor..."):
                documents = load_documents_from_folder("data")
                if not documents:
                    st.error("Hiç belge bulunamadı! Lütfen 'data' klasörüne belgelerinizi ekleyin.")
                    logging.error("Hiç belge bulunamadı")
                    st.stop()
                
                st.session_state.documents = documents
                logging.info(f"Toplam {len(documents)} belge yüklendi")
            
            # Vektör veritabanı oluştur
            with st.spinner("Vektör veritabanı oluşturuluyor..."):
                vector_db = create_vector_db(documents)
                if not vector_db:
                    st.error("Vektör veritabanı oluşturulamadı!")
                    logging.error("Vektör veritabanı oluşturulamadı")
                    st.stop()
                
                st.session_state.vector_db = vector_db
            
            # Chat zinciri oluştur
            with st.spinner("Sohbet sistemi hazırlanıyor..."):
                chat_chain = create_chat_chain(vector_db)
                if not chat_chain:
                    st.error("Sohbet sistemi oluşturulamadı!")
                    logging.error("Sohbet sistemi oluşturulamadı")
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
    cols = st.columns(4)
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
                
                # Düşünme animasyonu (opsiyonel)
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