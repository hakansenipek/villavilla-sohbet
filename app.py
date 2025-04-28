# streamlit_app.py

import os
import sys
import tempfile
import logging
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch  # Chroma yerine DocArrayInMemorySearch
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
# 3. OpenAI API Anahtarı Yönetimi
# ---------------------------------------
openai_api_key = st.secrets.get("openai", {}).get("api_key", None)
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.error("API anahtarı girilmedi. Lütfen OpenAI API anahtarınızı girin.")
        st.stop()

# ---------------------------------------
# 4. Belgeleri Yükleme Fonksiyonu (.docx)
# ---------------------------------------
def load_documents_from_folder(folder_path="data"):
    documents = []
    if not os.path.exists(folder_path):
        st.warning(f"Belge klasörü bulunamadı: {folder_path}. Örnek veri kullanılacak.")
        # Örnek belgeler oluştur
        return create_test_documents()

    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            full_path = os.path.join(folder_path, filename)
            try:
                docx = DocxDocument(full_path)
                text = "\n".join([p.text for p in docx.paragraphs if p.text.strip() != ""])
                documents.append(Document(page_content=text, metadata={"source": filename}))
            except Exception as e:
                logging.error(f"{filename} yüklenirken hata: {str(e)}")
    
    if not documents:
        st.warning("Belge bulunamadı. Örnek veri kullanılacak.")
        return create_test_documents()
    
    return documents

# ---------------------------------------
# 5. Test Belgeleri Oluşturma
# ---------------------------------------
def create_test_documents():
    documents = []
    
    yapilan_isler_content = """
    Villa Villa Organizasyon - Yapılan İşler
    
    Tarih: 15 Nisan 2025
    Müşteri: ABC Şirketi
    Etkinlik Türü: Kurumsal Yıl Dönümü
    Kişi Sayısı: 150
    Menü: Ana yemek, 5 çeşit meze, 2 çeşit tatlı
    Toplam Maliyet: 75,000 TL
    
    Tarih: 10 Nisan 2025
    Müşteri: XYZ Ltd.
    Etkinlik Türü: Ürün Lansmanı
    Kişi Sayısı: 80
    Menü: Kokteyl, 10 çeşit kanape
    Toplam Maliyet: 40,000 TL
    """
    
    genel_gider_content = """
    Villa Villa Organizasyon - Genel Giderler
    
    Nisan 2025:
    Kira: 15,000 TL
    Elektrik: 2,500 TL
    Su: 800 TL
    İnternet: 600 TL
    Personel Maaşları: 45,000 TL
    Toplam: 63,900 TL
    
    Mart 2025:
    Kira: 15,000 TL
    Elektrik: 2,800 TL
    Su: 750 TL
    İnternet: 600 TL
    Personel Maaşları: 45,000 TL
    Toplam: 64,150 TL
    """
    
    doc1 = Document(page_content=yapilan_isler_content, metadata={"source": "yapilan_isler.docx"})
    doc2 = Document(page_content=genel_gider_content, metadata={"source": "genel_gider.docx"})
    
    documents.extend([doc1, doc2])
    return documents

# ---------------------------------------
# 6. Vektör Veritabanı (DocArrayInMemorySearch)
# ---------------------------------------
def create_vector_db(documents):
    try:
        # Belgeleri parçalara böl
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        st.info(f"Belgeler {len(chunks)} parçaya bölündü")
        
        # Embeddings oluştur
        try:
            embeddings = OpenAIEmbeddings()
            
            # DocArrayInMemorySearch vektör veritabanı oluştur
            vector_db = DocArrayInMemorySearch.from_documents(
                documents=chunks,
                embedding=embeddings,
            )
            st.success("Vektör veritabanı başarıyla oluşturuldu")
            return vector_db
            
        except Exception as e:
            logging.error(f"Embedding hatası: {str(e)}")
            st.error(f"Embedding oluşturulurken hata: {str(e)}")
            return None
            
    except Exception as e:
        logging.error(f"Vektör veritabanı hatası: {str(e)}")
        st.error("Vektör veritabanı oluşturulamadı.")
        return None

# ---------------------------------------
# 7. Özel Prompt Şablonu
# ---------------------------------------
def create_qa_prompt():
    template = """
    Sen Villa Villa şirketinin yapay zekâ destekli bir asistanısın. Aşağıdaki içeriklere dayanarak müşterinin sorusunu yanıtla:

    Belgelerden Bilgiler:
    {context}

    Sohbet Geçmişi:
    {chat_history}

    Soru:
    {question}

    Notlar:
    - Yalnızca belgedeki bilgiler doğrultusunda cevap ver.
    - Bilgi bulunamazsa 'Bu konuda mevcut veri bulunmamaktadır.' yaz.
    - Profesyonel, açık ve kibar bir dil kullan.

    Yanıt:
    """
    return PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)

# ---------------------------------------
# 8. Chat Zinciri Kurulumu
def create_chat_chain(vector_db):
    try:
        llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4-turbo"  
        
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
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
        st.error("Sohbet sistemi oluşturulamadı.")
        return None
# ---------------------------------------
# 9. Ana Uygulama
# ---------------------------------------
def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Sidebar - Ayarlar
    with st.sidebar:
        st.header("Ayarlar")
        use_test_data = st.checkbox("Test verileri kullan", value=True)
        
        # Temizle butonu
        if st.button("Sohbeti Temizle"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Belgeler ve vektör veritabanı
    with st.spinner("Sistem hazırlanıyor..."):
        try:
            if use_test_data:
                documents = create_test_documents()
                st.info("Test belgeleri kullanılıyor")
            else:
                documents = load_documents_from_folder("data")
                
            if not documents:
                st.error("Hiç belge bulunamadı!")
                st.stop()
            
            vector_db = create_vector_db(documents)
            if not vector_db:
                st.error("Vektör veritabanı oluşturulamadı!")
                st.stop()
            
            chat_chain = create_chat_chain(vector_db)
            if not chat_chain:
                st.error("Sohbet sistemi oluşturulamadı!")
                st.stop()
                
        except Exception as e:
            logging.error(f"Sistem hazırlama hatası: {str(e)}")
            st.error("Sistem hazırlanırken bir hata oluştu!")
            st.stop()
    
    # Sohbet geçmişini görüntüle
    for i in range(0, len(st.session_state.chat_history), 2):
        if i < len(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(st.session_state.chat_history[i][1])
        
        if i+1 < len(st.session_state.chat_history):
            with st.chat_message("assistant"):
                st.markdown(st.session_state.chat_history[i+1][1])
    
    # Kullanıcı girişi
    user_input = st.chat_input("Sorunuzu yazınız...")
    
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
                response = chat_chain({
                    "question": user_input,
                    "chat_history": chat_formatted
                })
            
            # Yanıtı göster
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
                
                # Kullanılan kaynakları göster
                with st.expander("Kullanılan Kaynaklar", expanded=False):
                    for i, doc in enumerate(response["source_documents"]):
                        source = doc.metadata.get("source", "Bilinmeyen")
                        st.markdown(f"**Kaynak {i+1}:** {source}")
                        st.markdown(f"```\n{doc.page_content[:200]}...\n```")
            
            # Yanıtı geçmişe ekle
            st.session_state.chat_history.append(("assistant", response["answer"]))
            
        except Exception as e:
            logging.error(f"Yanıt hatası: {str(e)}")
            with st.chat_message("assistant"):
                st.error("Üzgünüm, yanıt oluşturulurken bir hata oluştu.")
            st.session_state.chat_history.append(("assistant", "Üzgünüm, bir hata oluştu."))

# ---------------------------------------
# 10. Uygulama Başlatılıyor
# ---------------------------------------
if __name__ == "__main__":
    main()