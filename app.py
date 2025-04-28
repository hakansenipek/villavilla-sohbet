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
        st.error(f"Belge klasörü bulunamadı: {folder_path}")
        return documents

    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            full_path = os.path.join(folder_path, filename)
            try:
                docx = DocxDocument(full_path)
                text = "\n".join([p.text for p in docx.paragraphs if p.text.strip() != ""])
                documents.append(Document(page_content=text, metadata={"source": filename}))
            except Exception as e:
                logging.error(f"{filename} yüklenirken hata: {str(e)}")
    
    return documents

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
# 6. Özel Prompt Şablonu
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
# 7. Chat Zinciri Kurulumu
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
    with st.spinner("Belgeler yükleniyor..."):
        documents = load_documents_from_folder("data")
        if not documents:
            st.error("Belge bulunamadı. 'data/' klasörüne .docx dosyalarınızı yükleyin.")
            st.stop()
        
        vector_db = create_vector_db(documents)
        if not vector_db:
            st.error("Vektör veritabanı oluşturulamadı.")
            st.stop()
        
        chat_chain = create_chat_chain(vector_db)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.success("Belgeler yüklendi. Şimdi sorularınızı sorabilirsiniz!")

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)
    
    user_input = st.chat_input("Sorunuzu yazınız...")

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
                st.error("Üzgünüm, yanıt oluşturulurken bir hata oluştu.")

# ---------------------------------------
# 9. Uygulama Başlatılıyor
# ---------------------------------------
if __name__ == "__main__":
    main()
