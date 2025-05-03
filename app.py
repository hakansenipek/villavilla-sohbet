import os
import time
import logging
import streamlit as st

from pinecone_utils import create_or_update_vector_db, load_vector_db_from_pinecone
from drive_utils import load_documents_from_urls
from chat_chain import create_chat_chain

# Loglama
if not os.path.exists("logs"):
    os.makedirs("logs")
logging.basicConfig(filename="logs/app.log", level=logging.INFO)

# Sayfa ayarları
st.set_page_config(page_title="Villa Villa Yapay Zeka", layout="wide")

# API Anahtarlarını yükle
def load_api_keys():
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
        os.environ["PINECONE_API_KEY"] = st.secrets["pinecone"]["api_key"]
        os.environ["PINECONE_ENVIRONMENT"] = st.secrets["pinecone"]["environment"]
        os.environ["PINECONE_INDEX_NAME"] = st.secrets["pinecone"]["index_name"]
        return True
    except Exception as e:
        st.error("API bilgileri eksik.")
        logging.error(str(e))
        return False

# Doküman URL’leri
DOC_URLS = {
    "gelen_faturalar": "https://docs.google.com/document/d/.../edit",
    "genel_gider": "https://docs.google.com/document/d/.../edit",
    "personel_giderleri": "https://docs.google.com/document/d/.../edit",
    "villa_villa_tanitim": "https://docs.google.com/document/d/.../edit",
    "yapilan_isler": "https://docs.google.com/document/d/.../edit"
}

def main():
    if not load_api_keys():
        st.stop()

    # Logo ve başlık
    st.markdown('<div style="display:flex; align-items:center;">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("assets/villa_villa_logo.jpg", width=100)
    with col2:
        st.title("Villa Villa Yapay Zeka Asistanı")
    st.markdown('</div>', unsafe_allow_html=True)

    # Açıklama
    with st.sidebar:
        st.subheader("🧠 Hakkında")
        st.info("Villa Villa şirketi verilerine dayalı akıllı asistan.")
        st.subheader("📊 Örnek Sorular")
        st.markdown("- Metro'dan ne kadar harcama yapılmış?\n- En pahalı tedarikçi kim?\n- Kasım ayında toplam gider nedir?")

    # Yükleme seçenekleri
    st.sidebar.subheader("⚙️ Veri Yükleme")
    option = st.sidebar.radio("Yükleme Seçeneği:", ["Pinecone'dan Yükle", "Google Drive'dan Yükle ve Kaydet"])
    namespace = st.sidebar.text_input("Namespace", value="villa_villa")
    if st.sidebar.button("📂 Verileri Yükle") or "vector_db" not in st.session_state:
        with st.spinner("Veriler işleniyor..."):
            if option == "Google Drive'dan Yükle ve Kaydet":
                docs = load_documents_from_urls(DOC_URLS)
                vector_db = create_or_update_vector_db(docs, namespace)
            else:
                vector_db = load_vector_db_from_pinecone(namespace)

            st.session_state.vector_db = vector_db
            st.session_state.chat_chain = create_chat_chain(vector_db)
            st.success("Veriler yüklendi.")

    # Sohbet alanı
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Bir soru yazın...")
    if user_input and "chat_chain" in st.session_state:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))

        with st.chat_message("assistant"):
            with st.spinner("Cevap hazırlanıyor..."):
                formatted_history = [
                    (st.session_state.chat_history[i][1], st.session_state.chat_history[i+1][1])
                    for i in range(0, len(st.session_state.chat_history)-1, 2)
                ]
                try:
                    response = st.session_state.chat_chain({
                        "question": user_input,
                        "chat_history": formatted_history
                    })
                    answer = response["answer"]
                except Exception as e:
                    logging.error(f"Yanıt hatası: {str(e)}")
                    answer = "Üzgünüm, yanıt oluşturulamadı."

                st.markdown(answer)
                st.session_state.chat_history.append(("assistant", answer))

if __name__ == "__main__":
    main()
