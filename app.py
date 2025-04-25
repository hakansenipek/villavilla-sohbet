import os
import streamlit as st
import requests
import io
import tempfile
from datetime import datetime, timedelta
import json
from PIL import Image
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Sayfa başlığı ve düzeni
st.set_page_config(page_title="Villa Villa Yapay Zeka ile Sohbet", layout="centered")

# Logoyu ve başlığı yatay sırada göster
col1, col2 = st.columns([1, 3])
with col1:
    try:
        # Logo dosyasını assets klasöründen yüklemeye çalış, yoksa geç
        st.image("assets/villa_villa_logo.jpg", width=100)
    except Exception:
        pass
with col2:
    st.title("Villa Villa Yapay Zeka ile Sohbet")

# Ayırıcı çizgi
st.markdown("---")

# API anahtarlarını Streamlit secrets'tan alma
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
except Exception:
    st.error("OpenAI API anahtarı secrets.toml dosyasında bulunamadı.")

# Google Drive API bağlantısı
def get_gdrive_service():
    """Google Drive API servisini döndürür"""
    try:
        # Session state'ten token bilgilerini al
        if "token" not in st.session_state:
            return None
        
        creds = Credentials(
            token=st.session_state.get("token"),
            refresh_token=st.session_state.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=st.secrets["google_auth"]["client_id"],
            client_secret=st.secrets["google_auth"]["client_secret"]
        )
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        st.error(f"Google Drive bağlantı hatası: {str(e)}")
        return None

def setup_google_oauth():
    """Google OAuth kimlik doğrulama akışını ayarlar"""
    from google_auth_oauthlib.flow import Flow
    
    # OAuth akışını ayarla
    client_config = {
        "web": {
            "client_id": st.secrets["google_auth"]["client_id"],
            "client_secret": st.secrets["google_auth"]["client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [st.secrets["app_url"] + "/oauth2callback"]
        }
    }
    
    # OAuth akışını başlat
    flow = Flow.from_client_config(
        client_config,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
        redirect_uri=st.secrets["app_url"] + "/oauth2callback"
    )
    
    # Kimlik doğrulama URL'sini oluştur
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    
    return auth_url, flow

# Paraşüt API bağlantısı
def get_parasut_token():
    """Paraşüt API'sinden access token alır"""
    try:
        client_id = st.secrets["parasut"]["client_id"]
        client_secret = st.secrets["parasut"]["client_secret"]
        
        # Token yoksa veya süresi dolduysa yeni token al
        if "parasut_token" not in st.session_state or \
        st.session_state.get("parasut_token_expires_at", datetime.now()) <= datetime.now():
            
            token_url = "https://api.parasut.com/oauth/token"
            
            payload = {
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": "urn:ietf:wg:oauth:2.0:oob",
                "grant_type": "client_credentials"
            }
            
            response = requests.post(token_url, data=payload)
            response.raise_for_status()
            
            token_data = response.json()
            
            # Token bilgilerini session_state'e kaydet
            st.session_state.parasut_token = token_data["access_token"]
            # Expire time'ı token süresinden biraz önceye ayarla
            expiry = datetime.now() + timedelta(seconds=token_data["expires_in"] - 60)
            st.session_state.parasut_token_expires_at = expiry
            
            return st.session_state.parasut_token
        
        # Mevcut token'ı döndür
        return st.session_state.parasut_token
    except Exception as e:
        st.error(f"Paraşüt API token alınırken hata: {str(e)}")
        return None

def get_invoices_from_parasut():
    """Paraşüt'ten fatura bilgilerini çeker"""
    token = get_parasut_token()
    
    if not token:
        return []
    
    try:
        company_id = st.secrets["parasut"]["company_id"]
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Son 30 günün satış faturalarını çek
        today = datetime.now()
        thirty_days_ago = today - timedelta(days=30)
        
        params = {
            "filter[issue_date]": f"{thirty_days_ago.strftime('%Y-%m-%d')}..{today.strftime('%Y-%m-%d')}",
            "include": "category,contact"
        }
        
        # Satış faturalarını çek
        invoices_url = f"https://api.parasut.com/v4/{company_id}/sales_invoices"
        response = requests.get(invoices_url, headers=headers, params=params)
        response.raise_for_status()
        
        return response.json()["data"]
    except Exception as e:
        st.error(f"Paraşüt API'den faturalar çekilirken hata: {str(e)}")
        return []

def prepare_parasut_data_for_langchain(invoices):
    """Paraşüt API'den gelen fatura verilerini LangChain belgelerine dönüştürür"""
    documents = []
    
    for invoice in invoices:
        attributes = invoice["attributes"]
        relationships = invoice.get("relationships", {})
        
        # Müşteri bilgisini al
        contact_name = "Bilinmeyen Müşteri"
        if "contact" in relationships and "data" in relationships["contact"]:
            contact_data = relationships["contact"]["data"]
            contact_name = contact_data.get("name", "Bilinmeyen Müşteri")
        
        # Fatura bilgilerini metin formatına dönüştür
        content = f"""
        Fatura No: {attributes.get('invoice_no', 'Belirtilmemiş')}
        Tarih: {attributes.get('issue_date', 'Belirtilmemiş')}
        Müşteri: {contact_name}
        Tutar: {attributes.get('total', '0')} TL
        Net Tutar: {attributes.get('net_total', '0')} TL
        Durum: {attributes.get('payment_status', 'Belirtilmemiş')}
        Açıklama: {attributes.get('description', '')}
        """
        
        # LangChain belgesi oluştur
        doc = Document(
            page_content=content,
            metadata={
                "source": "parasut_api",
                "invoice_id": invoice["id"],
                "date": attributes.get("issue_date", ""),
                "type": "invoice"
            }
        )
        documents.append(doc)
    
    return documents

# Google Drive'dan dosyaları indirme
@st.cache_data(ttl=3600)  # 1 saat önbellekte tut
def download_files_from_drive():
    """Google Drive'dan Villa Villa belgelerini indirir"""
    if "token" not in st.session_state:
        st.warning("Belgelere erişim için Google hesabınıza bağlanmalısınız")
        return {}
    
    service = get_gdrive_service()
    if not service:
        return {}
    
    # Dosya ID'lerini secrets'tan al
    try:
        file_ids = st.secrets["drive_files"]
    except Exception:
        st.error("Google Drive dosya ID'leri secrets dosyasında bulunamadı")
        return {}
    
    documents = {}
    for doc_name, file_id in file_ids.items():
        try:
            request = service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                
            file_content.seek(0)
            
            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                temp_file.write(file_content.getvalue())
                temp_path = temp_file.name
            
            # Langchain ile dosyayı yükle
            loader = Docx2txtLoader(temp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = doc_name
            
            documents[doc_name] = docs
            
            # Geçici dosyayı sil
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"{doc_name} dosyası indirilirken hata: {str(e)}")
    
    return documents

# Yerel dosyaları yükleme (geliştirme ve test amaçlı)
def load_local_documents():
    """Yerel klasördeki belgeleri yükler (API'ler kullanılamadığında)"""
    # Belge klasörü ve dosya listesi
    data_folder = "data"
    document_files = [
        "genel_gider.docx",
        "gelen_faturalar.docx",
        "personel_giderleri.docx",
        "villa_villa_tanitim.docx",
        "yapilan_isler.docx"
    ]
    
    documents = {}
    for doc_name in document_files:
        file_path = os.path.join(data_folder, doc_name)
        try:
            if os.path.exists(file_path):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = doc_name.replace(".docx", "")
                documents[doc_name.replace(".docx", "")] = docs
        except Exception as e:
            st.error(f"{doc_name} dosyası yüklenirken hata: {str(e)}")
    
    return documents

# Tüm veri kaynaklarını birleştirme
def load_all_data_sources():
    """Tüm veri kaynaklarını (Google Drive, Paraşüt API veya yerel) yükler"""
    documents = []
    
    # 1. Google Drive belgelerini yüklemeyi dene
    if "token" in st.session_state:
        drive_docs = download_files_from_drive()
        for doc_list in drive_docs.values():
            documents.extend(doc_list)
    
    # 2. Eğer Drive belgeleri bulunamazsa yerel dosyalara bak
    if not documents:
        local_docs = load_local_documents()
        for doc_list in local_docs.values():
            documents.extend(doc_list)
    
    # 3. Paraşüt API'sinden fatura bilgilerini yükle
    try:
        invoices = get_invoices_from_parasut()
        invoice_docs = prepare_parasut_data_for_langchain(invoices)
        documents.extend(invoice_docs)
    except Exception as e:
        st.error(f"Paraşüt verilerini yüklerken hata: {str(e)}")
    
    return documents

# Vektör veritabanı oluşturma
@st.cache_resource
def create_vector_db(documents, force_reload=False):
    """Belgelerden vektör veritabanı oluşturur"""
    if not documents:
        return None
    
    cache_path = "vector_db"
    
    # Yeniden yükleme istenirse veritabanını temizle
    if force_reload and os.path.exists(cache_path):
        import shutil
        try:
            shutil.rmtree(cache_path)
        except Exception as e:
            st.error(f"Veritabanı temizleme hatası: {str(e)}")
    
    try:
        # Belgeleri uygun parçalara böl
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        
        # Vektör embeddingler oluştur
        embeddings = OpenAIEmbeddings()
        
        # Vektör veritabanı oluştur
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=cache_path
        )
        
        # Verileri diske kaydet
        vector_db.persist()
        return vector_db
    except Exception as e:
        st.error(f"Vektör veritabanı oluşturulurken hata: {str(e)}")
        return None

# Özel prompt şablonu
def create_qa_prompt():
    """Villa Villa'ya özel sohbet şablonu oluşturur"""
    template = """
    Sen Villa Villa şirketi için bir yapay zeka asistanısın. Aşağıdaki belgelerden aldığın bilgilere dayanarak soruları yanıtla.
    
    Belgeler:
    {context}
    
    Mevcut Sohbet:
    {chat_history}
    
    Soru: {question}
    
    Villa Villa Yapay Zeka Destekli Chatbot'un Çalışma Sistemi:
    
    1. Tarih ve Yapılan İş Bilgisi Sorguları:
       - Yapılan işler tarih bazlı sorgulandığında, bilgi "yapilan_isler" adlı dosya üzerinden alınır.
       - En güncel iş, tarih sıralamasına göre tespit edilir ve detaylarıyla sunulur.
    
    2. Yapılan İşlere Göre Maliyet Hesaplama:
       - Talep edilen işle ilgili maliyet çıkarılırken "yapilan_isler" dosyasındaki işletme ve hizmet detayları temel alınır.
       - İşin türü, kişi sayısı, menü içeriği ve yer bilgileri dikkate alınarak değerlendirme yapılır.
    
    3. Aylık Gider ve Maliyet Analizi:
       - Belirli bir aya ilişkin gider sorgularında tüm dosyalar birlikte değerlendirilir: "genel_gider", "gelen_faturalar", "yapilan_isler", "personel_giderleri"
       - Bu dosyalar doğrultusunda, tedarikçi giderleri, fatura kalemleri ve iş bazlı maliyetler birleştirilerek kapsamlı bir analiz yapılır.
    
    4. Menü Teklifi Oluşturma:
       - Menü teklifi istenen işlerde, "yapilan_isler" dosyasında yer alan örnek menüler incelenir.
       - İşin niteliği (açılış, davet, kurumsal vb.) ve kişi sayısı göz önünde bulundurularak benzer işler temel alınır, uygun menü önerisi hazırlanır.
    
    Talimatlar:
    1. Villa Villa şirketinin belgeleri ve verilerine dayanarak yukarıdaki çalışma sistemine göre yanıt ver.
    2. Tüm yanıtlarında sadece belgelerden edindiğin bilgilere dayan, tahmin yürütme.
    3. Bilgi bulamadığın durumlarda bunu açıkça belirt.
    4. Sonuçları mümkün olduğunca yapılandırılmış ve okunması kolay bir şekilde sun.
    5. Müşteriye her zaman nazik ve profesyonel ol.
    
    Detaylı Yanıt:
    """
    
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

# Sohbet zinciri oluşturma
def create_chat_chain(vector_db):
    """LangChain sohbet zincirini oluşturur"""
    if not vector_db:
        return None
    
    try:
        # GPT model
        llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4-0125-preview",
            verbose=True
        )
        
        # Retriever tanımla
        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Özel prompt şablonu
        qa_prompt = create_qa_prompt()
        
        # Sohbet zinciri oluştur
        chat_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=True
        )
        
        return chat_chain
    except Exception as e:
        st.error(f"Sohbet zinciri oluşturulurken hata: {str(e)}")
        return None

# Google Drive bağlantısı kontrol
def check_google_auth():
    """Google Drive bağlantı durumunu kontrol eder"""
    if "token" not in st.session_state:
        auth_url, _ = setup_google_oauth()
        st.warning("Google Drive belgelerine erişim için lütfen oturum açın")
        st.markdown(f"[Google Hesabınızla Giriş Yapın]({auth_url})")
        return False
    return True

# Ana uygulama fonksiyonu
def main():
    # Sohbet geçmişi
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Yeniden yükleme bayrağı
    if "reload_data" not in st.session_state:
        st.session_state.reload_data = False
    
    # 1. Google Drive bağlantısını kontrol et
    # Not: Bu bölümü geliştirme sırasında devre dışı bırakabilirsiniz
    # is_authenticated = check_google_auth()
    is_authenticated = True  # Geliştirme sırasında doğrudan True yapabilirsiniz
    
    if is_authenticated:
        # 2. Tüm veri kaynaklarını yükle
        with st.spinner("Belgeler yükleniyor..."):
            documents = load_all_data_sources()
        
        if documents:
            # 3. Vektör veritabanı oluştur
            vector_db = create_vector_db(documents, force_reload=st.session_state.reload_data)
            
            # Yeniden yükleme bayrağını sıfırla
            if st.session_state.reload_data:
                st.session_state.reload_data = False
            
            if vector_db:
                # 4. Sohbet zincirini oluştur
                chat_chain = create_chat_chain(vector_db)
                
                if chat_chain:
                    # Temizleme butonu - sağ üstte küçük
                    col1, col2 = st.columns([5, 1])
                    with col2:
                        if st.button("🔄 Temizle", use_container_width=True):
                            st.session_state.chat_history = []
                            st.rerun()
                    
                    # 5. Sohbet geçmişini görüntüle
                    for i in range(0, len(st.session_state.chat_history), 2):
                        if i < len(st.session_state.chat_history):
                            with st.chat_message("user"):
                                st.markdown(st.session_state.chat_history[i][1])
                        
                        if i+1 < len(st.session_state.chat_history):
                            with st.chat_message("assistant"):
                                st.markdown(st.session_state.chat_history[i+1][1])
                    
                    # 6. Soru girişi
                    user_input = st.chat_input("Sorunuzu yazın")
                    
                    if user_input:
                        # Kullanıcı mesajını görüntüle
                        with st.chat_message("user"):
                            st.markdown(user_input)
                        
                        # Geçmişe ekle
                        st.session_state.chat_history.append(("user", user_input))
                        
                        # Sorguyu çalıştır
                        with st.spinner("Düşünüyor..."):
                            try:
                                # Sohbet geçmişini uygun formata dönüştür
                                if len(st.session_state.chat_history) > 1:
                                    chat_history = []
                                    for i in range(0, len(st.session_state.chat_history)-1, 2):
                                        if i+1 < len(st.session_state.chat_history):
                                            chat_history.append((st.session_state.chat_history[i][1], 
                                                                st.session_state.chat_history[i+1][1]))
                                else:
                                    chat_history = []
                                
                                # Sorguyu çalıştır
                                response = chat_chain({
                                    "question": user_input, 
                                    "chat_history": chat_history
                                })
                                
                                # Yanıtı görüntüle
                                with st.chat_message("assistant"):
                                    st.markdown(response["answer"])
                                    
                                    # Kullanılan kaynakları göster
                                    with st.expander("Kullanılan Kaynaklar", expanded=False):
                                        for i, doc in enumerate(response["source_documents"]):
                                            source = doc.metadata.get("source", "Bilinmeyen")
                                            st.markdown(f"**Kaynak {i+1}:** {source}")
                                            st.markdown(f"```\n{doc.page_content[:200]}...\n```")
                                
                                # Yanıtı geçmişe ekle
                                st.session_state.chat_history.append(("ai", response["answer"]))
                                
                            except Exception as e:
                                st.error(f"Yanıt oluşturulurken hata: {str(e)}")
                                # Kullanıcıya hata mesajı
                                with st.chat_message("assistant"):
                                    st.markdown("Üzgünüm, şu anda yanıt verirken teknik bir sorun yaşıyorum. Lütfen başka bir soru sorun veya daha sonra tekrar deneyin.")
                                # Hata mesajını geçmişe ekle
                                st.session_state.chat_history.append(("ai", "Üzgünüm, teknik bir sorun yaşandı."))
                else:
                    st.error("Sohbet zinciri oluşturulamadı. Lütfen daha sonra tekrar deneyin.")
            else:
                st.error("Vektör veritabanı oluşturulamadı. Lütfen daha sonra tekrar deneyin.")
        else:
            st.error("Belgeler yüklenemedi. Lütfen Google Drive bağlantınızı kontrol edin veya yerel dosyaların varlığını doğrulayın.")
    
    # Gelişmiş modda yönetici kontrolleri
    with st.expander("⚙️ Yönetici Ayarları", expanded=False):
        # Veriyi yeniden yükleme butonu
        if st.button("🔄 Veritabanını Yeniden Oluştur"):
            st.session_state.reload_data = True
            st.success("Veritabanı yeniden oluşturuluyor...")
            st.rerun()

# Uygulamayı çalıştır
if __name__ == "__main__":
    main()