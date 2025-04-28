import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma  # FAISS yerine Chroma kullanıyoruz
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

# API anahtarını doğrudan ayarlama (güvenlik için secrets kullanımı önerilir)
openai_api_key = st.secrets.get("openai", {}).get("api_key", None)
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print("API anahtarı secrets'tan ayarlandı")
else:
    # API anahtarını manuel giriş olarak ekleyebilirsiniz (test için)
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        print("API anahtarı manuel olarak ayarlandı")
    else:
        st.error("API anahtarı bulunamadı. Lütfen OpenAI API anahtarınızı girin.")

# Test amacıyla sabit veriler oluşturma
def create_test_documents():
    """Test için yapay belge verileri oluşturur"""
    documents = []
    
    # Yapılan işler örnek belgesi
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
    
    # Genel gider örnek belgesi
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
    
    # Belgeleri oluştur
    doc1 = Document(page_content=yapilan_isler_content, metadata={"source": "yapilan_isler"})
    doc2 = Document(page_content=genel_gider_content, metadata={"source": "genel_gider"})
    
    documents.extend([doc1, doc2])
    
    return documents

# Vektör veritabanı oluşturma - ChromaDB ile
def create_vector_db(documents):
    """Belgelerden vektör veritabanı oluşturur - ChromaDB kullanarak"""
    if not documents:
        return None
    
    try:
        # Tiktoken yüklü mü kontrol et (hata ayıklama)
        try:
            import tiktoken
            print(f"Tiktoken sürümü: {tiktoken.__version__}")
        except ImportError:
            print("Tiktoken yüklü değil! Yükleniyor...")
            import sys
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
            import tiktoken
            print(f"Tiktoken yüklendi: {tiktoken.__version__}")
        
        # Belgeleri uygun parçalara böl
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        print(f"Belgeler {len(chunks)} parçaya bölündü")
        
        # OPENAI_API_KEY kontrol et
        print(f"API Anahtarı ayarlandı mı: {'OPENAI_API_KEY' in os.environ}")
        
        # Vektör embeddingler oluştur
        try:
           embeddings = OpenAIEmbeddings()
print("OpenAIEmbeddings başarıyla oluşturuldu")
            
# Vektör veritabanı oluşturma
def create_vector_db(documents):
    """Belgelerden vektör veritabanı oluşturur - DocArrayInMemorySearch kullanarak"""
    if not documents:
        return None
    
    try:
        # Tiktoken yüklü mü kontrol et
        try:
            import tiktoken
            print(f"Tiktoken sürümü: {tiktoken.__version__}")
        except ImportError:
            print("Tiktoken yüklü değil! Yükleniyor...")
            import sys
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
            import tiktoken
        
        # Belgeleri uygun parçalara böl
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        print(f"Belgeler {len(chunks)} parçaya bölündü")
        
        # API anahtarı kontrol et
        print(f"API Anahtarı ayarlandı mı: {'OPENAI_API_KEY' in os.environ}")
        
        # Vektör embeddingler oluştur
        try:
            embeddings = OpenAIEmbeddings()
            print("OpenAIEmbeddings başarıyla oluşturuldu")
            
            # DocArrayInMemorySearch vektör veritabanı oluştur
            from langchain.vectorstores import DocArrayInMemorySearch
            
            vector_db = DocArrayInMemorySearch.from_documents(
                documents=chunks,
                embedding=embeddings,
            )
            print("DocArrayInMemorySearch vektör veritabanı başarıyla oluşturuldu")
            return vector_db
            
        except Exception as e:
            import traceback
            error_msg = f"Embedding oluşturma hatası: {str(e)}"
            print(error_msg)
            print(f"Hata detayı: {traceback.format_exc()}")
            st.error(f"Vektör veritabanı oluşturulurken hata: {str(e)}")
            return None
        
    except Exception as e:
        print(f"Genel hata: {str(e)}")
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
       - Belirli bir aya ilişkin gider sorgularında tüm dosyalar birlikte değerlendirilir.
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
        # API anahtarı kontrol et
        if "OPENAI_API_KEY" not in os.environ:
            st.error("OpenAI API anahtarı bulunamadı")
            return None
            
        # GPT model
        llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4-0125-preview"
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
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        
        print("Sohbet zinciri başarıyla oluşturuldu")
        return chat_chain
    except Exception as e:
        print(f"Sohbet zinciri hatası: {str(e)}")
        st.error(f"Sohbet zinciri oluşturulurken hata: {str(e)}")
        return None

# Ana uygulama fonksiyonu
def main():
    # Sohbet geçmişi
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Paket bağımlılıklarını kontrol et ve gerekirse yükle
    required_packages = ["tiktoken", "openai"]  # faiss-cpu çıkarıldı
    missing_packages = []
    
    # Gerekli paketleri kontrol et
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    # Eksik paketleri yükle
    if missing_packages:
        with st.spinner(f"Gerekli paketler yükleniyor: {', '.join(missing_packages)}..."):
            import sys
            import subprocess
            for package in missing_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            st.success("Paketler yüklendi. Sayfa yenileniyor...")
            st.rerun()  # experimental_rerun yerine rerun kullanılıyor
    
    # Test belgelerini oluştur
    with st.spinner("Belgeler hazırlanıyor..."):
        documents = create_test_documents()
    
    if documents:
        # Vektör veritabanı oluştur
        with st.spinner("Vektör veritabanı oluşturuluyor..."):
            vector_db = create_vector_db(documents)
        
        if vector_db:
            # Sohbet zincirini oluştur
            with st.spinner("Sohbet modeli hazırlanıyor..."):
                chat_chain = create_chat_chain(vector_db)
            
            if chat_chain:
                # Temizleme butonu - sağ üstte küçük
                col1, col2 = st.columns([5, 1])
                with col2:
                    if st.button("🔄 Temizle", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()  # experimental_rerun yerine rerun kullanılıyor
                
                # Sohbet geçmişini görüntüle
                for i in range(0, len(st.session_state.chat_history), 2):
                    if i < len(st.session_state.chat_history):
                        with st.chat_message("user"):
                            st.markdown(st.session_state.chat_history[i][1])
                    
                    if i+1 < len(st.session_state.chat_history):
                        with st.chat_message("assistant"):
                            st.markdown(st.session_state.chat_history[i+1][1])
                
                # Soru girişi
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
                            print(f"Yanıt hatası: {str(e)}")
                            st.error(f"Yanıt oluşturulurken hata: {str(e)}")
                            # Kullanıcıya hata mesajı
                            with st.chat_message("assistant"):
                                st.markdown("Üzgünüm, şu anda yanıt verirken teknik bir sorun yaşıyorum. Lütfen başka bir soru sorun veya daha sonra tekrar deneyin.")
                            # Hata mesajını geçmişe ekle
                            st.session_state.chat_history.append(("ai", "Üzgünüm, teknik bir sorun yaşandı."))
            else:
                st.error("Sohbet zinciri oluşturulamadı. API anahtarını kontrol edin.")
        else:
            st.error("Vektör veritabanı oluşturulamadı. Paket kurulumunu kontrol edin.")
    else:
        st.error("Belgeler hazırlanamadı.")
    
    # Gelişmiş modda yönetici kontrolleri
    with st.expander("⚙️ Yönetici Ayarları", expanded=False):
        st.warning("Şu anda test modu aktif - gerçek veriler yerine örnek veriler kullanılıyor.")
        # API durumu
        st.info(f"API Anahtarı durumu: {'Ayarlandı ✅' if 'OPENAI_API_KEY' in os.environ else 'Ayarlanmadı ❌'}")
        
        # Debug bilgisi
        if st.checkbox("Debug modunu aç"):
            st.code(f"""
            Paketler:
            - tiktoken: {__import__('importlib').util.find_spec('tiktoken') is not None}
            - langchain: {__import__('importlib').util.find_spec('langchain') is not None}
            - chromadb: {__import__('importlib').util.find_spec('chromadb') is not None}
            """)

# Uygulamayı çalıştır
if __name__ == "__main__":
    main()