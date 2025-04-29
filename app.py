# streamlit_app.py

import os
import sys
import tempfile
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
    st.info("Secrets dosyasında şu formatta API anahtarınızı tanımlayın: [openai] api_key = 'your-api-key'")
    return False

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
                logging.info(f"{filename} başarıyla yüklendi.")
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
    Menü: Ana yemek (Dana Bonfile), 5 çeşit meze (Humus, Cacık, Patlıcan Salatası, Acılı Ezme, Peynir Tabağı), 2 çeşit tatlı (Profiterol, Baklava)
    Yer: Beşiktaş Otel
    Toplam Maliyet: 75,000 TL
    
    Tarih: 10 Nisan 2025
    Müşteri: XYZ Ltd.
    Etkinlik Türü: Ürün Lansmanı
    Kişi Sayısı: 80
    Menü: Kokteyl, 10 çeşit kanape (Somon Füme, Peynirli Kraker, Mini Sandviç, Tavuk Şiş, Mini Pizza, İçli Köfte, Karides Kanepe, Sebzeli Kanepe, Meyve Çubukları, Çikolatalı Kurabiye)
    Yer: Nişantaşı Konferans Salonu
    Toplam Maliyet: 40,000 TL
    
    Tarih: 5 Nisan 2025
    Müşteri: 123 Holding
    Etkinlik Türü: Doğum Günü Kutlaması
    Kişi Sayısı: 30
    Menü: Açık Büfe (Izgara Balık, Tavuk Şiş, Karışık Izgara, Salata Bar, Meze Çeşitleri, Doğum Günü Pastası)
    Yer: Bebek Sahil Restoran
    Toplam Maliyet: 25,000 TL
    
    Tarih: 28 Mart 2025
    Müşteri: DEF AŞ
    Etkinlik Türü: Düğün
    Kişi Sayısı: 200
    Menü: Düğün Menüsü (Çorba, Ana Yemek (Kuzu Tandır), Pilav, Salata, 3 Çeşit Meze, Düğün Pastası)
    Yer: Yeniköy Düğün Salonu
    Toplam Maliyet: 120,000 TL
    """
    
    genel_gider_content = """
    Villa Villa Organizasyon - Genel Giderler
    
    Nisan 2025:
    Kira: 15,000 TL
    Elektrik: 2,500 TL
    Su: 800 TL
    İnternet: 600 TL
    Ofis Malzemeleri: 1,200 TL
    Araç Yakıt: 3,500 TL
    Bakım Onarım: 900 TL
    Toplam: 24,500 TL
    
    Mart 2025:
    Kira: 15,000 TL
    Elektrik: 2,800 TL
    Su: 750 TL
    İnternet: 600 TL
    Ofis Malzemeleri: 850 TL
    Araç Yakıt: 3,200 TL
    Bakım Onarım: 1,500 TL
    Toplam: 24,700 TL
    """
    
    personel_giderleri_content = """
    Villa Villa Organizasyon - Personel Giderleri
    
    Nisan 2025:
    Tam Zamanlı Çalışanlar (5 kişi): 35,000 TL
    Etkinlik Görevlileri (15 etkinlik): 22,500 TL
    SGK Ödemeleri: 12,800 TL
    Yemek Kartı: 3,500 TL
    Ulaşım Desteği: 2,500 TL
    Toplam: 76,300 TL
    
    Mart 2025:
    Tam Zamanlı Çalışanlar (5 kişi): 35,000 TL
    Etkinlik Görevlileri (12 etkinlik): 18,000 TL
    SGK Ödemeleri: 12,800 TL
    Yemek Kartı: 3,500 TL
    Ulaşım Desteği: 2,500 TL
    Toplam: 71,800 TL
    """
    
    gelen_faturalar_content = """
    Villa Villa Organizasyon - Gelen Faturalar
    
    Nisan 2025:
    Catering Hizmeti (ABC Catering): 25,000 TL (5 Nisan)
    Ses ve Işık Ekipmanları (XYZ Teknik): 15,000 TL (10 Nisan) 
    Dekorasyon Malzemeleri (Dekor AŞ): 12,000 TL (12 Nisan)
    Baskı ve Davetiye (Matbaa Ltd): 5,000 TL (8 Nisan)
    Çiçek Aranjmanları (Çiçekçi): 3,500 TL (15 Nisan)
    Ulaşım Hizmeti (Transfer Co): 8,000 TL (18 Nisan)
    Toplam: 68,500 TL
    
    Mart 2025:
    Catering Hizmeti (ABC Catering): 32,000 TL (28 Mart)
    Ses ve Işık Ekipmanları (XYZ Teknik): 18,000 TL (25 Mart)
    Dekorasyon Malzemeleri (Dekor AŞ): 9,500 TL (15 Mart)
    Baskı ve Davetiye (Matbaa Ltd): 7,000 TL (10 Mart)
    Çiçek Aranjmanları (Çiçekçi): 4,200 TL (22 Mart)
    Ulaşım Hizmeti (Transfer Co): 6,500 TL (18 Mart)
    Toplam: 77,200 TL
    """
    
    doc1 = Document(page_content=yapilan_isler_content, metadata={"source": "yapilan_isler.docx"})
    doc2 = Document(page_content=genel_gider_content, metadata={"source": "genel_gider.docx"})
    doc3 = Document(page_content=personel_giderleri_content, metadata={"source": "personel_giderleri.docx"})
    doc4 = Document(page_content=gelen_faturalar_content, metadata={"source": "gelen_faturalar.docx"})
    
    documents.extend([doc1, doc2, doc3, doc4])
    return documents

# ---------------------------------------
# 6. Vektör Veritabanı Oluşturma
# ---------------------------------------
def create_vector_db(documents):
    try:
        # Belgeleri parçalara böl - daha büyük parçalar ve daha fazla örtüşme
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Daha büyük chunk boyutu
            chunk_overlap=250,  # Daha fazla örtüşme
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        st.info(f"Belgeler {len(chunks)} parçaya bölündü")
        
        # Yüklenen belgelerin bir kısmını göster
        with st.expander("Yüklenen Belgeler", expanded=False):
            for doc in documents:
                st.markdown(f"**{doc.metadata.get('source', 'Bilinmeyen')}**")
                st.text(doc.page_content[:300] + "...")
                st.markdown("---")
        
        # Embeddings oluştur
        try:
            # API anahtarını kontrol et
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                st.error("API anahtarı bulunamadı!")
                return None
                
            # API anahtarının güvenli kontrolü
            print(f"API anahtarı Secrets'dan yüklendi ve kullanıma hazır")
            
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"  # Güncel embedding modeli
            )
            print("OpenAIEmbeddings başarıyla oluşturuldu")
            
            # DocArrayInMemorySearch vektör veritabanı oluştur
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
            st.error(f"Embedding oluşturulurken hata: {str(e)}")
            return None
            
    except Exception as e:
        logging.error(f"Vektör veritabanı hatası: {str(e)}")
        st.error(f"Vektör veritabanı oluşturulurken hata: {str(e)}")
        return None

# ---------------------------------------
# 7. Gelişmiş Prompt Şablonu
# ---------------------------------------
def create_qa_prompt():
    template = """
    Sen Villa Villa organizasyon şirketinin yapay zekâ destekli finans ve operasyon asistanısın.
    
    Villa Villa Yapay Zeka Destekli Chatbot Çalışma Sistemine göre yanıt vermelisin:
    
    1. Tarih ve Yapılan İş Bilgisi Sorguları:
       - Yapılan işler tarih bazlı sorgulandığında, bilgi "yapilan_isler.docx" adlı dosya üzerinden alınır.
       - En güncel iş, tarih sıralamasına göre tespit edilir ve detaylarıyla sunulur.
    
    2. Yapılan İşlere Göre Maliyet Hesaplama:
       - Talep edilen işle ilgili maliyet çıkarılırken "yapilan_isler.docx" dosyasındaki işletme ve hizmet detayları temel alınır.
       - İşin türü, kişi sayısı, menü içeriği ve yer bilgileri dikkate alınarak değerlendirme yapılır.
    
    3. Aylık Gider ve Maliyet Analizi:
       - Belirli bir aya ilişkin gider sorgularında dört dosya birlikte değerlendirilir:
         * "genel_gider.docx" - Ofis ve temel giderler
         * "personel_giderleri.docx" - Çalışan maaşları ve ödemeleri
         * "gelen_faturalar.docx" - Tedarikçilerden gelen faturalar
         * "yapilan_isler.docx" - Tamamlanan işler ve gelirleri
       - Bu dosyalar doğrultusunda, tedarikçi giderleri, fatura kalemleri ve iş bazlı maliyetler birleştirilerek kapsamlı bir analiz yapılır.
    
    4. Menü Teklifi Oluşturma:
       - Menü teklifi istenen işlerde, "yapilan_isler.docx" dosyasında yer alan örnek menüler incelenir.
       - İşin niteliği (açılış, davet, kurumsal vb.) ve kişi sayısı göz önünde bulundurularak benzer işler temel alınır, uygun menü önerisi hazırlanır.
    
    Aşağıdaki bilgiler doğrultusunda müşterinin sorusunu yanıtla:
    
    Belgelerden Bilgiler:
    {context}
    
    Sohbet Geçmişi:
    {chat_history}
    
    Soru:
    {question}
    
    Yanıtında şu noktalara dikkat et:
    - Yalnızca belgedeki gerçek bilgileri kullan ve belge isimlerini referans ver
    - İlgili belgede yanıt yoksa, hangi belgenin bu bilgiyi içermesi gerektiğini belirt
    - Tarih bazlı sorularda en güncel bilgileri öncelikle göster
    - Hesaplama gerektiren yanıtlarda detaylı olarak her kalemi göster
    - Sayısal verileri tablolar ile düzenli biçimde sun
    - Profesyonel, açık ve kibar bir dil kullan
    
    Yanıt:
    """
    return PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)

# ---------------------------------------
# 8. Chat Zinciri Kurulumu
# ---------------------------------------
def create_chat_chain(vector_db):
    try:
        llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4-turbo"  # Güncel model adı
        )
        
        # Gelişmiş retriever - MMR arama ve daha fazla belge
        retriever = vector_db.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance - daha çeşitli sonuçlar
            search_kwargs={"k": 8, "fetch_k": 15}  # Daha fazla belge getir
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
        st.error(f"Sohbet sistemi oluşturulamadı: {str(e)}")
        return None

# ---------------------------------------
# 9. Ana Uygulama
# ---------------------------------------
def main():
    # Global API anahtarı ayarla (sadece Secrets'dan)
    if not set_openai_api_key():
        st.stop()

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
            
        # Önbelleği temizleme butonu
        if st.button("Önbelleği Temizle"):
            try:
                st.cache_data.clear()
            except:
                pass
            try:
                st.cache_resource.clear()
            except:
                pass
            st.success("Önbellek temizlendi!")
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
    
    # Açıklama ekranı
    st.markdown("""
    ### 🤖 Villa Villa Yapay Zeka Asistanıyla Neler Yapabilirsiniz?
    
    **1. Tarih ve İş Bilgisi Sorguları:**
    - "15 Nisan'da hangi etkinlik düzenlendi?"
    - "Son yapılan etkinliğin detaylarını göster."
    
    **2. Maliyet Hesaplama:**
    - "100 kişilik bir kurumsal etkinliğin maliyeti ne olur?"
    - "Düğün organizasyonu için ortalama maliyet nedir?"
    
    **3. Aylık Analiz:**
    - "Nisan ayı toplam giderleri nelerdir?"
    - "Mart ve Nisan ayları arasında gider farkı ne kadar?"
    
    **4. Menü Önerileri:**
    - "50 kişilik doğum günü için menü önerisi verir misin?"
    - "Kurumsal lansman için kokteyl menüsü nasıl olmalı?"
    """)
    
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
                        st.markdown(f"```\n{doc.page_content[:300]}...\n```")
            
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