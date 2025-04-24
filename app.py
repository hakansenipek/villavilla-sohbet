import os
import streamlit as st
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from PIL import Image
import shutil

"openai" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]


# 📂 Belge klasörü - GitHub'a yüklenecek belgeler için göreceli yol kullan
# Proje klasörünüzde "belgeler" adında bir klasör oluşturun ve belgeleri oraya koyun
belge_klasoru = "belgeler"
belge_dosyalar = [
    "genel gider.docx",
    "gelen faturalar.docx",
    "yapılan işler.docx"
]

# 🧠 Sayfa başlığı ve düzeni
st.set_page_config(page_title="Villa Villa Yapay Zeka ile Sohbet", layout="centered")

# Logoyu ve başlığı yatay sırada göster
col1, col2 = st.columns([1, 3])
with col1:
    logo_path = "assets/villa_villa_logo.jpg"  # Logo dosyasını assets klasörüne koyun
    if os.path.exists(logo_path):
        st.image(Image.open(logo_path), width=100)
    else:
        st.write("Logo bulunamadı")
with col2:
    st.title("Villa Villa Yapay Zeka ile Sohbet")

# Ayırıcı çizgi
st.markdown("---")

# 🧠 Sohbet geçmişi
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Widget'lar dışarı alındı ve cached function iyileştirildi
def temizle_veritabani(cache_path):
    """Veritabanını temizleyen yardımcı fonksiyon."""
    if os.path.exists(cache_path):
        try:
            shutil.rmtree(cache_path)
            return True, f"Veritabanı temizlendi: {cache_path}"
        except Exception as e:
            return False, f"Veritabanı temizleme hatası: {str(e)}"
    return False, "Veritabanı bulunamadı"

# 📄 Belgeleri yükle ve vektörleştir (arka planda)
@st.cache_resource
def yukle_ve_vektorlestir(force_reload=False):
    """
    Belgeleri yükleyip vektörleştiren ana fonksiyon.
    Widget'lar bu fonksiyonun dışında kullanılmalı.
    """
    cache_path = "vektor_db"
    
    # force_reload burada sadece bir parametre, widget içermiyor
    if force_reload:
        success, message = temizle_veritabani(cache_path)
        print(message)
    
    try:
        dokumanlar = []
        for dosya in belge_dosyalar:
            dosya_yolu = os.path.join(belge_klasoru, dosya)
            try:
                loader = Docx2txtLoader(dosya_yolu)
                belge = loader.load()
                # Dosya adını metadata olarak ekle
                for doc in belge:
                    doc.metadata["source"] = dosya
                dokumanlar.extend(belge)
                print(f"Belge yüklendi: {dosya} - {len(belge)} bölüm")
            except Exception as e:
                print(f"Belge yükleme hatası: {dosya} - {str(e)}")
        
        if dokumanlar:
            # 🔠 Gelişmiş bölme ve vektörleme
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Daha küçük parçalar
                chunk_overlap=150,  # Daha fazla örtüşme
                separators=["\n\n", "\n", ". ", " ", ""],  # Özel ayırıcılar
                length_function=len
            )
            parcalar = splitter.split_documents(dokumanlar)
            print(f"Toplam {len(parcalar)} parça oluşturuldu")
            
            # OpenAI embeddings modelini kullan - varsayılan model
            embeddings = OpenAIEmbeddings()
            
            # Chroma veritabanını oluştur
            vektordb = Chroma.from_documents(
                documents=parcalar,
                embedding=embeddings,
                persist_directory=cache_path
            )
            
            # Verileri diske kaydet
            vektordb.persist()
            return vektordb
    except Exception as e:
        print(f"Vektörleştirme hatası: {str(e)}")
    return None

# Session state değişkenini başlat
if "yeniden_yukle" not in st.session_state:
    st.session_state.yeniden_yukle = False

# Kontrol panelini gizli bir şekilde ekleyelim - Widget'lar cached function dışında
with st.expander("⚙️ Yönetici Ayarları", expanded=False):
    yeniden_yukle_butonu = st.button("🔄 Veritabanını Yeniden Oluştur")
    if yeniden_yukle_butonu:
        st.session_state.yeniden_yukle = True
        st.success("Veritabanı yeniden oluşturuluyor...")
        st.rerun()

# Veritabanını yükle - cached function'a parametre olarak session state'i geçiriyoruz
force_reload_param = st.session_state.yeniden_yukle
vektordb = yukle_ve_vektorlestir(force_reload=force_reload_param)

# Yeniden yükleme bayrağını sıfırla
if st.session_state.yeniden_yukle:
    st.session_state.yeniden_yukle = False

if vektordb:
    # ChatOpenAI modelini GPT-4 olarak ayarla
    llm = ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4-0125-preview",
        verbose=True
    )
    
    # Retriever'ı yapılandır
    retriever = vektordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Sorgu zinciri oluştur - özel bir prompt şablonu kullanarak
    template = """
    Sen Villa Villa şirketi için bir yapay zeka asistanısın. Aşağıdaki belgelerden aldığın bilgilere dayanarak soruları yanıtla.
    
    Belgeler:
    {context}
    
    Mevcut Sohbet:
    {chat_history}
    
    Soru: {question}
    
    Villa Villa Yapay Zeka Destekli Chatbot'un Çalışma Sistemi:
    
    1. Tarih ve Yapılan İş Bilgisi Sorguları:
       - Yapılan işler tarih bazlı sorgulandığında, bilgi "yapılan işler" adlı dosya üzerinden alınır.
       - En güncel iş, tarih sıralamasına göre tespit edilir ve detaylarıyla sunulur.
    
    2. Yapılan İşlere Göre Maliyet Hesaplama:
       - Talep edilen işle ilgili maliyet çıkarılırken "yapılan işler" dosyasındaki işletme ve hizmet detayları temel alınır.
       - İşin türü, kişi sayısı, menü içeriği ve yer bilgileri dikkate alınarak değerlendirme yapılır.
    
    3. Aylık Gider ve Maliyet Analizi:
       - Belirli bir aya ilişkin gider sorgularında üç dosya birlikte değerlendirilir: "genel gider", "gelen faturalar", "yapılan işler"
       - Bu dosyalar doğrultusunda, tedarikçi giderleri, fatura kalemleri ve iş bazlı maliyetler birleştirilerek kapsamlı bir analiz yapılır.
    
    4. Menü Teklifi Oluşturma:
       - Menü teklifi istenen işlerde, "yapılan işler" dosyasında yer alan örnek menüler incelenir.
       - İşin niteliği (açılış, davet, kurumsal vb.) ve kişi sayısı göz önünde bulundurularak benzer işler temel alınır, uygun menü önerisi hazırlanır.
    
    Talimatlar:
    1. Villa Villa şirketinin belgeleri ve verilerine dayanarak yukarıdaki çalışma sistemine göre yanıt ver.
    2. Tüm yanıtlarında sadece belgelerden edindiğin bilgilere dayan, tahmin yürütme.
    3. Bilgi bulamadığın durumlarda bunu açıkça belirt.
    4. Sonuçları mümkün olduğunca yapılandırılmış ve okunması kolay bir şekilde sun.
    5. Müşteriye her zaman nazik ve profesyonel ol.
    
    Detaylı Yanıt:"""
    
    QA_PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )
    
    sorgu = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=True
    )
    
    # Temizleme butonu - sağ üstte küçük
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🔄 Temizle", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # 🧠 Sohbet geçmişini görüntüle
    for i in range(0, len(st.session_state.chat_history), 2):
        if i < len(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(st.session_state.chat_history[i][1])
            
        if i+1 < len(st.session_state.chat_history):
            with st.chat_message("assistant"):
                st.markdown(st.session_state.chat_history[i+1][1])
    
    # ✏️ Soru girişi
    user_input = st.chat_input("Sorunuzu yazın")
    
    if user_input:
        # Kullanıcı mesajını ekrana ve geçmişe ekle
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append(("user", user_input))
        
        # Sorguyu çalıştır
        with st.spinner("Düşünüyor..."):
            try:
                # Sohbet geçmişini doğru formatta hazırla
                if len(st.session_state.chat_history) > 1:
                    gecmis = []
                    for i in range(0, len(st.session_state.chat_history)-1, 2):
                        if i+1 < len(st.session_state.chat_history):
                            gecmis.append((st.session_state.chat_history[i][1], 
                                           st.session_state.chat_history[i+1][1]))
                else:
                    gecmis = []
                
                # Sorguyu çalıştır
                yanit = sorgu({
                    "question": user_input, 
                    "chat_history": gecmis
                })
                
                # Yanıtı göster
                with st.chat_message("assistant"):
                    st.markdown(yanit["answer"])
                    
                    # Kullanılan kaynakları göster (isteğe bağlı, kolayca açılabilir)
                    with st.expander("Kullanılan Kaynaklar", expanded=False):
                        for i, doc in enumerate(yanit["source_documents"]):
                            st.markdown(f"**Kaynak {i+1}:** {doc.metadata.get('source', 'Bilinmeyen')}")
                            st.markdown(f"```\n{doc.page_content[:200]}...\n```")
                
                # Yanıtı geçmişe ekle
                st.session_state.chat_history.append(("ai", yanit["answer"]))
                
            except Exception as e:
                st.error(f"Yanıt oluşturulurken bir hata oluştu: {str(e)}")
                # Basit bir yanıt vererek kullanıcıyı bilgilendir
                with st.chat_message("assistant"):
                    st.markdown("Üzgünüm, şu anda yanıt verirken bir sorun yaşıyorum. Lütfen başka bir soru sorun veya daha sonra tekrar deneyin.")
                # Hata mesajını geçmişe ekle
                st.session_state.chat_history.append(("ai", "Üzgünüm, teknik bir sorun yaşandı."))
else:
    st.error("Belgeler yüklenemedi. Lütfen belgeler klasöründeki dosyaları kontrol ediniz.")