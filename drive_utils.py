import re
import tempfile
import logging
import requests
from docx import Document as DocxDocument
from langchain.docstore.document import Document
import streamlit as st
import os

def extract_document_id(url):
    pattern = r"/d/([a-zA-Z0-9-_]+)"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def download_google_doc(doc_url, file_format="docx"):
    try:
        doc_id = extract_document_id(doc_url)
        if not doc_id:
            return None, None
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format={file_format}"
        response = requests.get(export_url)
        if response.status_code != 200:
            return None, None
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name, doc_id
    except Exception as e:
        logging.error(f"Download error: {str(e)}")
        return None, None

def process_document(file_path, doc_name, file_format="docx"):
    try:
        if file_format == "docx":
            docx = DocxDocument(file_path)
            text = "\n".join([p.text for p in docx.paragraphs if p.text.strip()])
            for table in docx.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text for cell in row.cells])
                    text += f"\n{row_text}"
            return Document(page_content=text, metadata={"source": doc_name})
        elif file_format == "txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return Document(page_content=text, metadata={"source": doc_name})
        else:
            return None
    finally:
        try:
            os.remove(file_path)
        except:
            pass

def load_documents_from_urls(doc_urls):
    documents = []
    progress_bar = st.progress(0)
    successful, failed = [], []
    for idx, (name, url) in enumerate(doc_urls.items()):
        file_path, _ = download_google_doc(url, "docx")
        if not file_path:
            file_path, _ = download_google_doc(url, "txt")
        if not file_path:
            st.error(f"{name} indirilemedi.")
            failed.append(name)
            continue
        doc = process_document(file_path, name, "docx")
        if doc:
            documents.append(doc)
            successful.append(name)
        else:
            failed.append(name)
        progress_bar.progress((idx + 1) / len(doc_urls))
    if successful:
        st.success(f"{len(successful)} belge yüklendi: {', '.join(successful)}")
    if failed:
        st.warning(f"{len(failed)} belge başarısız: {', '.join(failed)}")
    return documents
