import re, os, tempfile, logging, requests
from docx import Document as DocxDocument
from langchain.docstore.document import Document

def extract_document_id(url):
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    return match.group(1) if match else None

def download_google_doc(doc_url, file_format="docx"):
    doc_id = extract_document_id(doc_url)
    if not doc_id: return None, None
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format={file_format}"
    response = requests.get(export_url)
    if response.status_code != 200: return None, None
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}")
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name, doc_id

def process_document(file_path, doc_name, file_format="docx"):
    try:
        if file_format == "docx":
            docx = DocxDocument(file_path)
            text = "\n".join(p.text for p in docx.paragraphs if p.text.strip())
            for table in docx.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text for cell in row.cells)
                    text += f"\n{row_text}"
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        return Document(page_content=text, metadata={"source": doc_name})
    except Exception as e:
        logging.error(f"process_document error: {str(e)}")
        return None
    finally:
        try: os.remove(file_path)
        except: pass
