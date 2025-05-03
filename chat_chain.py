from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from prompts import create_qa_prompt
import logging

def create_chat_chain(vector_db):
    try:
        llm = ChatOpenAI(temperature=0.2, model_name="gpt-4-turbo", streaming=True)
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        qa_prompt = create_qa_prompt()
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=True
        )
    except Exception as e:
        logging.error(f"create_chat_chain error: {str(e)}")
        return None
