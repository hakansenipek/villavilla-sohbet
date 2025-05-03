from langchain.prompts import PromptTemplate

def create_qa_prompt():
    template = """
    Sen Villa Villa şirketinin finans ve operasyon asistanısın...
    (diğer kurallar olduğu gibi buraya kopyalanır)
    """
    return PromptTemplate(input_variables=["context", "chat_history", "question"], template=template)
