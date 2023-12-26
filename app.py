from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.schema import retriever
from langchain.chains import RetrievalQA
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.llms import OpenAI
 
from dotenv import load_dotenv
import os 
import qdrant_client
import streamlit as st
# load keys from .env file
load_dotenv()
mistral_api_key = os.getenv('MISTRALAI_API_KEY')
qdrant_uri = os.getenv('QDRANT_URI')
qdrant_api_key = os.getenv('QDRANT_API_KEY')

def get_vectore_store():
    
    client = qdrant_client.QdrantClient(
        qdrant_uri,
        api_key=qdrant_api_key
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectore_store = Qdrant(
        client=client,
        collection_name="my-collection",
        embeddings=embeddings
    )
      
    return vectore_store
  
def main():
    st.set_page_config(page_title="MLOps Guide")
    st.header("I'm your MLOps teacher, ask about MLOps💭")
    
    vectore_store = get_vectore_store()
    llm = ChatMistralAI(mistral_api_key=mistral_api_key,
                        model='mistral-small')
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectore_store.as_retriever()
    )

    # show user input
    user_question = st.text_input("Ask a question: ")
    if user_question:
        st.write(f"Question: {user_question}")
        answer = qa.run(user_question)
        st.write(f"Answer: {answer}")

if __name__ == '__main__':
    main()
