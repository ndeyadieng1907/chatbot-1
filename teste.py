import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import HuggingFaceChatModel
import asyncio

load_dotenv()

st.set_page_config(page_title="ANSD'S bot", page_icon=":ansd-sn:", layout="wide")

def load_db(file, chain_type, k):
    async def load_and_process():
        try:
            loader = PyPDFLoader(file)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")  # Utilisation de Hugging Face pour les embeddings
            db = DocArrayInMemorySearch.from_documents(docs, embeddings)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
            qa = ConversationalRetrievalChain.from_llm(
                llm=HuggingFaceChatModel(model_name="gpt2", temperature=0),  # Utilisation d'un modèle de Hugging Face
                chain_type=chain_type,
                retriever=retriever,
                return_source_documents=True,
                return_generated_question=True,
            )
            return qa
        except Exception as e:
            st.error(f"Erreur lors du chargement de la base de données : {str(e)}")
            return None

    return asyncio.run(load_and_process())

async def chat_with_bot(cb, query):
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: cb({"question": query, "chat_history": []}))
        return response
    except Exception as e:
        st.error(f"Erreur lors de la communication avec le chatbot : {str(e)}")
        return {"answer": "Désolé, une erreur est survenue."}

def main():
    st.title("ANSD's bot ✨")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Je suis votre assistant dédié à l'ANSD. N'hésitez pas à me poser des questions sur les informations statistiques du Sénégal. Je suis là pour vous fournir des réponses précises et utiles.")
    with col2:
        original_image = Image.open("image.png")
        st.image(original_image)

    uploaded_file = st.file_uploader("MORTALITE-Rapport-Provisoire-RGPH5_juillet2024.pdf", type="pdf")
    if uploaded_file:
        file_path = os.path.join("./", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        cb = load_db(file_path, "stuff", 4)

        query = st.text_input("Poser une question:")
        if st.button("Demander"):
            result = asyncio.run(chat_with_bot(cb, query))
            response = result.get("answer", "Désolé, je n'ai pas pu répondre à votre question.")
            st.write("ChatBot:", response)

if __name__ == '__main__':
    main()
