import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import os

# Fonction pour lire le fichier PDF et créer un dictionnaire
def lire_dictionnaire(fichier_pdf):
    loader = PyPDFLoader(fichier_pdf)
    data = loader.load()
    dictionnaire = {}

    for page in data:
        for line in page.page_content.split('\n'):
            if ':' in line:
                francais, wolof = line.split(':', 1)
                dictionnaire[francais.strip().lower()] = wolof.strip()

    return dictionnaire

# Fonction pour charger le document
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# Fonction pour diviser les données en morceaux
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# Fonction pour créer des embeddings
def create_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-uncased")
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

# Fonction pour poser une question et obtenir une réponse
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import HuggingFaceChatModel

    llm = HuggingFaceChatModel(model_name='gpt2')  # Utiliser un modèle de Hugging Face
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer

# Fonction pour calculer le coût des embeddings
def calculate_embedding_cost(texts):
    # Cette fonction peut être adaptée selon la méthode de calcul des coûts
    total_tokens = sum(len(page.page_content.split()) for page in texts)
    return total_tokens, total_tokens / 1000 * 0.0004  # Exemple de coût

# Fonction pour traduire un message en wolof
def traduire_wolof(message, dictionnaire):
    mots = message.split()
    traduction = []
    for mot in mots:
        traduction.append(dictionnaire.get(mot.lower(), mot))
    return ' '.join(traduction)

# Fonction pour effacer l'historique de chat
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        api_key = st.text_input('Hugging Face API Key:', type='password')
        if api_key:
            os.environ['HUGGING_FACE_TOKEN'] = api_key

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # Charger le dictionnaire à partir du fichier PDF
    dictionnaire = {}
    dictionnaire_file = st.file_uploader('Upload your French-Wolof dictionary (PDF file):', type=['pdf'])
    if dictionnaire_file:
        dictionnaire = lire_dictionnaire(dictionnaire_file)
        st.success('Dictionnaire chargé avec succès.')

    # Entrée de question de l'utilisateur
    q = st.text_input('Ask a question about the content of your file:')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            answer = ask_and_get_answer(vector_store, q, k)
            st.write(f"Chatbot : {answer}")

            # Traduire la réponse en wolof
            if dictionnaire:
                answer_wolof = traduire_wolof(answer, dictionnaire)
                st.write(f"Chatbot en Wolof : {answer_wolof}")
