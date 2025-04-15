# ----------------------- Importations améliorées -----------------------
import os
import magic
import torch
from gtts import gTTS
from pydub import AudioSegment
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# LangChain imports
from langchain.document_loaders import (
    TextLoader, PDFMinerLoader, CSVLoader, 
    UnstructuredExcelLoader, UnstructuredImageLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# ----------------------- Configuration initiale -----------------------
# Chemin vers le dossier contenant les documents
folder_path = "/content/drive/MyDrive/data-ansd"

# ----------------------- Fonctions améliorées -----------------------

def speech_to_text_wolof(audio_file_path, language="wo-SN"):
    """Convertit un fichier audio en texte avec meilleure gestion des erreurs."""
    r = sr.Recognizer()
    try:
        # Conversion du format audio si nécessaire
        sound = AudioSegment.from_file(audio_file_path)
        sound.export("temp_audio.wav", format="wav")
        
        with sr.AudioFile("temp_audio.wav") as source:
            audio = r.record(source)
            
        text = r.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        return "Je n'ai pas compris l'audio."
    except sr.RequestError as e:
        return f"Erreur de service: {e}"
    except Exception as e:
        return f"Erreur inattendue: {e}"
    finally:
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

def text_to_speech_wolof(text, output_file="output_wolof.mp3"):
    """Synthèse vocale avec gestion des erreurs."""
    try:
        tts = gTTS(text=text, lang='wo', slow=False)
        tts.save(output_file)
        return output_file
    except Exception as e:
        print(f"Erreur TTS: {e}")
        return None

def record_audio(filename="user_input.wav", duration=5, fs=44100):
    """Enregistrement audio avec feedback visuel."""
    print("Enregistrement en cours... parlez maintenant")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Attendre la fin
        wav.write(filename, fs, recording)
        print("Enregistrement terminé")
        return filename
    except Exception as e:
        print(f"Erreur d'enregistrement: {e}")
        return None

def load_documents_from_folder(folder_path):
    """Charge les documents avec meilleure détection des types."""
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            file_type = magic.from_file(file_path, mime=True)
            
            if file_type == "application/pdf":
                loader = PDFMinerLoader(file_path)
            elif file_type == "text/csv":
                loader = CSVLoader(file_path)
            elif file_type in ["application/vnd.ms-excel", 
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                loader = UnstructuredExcelLoader(file_path)
            elif file_type.startswith("image/"):
                loader = UnstructuredImageLoader(file_path)
            elif file_type == "text/plain":
                loader = TextLoader(file_path)
            else:
                print(f"Type non supporté: {filename}")
                continue
                
            documents.extend(loader.load())
        except Exception as e:
            print(f"Erreur avec {filename}: {str(e)}")
    
    return documents

def create_index_from_documents(documents):
    """Crée l'index vectoriel avec des paramètres optimisés."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(documents)
    
    # Embeddings multilingues pour Wolof/Français
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def initialize_llm():
    """Initialise un modèle plus léger et accessible."""
    model_name = "facebook/mbart-large-50"  # Modèle multilingue
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=256
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        print(f"Erreur de chargement du modèle: {e}")
        return None

def create_qa_chain(llm, retriever):
    """Crée la chaîne QA avec des paramètres optimisés."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

def chatbot():
    """Fonction principale améliorée avec gestion des erreurs."""
    # Initialisation des composants
    print("Initialisation du chatbot...")
    
    llm = initialize_llm()
    if not llm:
        print("Échec de l'initialisation du modèle de langage")
        return
    
    print("Chargement des documents...")
    documents = load_documents_from_folder(folder_path)
    if not documents:
        print("Aucun document chargé")
        return
    
    print("Création de l'index...")
    retriever = create_index_from_documents(documents)
    
    print("Création de la chaîne QA...")
    qa_chain = create_qa_chain(llm, retriever)
    
    print("Chatbot prêt! Parlez ou tapez 'quitter' pour terminer.")
    
    # Boucle de conversation
    while True:
        # Enregistrement audio
        audio_file = record_audio()
        if not audio_file:
            continue
            
        # Reconnaissance vocale
        question_wolof = speech_to_text_wolof(audio_file)
        print(f"Vous: {question_wolof}")
        
        if question_wolof.lower() in ["quitter", "stop", "au revoir"]:
            print("Au revoir!")
            break
            
        # Traitement de la question
        try:
            result = qa_chain({"query": question_wolof})
            answer = result["result"]
            print(f"Réponse: {answer}")
            
            # Synthèse vocale
            audio_response = text_to_speech_wolof(answer)
            if audio_response:
                os.system(f"start {audio_response}")  # Windows
        except Exception as e:
            print(f"Erreur: {str(e)}")
            answer = "Désolé, une erreur s'est produite"
            
# ----------------------- Lancement -----------------------
if __name__ == "__main__":
    chatbot()
