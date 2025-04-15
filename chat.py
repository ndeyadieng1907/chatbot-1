import streamlit as st
from audio_recorder_streamlit import audio_recorder

st.title("Chatbot Vocal Wolof")

audio_bytes = audio_recorder()
if audio_bytes:
    with open("user_input.wav", "wb") as f:
        f.write(audio_bytes)
    
    question = speech_to_text_wolof("user_input.wav")
    st.write(f"Question: {question}")
    
    # Ici votre logique de génération de réponse
    answer = "Ceci est une réponse exemple"
    
    st.write(f"Réponse: {answer}")
    audio_response = text_to_speech_wolof(answer)
    st.audio(audio_response)
