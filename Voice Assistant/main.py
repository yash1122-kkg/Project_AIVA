import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import openai
import pyttsx3
import os
import speech_recognition as sr
from googletrans import Translator
from htmlTempletes import css, bot_template, user_template

# Load your OpenAI API key from environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_file_path):
    text = ""
    pdf_reader = PdfReader(pdf_file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=40,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def translate_text(text, source_lang, dest_lang):
    translator = Translator()
    translated_text = translator.translate(text, src=source_lang, dest=dest_lang).text
    return translated_text

def recognize_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say something...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    
    try:
        user_input = r.recognize_google(audio)
        return user_input
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand your audio.")
        return ""
    except sr.RequestError as e:
        st.write("Error fetching results: {0}".format(e))
        return ""

def handle_userinput(user_question, conversation_chain, language):
    response = conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            if language == 'Hindi':
                translated_response = translate_text(message.content, 'en', 'hi')
                st.write(bot_template.replace(
                    "{{MSG}}", translated_response), unsafe_allow_html=True)
                
                # Convert bot response to speech in Hindi
                engine = pyttsx3.init()

                # Select a female voice for Hindi (if available)
                voices = engine.getProperty('voices')
                for voice in voices:
                    if "hindi" in voice.name.lower() and "female" in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                else:
                    engine.setProperty('voice', voices[1].id)  # Default to second voice if no suitable Hindi voice found

                # Set speech parameters for a more human-like voice
                engine.setProperty('rate', 190)  # Adjust the rate (words per minute)
                engine.setProperty('volume', 1.0)  # Adjust the volume (0.0 to 1.0)
                
                engine.say(translated_response)
                engine.runAndWait()
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
                
                # Convert bot response to speech in English
                engine = pyttsx3.init()

                # Select a female voice for English
                voices = engine.getProperty('voices')
                for voice in voices:
                    if "english" in voice.name.lower() and "female" in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                else:
                    engine.setProperty('voice', voices[3].id)  # Default to first voice if no suitable English voice found

                # Set speech parameters for a more human-like voice
                engine.setProperty('rate', 150)  # Adjust the rate (words per minute)
                engine.setProperty('volume', 1.0)  # Adjust the volume (0.0 to 1.0)
                
                engine.say(message.content)
                engine.runAndWait()

def main():
    load_dotenv()
    st.set_page_config(page_title="AIVA",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "language" not in st.session_state:
        st.session_state.language = 'English'

    st.image("aiva-high-resolution-logo-transparent.png", width=275)  # Add this line to display logo above the header

    st.header("AIVA :books:")

    # Language selection buttons
    if st.button("Hindi"):
        st.session_state.language = 'Hindi'
    if st.button("English"):
        st.session_state.language = 'English'

    st.write(f"Current language: {st.session_state.language}")

    # File path to default PDF file
    pdf_file_path = "DYPCET v3.0.pdf"

    # Read PDF file
    pdf_text = get_pdf_text(pdf_file_path)

    # get the text chunks
    text_chunks = get_text_chunks(pdf_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # create conversation chain
    st.session_state.conversation = get_conversation_chain(
        vectorstore)

    # Voice input button
    if st.button("Voice Input"):
        user_question = recognize_voice()
        if user_question:
            handle_userinput(user_question, st.session_state.conversation, st.session_state.language)

    # Sidebar with FAQ section
    st.sidebar.header("FAQs")
    faq_questions = {
        "What engineering programs do you offer?": "What engineering programs do you offer?",
        "Can you provide an overview of the engineering curriculum?": "Can you provide an overview of the engineering curriculum?",
        "Overview of DYPCET college?": "Overview of DYPCET college?",
        "About Training and Placement Cell of DYPCET?": "About Training and Placement Cell of DYPCET?",
        "Top recruiters of Campus Placement?":"Top recruiters of Campus Placement?",
        "Does College provide any Foreign Language programs?":"Does College provide any Foreign Language programs?",
        "What are various clubs in DYPCET?":"What are various clubs in DYPCET?"
    }
    for faq_question, input_question in faq_questions.items():
        if st.sidebar.button(faq_question):
            handle_userinput(input_question, st.session_state.conversation, st.session_state.language)

if __name__ == '__main__':
    main()
