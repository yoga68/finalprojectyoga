import os
import streamlit as st
import pandas as pd 

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_utils import (
    extract_text_from_pdf, 
    split_text, 
    create_vectorstore, 
    create_rag_chain,
    save_uploaded_file,
    cleanup_temp_dir,
    process_multiple_pdfs
)



st.title("GEBRAK DOBRAK AI")

def load_data(data):
    return pd.read_csv(data)

def validate_google_api_key(api_key: str) -> tuple[bool, str]:
    """Validate Google API key by attempting to initialize and use the model."""
    if not api_key or len(api_key.strip()) == 0:
        return False, "API Key tidak boleh kosong."
    
    try:
        # Coba inisialisasi model dengan API key
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
        
        # Coba gunakan model untuk memastikan API key benar-benar valid
        test_response = llm.invoke("Test")
        if test_response:
            return True, "API Key valid"
        return False, "API Key tidak dapat digunakan untuk mengakses model."
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid api key" in error_msg or "unauthorized" in error_msg:
            return False, "API Key tidak valid atau tidak memiliki akses yang diperlukan."
        return False, f"Gagal memverifikasi API Key: {str(e)}"

def get_api_key_input():
    local_logo_path = os.path.join("assets", "logo.png")
    st.image(local_logo_path, width=200)
    
    st.write("Masukkan Google API Key")
    
    if "GOOGLE_API_KEY" not in st.session_state:
        st.session_state["GOOGLE_API_KEY"] = ""
    
    # Container untuk form dan pesan error
    form_container = st.container()
    
    # Container untuk pesan error/success
    message_container = st.container()
    
    with form_container:
        col1, col2 = st.columns((80, 20))
        with col1:
            api_key = st.text_input("", label_visibility="collapsed", type="password")

        with col2:
            is_submit_pressed = st.button("Submit")
    
    # Tampilkan pesan di container yang sudah ditetapkan
    if is_submit_pressed:
        with message_container:
            # Create placeholder for animated transition
            message_placeholder = st.empty()
            
            # Show spinner in placeholder
            with message_placeholder:
                with st.spinner("🔄 Memverifikasi API Key..."):
                    # Validate the API key
                    is_valid, message = validate_google_api_key(api_key)
            
            # Add slight delay for smooth transition
            import time
            time.sleep(0.5)
            
            # Replace spinner with result message
            with message_placeholder:
                if is_valid:
                    st.session_state["GOOGLE_API_KEY"] = api_key
                    os.environ["GOOGLE_API_KEY"] = api_key
                    st.success("✅ API Key berhasil diverifikasi.")
                else:
                    st.error(f"❌ {message}")
                    st.session_state["GOOGLE_API_KEY"] = ""
                    os.environ["GOOGLE_API_KEY"] = ""
                    st.stop()
                    return

# API Key input hanya tampil jika belum diverifikasi
if not st.session_state.get("GOOGLE_API_KEY"):
    get_api_key_input()
    st.stop()
else:
    st.success("API Key sudah terverifikasi, saat ini anda menggunakan metode full knowledge.")
    st.info("Untuk mengaktifkan batasan knowledge, silahkan unggah file dokumen PDF dibawah ini")
# Setelah API key diverifikasi, baru tampilkan upload dokumen dan tombol clear
with st.form("upload_form"):
    uploaded_files = st.file_uploader("Upload PDF File(s)", type=["pdf"], accept_multiple_files=True)
    submitted = st.form_submit_button("Submit")

# Tombol clear untuk menghapus batasan RAG dan temporary files
if st.button("Clear PDF Context"):
    cleanup_temp_dir()  # Hapus semua file temporary
    st.session_state["conversation"] = None
    st.session_state["chat_history"] = []
    st.success("Batasan knowledge dari PDF telah dihapus dan file temporary dibersihkan. LLM kembali ke mode full knowledge.")

if submitted and uploaded_files:
    with st.spinner("Penyesuaian Knowledge atas PDF(s)..."):
        # Save uploaded files
        for file in uploaded_files:
            save_uploaded_file(file)
        
        # Process all PDFs
        raw_text = process_multiple_pdfs(uploaded_files)
        text_chunks = split_text(raw_text)
        vectorstore = create_vectorstore(text_chunks)
        st.session_state["conversation"] = create_rag_chain(vectorstore)
        st.session_state["chat_history"] = []  # Reset chat history
        st.success(f"{len(uploaded_files)} PDF(s) Berhasil Diproses! Chat history telah direset.")
        st.info("Mode terbatas: Jawablah pertanyaan yang hanya relevan dengan dokumen PDF yang diupload.")

def load_llm():
    if "llm" not in st.session_state:
        st.session_state["llm"] = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    return st.session_state["llm"]


def get_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    return st.session_state["chat_history"]


def display_chat_message(message):
    
    if type(message) is HumanMessage:
        role = "User"
        avatar = "💻"
    elif type(message) is AIMessage:
        role = "AI"
        avatar ="💭"
    else:
        role = "Unknown"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message.content)


def user_query_to_llm(chat_history):
    prompt = st.chat_input("Yuk Tanya Mindi")
    if not prompt:
        st.stop()
    chat_history.append(HumanMessage(content=prompt))
    display_chat_message(chat_history[-1])
    with st.spinner("Thinking..."):
        if "conversation" in st.session_state and st.session_state["conversation"] is not None:
            # RAG mode: batasi knowledge ke PDF
            response = st.session_state["conversation"]({
                "question": prompt,
                "chat_history": [(msg.content, resp.content) for msg, resp in zip(chat_history[::2], chat_history[1::2])]
            })
            ai_msg = AIMessage(content=response["answer"])
        else:
            # LLM full knowledge
            response = llm.invoke(chat_history)
            ai_msg = response
    chat_history.append(ai_msg)
    display_chat_message(chat_history[-1])
    st.session_state["chat_history"] = chat_history

# Main code
llm = load_llm()
chat_history = get_chat_history()
for chat in chat_history:
    display_chat_message(chat)
user_query_to_llm(chat_history)