import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

st.title("DAMRI AI")



def get_api_key_input():
    st.write("Masukkan Google API Key")

    if "GOOGLE_API_KEY" not in st.session_state:
        st.session_state["GOOGLE_API_KEY"] = ""

    col1, col2 = st.columns((80, 20))
    with col1:
        api_key = st.text_input("", label_visibility="collapsed", type="password")

    with col2:
        is_submit_pressed = st.button("Submit")
        if is_submit_pressed:
            st.session_state["GOOGLE_API_KEY"] = api_key

    os.environ["GOOGLE_API_KEY"] = st.session_state["GOOGLE_API_KEY"]


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
    elif type(message) is AIMessage:
        role = "AI"
        content="Kamu adalah seorang assistant yang ramah dan paham dunia transportasi terutama bus."
    else:
        role = "Unknown"
    with st.chat_message(role):
        st.markdown(message.content)


def user_query_to_llm(chat_history):
    prompt = st.chat_input("Yuk Tanya Mindi")
    if not prompt:
        st.stop()
    chat_history.append(HumanMessage(content=prompt))
    display_chat_message(chat_history[-1])

    response = llm.invoke(chat_history)
    chat_history.append(response)
    display_chat_message(chat_history[-1])


# Main code
get_api_key_input()
if not os.environ["GOOGLE_API_KEY"]:
    st.stop()
llm = load_llm()
chat_history = get_chat_history()
for chat in chat_history:
    display_chat_message(chat)
user_query_to_llm(chat_history)