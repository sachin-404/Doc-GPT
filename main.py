from langchain_main import agent_executor
import streamlit as st

st.set_page_config(page_title="🩺 DocGPT")

with st.sidebar:
    st.title('🩺 &nbsp; DocGPT')
    st.subheader('A friendly virtual companion offering medical insights and guidance.', divider='rainbow')
    st.write('DocGPT is your trusted virtual companion on the path to wellness. With a blend of medical expertise and compassion, DocGPT is designed to provide thoughtful guidance and support for your health-related questions.')
    desc_container = st.container(border=True)
    desc_container.write("DocGPT leverages the power of Google's Gemini LLM for intelligent and context-aware responses.")
    st.warning('DocGPT offers information and guidance based on available data. It may generate responses that are not a substitute for professional medical advice.', icon="⚠️")



with st.chat_message(name="AI Doc", avatar="👨‍⚕️"):
    st.write("Hello 👋! How can I help you today?")

if "messages" not in st.session_state:
    st.session_state.messages = []

    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt =st.chat_input("How can I help?")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke(prompt)['output']
            message_placeholder =st.empty()
            full_response = ""
            for item in response:
                full_response += item
                message_placeholder.markdown(full_response + "| ")
            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
