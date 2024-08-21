import os
import re
import time
import hmac
import uuid
from typing import Dict, List, Optional

import boto3
import streamlit as st
from botocore.exceptions import ClientError
from langchain.chains import ConversationChain
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory

# Constants
S3_BUCKET_NAME = 'chatdshs'
AWS_REGION = 'us-west-2'
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
ALLOWED_FILE_TYPES = ["pdf", "png"]

# AWS Configuration
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets.aws_credentials.aws_access_key_id
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets.aws_credentials.aws_secret_access_key

# Initialize AWS clients
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
textract_client = boto3.client(service_name="textract", region_name=AWS_REGION)
s3_client = boto3.client(service_name="s3", region_name=AWS_REGION)

def load_css():
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def check_password() -> bool:
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        subcol1, subcol2, subcol3 = st.columns([1,2,1])
        with subcol2:
            st.text_input("Password", type="password", on_change=password_entered, key="password")
            if "password_correct" in st.session_state:
                st.error("😕 Password incorrect")
    return False

@st.cache_resource
def load_llm() -> ConversationChain:
    llm = BedrockChat(
        client=bedrock_runtime,
        model_id=MODEL_ID,
        model_kwargs={
            "temperature": 0.7,
            "top_p": 1,
            "max_tokens": 10000
        }
    )
    return ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())

def upload_to_s3(file) -> Optional[str]:
    file_key = f"{uuid.uuid4()}.pdf"
    try:
        s3_client.upload_fileobj(file, S3_BUCKET_NAME, file_key)
        return file_key
    except ClientError as e:
        st.error(f"Could not upload file to S3: {e}")
        return None

def extract_text_from_s3(file_key: str) -> Optional[str]:
    try:
        response = textract_client.start_document_text_detection(
            DocumentLocation={'S3Object': {'Bucket': S3_BUCKET_NAME, 'Name': file_key}}
        )
        job_id = response['JobId']
        
        while True:
            response = textract_client.get_document_text_detection(JobId=job_id)
            if response['JobStatus'] in ['SUCCEEDED', 'FAILED']:
                break
            time.sleep(1)
        
        if response['JobStatus'] == 'SUCCEEDED':
            return '\n'.join(item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE')
        else:
            st.error("Failed to process document.")
            return None
    except ClientError as e:
        st.error(f"An error occurred: {e}")
        return None

def delete_from_s3(file_key: str) -> None:
    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=file_key)
    except ClientError as e:
        st.error(f"Could not delete file from S3: {e}")

def process_uploaded_file(uploaded_file) -> None:
    with st.spinner("Processing uploaded file..."):
        file_key = upload_to_s3(uploaded_file)
        if file_key:
            st.session_state.file_key = file_key
            st.session_state.file_content = extract_text_from_s3(file_key)
            if st.session_state.file_content:
                st.success("File content has been extracted and is ready for use.")
            else:
                st.error("Failed to extract content from the file.")
        else:
            st.error("Failed to upload the file.")

def display_typing_indicator():
    st.markdown("""
    <div class="typing-indicator">
        <span class="turtle">🐢</span>
        <span class="typing-text">Turtle is typing</span>
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
    </div>
    """, unsafe_allow_html=True)

def get_ai_response(model: ConversationChain, prompt: str, file_content: Optional[str]) -> str:
    system_prompt = '''Human: You are Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest. Please provide accurate and relevant responses to user queries. If file content is provided, use it as a reference when appropriate, but don't hesitate to use your general knowledge for questions not directly related to the file. Aim for clear and concise answers.'''

    if file_content:
        combined_input = f'''{system_prompt}

                            Here is the content of an uploaded file:

                            {file_content}

                            User's question: {prompt}

                            Please concisely respond to the user's question. You may use the file content if relevant, but be sure to draw on your general knowledge as needed.'''
    else:
        combined_input = f'''{system_prompt}

                            User's question: {prompt}

                            Please concisely respond to the user's question based on your general knowledge.'''

    response = model.predict(input=combined_input)
    return re.sub(r'\[/?INST\]', '', response).strip()

def display_chat_message(role: str, content: str):
    with st.container():
        st.markdown(f"""
        <div class="chat-message {role}">
            <div class="avatar">{'🧑🏼' if role == 'user' else '🐢'}</div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)

def display_chat_interface(model: ConversationChain) -> None:
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(message["role"], message["content"])
        
        # Add a placeholder for the typing indicator
        typing_indicator = st.empty()

    if prompt := st.chat_input("Enter text"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            display_chat_message("user", prompt)

        with typing_indicator:
            display_typing_indicator()

        with st.spinner(text=''):
            result = get_ai_response(model, prompt, st.session_state.file_content)

        typing_indicator.empty()

        with chat_container:
            display_chat_message("assistant", result)
        st.session_state.messages.append({"role": "assistant", "content": result})

def display_clear_button() -> None:
    if st.button("🗑️ Clear Conversation", key="clear_button"):
        if st.session_state.file_key:
            delete_from_s3(st.session_state.file_key)
        st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]
        st.session_state.file_content = ""
        st.session_state.file_key = ""
        st.session_state.uploaded_file = None
        st.session_state.file_uploader_key += 1
        st.rerun()

def main():
    st.set_page_config(page_title="🐢 Turtle Chat 🐢", layout="wide")
    load_css()

    if not check_password():
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]
    if "file_content" not in st.session_state:
        st.session_state.file_content = ""
    if "file_key" not in st.session_state:
        st.session_state.file_key = ""
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

    model = load_llm()
    
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "📎 Upload a document",
            type=ALLOWED_FILE_TYPES,
            key=f"file_uploader_{st.session_state.file_uploader_key}"
        )

    if uploaded_file and not st.session_state.file_key:
        process_uploaded_file(uploaded_file)
    
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        display_chat_interface(model)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.messages or st.session_state.file_content:
        display_clear_button()

if __name__ == "__main__":
    main()