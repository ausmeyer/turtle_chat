import os
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
MODEL_ID = "meta.llama3-1-405b-instruct-v1:0"
ALLOWED_FILE_TYPES = ["pdf", "png", "jpg", "jpeg"]

# Set AWS credentials from Streamlit secrets
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets.aws_credentials.aws_access_key_id
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets.aws_credentials.aws_secret_access_key

# Initialize AWS clients
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
textract_client = boto3.client(service_name="textract", region_name=AWS_REGION)
s3_client = boto3.client(service_name="s3", region_name=AWS_REGION)

def check_password() -> bool:
    """
    Implement a simple password protection for the Streamlit app.
    
    Returns:
        bool: True if the password is correct, False otherwise.
    """
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

@st.cache_resource
def load_llm() -> ConversationChain:
    """
    Initialize and return the language model for conversation.
    
    Returns:
        ConversationChain: Initialized conversation model.
    """
    llm = BedrockChat(
        client=bedrock_runtime,
        model_id=MODEL_ID,
        model_kwargs={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_gen_len": 4096,
            "stop": ["Human:", "Assistant:", "User's question:"]
        }
    )
    return ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())

def upload_to_s3(file) -> Optional[str]:
    """
    Upload a file to S3 bucket.
    
    Args:
        file: File object to upload.
    
    Returns:
        Optional[str]: The file key if upload is successful, None otherwise.
    """
    file_key = f"{uuid.uuid4()}.pdf"
    try:
        s3_client.upload_fileobj(file, S3_BUCKET_NAME, file_key)
        return file_key
    except ClientError as e:
        st.error(f"Could not upload file to S3: {e}")
        return None

def extract_text_from_s3(file_key: str) -> Optional[str]:
    """
    Extract text from a document stored in S3 using Amazon Textract.
    
    Args:
        file_key (str): The S3 object key of the document.
    
    Returns:
        Optional[str]: Extracted text if successful, None otherwise.
    """
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
    """
    Delete a file from S3 bucket.
    
    Args:
        file_key (str): The S3 object key of the file to delete.
    """
    try:
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=file_key)
    except ClientError as e:
        st.error(f"Could not delete file from S3: {e}")

def process_uploaded_file(uploaded_file) -> None:
    """
    Process the uploaded file: upload to S3, extract text, and update session state.
    
    Args:
        uploaded_file: The file uploaded by the user.
    """
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

def get_ai_response(model: ConversationChain, prompt: str, file_content: Optional[str]) -> str:
    """
    Get AI response based on the user's prompt and file content if available.
    
    Args:
        model (ConversationChain): The conversation model.
        prompt (str): User's input prompt.
        file_content (Optional[str]): Content of the uploaded file, if any.
    
    Returns:
        str: AI-generated response.
    """
    system_prompt = '''You are a helpful AI assistant with broad knowledge. Provide accurate and relevant responses to user queries. If file content is provided, use it as a reference when appropriate, but don't hesitate to use your general knowledge for questions not directly related to the file. Aim for clear and concise answers.'''

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

    return model.predict(input=combined_input)

def display_chat_interface(model: ConversationChain) -> None:
    """
    Display and handle the chat interface.
    
    Args:
        model (ConversationChain): The conversation model.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter text"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            result = get_ai_response(model, prompt, st.session_state.file_content)

            for chunk in result.split():
                full_response += chunk + ' '
                time.sleep(0.01)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

def display_clear_button() -> None:
    """Display the clear conversation button and handle its functionality."""
    st.markdown("""
        <style>
        .stButton button {
            margin-left: 90px;
        }
        </style>
    """, unsafe_allow_html=True)

    clear_button_placeholder = st.empty()
    with clear_button_placeholder.container():
        _, clear_button_col, _ = st.columns([1, 2, 1])
        with clear_button_col:
            if st.button("Clear Conversation"):
                if st.session_state.file_key:
                    delete_from_s3(st.session_state.file_key)
                st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]
                st.session_state.file_content = ""
                st.session_state.file_key = ""
                st.session_state.uploaded_file = None
                st.session_state.file_uploader_key += 1
                #st.experimental_rerun()
                st.rerun()

def main():
    """Main function to run the Streamlit app."""
    st.title("Turtle Chat")

    if not check_password():
        st.stop()

    # Initialize session state
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

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload a document file",
        type=ALLOWED_FILE_TYPES,
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )

    if uploaded_file and not st.session_state.file_key:
        process_uploaded_file(uploaded_file)

    display_chat_interface(model)

    if st.session_state.messages or st.session_state.file_content:
        display_clear_button()

if __name__ == "__main__":
    main()