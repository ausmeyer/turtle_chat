import os
import time
import hmac
import boto3
import streamlit as st
from botocore.exceptions import ClientError
from langchain.chains import ConversationChain
from langchain_community.chat_models import BedrockChat
from langchain.memory import ConversationBufferMemory
import uuid

# Set AWS credentials from Streamlit secrets
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets.aws_credentials.aws_access_key_id
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets.aws_credentials.aws_secret_access_key

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

st.title("Turtle Chat")

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)

textract_client = boto3.client(
    service_name="textract",
    region_name="us-west-2", 
)

s3_client = boto3.client(
    service_name="s3",
    region_name="us-west-2",
)

s3_bucket_name = 'chatdshs'  # Replace with your S3 bucket name

@st.cache_resource
def load_llm():
    llm = BedrockChat(client=bedrock_runtime, model_id="meta.llama3-1-405b-instruct-v1:0")
    llm.model_kwargs = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_gen_len": 4096,
        "stop": ["Human:", "Assistant:", "User's question:"]  # To prevent run-on answers
    }

    model = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())

    return model

model = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_content" not in st.session_state:
    st.session_state.file_content = ""

if "file_key" not in st.session_state:
    st.session_state.file_key = ""

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0

# Add initial greeting if messages are empty
if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "How can I help?"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def upload_to_s3(file):
    file_key = f"{uuid.uuid4()}.pdf"  # Generate a unique key for the file
    try:
        s3_client.upload_fileobj(file, s3_bucket_name, file_key)
        return file_key
    except ClientError as e:
        st.error(f"Could not upload file to S3: {e}")
        return None

def extract_text_from_s3(file_key):
    try:
        response = textract_client.start_document_text_detection(DocumentLocation={'S3Object': {'Bucket': s3_bucket_name, 'Name': file_key}})
        job_id = response['JobId']
        
        while True:
            response = textract_client.get_document_text_detection(JobId=job_id)
            status = response['JobStatus']
            
            if status in ['SUCCEEDED', 'FAILED']:
                break
            time.sleep(1)
        
        if status == 'FAILED':
            st.error("Failed to process document.")
            return None

        text = ''
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                text += item['Text'] + '\n'
        return text
    except ClientError as e:
        st.error(f"An error occurred: {e}")
        return None

def delete_from_s3(file_key):
    try:
        s3_client.delete_object(Bucket=s3_bucket_name, Key=file_key)
    except ClientError as e:
        st.error(f"Could not delete file from S3: {e}")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a document file", 
                                         type=["pdf", "png", "jpg", "jpeg"],
                                         key=f"file_uploader_{st.session_state.file_uploader_key}")

# Process the file immediately after upload
if uploaded_file and not st.session_state.file_key:
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

# Display the prompt entry box
prompt = st.chat_input("Enter text")

if prompt:
    # Simplified system prompt
    system_prompt = '''You are a helpful AI assistant with broad knowledge. Provide accurate and relevant responses to user queries. If file content is provided, use it as a reference when appropriate, but don't hesitate to use your general knowledge for questions not directly related to the file. Aim for clear and concise answers.'''

    if st.session_state.file_content:
        combined_input = f'''{system_prompt}

Here is the content of an uploaded file:

{st.session_state.file_content}

User's question: {prompt}

Please concisely respond to the user's question. You may use the file content if relevant, but be sure to draw on your general knowledge as needed.'''
    else:
        combined_input = f'''{system_prompt}

User's question: {prompt}

Please concisely respond to the user's question based on your general knowledge.'''

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        result = model.predict(input=combined_input)

        # Simulate stream of response with milliseconds delay
        for chunk in result.split(' '):
            full_response += chunk + ' '
            if chunk.endswith('\n'):
                full_response += ' '
            time.sleep(0.01)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add CSS to style the clear button and move it 50 pixels to the right
st.markdown("""
    <style>
    .stButton button {
        margin-left: 90px;
    }
    </style>
""", unsafe_allow_html=True)

# Conditionally display the clear button if there are messages or a file has been uploaded
if st.session_state.messages or st.session_state.file_content:
    clear_button_placeholder = st.empty()
    with clear_button_placeholder.container():
        clear_button_col1, clear_button_col2, clear_button_col3 = st.columns([1, 2, 1])
        with clear_button_col2:
            if st.button("Clear Conversation"):
                if st.session_state.file_key:
                    delete_from_s3(st.session_state.file_key)
                st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]
                st.session_state.file_content = ""
                st.session_state.file_key = ""
                st.session_state.uploaded_file = None
                # Increment the file uploader key to force a reset
                st.session_state.file_uploader_key += 1
                st.experimental_rerun()
