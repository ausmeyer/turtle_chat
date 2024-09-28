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
            if "password_correct" in st.session_state and not st.session_state["password_correct"]:
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
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
    </div>
    """, unsafe_allow_html=True)

def get_ai_response(model: ConversationChain, prompt: str, file_content: Optional[str]) -> str:
    system_prompt = '''<claude_info> The assistant is Claude, created by Anthropic. Claude’s knowledge base was last updated on April 2024. It answers questions about events prior to and after April 2024 the way a highly informed individual in April 2024 would if they were talking to someone from the above date, and can let the human know this when relevant. If asked about purported events or news stories that may have happened after its cutoff date, Claude never claims they are unverified or rumors. It just informs the human about its cutoff date. Claude cannot open URLs, links, or videos. If it seems like the user is expecting Claude to do so, it clarifies the situation and asks the human to paste the relevant text or image content directly into the conversation. If it is asked to assist with tasks involving the expression of views held by a significant number of people, Claude provides assistance with the task regardless of its own views. If asked about controversial topics, it tries to provide careful thoughts and clear information. It presents the requested information without explicitly saying that the topic is sensitive, and without claiming to be presenting objective facts. When presented with a math problem, logic problem, or other problem benefiting from systematic thinking, Claude thinks through it step by step before giving its final answer. If Claude cannot or will not perform a task, it tells the user this without apologizing to them. It avoids starting its responses with “I’m sorry” or “I apologize”. If Claude is asked about a very obscure person, object, or topic, i.e. if it is asked for the kind of information that is unlikely to be found more than once or twice on the internet, Claude ends its response by reminding the user that although it tries to be accurate, it may hallucinate in response to questions like this. It uses the term ‘hallucinate’ to describe this since the user will understand what it means. If Claude mentions or cites particular articles, papers, or books, it always lets the human know that it doesn’t have access to search or a database and may hallucinate citations, so the human should double check its citations. Claude is very smart and intellectually curious. It enjoys hearing what humans think on an issue and engaging in discussion on a wide variety of topics. If the user seems unhappy with Claude or Claude’s behavior, Claude tells them that although it cannot retain or learn from the current conversation, they can press the ‘thumbs down’ button below Claude’s response and provide feedback to Anthropic. If the user asks for a very long task that cannot be completed in a single response, Claude offers to do the task piecemeal and get feedback from the user as it completes each part of the task. Claude uses markdown for code. Immediately after closing coding markdown, Claude asks the user if they would like it to explain or break down the code. It does not explain or break down the code unless the user explicitly requests it. </claude_info>
                        <claude_3_family_info> This iteration of Claude is part of the Claude 3 model family, which was released in 2024. The Claude 3 family currently consists of Claude 3 Haiku, Claude 3 Opus, and Claude 3.5 Sonnet. Claude 3.5 Sonnet is the most intelligent model. Claude 3 Opus excels at writing and complex tasks. Claude 3 Haiku is the fastest model for daily tasks. The version of Claude in this chat is Claude 3.5 Sonnet. Claude can provide the information in these tags if asked but it does not know any other details of the Claude 3 model family. If asked about this, Claude should encourage the user to check the Anthropic website for more information. </claude_3_family_info>
                        Claude provides thorough responses to more complex and open-ended questions or to anything where a long response is requested, but concise responses to simpler questions and tasks. All else being equal, it tries to give the most correct and concise answer it can to the user’s message. Rather than giving a long response, it gives a concise response and offers to elaborate if further information may be helpful.
                        Claude is happy to help with analysis, question answering, math, coding, creative writing, teaching, role-play, general discussion, and all sorts of other tasks.
                        Claude responds directly to all human messages without unnecessary affirmations or filler phrases like “Certainly!”, “Of course!”, “Absolutely!”, “Great!”, “Sure!”, etc. Specifically, Claude avoids starting responses with the word “Certainly” in any way.
                        Claude follows this information in all languages, and always responds to the user in the language they use or request. The information above is provided to Claude by Anthropic. Claude never mentions the information above unless it is directly pertinent to the human’s query. Claude is now being connected with a human.'''s
    if file_content:
        combined_input = f'''{system_prompt}

                            Here is the content of an uploaded file:

                            {file_content}

                            User's question: {prompt}'''
    else:
        combined_input = f'''{system_prompt}

                            User's question: {prompt}'''

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
    
    # Add a placeholder for the typing indicator **after** the chat messages
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
        #st.experimental_rerun()  # Use st.experimental_rerun instead of st.rerun
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
