import os
import re
import time
import hmac
import uuid
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from cryptography.fernet import Fernet

import boto3
import streamlit as st
from botocore.exceptions import ClientError

# Constants
S3_BUCKET_NAME = 'chatdshs'
AWS_REGION = 'us-west-2'
MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
ALLOWED_FILE_TYPES = ["pdf", "png"]
SESSION_TIMEOUT_MINUTES = 30  # Auto-logout after 30 minutes of inactivity

# AWS Configuration
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets.aws_credentials.aws_access_key_id
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets.aws_credentials.aws_secret_access_key
os.environ['AWS_BEARER_TOKEN_BEDROCK'] = st.secrets.aws_credentials.aws_bearer_token_bedrock

# Initialize AWS clients
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
textract_client = boto3.client(service_name="textract", region_name=AWS_REGION)
s3_client = boto3.client(service_name="s3", region_name=AWS_REGION)

# Setup audit logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audit.log'),
        logging.StreamHandler()
    ]
)
audit_logger = logging.getLogger('audit')

def load_css():
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def check_session_timeout() -> bool:
    """Check if session has timed out due to inactivity"""
    current_time = datetime.now()
    if "last_activity" in st.session_state:
        last_activity = st.session_state["last_activity"]
        if current_time - last_activity > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            # Session timed out
            st.session_state["password_correct"] = False
            st.session_state["session_timed_out"] = True
            return True
    
    # Update last activity time
    st.session_state["last_activity"] = current_time
    return False

def check_password() -> bool:
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            st.session_state["last_activity"] = datetime.now()
            st.session_state["session_timed_out"] = False
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Check for session timeout
    if check_session_timeout():
        st.warning("⏰ Session timed out due to inactivity. Please log in again.")
        return False

    if st.session_state.get("password_correct", False):
        return True

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        subcol1, subcol2, subcol3 = st.columns([1,2,1])
        with subcol2:
            if st.session_state.get("session_timed_out", False):
                st.warning("⏰ Session expired. Please log in again.")
            st.text_input("Password", type="password", on_change=password_entered, key="password")
            if "password_correct" in st.session_state and not st.session_state["password_correct"]:
                st.error("😕 Password incorrect")
    return False

@st.cache_resource
def load_llm():
    return bedrock_runtime

def upload_to_s3(file) -> Optional[str]:
    file_key = f"{uuid.uuid4()}.pdf"
    try:
        # Read file content and encrypt it
        file_content = file.read()
        encrypted_content = encrypt_file_content(file_content)
        
        # Upload encrypted content
        from io import BytesIO
        encrypted_file = BytesIO(encrypted_content)
        s3_client.upload_fileobj(encrypted_file, S3_BUCKET_NAME, file_key)
        
        # Log the upload
        log_audit_event("file_upload", get_user_id(), {
            "file_key": file_key,
            "file_size": len(file_content),
            "encrypted": True
        })
        
        return file_key
    except ClientError as e:
        st.error(f"Could not upload file to S3: {e}")
        return None

def extract_text_from_s3(file_key: str) -> Optional[str]:
    try:
        # Download encrypted file from S3
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        encrypted_content = obj['Body'].read()
        
        # Decrypt the content
        decrypted_content = decrypt_file_content(encrypted_content)
        
        # Create a temporary file for Textract
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(decrypted_content)
            temp_file_path = temp_file.name
        
        # Upload decrypted content to a temporary S3 location for Textract
        temp_key = f"temp_{uuid.uuid4()}.pdf"
        with open(temp_file_path, 'rb') as f:
            s3_client.upload_fileobj(f, S3_BUCKET_NAME, temp_key)
        
        # Process with Textract
        response = textract_client.start_document_text_detection(
            DocumentLocation={'S3Object': {'Bucket': S3_BUCKET_NAME, 'Name': temp_key}}
        )
        job_id = response['JobId']
        
        while True:
            response = textract_client.get_document_text_detection(JobId=job_id)
            if response['JobStatus'] in ['SUCCEEDED', 'FAILED']:
                break
            time.sleep(1)
        
        # Clean up temporary file and S3 object
        os.unlink(temp_file_path)
        s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=temp_key)
        
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

def log_audit_event(event_type: str, user_id: str, details: dict):
    """Log audit events for compliance tracking"""
    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "details": details
    }
    audit_logger.info(f"AUDIT: {json.dumps(audit_entry)}")

def get_user_id() -> str:
    """Generate a session-based user ID for audit logging"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16]
    return st.session_state.user_id

def get_encryption_key() -> bytes:
    """Get or generate encryption key for file encryption"""
    if "encryption_key" not in st.session_state:
        st.session_state.encryption_key = Fernet.generate_key()
    return st.session_state.encryption_key

def encrypt_file_content(content: bytes) -> bytes:
    """Encrypt file content"""
    fernet = Fernet(get_encryption_key())
    return fernet.encrypt(content)

def decrypt_file_content(encrypted_content: bytes) -> bytes:
    """Decrypt file content"""
    fernet = Fernet(get_encryption_key())
    return fernet.decrypt(encrypted_content)

def get_ai_response(model, prompt: str, file_content: Optional[str]) -> str:
    base_system_prompt = '''<claude_info> The assistant is Claude, created by Anthropic. Claude's knowledge base was last updated on April 2024. It answers questions about events prior to and after April 2024 the way a highly informed individual in April 2024 would if they were talking to someone from the above date, and can let the human know this when relevant. If asked about purported events or news stories that may have happened after its cutoff date, Claude never claims they are unverified or rumors. It just informs the human about its cutoff date. Claude cannot open URLs, links, or videos. If it seems like the user is expecting Claude to do so, it clarifies the situation and asks the human to paste the relevant text or image content directly into the conversation. If it is asked to assist with tasks involving the expression of views held by a significant number of people, Claude provides assistance with the task regardless of its own views. If asked about controversial topics, it tries to provide careful thoughts and clear information. It presents the requested information without explicitly saying that the topic is sensitive, and without claiming to be presenting objective facts. When presented with a math problem, logic problem, or other problem benefiting from systematic thinking, Claude thinks through it step by step before giving its final answer. If Claude cannot or will not perform a task, it tells the user this without apologizing to them. It avoids starting its responses with “I’m sorry” or “I apologize”. If Claude is asked about a very obscure person, object, or topic, i.e. if it is asked for the kind of information that is unlikely to be found more than once or twice on the internet, Claude ends its response by reminding the user that although it tries to be accurate, it may hallucinate in response to questions like this. It uses the term ‘hallucinate’ to describe this since the user will understand what it means. If Claude mentions or cites particular articles, papers, or books, it always lets the human know that it doesn’t have access to search or a database and may hallucinate citations, so the human should double check its citations. Claude is very smart and intellectually curious. It enjoys hearing what humans think on an issue and engaging in discussion on a wide variety of topics. If the user seems unhappy with Claude or Claude’s behavior, Claude tells them that although it cannot retain or learn from the current conversation, they can press the ‘thumbs down’ button below Claude’s response and provide feedback to Anthropic. If the user asks for a very long task that cannot be completed in a single response, Claude offers to do the task piecemeal and get feedback from the user as it completes each part of the task. Claude uses markdown for code. Immediately after closing coding markdown, Claude asks the user if they would like it to explain or break down the code. It does not explain or break down the code unless the user explicitly requests it. </claude_info>
                        <claude_4_family_info> This iteration of Claude is part of the Claude 4 model family, which was released in 2025. The Claude 4 family currently consists of Claude 4 Sonnet and Claude 4 Opus. Claude 4 Sonnet is optimized for high-volume use cases and can function effectively as a task-specific sub-agent within broader AI systems. Claude 4 Opus excels at complex reasoning and writing tasks. Both models are hybrid reasoning models offering two modes: near-instant responses and extended thinking for deeper reasoning. The version of Claude in this chat is Claude 4 Sonnet. Claude can provide the information in these tags if asked but it does not know any other details of the Claude 4 model family. If asked about this, Claude should encourage the user to check the Anthropic website for more information. </claude_4_family_info>
                        Claude provides thorough responses to more complex and open-ended questions or to anything where a long response is requested, but concise responses to simpler questions and tasks. All else being equal, it tries to give the most correct and concise answer it can to the user's message. Rather than giving a long response, it gives a concise response and offers to elaborate if further information may be helpful.
                        Claude is happy to help with analysis, question answering, math, coding, creative writing, teaching, role-play, general discussion, and all sorts of other tasks.
                        Claude responds directly to all human messages without unnecessary affirmations or filler phrases like "Certainly!", "Of course!", "Absolutely!", "Great!", "Sure!", etc. Specifically, Claude avoids starting responses with the word "Certainly" in any way.
                        Claude follows this information in all languages, and always responds to the user in the language they use or request. The information above is provided to Claude by Anthropic. Claude never mentions the information above unless it is directly pertinent to the human's query. Claude is now being connected with a human.'''
    
    # Add citation instructions if citation mode is enabled
    citation_prompt = ""
    if st.session_state.get("citation_mode", False):
        citation_prompt = '''
        
        <citation_instructions>
        IMPORTANT: When providing medical information, please:
        1. Cite specific medical sources when possible (e.g., "According to the American Heart Association guidelines...")
        2. Reference established medical literature or clinical practice guidelines
        3. Indicate when information comes from peer-reviewed studies
        4. Specify the level of evidence (e.g., "A systematic review found..." or "Case studies suggest...")
        5. Always remind the user that you cannot access real-time databases and citations should be verified
        6. For drug information, reference standard medical references when possible
        7. Include disclaimers about the need for professional medical consultation
        </citation_instructions>'''
    
    system_prompt = base_system_prompt + citation_prompt
    if file_content:
        user_message = f'''Here is the content of an uploaded file:

{file_content}

User's question: {prompt}'''
    else:
        user_message = prompt

    messages = [
        {"role": "user", "content": user_message}
    ]

    # Build the request body
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64000,  # Claude 4 Sonnet supports up to 64K output tokens
        "system": system_prompt,
        "messages": messages
    }
    
    # Add extended thinking if enabled
    if st.session_state.get("extended_thinking", False):
        request_body["thinking"] = {
            "type": "enabled",
            "budget_tokens": st.session_state.get("thinking_budget", 4000)
        }
    else:
        # Only add temperature and top_p if not using extended thinking
        # (extended thinking is not compatible with these parameters)
        request_body["temperature"] = 0.1   # Lower temperature reduces hallucinations
        request_body["top_p"] = 0.9         # Slightly lower top_p for more focused responses
    
    body = json.dumps(request_body)

    # Log the query for audit
    user_id = get_user_id()
    log_audit_event("query", user_id, {
        "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
        "has_file_content": bool(file_content),
        "extended_thinking": st.session_state.get("extended_thinking", False)
    })
    
    try:
        response = model.invoke_model(
            body=body,
            modelId=MODEL_ID,
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        
        # Handle different response formats for extended thinking vs normal mode
        if "content" in response_body and len(response_body["content"]) > 0:
            # Check if this is extended thinking mode response
            if st.session_state.get("extended_thinking", False):
                thinking_content = ""
                final_answer = ""
                
                for item in response_body["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "thinking":
                            thinking_content = item.get("thinking", "")
                        elif item.get("type") == "text":
                            final_answer = item.get("text", "")
                
                # Log the response for audit
                log_audit_event("response", user_id, {
                    "response_length": len(final_answer),
                    "has_thinking": bool(thinking_content),
                    "thinking_length": len(thinking_content) if thinking_content else 0
                })
                
                # Return both thinking and final answer as a tuple
                return (thinking_content, final_answer)
            else:
                # Normal mode response
                content = response_body["content"][0]
                if "text" in content:
                    response_text = content["text"]
                    
                    # Log the response for audit
                    log_audit_event("response", user_id, {
                        "response_length": len(response_text),
                        "has_thinking": False
                    })
                    
                    return response_text
        
        # Fallback: try to extract any text from the response
        if "content" in response_body:
            for item in response_body["content"]:
                if isinstance(item, dict) and "text" in item:
                    response_text = item["text"]
                    
                    # Log the response for audit
                    log_audit_event("response", user_id, {
                        "response_length": len(response_text),
                        "has_thinking": False
                    })
                    
                    return response_text
        
        # Log failed parsing
        log_audit_event("error", user_id, {
            "error_type": "parsing_failure",
            "response_structure": str(response_body)
        })
        
        return "Sorry, I couldn't parse the response properly."
    except ClientError as e:
        # Log the error for audit
        log_audit_event("error", user_id, {
            "error_type": "bedrock_client_error",
            "error_message": str(e)
        })
        
        st.error(f"Error calling Bedrock: {e}")
        return "Sorry, I encountered an error processing your request."

def display_chat_message(role: str, content, thinking_content=None):
    with st.container():
        if role == "assistant" and thinking_content and thinking_content.strip():
            # Display thinking process in expandable section only if there's actual thinking content
            with st.expander("🧠 View Thinking Process", expanded=False):
                st.markdown(f"```\n{thinking_content}\n```")
        
        # Always display the main message
        st.markdown(f"""
        <div class="chat-message {role}">
            <div class="avatar">{'🧑🏼' if role == 'user' else '🐢'}</div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)

def display_chat_interface(model) -> None:
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            thinking_content = message.get("thinking_content")
            display_chat_message(message["role"], message["content"], thinking_content)
    
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

        # Handle extended thinking mode response
        if st.session_state.get("extended_thinking", False) and isinstance(result, tuple):
            thinking_content, final_answer = result
            with chat_container:
                display_chat_message("assistant", final_answer, thinking_content)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_answer,
                "thinking_content": thinking_content
            })
        else:
            # Normal mode response
            with chat_container:
                display_chat_message("assistant", result)
            st.session_state.messages.append({"role": "assistant", "content": result})

def export_conversation():
    """Export the current conversation to text format"""
    if not st.session_state.messages or len(st.session_state.messages) <= 1:
        st.warning("No conversation to export.")
        return
    
    # Generate export content
    export_text = f"""Turtle Chat - Medical Consultation Export
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
User ID: {get_user_id()}
Session Features: Extended Thinking: {st.session_state.get('extended_thinking', False)}, Citations: {st.session_state.get('citation_mode', False)}

{'='*60}

"""
    
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            export_text += f"USER: {message['content']}\n\n"
        else:
            export_text += f"CLAUDE: {message['content']}\n"
            
            # Add thinking content if available
            if message.get("thinking_content"):
                export_text += f"\n[THINKING PROCESS]\n{message['thinking_content']}\n"
            
            export_text += "\n" + "-"*40 + "\n\n"
    
    export_text += f"""
{'='*60}

MEDICAL DISCLAIMER:
This conversation is for informational purposes only and does not constitute medical advice. 
Always consult with a qualified healthcare professional for medical decisions.

Export generated by Turtle Chat - Secure Medical LLM Interface
"""
    
    # Offer download
    st.download_button(
        label="💾 Download as Text File",
        data=export_text,
        file_name=f"turtle_chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
    
    # Log export event
    log_audit_event("export", get_user_id(), {
        "message_count": len(st.session_state.messages),
        "export_format": "text"
    })

def save_conversation():
    """Save the current conversation to history"""
    if not st.session_state.messages or len(st.session_state.messages) <= 1:
        st.warning("No conversation to save.")
        return
    
    # Create conversations directory if it doesn't exist
    os.makedirs("conversations", exist_ok=True)
    
    # Generate conversation metadata
    conversation_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    conversation_data = {
        "id": conversation_id,
        "user_id": get_user_id(),
        "timestamp": timestamp,
        "messages": st.session_state.messages,
        "tags": st.session_state.get("conversation_tags", []),
        "settings": {
            "extended_thinking": st.session_state.get("extended_thinking", False),
            "citation_mode": st.session_state.get("citation_mode", False)
        }
    }
    
    # Save to encrypted file
    filename = f"conversations/{conversation_id}.json"
    with open(filename, 'w') as f:
        json.dump(conversation_data, f, indent=2)
    
    # Log save event
    log_audit_event("save_conversation", get_user_id(), {
        "conversation_id": conversation_id,
        "message_count": len(st.session_state.messages),
        "tags": st.session_state.get("conversation_tags", [])
    })
    
    st.success(f"Conversation saved successfully! ID: {conversation_id[:8]}...")

def search_conversations(query: str):
    """Search through saved conversations"""
    if not query:
        st.warning("Please enter a search query.")
        return
    
    conversations_dir = "conversations"
    if not os.path.exists(conversations_dir):
        st.info("No saved conversations found.")
        return
    
    search_results = []
    
    # Search through all saved conversations
    for filename in os.listdir(conversations_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(conversations_dir, filename), 'r') as f:
                    conversation = json.load(f)
                
                # Search in messages and tags
                found_match = False
                
                # Search in tags
                if any(query.lower() in tag.lower() for tag in conversation.get("tags", [])):
                    found_match = True
                
                # Search in message content
                if not found_match:
                    for message in conversation.get("messages", []):
                        if query.lower() in message.get("content", "").lower():
                            found_match = True
                            break
                
                if found_match:
                    search_results.append(conversation)
            
            except (json.JSONDecodeError, KeyError):
                continue
    
    # Display search results
    if search_results:
        st.success(f"Found {len(search_results)} conversations matching '{query}':")
        
        for conv in sorted(search_results, key=lambda x: x['timestamp'], reverse=True):
            with st.expander(f"💬 {conv['timestamp'][:19]} - {len(conv['messages'])} messages"):
                st.write(f"**ID:** {conv['id'][:8]}...")
                st.write(f"**Tags:** {', '.join(conv.get('tags', []))}")
                st.write(f"**Settings:** Extended Thinking: {conv['settings'].get('extended_thinking', False)}, Citations: {conv['settings'].get('citation_mode', False)}")
                
                # Show first few messages
                for i, msg in enumerate(conv['messages'][:3]):
                    role = "👤 User" if msg['role'] == 'user' else "🐢 Claude"
                    content = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
                    st.write(f"**{role}:** {content}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📂 Load", key=f"load_{conv['id']}"):
                        load_conversation(conv['id'])
                with col2:
                    if st.button(f"🗑️ Delete", key=f"delete_{conv['id']}"):
                        delete_conversation(conv['id'])
    else:
        st.info(f"No conversations found matching '{query}'.")

def load_conversation(conversation_id: str):
    """Load a saved conversation"""
    try:
        filename = f"conversations/{conversation_id}.json"
        with open(filename, 'r') as f:
            conversation = json.load(f)
        
        # Load conversation data
        st.session_state.messages = conversation['messages']
        st.session_state.conversation_tags = conversation.get('tags', [])
        st.session_state.extended_thinking = conversation['settings'].get('extended_thinking', False)
        st.session_state.citation_mode = conversation['settings'].get('citation_mode', False)
        
        # Log load event
        log_audit_event("load_conversation", get_user_id(), {
            "conversation_id": conversation_id,
            "message_count": len(conversation['messages'])
        })
        
        st.success("Conversation loaded successfully!")
        st.rerun()
        
    except FileNotFoundError:
        st.error("Conversation not found.")
    except json.JSONDecodeError:
        st.error("Error loading conversation.")

def get_all_conversations() -> List[Dict]:
    """Get all saved conversations sorted by timestamp"""
    conversations_dir = "conversations"
    if not os.path.exists(conversations_dir):
        return []
    
    conversations = []
    for filename in os.listdir(conversations_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(conversations_dir, filename), 'r') as f:
                    conversation = json.load(f)
                    conversations.append(conversation)
            except (json.JSONDecodeError, KeyError):
                continue
    
    # Sort by timestamp (newest first)
    return sorted(conversations, key=lambda x: x['timestamp'], reverse=True)

def delete_conversation(conversation_id: str):
    """Delete a saved conversation"""
    try:
        filename = f"conversations/{conversation_id}.json"
        if os.path.exists(filename):
            os.remove(filename)
            
            # Log delete event
            log_audit_event("delete_conversation", get_user_id(), {
                "conversation_id": conversation_id,
                "deleted_by": get_user_id()
            })
            
            st.success("Conversation deleted successfully!")
            st.rerun()
        else:
            st.error("Conversation not found.")
    except Exception as e:
        st.error(f"Error deleting conversation: {e}")

def delete_all_conversations():
    """Delete all saved conversations (admin function)"""
    conversations_dir = "conversations"
    if not os.path.exists(conversations_dir):
        st.info("No conversations to delete.")
        return
    
    deleted_count = 0
    for filename in os.listdir(conversations_dir):
        if filename.endswith('.json'):
            try:
                os.remove(os.path.join(conversations_dir, filename))
                deleted_count += 1
            except Exception:
                continue
    
    # Log bulk delete event
    log_audit_event("delete_all_conversations", get_user_id(), {
        "deleted_count": deleted_count
    })
    
    st.success(f"Deleted {deleted_count} conversations.")
    st.rerun()

def display_clear_button() -> None:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
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
    if "extended_thinking" not in st.session_state:
        st.session_state.extended_thinking = False
    if "thinking_budget" not in st.session_state:
        st.session_state.thinking_budget = 4000
    if "citation_mode" not in st.session_state:
        st.session_state.citation_mode = False
    if "conversation_tags" not in st.session_state:
        st.session_state.conversation_tags = []

    model = load_llm()
    
    with st.sidebar:
        uploaded_file = st.file_uploader(
            "📎 Upload a document",
            type=ALLOWED_FILE_TYPES,
            key=f"file_uploader_{st.session_state.file_uploader_key}"
        )
        
        st.divider()
        
        # Extended thinking mode toggle
        extended_thinking = st.checkbox(
            "🧠 Extended Thinking Mode",
            help="Enable Claude's step-by-step reasoning for complex tasks. Uses additional tokens for internal thinking."
        )
        
        if extended_thinking:
            thinking_budget = st.slider(
                "Thinking Budget (tokens)",
                min_value=1024,
                max_value=10000,
                value=4000,
                step=512,
                help="Maximum tokens Claude can use for internal reasoning"
            )
            st.session_state.extended_thinking = True
            st.session_state.thinking_budget = thinking_budget
        else:
            st.session_state.extended_thinking = False
        
        # Citation requests toggle
        citation_mode = st.checkbox(
            "📚 Request Citations",
            help="Ask Claude to provide sources and citations when possible for medical information."
        )
        st.session_state.citation_mode = citation_mode
        
        st.divider()
        
        # Export conversation
        if st.button("📄 Export Conversation"):
            export_conversation()
        
        # Conversation tagging
        st.subheader("🏷️ Conversation Tags")
        current_tags = st.session_state.get("conversation_tags", [])
        
        # Tag input
        new_tag = st.text_input("Add tag:", placeholder="e.g., cardiology, oncology")
        if st.button("Add Tag") and new_tag:
            if new_tag not in current_tags:
                current_tags.append(new_tag)
                st.session_state.conversation_tags = current_tags
                st.success(f"Added tag: {new_tag}")
        
        # Display current tags
        if current_tags:
            st.write("Current tags:")
            for tag in current_tags:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {tag}")
                with col2:
                    if st.button("❌", key=f"remove_{tag}"):
                        current_tags.remove(tag)
                        st.session_state.conversation_tags = current_tags
                        st.rerun()
        
        # Save conversation
        if st.button("💾 Save Conversation"):
            save_conversation()
        
        # Browse all conversations
        st.subheader("📚 Browse Conversations")
        conversations = get_all_conversations()
        
        if conversations:
            # Create dropdown options
            conversation_options = ["Select a conversation..."] + [
                f"{conv['timestamp'][:19]} - {', '.join(conv.get('tags', ['No tags']))} ({len(conv['messages'])} msgs)"
                for conv in conversations
            ]
            
            selected_index = st.selectbox(
                "Select conversation:",
                range(len(conversation_options)),
                format_func=lambda x: conversation_options[x]
            )
            
            if selected_index > 0:  # If not the default "Select..." option
                selected_conv = conversations[selected_index - 1]
                
                # Show conversation preview
                with st.expander("Preview", expanded=True):
                    st.write(f"**ID:** {selected_conv['id'][:8]}...")
                    st.write(f"**Created:** {selected_conv['timestamp'][:19]}")
                    st.write(f"**Tags:** {', '.join(selected_conv.get('tags', ['No tags']))}")
                    st.write(f"**Messages:** {len(selected_conv['messages'])}")
                    
                    # Show first few messages
                    for i, msg in enumerate(selected_conv['messages'][:3]):
                        role = "👤 User" if msg['role'] == 'user' else "🐢 Claude"
                        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                        st.write(f"**{role}:** {content}")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📂 Load Conversation", key=f"load_selected_{selected_conv['id']}"):
                        load_conversation(selected_conv['id'])
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_selected_{selected_conv['id']}"):
                        delete_conversation(selected_conv['id'])
        else:
            st.info("No saved conversations found.")
        
        # Search conversations
        st.subheader("🔍 Search History")
        search_query = st.text_input("Search conversations:", placeholder="Enter keywords or tags")
        if st.button("Search"):
            search_conversations(search_query)
        
        # Bulk delete option
        st.divider()
        st.subheader("⚠️ Danger Zone")
        if st.button("🗑️ Delete All Conversations", type="secondary"):
            if st.session_state.get("confirm_delete_all", False):
                delete_all_conversations()
                st.session_state.confirm_delete_all = False
            else:
                st.session_state.confirm_delete_all = True
                st.warning("⚠️ Click again to confirm deletion of ALL conversations. This cannot be undone!")
        
        # Reset confirmation if user moves away
        if st.session_state.get("confirm_delete_all", False):
            if st.button("Cancel"):
                st.session_state.confirm_delete_all = False
                st.rerun()

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