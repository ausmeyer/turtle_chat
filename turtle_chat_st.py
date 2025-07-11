"""
Turtle Chat - Advanced Medical LLM Interface
Improved version with better structure, error handling, and UI/UX
"""

import os
import json
import logging
import hashlib
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from io import BytesIO

import boto3
import streamlit as st
from botocore.exceptions import ClientError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import our custom modules
from config import MODEL_CONFIGS, BASE_SYSTEM_PROMPT, CITATION_INSTRUCTIONS, DEFAULT_SETTINGS, EXPORT_DISCLAIMER
from constants import *
from utils import *
from exceptions import *


class TurtleChatApp:
    """Main application class for Turtle Chat."""
    
    def __init__(self):
        self.bedrock_runtime = None
        self.textract_client = None
        self.s3_client = None
        self.logger = self._setup_logging()
        self._initialize_aws_clients()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            handlers=[
                logging.FileHandler(AUDIT_LOG_FILE),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('turtle_chat')
    
    def _initialize_aws_clients(self) -> None:
        """Initialize AWS clients if credentials are available."""
        try:
            if hasattr(st.secrets, 'aws_credentials'):
                os.environ['AWS_ACCESS_KEY_ID'] = st.secrets.aws_credentials.aws_access_key_id
                os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets.aws_credentials.aws_secret_access_key
                os.environ['AWS_BEARER_TOKEN_BEDROCK'] = st.secrets.aws_credentials.aws_bearer_token_bedrock
                
                self.bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
                self.textract_client = boto3.client(service_name="textract", region_name=AWS_REGION)
                self.s3_client = boto3.client(service_name="s3", region_name=AWS_REGION)
                
                self.logger.info("AWS clients initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS clients: {e}")
            raise ConfigurationError(f"AWS configuration failed: {e}")
    
    def run(self) -> None:
        """Main application entry point."""
        try:
            st.set_page_config(
                page_title="🐢 Turtle Chat 🐢", 
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Load CSS and initialize session
            self._load_css()
            SessionUtils.initialize_session_state()
            
            # Check authentication
            if not self._check_authentication():
                st.stop()
            
            # Main application flow
            self._render_main_interface()
            
        except Exception as e:
            self.logger.error(f"Application error: {e}", exc_info=True)
            st.error(handle_exception(e, "application startup"))
    
    def _load_css(self) -> None:
        """Load and apply custom CSS."""
        try:
            css_path = os.path.join(os.path.dirname(__file__), CSS_FILE)
            with open(css_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
            
            # Apply CSS
            st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
            
            # Add enhanced CSS for new features
            self._add_enhanced_css()
            
        except FileNotFoundError:
            self.logger.warning("CSS file not found, using default styling")
            st.error("CSS file not found. Using default styling.")
    
    def _add_enhanced_css(self) -> None:
        """Add enhanced CSS for new features."""
        enhanced_css = """
        /* Enhanced message bubbles with animations */
        .message-content {
            position: relative;
            animation: fadeInUp 0.3s ease-out;
        }
        
        .message-content::before {
            content: '';
            position: absolute;
            top: 15px;
            left: -8px;
            width: 0;
            height: 0;
            border-top: 8px solid transparent;
            border-bottom: 8px solid transparent;
            border-right: 8px solid #ffffff;
        }
        
        .chat-message.user .message-content::before {
            border-right-color: #3b82f6;
        }
        
        /* Enhanced typing indicator */
        .typing-indicator-enhanced {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 2rem auto;
            padding: 1.5rem;
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            border-radius: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid #e5e7eb;
            max-width: 200px;
            position: relative;
        }
        
        .typing-dots {
            display: flex;
            margin-bottom: 0.5rem;
        }
        
        .typing-message {
            font-size: 0.875rem;
            color: #6b7280;
            text-align: center;
            font-weight: 500;
        }
        
        /* Better code blocks */
        .message-content pre {
            background-color: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 0.875rem;
        }
        
        .message-content code {
            background-color: #f1f5f9;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 0.875rem;
        }
        
        /* Enhanced file upload zone */
        .file-upload-zone {
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            background: linear-gradient(135deg, #f8fafc, #ffffff);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .file-upload-zone:hover {
            border-color: #3b82f6;
            background: linear-gradient(135deg, #eff6ff, #f0f9ff);
        }
        
        .upload-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.6;
        }
        
        .upload-text strong {
            color: #1f2937;
            font-size: 1.1rem;
        }
        
        .upload-text p {
            color: #6b7280;
            margin-top: 0.5rem;
            font-size: 0.875rem;
        }
        
        /* Smooth animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Enhanced scrollbar */
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 4px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }
        
        .chat-container::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
        
        /* Dark mode support */
        .dark-mode {
            background-color: #111827 !important;
            color: #f9fafb !important;
        }
        
        .dark-mode .message-content {
            background-color: #374151 !important;
            color: #f9fafb !important;
            border-color: #4b5563 !important;
        }
        
        .dark-mode .chat-message.user .message-content {
            background-color: #3b82f6 !important;
        }
        
        /* Keyboard shortcuts display */
        .shortcuts-hint {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.75rem;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .shortcuts-hint.visible {
            opacity: 1;
        }
        """
        
        st.markdown(f'<style>{enhanced_css}</style>', unsafe_allow_html=True)
    
    def _check_authentication(self) -> bool:
        """Check user authentication."""
        try:
            # Check for session timeout
            if SessionUtils.check_session_timeout():
                st.warning("⏰ Session timed out due to inactivity. Please log in again.")
                return False
            
            # Check if already authenticated
            if st.session_state.get("password_correct", False):
                SessionUtils.update_activity()
                return True
            
            # Show login form
            return self._show_login_form()
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            st.error(handle_exception(e, "authentication"))
            return False
    
    def _show_login_form(self) -> bool:
        """Display login form."""
        def password_entered():
            try:
                import hmac
                entered_password = st.session_state["password"].encode('utf-8')
                stored_password = st.secrets["password"].encode('utf-8')
                
                if hmac.compare_digest(entered_password, stored_password):
                    st.session_state["password_correct"] = True
                    st.session_state["last_activity"] = datetime.now()
                    st.session_state["session_timed_out"] = False
                    del st.session_state["password"]
                    SessionUtils.update_activity()
                else:
                    st.session_state["password_correct"] = False
                    raise AuthenticationError("Invalid password")
            except Exception as e:
                st.session_state["password_correct"] = False
                self.logger.warning(f"Login attempt failed: {e}")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            subcol1, subcol2, subcol3 = st.columns([1, 2, 1])
            with subcol2:
                if st.session_state.get("session_timed_out", False):
                    st.warning("⏰ Session expired. Please log in again.")
                
                st.text_input(
                    "Password", 
                    type="password", 
                    on_change=password_entered, 
                    key="password",
                    placeholder="Enter your password"
                )
                
                if "password_correct" in st.session_state and not st.session_state["password_correct"]:
                    st.error("😕 Password incorrect")
        
        return False
    
    def _render_main_interface(self) -> None:
        """Render the main application interface."""
        # Sidebar
        self._render_sidebar()
        
        # Main chat interface
        self._render_chat_interface()
        
        # Action buttons
        if st.session_state.messages or st.session_state.file_content:
            self._render_action_buttons()
        
        # Keyboard shortcuts
        self._add_keyboard_shortcuts()
    
    def _render_sidebar(self) -> None:
        """Render the sidebar with controls."""
        with st.sidebar:
            # Model selection
            self._render_model_selection()
            st.divider()
            
            # File upload
            self._render_file_upload()
            st.divider()
            
            # AI settings
            self._render_ai_settings()
            st.divider()
            
            # Current session
            self._render_session_controls()
            st.divider()
            
            # Conversation history
            self._render_conversation_history()
            st.divider()
            
            # Settings and utilities
            self._render_settings()
    
    def _render_model_selection(self) -> None:
        """Render model selection interface."""
        st.markdown("### 🤖 AI Model Selection")
        
        # Get available models
        available_models = self._get_available_models()
        
        if not available_models:
            st.error("No AI models available. Please configure your API credentials.")
            self._show_model_troubleshooting()
            st.stop()
        
        # Model selector
        model_options = {key: MODEL_CONFIGS[key]["name"] for key in available_models}
        selected_model = st.selectbox(
            "Select AI Model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0 if st.session_state.selected_model not in available_models else available_models.index(st.session_state.selected_model)
        )
        st.session_state.selected_model = selected_model
        
        # Display model info
        self._display_model_info(selected_model)
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models based on configuration."""
        available_models = []
        
        # Check AWS Bedrock
        if self.bedrock_runtime is not None:
            available_models.append("claude-sonnet-4")
        
        # Check xAI
        if self._check_xai_credentials():
            available_models.append("grok-4")
        
        return available_models
    
    def _check_xai_credentials(self) -> bool:
        """Check if xAI credentials are available."""
        try:
            return ("xai_credentials" in st.secrets and 
                   "api_key" in st.secrets["xai_credentials"])
        except Exception:
            return False
    
    def _display_model_info(self, model_key: str) -> None:
        """Display information about the selected model."""
        model_config = MODEL_CONFIGS[model_key]
        
        info_text = f"**Service:** {model_config['service'].upper()}"
        if 'context_window' in model_config:
            info_text += f"\n**Context Window:** {model_config['context_window']:,} tokens"
        if 'max_tokens' in model_config:
            info_text += f"\n**Max Output:** {model_config['max_tokens']:,} tokens"
        
        st.info(info_text)
    
    def _show_model_troubleshooting(self) -> None:
        """Show troubleshooting info for model configuration."""
        with st.expander("🔧 Troubleshooting", expanded=True):
            st.write("**Configuration Issues:**")
            st.write("1. Check AWS credentials for Bedrock access")
            st.write("2. Verify xAI API key in secrets")
            st.write("3. Ensure proper permissions are set")
    
    def _render_file_upload(self) -> None:
        """Render file upload interface."""
        st.markdown("### 📎 Document Upload")
        
        model_config = MODEL_CONFIGS[st.session_state.selected_model]
        
        if not model_config["supports_file_upload"]:
            st.info(f"File upload not supported by {model_config['name']}")
            return
        
        # Enhanced file uploader
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=ALLOWED_FILE_TYPES,
            key=f"file_uploader_{st.session_state.file_uploader_key}",
            help=f"Supported formats: {', '.join([ft.upper() for ft in ALLOWED_FILE_TYPES])}\\nMax size: {MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file:
            self._handle_file_upload(uploaded_file)
    
    def _handle_file_upload(self, uploaded_file) -> None:
        """Handle file upload with enhanced error handling."""
        try:
            # Validate file
            is_valid, error_msg = ValidationUtils.validate_file_type(uploaded_file.name)
            if not is_valid:
                raise UnsupportedFileTypeError(error_msg, file_type=uploaded_file.name.split('.')[-1])
            
            # Check file size
            file_content = uploaded_file.read()
            if not FileUtils.validate_file_size(file_content):
                raise FileSizeError(
                    f"File size exceeds {MAX_FILE_SIZE_MB}MB limit", 
                    file_size=len(file_content)
                )
            
            # Process file
            uploaded_file.seek(0)  # Reset file pointer
            self._process_uploaded_file(uploaded_file)
            
        except Exception as e:
            st.error(handle_exception(e, "file upload"))
            self.logger.error(f"File upload error: {e}")
    
    def _process_uploaded_file(self, uploaded_file) -> None:
        """Process uploaded file with progress indicator."""
        with st.spinner("Processing uploaded file..."):
            try:
                file_name = uploaded_file.name
                file_content = uploaded_file.read()
                
                if FileUtils.is_image_file(file_name):
                    self._process_image_file(file_name, file_content)
                else:
                    self._process_document_file(uploaded_file, file_content)
                    
            except Exception as e:
                raise FileProcessingError(f"Failed to process file: {e}", file_name=file_name)
    
    def _process_image_file(self, file_name: str, file_content: bytes) -> None:
        """Process image file."""
        file_type = file_name.split('.')[-1].lower()
        mime_type = FileUtils.get_mime_type(file_type)
        base64_data = FileUtils.encode_image_to_base64(file_content)
        
        st.session_state.file_data = {
            "is_image": True,
            "base64": base64_data,
            "mime_type": mime_type,
            "filename": file_name
        }
        st.session_state.file_content = None
        st.success(f"✅ Image '{file_name}' uploaded successfully")
    
    def _process_document_file(self, uploaded_file, file_content: bytes) -> None:
        """Process document file using S3 and Textract."""
        if not self.s3_client:
            raise ConfigurationError("S3 client not configured for document processing")
        
        file_key = self._upload_to_s3(uploaded_file)
        if file_key:
            st.session_state.file_key = file_key
            st.session_state.file_content = self._extract_text_from_s3(file_key)
            st.session_state.file_data = None
            
            if st.session_state.file_content:
                st.success("✅ Document content extracted successfully")
            else:
                st.error("❌ Failed to extract content from document")
        else:
            st.error("❌ Failed to upload document")
    
    def _upload_to_s3(self, file) -> Optional[str]:
        """Upload file to S3 with encryption."""
        try:
            import uuid
            file_key = f"{uuid.uuid4()}.{file.name.split('.')[-1]}"
            
            # Read and encrypt file content
            file_content = file.read()
            encrypted_content = SecurityUtils.encrypt_content(file_content)
            
            # Upload to S3
            encrypted_file = BytesIO(encrypted_content)
            self.s3_client.upload_fileobj(encrypted_file, S3_BUCKET_NAME, file_key)
            
            # Log upload
            self._log_audit_event("file_upload", SessionUtils.get_user_id(), {
                "file_key": file_key,
                "file_size": len(file_content),
                "encrypted": True
            })
            
            return file_key
            
        except Exception as e:
            self.logger.error(f"S3 upload failed: {e}")
            raise StorageError(f"Failed to upload to S3: {e}")
    
    def _extract_text_from_s3(self, file_key: str) -> Optional[str]:
        """Extract text from S3 file using Textract."""
        try:
            # Download and decrypt file
            obj = self.s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
            encrypted_content = obj['Body'].read()
            decrypted_content = SecurityUtils.decrypt_content(encrypted_content)
            
            # Create temporary file for Textract
            temp_file_path = FileUtils.create_temp_file(decrypted_content, '.pdf')
            
            # Upload to temp S3 location for Textract
            import uuid
            temp_key = f"temp_{uuid.uuid4()}.pdf"
            with open(temp_file_path, 'rb') as f:
                self.s3_client.upload_fileobj(f, S3_BUCKET_NAME, temp_key)
            
            # Process with Textract
            response = self.textract_client.start_document_text_detection(
                DocumentLocation={'S3Object': {'Bucket': S3_BUCKET_NAME, 'Name': temp_key}}
            )
            job_id = response['JobId']
            
            # Wait for completion
            while True:
                response = self.textract_client.get_document_text_detection(JobId=job_id)
                if response['JobStatus'] in ['SUCCEEDED', 'FAILED']:
                    break
                time.sleep(1)
            
            # Clean up
            os.unlink(temp_file_path)
            self.s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=temp_key)
            
            if response['JobStatus'] == 'SUCCEEDED':
                return '\\n'.join(item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE')
            else:
                raise Exception("Textract processing failed")
                
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            raise FileProcessingError(f"Failed to extract text: {e}")
    
    def _render_ai_settings(self) -> None:
        """Render AI settings panel."""
        st.markdown("### ⚙️ AI Settings")
        
        model_config = MODEL_CONFIGS[st.session_state.selected_model]
        
        # Extended thinking
        if model_config["supports_extended_thinking"]:
            extended_thinking = st.checkbox(
                "🧠 Extended Thinking",
                value=st.session_state.extended_thinking,
                help="Enable step-by-step reasoning for complex tasks"
            )
            
            if extended_thinking:
                thinking_budget = st.slider(
                    "Thinking Budget",
                    min_value=1024,
                    max_value=MAX_THINKING_TOKENS,
                    value=st.session_state.thinking_budget,
                    step=512,
                    help="Maximum tokens for internal reasoning"
                )
                st.session_state.thinking_budget = thinking_budget
            
            st.session_state.extended_thinking = extended_thinking
        else:
            st.info(f"Extended thinking not available for {model_config['name']}")
            st.session_state.extended_thinking = False
        
        # Citations
        if model_config["supports_citations"]:
            citation_mode = st.checkbox(
                "📚 Request Citations",
                value=st.session_state.citation_mode,
                help="Ask for sources and citations when possible"
            )
            st.session_state.citation_mode = citation_mode
        else:
            st.session_state.citation_mode = False
    
    def _render_session_controls(self) -> None:
        """Render current session controls."""
        st.markdown("### 💬 Current Session")
        
        # Message count
        message_count = len(st.session_state.messages) - 1  # Exclude initial message
        st.metric("Messages", message_count)
        
        # Conversation tagging
        with st.expander("🏷️ Add Tags", expanded=False):
            self._render_conversation_tagging()
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", use_container_width=True, help="Save current conversation"):
                self._save_current_conversation()
        with col2:
            if st.button("📄 Export", use_container_width=True, help="Export conversation"):
                self._export_current_conversation()
    
    def _render_conversation_tagging(self) -> None:
        """Render conversation tagging interface."""
        current_tags = st.session_state.get("conversation_tags", [])
        
        # Tag input
        new_tag = st.text_input(
            "Add tag:", 
            placeholder="e.g., cardiology, research",
            key="new_tag_input"
        )
        
        if st.button("Add Tag", use_container_width=True) and new_tag:
            if new_tag not in current_tags:
                current_tags.append(new_tag)
                st.session_state.conversation_tags = current_tags
                st.success(f"Added: {new_tag}")
                st.rerun()
        
        # Display current tags
        if current_tags:
            st.markdown("**Current tags:**")
            for tag in current_tags:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"• {tag}")
                with col2:
                    if st.button("✕", key=f"remove_{tag}", help=f"Remove {tag}"):
                        current_tags.remove(tag)
                        st.session_state.conversation_tags = current_tags
                        st.rerun()
    
    def _save_current_conversation(self) -> None:
        """Save current conversation."""
        try:
            if not st.session_state.messages or len(st.session_state.messages) <= 1:
                st.warning("No conversation to save.")
                return
            
            conversation_id = ConversationUtils.save_conversation(
                st.session_state.messages,
                st.session_state.get("conversation_tags", []),
                {
                    "extended_thinking": st.session_state.get("extended_thinking", False),
                    "citation_mode": st.session_state.get("citation_mode", False),
                    "selected_model": st.session_state.get("selected_model", "claude-sonnet-4")
                }
            )
            
            self._log_audit_event("save_conversation", SessionUtils.get_user_id(), {
                "conversation_id": conversation_id,
                "message_count": len(st.session_state.messages)
            })
            
            st.success(f"Conversation saved! ID: {conversation_id[:8]}...")
            
        except Exception as e:
            st.error(handle_exception(e, "conversation save"))
    
    def _export_current_conversation(self) -> None:
        """Export current conversation."""
        try:
            if not st.session_state.messages or len(st.session_state.messages) <= 1:
                st.warning("No conversation to export.")
                return
            
            # Export format selection
            export_format = st.selectbox(
                "Export format:",
                options=["txt", "json", "csv"],
                format_func=lambda x: x.upper()
            )
            
            if st.button("Download", use_container_width=True):
                export_content = ConversationUtils.export_conversation(
                    st.session_state.messages, 
                    export_format
                )
                
                if export_format == "txt":
                    export_content += EXPORT_DISCLAIMER
                
                st.download_button(
                    label=f"💾 Download {export_format.upper()} File",
                    data=export_content,
                    file_name=f"turtle_chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                    mime=EXPORT_FORMATS[export_format]
                )
                
                self._log_audit_event("export", SessionUtils.get_user_id(), {
                    "format": export_format,
                    "message_count": len(st.session_state.messages)
                })
                
        except Exception as e:
            st.error(handle_exception(e, "conversation export"))
    
    def _render_conversation_history(self) -> None:
        """Render conversation history interface."""
        st.markdown("### 📚 Conversation History")
        
        # Search conversations
        with st.expander("🔍 Search Conversations", expanded=False):
            search_query = st.text_input(
                "Search:", 
                placeholder="Enter keywords or tags",
                key="search_input"
            )
            if st.button("Search", use_container_width=True):
                self._search_conversations(search_query)
        
        # Browse conversations
        with st.expander("📂 Browse Conversations", expanded=False):
            self._render_conversation_browser()
    
    def _search_conversations(self, query: str) -> None:
        """Search through conversations."""
        try:
            if not query:
                st.warning("Please enter a search query.")
                return
            
            results = ConversationUtils.search_conversations(query)
            
            if results:
                st.success(f"Found {len(results)} conversations matching '{query}':")
                
                for conv in results:
                    with st.expander(f"💬 {UIUtils.format_timestamp(conv['timestamp'])} - {len(conv['messages'])} messages"):
                        self._render_conversation_preview(conv)
            else:
                st.info(f"No conversations found matching '{query}'.")
                
        except Exception as e:
            st.error(handle_exception(e, "conversation search"))
    
    def _render_conversation_browser(self) -> None:
        """Render conversation browser."""
        try:
            conversations = ConversationUtils.get_all_conversations()
            
            if not conversations:
                st.info("No saved conversations found.")
                return
            
            # Create dropdown
            conversation_options = ["Select a conversation..."] + [
                f"{UIUtils.format_timestamp(conv['timestamp'])} - {len(conv['messages'])} msgs"
                for conv in conversations
            ]
            
            selected_index = st.selectbox(
                "Select:",
                range(len(conversation_options)),
                format_func=lambda x: conversation_options[x],
                key="conversation_selector"
            )
            
            if selected_index > 0:
                selected_conv = conversations[selected_index - 1]
                self._render_conversation_preview(selected_conv, show_actions=True)
                
        except Exception as e:
            st.error(handle_exception(e, "conversation browser"))
    
    def _render_conversation_preview(self, conversation: Dict, show_actions: bool = False) -> None:
        """Render conversation preview."""
        st.markdown(f"**Created:** {UIUtils.format_timestamp(conversation['timestamp'])}")
        st.markdown(f"**Tags:** {', '.join(conversation.get('tags', ['No tags']))}")
        st.markdown(f"**Messages:** {len(conversation['messages'])}")
        
        # Show first few messages
        for i, msg in enumerate(conversation['messages'][:3]):
            preview = UIUtils.get_message_preview(msg)
            st.write(preview)
        
        if show_actions:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 Load", key=f"load_{conversation['id']}", use_container_width=True):
                    self._load_conversation(conversation['id'])
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{conversation['id']}", use_container_width=True):
                    self._delete_conversation(conversation['id'])
    
    def _load_conversation(self, conversation_id: str) -> None:
        """Load a conversation."""
        try:
            conversation = ConversationUtils.load_conversation(conversation_id)
            if not conversation:
                st.error("Conversation not found.")
                return
            
            # Load conversation data
            st.session_state.messages = conversation['messages']
            st.session_state.conversation_tags = conversation.get('tags', [])
            st.session_state.extended_thinking = conversation['settings'].get('extended_thinking', False)
            st.session_state.citation_mode = conversation['settings'].get('citation_mode', False)
            
            self._log_audit_event("load_conversation", SessionUtils.get_user_id(), {
                "conversation_id": conversation_id,
                "message_count": len(conversation['messages'])
            })
            
            st.success("Conversation loaded successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(handle_exception(e, "conversation load"))
    
    def _delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation."""
        try:
            success = ConversationUtils.delete_conversation(conversation_id)
            if success:
                self._log_audit_event("delete_conversation", SessionUtils.get_user_id(), {
                    "conversation_id": conversation_id
                })
                st.success("Conversation deleted successfully!")
                st.rerun()
            else:
                st.error("Failed to delete conversation.")
                
        except Exception as e:
            st.error(handle_exception(e, "conversation delete"))
    
    def _render_settings(self) -> None:
        """Render settings and utilities."""
        with st.expander("⚙️ Settings", expanded=False):
            # Theme selection
            theme = st.selectbox(
                "Theme:",
                options=["light", "dark"],
                index=0 if st.session_state.get("theme", "light") == "light" else 1
            )
            st.session_state.theme = theme
            
            # Auto-scroll
            auto_scroll = st.checkbox(
                "Auto-scroll to new messages",
                value=st.session_state.get("auto_scroll", True)
            )
            st.session_state.auto_scroll = auto_scroll
            
            # Show thinking
            show_thinking = st.checkbox(
                "Show thinking process",
                value=st.session_state.get("show_thinking", True)
            )
            st.session_state.show_thinking = show_thinking
            
            # Keyboard shortcuts
            enable_shortcuts = st.checkbox(
                "Enable keyboard shortcuts",
                value=st.session_state.get("enable_shortcuts", True)
            )
            st.session_state.enable_shortcuts = enable_shortcuts
        
        # Danger zone
        with st.expander("⚠️ Danger Zone", expanded=False):
            st.warning("Destructive actions - use with caution!")
            
            if st.button("🗑️ Delete All Conversations", type="secondary", use_container_width=True):
                self._handle_delete_all_conversations()
    
    def _handle_delete_all_conversations(self) -> None:
        """Handle deletion of all conversations."""
        if st.session_state.get("confirm_delete_all", False):
            try:
                # Delete all conversations
                conversations = ConversationUtils.get_all_conversations()
                deleted_count = 0
                
                for conv in conversations:
                    if ConversationUtils.delete_conversation(conv['id']):
                        deleted_count += 1
                
                self._log_audit_event("delete_all_conversations", SessionUtils.get_user_id(), {
                    "deleted_count": deleted_count
                })
                
                st.success(f"Deleted {deleted_count} conversations.")
                st.session_state.confirm_delete_all = False
                st.rerun()
                
            except Exception as e:
                st.error(handle_exception(e, "bulk delete"))
        else:
            st.session_state.confirm_delete_all = True
            st.error("⚠️ Click again to confirm deletion of ALL conversations!")
    
    def _render_chat_interface(self) -> None:
        """Render the main chat interface."""
        chat_container = st.container()
        
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display messages
            for message in st.session_state.messages:
                thinking_content = message.get("thinking_content") if st.session_state.get("show_thinking", True) else None
                self._display_chat_message(message["role"], message["content"], thinking_content)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        self._render_chat_input(chat_container)
    
    def _render_chat_input(self, chat_container) -> None:
        """Render chat input with enhanced features."""
        # Add typing indicator placeholder
        typing_indicator = st.empty()
        
        # Chat input
        if prompt := st.chat_input(CHAT_INPUT_PLACEHOLDER):
            # Validate input
            is_valid, error_msg = ValidationUtils.validate_message_content(prompt)
            if not is_valid:
                st.error(error_msg)
                return
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with chat_container:
                self._display_chat_message("user", prompt)
            
            # Show enhanced typing indicator
            with typing_indicator:
                self._display_enhanced_typing_indicator()
            
            # Get AI response
            try:
                with st.spinner():
                    response = self._get_ai_response(prompt)
                
                typing_indicator.empty()
                
                # Handle response
                if st.session_state.get("extended_thinking", False) and isinstance(response, tuple):
                    thinking_content, final_answer = response
                    with chat_container:
                        thinking_display = thinking_content if st.session_state.get("show_thinking", True) else None
                        self._display_chat_message("assistant", final_answer, thinking_display)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "thinking_content": thinking_content
                    })
                else:
                    with chat_container:
                        self._display_chat_message("assistant", response)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                
                # Auto-scroll
                if st.session_state.get("auto_scroll", True):
                    st.rerun()
                    
            except Exception as e:
                typing_indicator.empty()
                error_msg = handle_exception(e, "AI response")
                st.error(error_msg)
                self.logger.error(f"AI response error: {e}")
    
    def _display_enhanced_typing_indicator(self) -> None:
        """Display enhanced typing indicator."""
        import random
        message = random.choice(TYPING_INDICATOR_MESSAGES)
        
        st.markdown(f"""
        <div class="typing-indicator-enhanced">
            <div class="typing-dots">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </div>
            <div class="typing-message">{message}</div>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_chat_message(self, role: str, content: str, thinking_content: str = None) -> None:
        """Display a chat message with enhanced formatting."""
        with st.container():
            # Show thinking process if available
            if role == "assistant" and thinking_content and thinking_content.strip():
                with st.expander("🧠 View Thinking Process", expanded=False):
                    st.markdown(f"```\n{thinking_content}\n```")
            
            # Display main message
            st.markdown(f"""
            <div class="chat-message {role}">
                <div class="avatar">{'🧑🏼' if role == 'user' else '🐢'}</div>
                <div class="message-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add copy functionality for assistant messages
            if role == "assistant":
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    with st.expander("📋 Copy Text", expanded=False):
                        st.text_area(
                            "Select all and copy:",
                            value=content,
                            height=120,
                            key=f"copy_{hashlib.md5(content.encode()).hexdigest()[:8]}",
                            help="Select all (Ctrl+A/Cmd+A) and copy (Ctrl+C/Cmd+C)"
                        )
    
    def _get_ai_response(self, prompt: str) -> Union[str, Tuple[str, str]]:
        """Get AI response with enhanced error handling."""
        try:
            selected_model = st.session_state.get("selected_model", "claude-sonnet-4")
            model_config = MODEL_CONFIGS[selected_model]
            
            # Route to appropriate service
            if model_config["service"] == "xai":
                return self._get_xai_response(prompt, model_config)
            else:
                return self._get_bedrock_response(prompt, model_config)
                
        except Exception as e:
            self.logger.error(f"AI response error: {e}")
            raise ModelServiceError(f"Failed to get AI response: {e}")
    
    def _get_xai_response(self, prompt: str, model_config: Dict) -> str:
        """Get response from xAI API."""
        try:
            # Get API key
            if not self._check_xai_credentials():
                raise ConfigurationError("xAI API key not configured")
            
            xai_api_key = st.secrets["xai_credentials"]["api_key"]
            
            # Build messages
            messages = self._build_conversation_messages(prompt)
            
            # Prepare request
            payload = {
                "model": model_config["model_id"],
                "messages": messages,
                "max_tokens": model_config["max_tokens"],
                "temperature": model_config.get("temperature", 0.1),
                "top_p": model_config.get("top_p", 0.9)
            }
            
            # Make request with retry logic
            response = self._make_xai_request(payload, xai_api_key)
            
            # Log request
            self._log_audit_event("xai_request", SessionUtils.get_user_id(), {
                "model": model_config["model_id"],
                "prompt_length": len(prompt),
                "response_length": len(response)
            })
            
            return response
            
        except Exception as e:
            raise XAIError(f"xAI API error: {e}")
    
    def _make_xai_request(self, payload: Dict, api_key: str) -> str:
        """Make request to xAI API with retry logic."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            status_forcelist=RETRY_STATUS_CODES,
            allowed_methods=["POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Detect timeout based on environment
        timeout = REQUEST_TIMEOUT_CLOUD if self._is_cloud_environment() else REQUEST_TIMEOUT_LOCAL
        
        try:
            response = session.post(
                XAI_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                raise NetworkError(
                    f"xAI API error: {response.status_code} - {response.text}",
                    endpoint=XAI_API_URL,
                    status_code=response.status_code
                )
                
        except requests.exceptions.Timeout:
            raise TimeoutError("Request timed out", timeout=timeout)
        except requests.exceptions.ConnectionError:
            raise NetworkError("Connection error")
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error: {e}")
    
    def _is_cloud_environment(self) -> bool:
        """Check if running in cloud environment."""
        return (os.environ.get("STREAMLIT_CLOUD") or 
                "streamlit.app" in os.environ.get("SERVER_NAME", ""))
    
    def _get_bedrock_response(self, prompt: str, model_config: Dict) -> Union[str, Tuple[str, str]]:
        """Get response from AWS Bedrock."""
        try:
            if not self.bedrock_runtime:
                raise ConfigurationError("AWS Bedrock not configured")
            
            # Build system prompt
            system_prompt = self._build_system_prompt()
            
            # Build messages
            messages = self._build_conversation_messages(prompt)
            
            # Build request body
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": model_config["max_tokens"],
                "system": system_prompt,
                "messages": messages
            }
            
            # Add extended thinking if enabled
            if st.session_state.get("extended_thinking", False):
                request_body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": st.session_state.get("thinking_budget", DEFAULT_THINKING_TOKENS)
                }
            else:
                request_body["temperature"] = model_config.get("temperature", 0.1)
                request_body["top_p"] = model_config.get("top_p", 0.9)
            
            # Make request
            response = self.bedrock_runtime.invoke_model(
                body=json.dumps(request_body),
                modelId=model_config["model_id"],
                accept="application/json",
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response.get("body").read())
            
            # Log request
            self._log_audit_event("bedrock_request", SessionUtils.get_user_id(), {
                "model": model_config["model_id"],
                "prompt_length": len(prompt),
                "extended_thinking": st.session_state.get("extended_thinking", False)
            })
            
            return self._parse_bedrock_response(response_body)
            
        except ClientError as e:
            raise BedrockError(f"AWS Bedrock error: {e}")
        except Exception as e:
            raise BedrockError(f"Bedrock request failed: {e}")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with optional citations."""
        system_prompt = BASE_SYSTEM_PROMPT
        
        if st.session_state.get("citation_mode", False):
            system_prompt += CITATION_INSTRUCTIONS
        
        return system_prompt
    
    def _build_conversation_messages(self, current_prompt: str) -> List[Dict]:
        """Build conversation messages for API request."""
        messages = []
        
        # Add conversation history
        conversation_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else []
        
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
        
        # Add current message with file content if available
        if st.session_state.get("file_content"):
            current_message = f"""Here is the content of an uploaded file:

{st.session_state.file_content}

User's question: {current_prompt}"""
        elif st.session_state.get("file_data") and st.session_state.file_data.get("is_image"):
            # For image files, add image data (for models that support it)
            current_message = current_prompt
        else:
            current_message = current_prompt
        
        messages.append({"role": "user", "content": current_message})
        
        return messages
    
    def _parse_bedrock_response(self, response_body: Dict) -> Union[str, Tuple[str, str]]:
        """Parse response from Bedrock."""
        if "content" not in response_body or not response_body["content"]:
            raise BedrockError("Invalid response format from Bedrock")
        
        # Handle extended thinking mode
        if st.session_state.get("extended_thinking", False):
            thinking_content = ""
            final_answer = ""
            
            for item in response_body["content"]:
                if isinstance(item, dict):
                    if item.get("type") == "thinking":
                        thinking_content = item.get("thinking", "")
                    elif item.get("type") == "text":
                        final_answer = item.get("text", "")
            
            return thinking_content, final_answer
        else:
            # Normal mode
            content = response_body["content"][0]
            if "text" in content:
                return content["text"]
            else:
                raise BedrockError("No text content in response")
    
    def _render_action_buttons(self) -> None:
        """Render action buttons below chat."""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🗑️ Clear Conversation", key="clear_button", use_container_width=True, type="secondary"):
                self._clear_conversation()
    
    def _clear_conversation(self) -> None:
        """Clear current conversation."""
        try:
            # Clean up S3 file if exists
            if st.session_state.get("file_key"):
                self.s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=st.session_state.file_key)
            
            # Reset session state
            st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]
            st.session_state.file_content = ""
            st.session_state.file_key = ""
            st.session_state.file_data = None
            st.session_state.uploaded_file = None
            st.session_state.file_uploader_key += 1
            
            # Log clear action
            self._log_audit_event("clear_conversation", SessionUtils.get_user_id(), {})
            
            st.rerun()
            
        except Exception as e:
            self.logger.error(f"Clear conversation error: {e}")
            st.error(handle_exception(e, "clear conversation"))
    
    def _add_keyboard_shortcuts(self) -> None:
        """Add keyboard shortcuts support."""
        if not st.session_state.get("enable_shortcuts", True):
            return
        
        # Add JavaScript for keyboard shortcuts
        shortcuts_js = """
        <script>
        document.addEventListener('keydown', function(e) {
            // Ctrl+Enter to send message
            if (e.ctrlKey && e.key === 'Enter') {
                const chatInput = document.querySelector('input[data-testid="stChatInput"]');
                if (chatInput) {
                    chatInput.dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter'}));
                }
            }
            
            // Ctrl+L to clear conversation
            if (e.ctrlKey && e.key === 'l') {
                e.preventDefault();
                const clearButton = document.querySelector('button[data-testid="stButton"][key="clear_button"]');
                if (clearButton) {
                    clearButton.click();
                }
            }
            
            // Ctrl+N for new conversation
            if (e.ctrlKey && e.key === 'n') {
                e.preventDefault();
                const clearButton = document.querySelector('button[data-testid="stButton"][key="clear_button"]');
                if (clearButton) {
                    clearButton.click();
                }
            }
        });
        
        // Show shortcuts hint
        if (document.querySelector('.shortcuts-hint')) {
            setTimeout(() => {
                document.querySelector('.shortcuts-hint').classList.add('visible');
            }, 2000);
            
            setTimeout(() => {
                document.querySelector('.shortcuts-hint').classList.remove('visible');
            }, 8000);
        }
        </script>
        
        <div class="shortcuts-hint">
            💡 Shortcuts: Ctrl+Enter (Send), Ctrl+L (Clear), Ctrl+N (New)
        </div>
        """
        
        st.markdown(shortcuts_js, unsafe_allow_html=True)
    
    def _log_audit_event(self, event_type: str, user_id: str, details: Dict) -> None:
        """Log audit event."""
        try:
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "user_id": user_id,
                "details": details
            }
            self.logger.info(f"AUDIT: {json.dumps(audit_entry)}")
        except Exception as e:
            self.logger.error(f"Audit logging failed: {e}")


# Main application entry point
def main():
    """Main application entry point."""
    try:
        app = TurtleChatApp()
        app.run()
    except Exception as e:
        st.error(f"Fatal error: {e}")
        logging.error(f"Fatal application error: {e}", exc_info=True)


if __name__ == "__main__":
    main()