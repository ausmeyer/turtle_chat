"""Utility functions for Turtle Chat application."""

import os
import re
import time
import uuid
import json
import base64
import hashlib
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from cryptography.fernet import Fernet
import streamlit as st

from constants import (
    MIME_TYPES, IMAGE_EXTENSIONS, SESSION_TIMEOUT_MINUTES,
    MAX_FILE_SIZE_BYTES, ENCRYPTION_KEY_LENGTH, EXPORT_FORMATS,
    CONVERSATIONS_DIR, TEMP_DIR
)


class SecurityUtils:
    """Security-related utility functions."""
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate a secure session ID."""
        return hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:16]
    
    @staticmethod
    def get_encryption_key() -> bytes:
        """Get or generate encryption key for file encryption."""
        if "encryption_key" not in st.session_state:
            st.session_state.encryption_key = Fernet.generate_key()
        return st.session_state.encryption_key
    
    @staticmethod
    def encrypt_content(content: bytes) -> bytes:
        """Encrypt content using session key."""
        fernet = Fernet(SecurityUtils.get_encryption_key())
        return fernet.encrypt(content)
    
    @staticmethod
    def decrypt_content(encrypted_content: bytes) -> bytes:
        """Decrypt content using session key."""
        fernet = Fernet(SecurityUtils.get_encryption_key())
        return fernet.decrypt(encrypted_content)
    
    @staticmethod
    def secure_filename(filename: str) -> str:
        """Generate a secure filename."""
        # Remove path components and dangerous characters
        filename = os.path.basename(filename)
        filename = re.sub(r'[^\w\-_\.]', '', filename)
        # Add timestamp to prevent conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        return f"{name}_{timestamp}{ext}"


class FileUtils:
    """File handling utility functions."""
    
    @staticmethod
    def is_image_file(filename: str) -> bool:
        """Check if file is an image."""
        file_ext = filename.split('.')[-1].lower()
        return file_ext in IMAGE_EXTENSIONS
    
    @staticmethod
    def get_mime_type(file_type: str) -> str:
        """Get MIME type from file extension."""
        return MIME_TYPES.get(file_type.lower(), "application/octet-stream")
    
    @staticmethod
    def encode_image_to_base64(file_content: bytes) -> str:
        """Convert image file to base64 string."""
        return base64.b64encode(file_content).decode('utf-8')
    
    @staticmethod
    def validate_file_size(file_content: bytes) -> bool:
        """Validate file size against maximum allowed."""
        return len(file_content) <= MAX_FILE_SIZE_BYTES
    
    @staticmethod
    def create_temp_file(content: bytes, suffix: str = '') -> str:
        """Create a temporary file with given content."""
        os.makedirs(TEMP_DIR, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=suffix, 
            dir=TEMP_DIR
        )
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def cleanup_temp_files(max_age_hours: int = 24) -> int:
        """Clean up temporary files older than specified hours."""
        if not os.path.exists(TEMP_DIR):
            return 0
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_count = 0
        
        for filename in os.listdir(TEMP_DIR):
            filepath = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                try:
                    os.remove(filepath)
                    cleaned_count += 1
                except OSError:
                    pass
        
        return cleaned_count


class SessionUtils:
    """Session management utility functions."""
    
    @staticmethod
    def check_session_timeout() -> bool:
        """Check if session has timed out due to inactivity."""
        current_time = datetime.now()
        if "last_activity" in st.session_state:
            last_activity = st.session_state["last_activity"]
            if current_time - last_activity > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                st.session_state["password_correct"] = False
                st.session_state["session_timed_out"] = True
                return True
        
        st.session_state["last_activity"] = current_time
        return False
    
    @staticmethod
    def update_activity() -> None:
        """Update last activity timestamp."""
        st.session_state["last_activity"] = datetime.now()
    
    @staticmethod
    def get_user_id() -> str:
        """Get or generate user ID for current session."""
        if "user_id" not in st.session_state:
            st.session_state.user_id = SecurityUtils.generate_session_id()
        return st.session_state.user_id
    
    @staticmethod
    def initialize_session_state() -> None:
        """Initialize all session state variables."""
        defaults = {
            "messages": [{"role": "assistant", "content": "How can I help?"}],
            "file_content": "",
            "file_key": "",
            "uploaded_file": None,
            "file_uploader_key": 0,
            "extended_thinking": False,
            "thinking_budget": 4000,
            "citation_mode": False,
            "conversation_tags": [],
            "selected_model": "claude-sonnet-4",
            "file_data": None,
            "theme": "light",
            "auto_scroll": True,
            "show_thinking": True,
            "enable_shortcuts": True,
            "confirm_delete_all": False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value


class ConversationUtils:
    """Conversation management utility functions."""
    
    @staticmethod
    def save_conversation(messages: List[Dict], tags: List[str], settings: Dict) -> str:
        """Save conversation to file and return conversation ID."""
        os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
        
        conversation_id = str(uuid.uuid4())
        conversation_data = {
            "id": conversation_id,
            "user_id": SessionUtils.get_user_id(),
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "tags": tags,
            "settings": settings,
            "message_count": len(messages),
            "created_at": datetime.now().isoformat()
        }
        
        filename = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        return conversation_id
    
    @staticmethod
    def load_conversation(conversation_id: str) -> Optional[Dict]:
        """Load conversation from file."""
        try:
            filename = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    @staticmethod
    def delete_conversation(conversation_id: str) -> bool:
        """Delete conversation file."""
        try:
            filename = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
            if os.path.exists(filename):
                os.remove(filename)
                return True
            return False
        except OSError:
            return False
    
    @staticmethod
    def get_all_conversations() -> List[Dict]:
        """Get all saved conversations sorted by timestamp."""
        if not os.path.exists(CONVERSATIONS_DIR):
            return []
        
        conversations = []
        for filename in os.listdir(CONVERSATIONS_DIR):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(CONVERSATIONS_DIR, filename), 'r', encoding='utf-8') as f:
                        conversation = json.load(f)
                        conversations.append(conversation)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return sorted(conversations, key=lambda x: x['timestamp'], reverse=True)
    
    @staticmethod
    def search_conversations(query: str) -> List[Dict]:
        """Search through saved conversations."""
        if not query:
            return []
        
        conversations = ConversationUtils.get_all_conversations()
        results = []
        
        for conv in conversations:
            # Search in tags
            if any(query.lower() in tag.lower() for tag in conv.get("tags", [])):
                results.append(conv)
                continue
            
            # Search in message content
            for message in conv.get("messages", []):
                if query.lower() in message.get("content", "").lower():
                    results.append(conv)
                    break
        
        return results
    
    @staticmethod
    def export_conversation(messages: List[Dict], format_type: str = "txt") -> str:
        """Export conversation to specified format."""
        if format_type == "txt":
            return ConversationUtils._export_to_text(messages)
        elif format_type == "json":
            return ConversationUtils._export_to_json(messages)
        elif format_type == "csv":
            return ConversationUtils._export_to_csv(messages)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    @staticmethod
    def _export_to_text(messages: List[Dict]) -> str:
        """Export conversation to text format."""
        export_text = f"""Turtle Chat - Medical Consultation Export
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
User ID: {SessionUtils.get_user_id()}
Session Features: Extended Thinking: {st.session_state.get('extended_thinking', False)}, Citations: {st.session_state.get('citation_mode', False)}

{'='*60}

"""
        
        for message in messages:
            if message["role"] == "user":
                export_text += f"USER: {message['content']}\n\n"
            else:
                export_text += f"CLAUDE: {message['content']}\n"
                
                if message.get("thinking_content"):
                    export_text += f"\n[THINKING PROCESS]\n{message['thinking_content']}\n"
                
                export_text += "\n" + "-"*40 + "\n\n"
        
        return export_text
    
    @staticmethod
    def _export_to_json(messages: List[Dict]) -> str:
        """Export conversation to JSON format."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "user_id": SessionUtils.get_user_id(),
            "session_settings": {
                "extended_thinking": st.session_state.get('extended_thinking', False),
                "citation_mode": st.session_state.get('citation_mode', False),
                "selected_model": st.session_state.get('selected_model', 'claude-sonnet-4')
            },
            "messages": messages,
            "message_count": len(messages)
        }
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def _export_to_csv(messages: List[Dict]) -> str:
        """Export conversation to CSV format."""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Timestamp', 'Role', 'Content', 'Has_Thinking', 'Thinking_Content'])
        
        # Write messages
        for i, message in enumerate(messages):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            role = message["role"]
            content = message["content"]
            has_thinking = bool(message.get("thinking_content"))
            thinking_content = message.get("thinking_content", "")
            
            writer.writerow([timestamp, role, content, has_thinking, thinking_content])
        
        return output.getvalue()


class ValidationUtils:
    """Input validation utility functions."""
    
    @staticmethod
    def validate_message_content(content: str) -> Tuple[bool, str]:
        """Validate message content."""
        if not content or not content.strip():
            return False, "Message cannot be empty"
        
        if len(content) > 50000:  # 50K character limit
            return False, "Message too long. Please keep messages under 50,000 characters."
        
        # Check for potential injection attempts
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'on\w+\s*=',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False, "Message contains potentially unsafe content"
        
        return True, ""
    
    @staticmethod
    def validate_file_type(filename: str) -> Tuple[bool, str]:
        """Validate file type."""
        if not filename:
            return False, "No filename provided"
        
        file_ext = filename.split('.')[-1].lower()
        from constants import ALLOWED_FILE_TYPES
        
        if file_ext not in ALLOWED_FILE_TYPES:
            return False, f"Unsupported file type. Allowed: {', '.join(ALLOWED_FILE_TYPES)}"
        
        return True, ""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename


class UIUtils:
    """UI helper utility functions."""
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    @staticmethod
    def format_timestamp(timestamp: str) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return timestamp
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100) -> str:
        """Truncate text to specified length."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    @staticmethod
    def get_message_preview(message: Dict) -> str:
        """Get a preview of a message for display."""
        content = message.get("content", "")
        role = "👤 User" if message.get("role") == "user" else "🐢 Claude"
        preview = UIUtils.truncate_text(content, 150)
        return f"{role}: {preview}"


class PerformanceUtils:
    """Performance optimization utility functions."""
    
    @staticmethod
    def cleanup_old_conversations(days: int = 30) -> int:
        """Clean up conversations older than specified days."""
        if not os.path.exists(CONVERSATIONS_DIR):
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        for filename in os.listdir(CONVERSATIONS_DIR):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(CONVERSATIONS_DIR, filename)
                    with open(filepath, 'r') as f:
                        conversation = json.load(f)
                    
                    created_date = datetime.fromisoformat(conversation.get('timestamp', ''))
                    if created_date < cutoff_date:
                        os.remove(filepath)
                        cleaned_count += 1
                        
                except (json.JSONDecodeError, KeyError, ValueError, OSError):
                    continue
        
        return cleaned_count
    
    @staticmethod
    def get_memory_usage() -> Dict[str, int]:
        """Get current memory usage statistics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        return {
            "rss": process.memory_info().rss,
            "vms": process.memory_info().vms,
            "percent": process.memory_percent()
        }
    
    @staticmethod
    def should_cleanup_session() -> bool:
        """Determine if session cleanup is needed."""
        # Check if we have too many messages
        if len(st.session_state.get("messages", [])) > 1000:
            return True
        
        # Check if we've been running for too long
        if "session_start_time" in st.session_state:
            session_duration = time.time() - st.session_state["session_start_time"]
            if session_duration > 7200:  # 2 hours
                return True
        
        return False