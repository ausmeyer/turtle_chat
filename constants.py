"""Constants used throughout the Turtle Chat application."""

# AWS Configuration
S3_BUCKET_NAME = 'chatdshs'
AWS_REGION = 'us-west-2'
CLAUDE_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# File handling
ALLOWED_FILE_TYPES = ["pdf", "png", "jpg", "jpeg", "gif", "webp"]
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Session management
SESSION_TIMEOUT_MINUTES = 30
MAX_CONVERSATION_HISTORY = 100
MAX_THINKING_TOKENS = 10000
DEFAULT_THINKING_TOKENS = 4000

# API Configuration
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
REQUEST_TIMEOUT_LOCAL = 60
REQUEST_TIMEOUT_CLOUD = 300  # 5 minutes for cloud - very generous
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# UI Constants
CHAT_INPUT_PLACEHOLDER = "Enter your message..."
TYPING_INDICATOR_MESSAGES = [
    "Claude is thinking...",
    "Processing your request...",
    "Analyzing the content...",
    "Generating response..."
]

# Logging
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
AUDIT_LOG_FILE = 'audit.log'
MAX_LOG_SIZE_MB = 100

# Security
MIN_PASSWORD_LENGTH = 8
SESSION_COOKIE_NAME = 'turtle_chat_session'
ENCRYPTION_KEY_LENGTH = 32

# Performance
CACHE_TTL_SECONDS = 3600  # 1 hour
MAX_CACHED_CONVERSATIONS = 50
CONVERSATION_CLEANUP_DAYS = 30

# File paths
CONVERSATIONS_DIR = "conversations"
TEMP_DIR = "temp"
LOGS_DIR = "logs"
CSS_FILE = "style.css"

# MIME types
MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg", 
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "pdf": "application/pdf"
}

# Image file extensions
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "webp"]

# Export formats
EXPORT_FORMATS = {
    "txt": "text/plain",
    "json": "application/json",
    "csv": "text/csv"
}

# Keyboard shortcuts
KEYBOARD_SHORTCUTS = {
    "send_message": "Ctrl+Enter",
    "new_conversation": "Ctrl+N",
    "save_conversation": "Ctrl+S",
    "toggle_sidebar": "Ctrl+B",
    "toggle_theme": "Ctrl+T",
    "copy_last_response": "Ctrl+C",
    "clear_conversation": "Ctrl+L"
}

# Rate limiting
RATE_LIMIT_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_TOKENS_PER_MINUTE = 100000

# Error messages
ERROR_MESSAGES = {
    "file_too_large": f"File size exceeds {MAX_FILE_SIZE_MB}MB limit",
    "unsupported_file_type": f"Unsupported file type. Allowed: {', '.join(ALLOWED_FILE_TYPES)}",
    "upload_failed": "Failed to upload file. Please try again.",
    "processing_failed": "Failed to process your request. Please try again.",
    "session_expired": "Session expired. Please log in again.",
    "rate_limit_exceeded": "Rate limit exceeded. Please wait a moment.",
    "network_error": "Network error. Please check your connection.",
    "invalid_input": "Invalid input. Please check your message.",
    "model_unavailable": "Selected model is currently unavailable.",
    "configuration_error": "Configuration error. Please check your settings."
}

# Success messages
SUCCESS_MESSAGES = {
    "file_uploaded": "File uploaded successfully",
    "conversation_saved": "Conversation saved successfully",
    "conversation_loaded": "Conversation loaded successfully",
    "conversation_deleted": "Conversation deleted successfully",
    "export_complete": "Export completed successfully",
    "settings_updated": "Settings updated successfully"
}

# Theme colors
THEMES = {
    "light": {
        "background": "#f8f9fa",
        "sidebar": "#ffffff",
        "chat_bg": "#ffffff",
        "user_bubble": "#3b82f6",
        "assistant_bubble": "#ffffff",
        "text_primary": "#1f2937",
        "text_secondary": "#6b7280",
        "border": "#e5e7eb",
        "accent": "#10b981"
    },
    "dark": {
        "background": "#111827",
        "sidebar": "#1f2937",
        "chat_bg": "#1f2937",
        "user_bubble": "#3b82f6",
        "assistant_bubble": "#374151",
        "text_primary": "#f9fafb",
        "text_secondary": "#d1d5db",
        "border": "#4b5563",
        "accent": "#10b981"
    }
}

# Animation timing
ANIMATION_TIMING = {
    "fast": 0.15,
    "normal": 0.3,
    "slow": 0.5
}

# Responsive breakpoints
BREAKPOINTS = {
    "mobile": 480,
    "tablet": 768,
    "desktop": 1024,
    "large": 1200
}