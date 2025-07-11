"""Custom exceptions for Turtle Chat application."""


class TurtleChatException(Exception):
    """Base exception for all Turtle Chat related errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class AuthenticationError(TurtleChatException):
    """Authentication related errors."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        kwargs.setdefault("error_code", "AUTH_ERROR")
        super().__init__(message, **kwargs)


class SessionExpiredError(TurtleChatException):
    """Session timeout or expiration errors."""
    
    def __init__(self, message: str = "Session has expired", **kwargs):
        kwargs.setdefault("error_code", "SESSION_EXPIRED")
        super().__init__(message, **kwargs)


class FileProcessingError(TurtleChatException):
    """File upload, processing, or handling errors."""
    
    def __init__(self, message: str, file_name: str = None, **kwargs):
        kwargs.setdefault("error_code", "FILE_ERROR")
        super().__init__(message, **kwargs)
        self.file_name = file_name
        if file_name:
            self.details["file_name"] = file_name


class FileSizeError(FileProcessingError):
    """File size exceeds limits."""
    
    def __init__(self, message: str = "File size exceeds limit", file_size: int = None, **kwargs):
        kwargs.setdefault("error_code", "FILE_SIZE_ERROR")
        super().__init__(message, **kwargs)
        if file_size:
            self.details["file_size"] = file_size


class UnsupportedFileTypeError(FileProcessingError):
    """Unsupported file type error."""
    
    def __init__(self, message: str = "Unsupported file type", file_type: str = None, **kwargs):
        kwargs.setdefault("error_code", "UNSUPPORTED_FILE_TYPE")
        super().__init__(message, **kwargs)
        if file_type:
            self.details["file_type"] = file_type


class ModelServiceError(TurtleChatException):
    """AI model service related errors."""
    
    def __init__(self, message: str, service: str = None, model: str = None, **kwargs):
        kwargs.setdefault("error_code", "MODEL_SERVICE_ERROR")
        super().__init__(message, **kwargs)
        self.service = service
        self.model = model
        if service:
            self.details["service"] = service
        if model:
            self.details["model"] = model


class BedrockError(ModelServiceError):
    """AWS Bedrock specific errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "BEDROCK_ERROR")
        super().__init__(message, service="bedrock", **kwargs)


class XAIError(ModelServiceError):
    """xAI API specific errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("error_code", "XAI_ERROR")
        super().__init__(message, service="xai", **kwargs)


class NetworkError(TurtleChatException):
    """Network connectivity and API communication errors."""
    
    def __init__(self, message: str, endpoint: str = None, status_code: int = None, **kwargs):
        kwargs.setdefault("error_code", "NETWORK_ERROR")
        super().__init__(message, **kwargs)
        self.endpoint = endpoint
        self.status_code = status_code
        if endpoint:
            self.details["endpoint"] = endpoint
        if status_code:
            self.details["status_code"] = status_code


class TimeoutError(NetworkError):
    """Request timeout errors."""
    
    def __init__(self, message: str = "Request timed out", timeout: int = None, **kwargs):
        kwargs.setdefault("error_code", "TIMEOUT_ERROR")
        super().__init__(message, **kwargs)
        if timeout:
            self.details["timeout"] = timeout


class RateLimitError(NetworkError):
    """Rate limiting errors."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None, **kwargs):
        kwargs.setdefault("error_code", "RATE_LIMIT_ERROR")
        super().__init__(message, **kwargs)
        if retry_after:
            self.details["retry_after"] = retry_after


class ConfigurationError(TurtleChatException):
    """Configuration and setup errors."""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        kwargs.setdefault("error_code", "CONFIG_ERROR")
        super().__init__(message, **kwargs)
        self.config_key = config_key
        if config_key:
            self.details["config_key"] = config_key


class ValidationError(TurtleChatException):
    """Input validation errors."""
    
    def __init__(self, message: str, field: str = None, value: str = None, **kwargs):
        kwargs.setdefault("error_code", "VALIDATION_ERROR")
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value
        if field:
            self.details["field"] = field
        if value:
            self.details["value"] = str(value)[:100]  # Truncate long values


class ConversationError(TurtleChatException):
    """Conversation management errors."""
    
    def __init__(self, message: str, conversation_id: str = None, **kwargs):
        kwargs.setdefault("error_code", "CONVERSATION_ERROR")
        super().__init__(message, **kwargs)
        self.conversation_id = conversation_id
        if conversation_id:
            self.details["conversation_id"] = conversation_id


class StorageError(TurtleChatException):
    """Storage and persistence errors."""
    
    def __init__(self, message: str, operation: str = None, **kwargs):
        kwargs.setdefault("error_code", "STORAGE_ERROR")
        super().__init__(message, **kwargs)
        self.operation = operation
        if operation:
            self.details["operation"] = operation


class EncryptionError(TurtleChatException):
    """Encryption and decryption errors."""
    
    def __init__(self, message: str = "Encryption operation failed", **kwargs):
        kwargs.setdefault("error_code", "ENCRYPTION_ERROR")
        super().__init__(message, **kwargs)


class ExportError(TurtleChatException):
    """Export operation errors."""
    
    def __init__(self, message: str, export_format: str = None, **kwargs):
        kwargs.setdefault("error_code", "EXPORT_ERROR")
        super().__init__(message, **kwargs)
        self.export_format = export_format
        if export_format:
            self.details["export_format"] = export_format


class UIError(TurtleChatException):
    """User interface related errors."""
    
    def __init__(self, message: str, component: str = None, **kwargs):
        kwargs.setdefault("error_code", "UI_ERROR")
        super().__init__(message, **kwargs)
        self.component = component
        if component:
            self.details["component"] = component


# Exception handling utilities
def handle_exception(exception: Exception, context: str = None) -> str:
    """
    Handle exceptions gracefully and return user-friendly error messages.
    
    Args:
        exception: The exception to handle
        context: Additional context about where the error occurred
        
    Returns:
        User-friendly error message
    """
    # Log the full exception details
    import logging
    logger = logging.getLogger(__name__)
    
    error_context = f" in {context}" if context else ""
    logger.error(f"Exception occurred{error_context}: {str(exception)}", exc_info=True)
    
    # Return appropriate user message based on exception type
    if isinstance(exception, AuthenticationError):
        return "Authentication failed. Please check your credentials."
    
    elif isinstance(exception, SessionExpiredError):
        return "Your session has expired. Please log in again."
    
    elif isinstance(exception, FileSizeError):
        return f"File is too large. Maximum size allowed is {exception.details.get('max_size', 'unknown')}."
    
    elif isinstance(exception, UnsupportedFileTypeError):
        return f"File type '{exception.details.get('file_type', 'unknown')}' is not supported."
    
    elif isinstance(exception, FileProcessingError):
        return f"Error processing file: {exception.message}"
    
    elif isinstance(exception, TimeoutError):
        return "Request timed out. Please try again."
    
    elif isinstance(exception, RateLimitError):
        retry_after = exception.details.get('retry_after', 60)
        return f"Rate limit exceeded. Please wait {retry_after} seconds before trying again."
    
    elif isinstance(exception, NetworkError):
        if "Streamlit Cloud cannot connect to xAI service" in str(exception):
            return "Streamlit Cloud cannot reach the xAI/Grok service due to networking restrictions. Please use Claude Sonnet 4 instead, which works reliably on Streamlit Cloud."
        elif "Cannot connect to xAI service" in str(exception):
            return "Cannot connect to Grok/xAI service. Please check your internet connection or try again later. The service may be temporarily unavailable."
        return "Network error occurred. Please check your internet connection and try again."
    
    elif isinstance(exception, BedrockError):
        return "Error with AWS Bedrock service. Please try again later."
    
    elif isinstance(exception, XAIError):
        return "Error with xAI service. Please try again later."
    
    elif isinstance(exception, ConfigurationError):
        return "Configuration error. Please check your settings."
    
    elif isinstance(exception, ValidationError):
        return f"Invalid input: {exception.message}"
    
    elif isinstance(exception, ConversationError):
        return f"Conversation error: {exception.message}"
    
    elif isinstance(exception, StorageError):
        return "Storage error occurred. Please try again."
    
    elif isinstance(exception, EncryptionError):
        return "Security error occurred. Please try again."
    
    elif isinstance(exception, ExportError):
        return f"Export failed: {exception.message}"
    
    elif isinstance(exception, UIError):
        return f"Interface error: {exception.message}"
    
    elif isinstance(exception, TurtleChatException):
        return exception.message
    
    else:
        # Generic error message for unknown exceptions
        return "An unexpected error occurred. Please try again."


def create_error_response(exception: Exception, context: str = None) -> dict:
    """
    Create a structured error response for API-like usage.
    
    Args:
        exception: The exception to handle
        context: Additional context about where the error occurred
        
    Returns:
        Dictionary with error details
    """
    error_message = handle_exception(exception, context)
    
    response = {
        "success": False,
        "error": {
            "message": error_message,
            "type": type(exception).__name__,
            "context": context
        }
    }
    
    if isinstance(exception, TurtleChatException):
        response["error"]["code"] = exception.error_code
        response["error"]["details"] = exception.details
    
    return response