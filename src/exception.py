import sys

def error_message_details(error, error_detail:sys):
    """Return a formatted error message with details from the exception."""
    
    _, _, exc_traceback = sys.exc_info()
    file_name = exc_traceback.tb_frame.f_code.co_filename
    line_number = exc_traceback.tb_lineno

    error_message = f"Error occured in python script name {file_name} in \
     line number {line_number} error message {str(error)}\n"

    return error_message

class CustomException(Exception):
    """Custom exception class to handle exceptions with detailed messages."""
    
    def __init__(self, error_message, error_detail:sys):
        """Initialize the custom exception with an error message and details."""
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        """Return the string representation of the custom exception."""
        return self.error_message
    