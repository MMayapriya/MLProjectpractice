import sys
from src.logger import logging


def get_detailed_error_message(error, error_detail: sys):
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in file: {file_name} at line: {line_number} with error message: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = get_detailed_error_message(error_message, error_detail)

    def __str__(self):
        return f"{self.error_message}"
    
