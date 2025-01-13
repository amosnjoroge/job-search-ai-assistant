from pydantic import BaseModel, HttpUrl, StringConstraints, field_validator
from typing import Optional, Annotated
import tempfile
import os
from streamlit.runtime.uploaded_file_manager import UploadedFile

class InitialJobPostingData(BaseModel):
    resume: str
    posting_text: Optional[Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]] = None
    posting_url: Optional[HttpUrl] = None

    @field_validator('posting_url')
    def validate_exclusive_fields(cls, v, values):
        posting_text = values.data.get('posting_text')
        
        # Check if both are None or empty
        if not v and not posting_text:
            raise ValueError('Must provide either posting text or URL')
        # Check if both are provided
        if v and posting_text:
            raise ValueError('Cannot provide both posting text and URL')
        return v

def validate_job_application(
    resume: str | None,
    posting_text: str | None = None,
    posting_url: str | None = None
) -> tuple[bool, str, dict]:
    """
    Validates job application data and returns validation status.
    
    Args:
        resume: Text of the CV file uploaded by user
        posting_text: The job posting text if provided
        posting_url: The job posting URL if provided
        
    Returns:
        tuple containing:
        - bool: Whether validation passed
        - str: Error message if validation failed, empty string if passed
        - dict: Validated data if passed, empty dict if failed
    """
    try:
        if not resume:
            return False, "Please upload a CV file", {}
        # Prepare data for validation
        validation_data = {
            "resume": resume,
            "posting_text": posting_text,
            "posting_url": posting_url
        }

        # Validate using Pydantic model
        job_data = InitialJobPostingData(**validation_data)
        
        return True, "", job_data.model_dump()

    except ValueError as e:
        return False, f"Validation error: {str(e)}", {}
    except Exception as e:
        return False, f"An error occurred: {str(e)}", {}
    