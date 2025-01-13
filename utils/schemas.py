import os
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, Dict, Any, List, Union
from enum import Enum


class AgentAction(str, Enum):
    """Valid actions for agent responses"""

    RESUME_CLEAN = "resume_clean"
    JOB_POSTING_CLEAN = "job_posting_clean"
    APPLICATION_PARSE = "application_parse"
    MATCH_ANALYZE = "match_analyze"
    RESUME_UPDATE = "resume_update"
    DOCUMENT_GENERATE = "cover_letter_resume_generate"


class Status(str, Enum):
    """Valid statuses for job applications"""

    DRAFT = "draft"
    NEW = "new"
    PARSED = "parsed"
    MATCHED = "matched"
    DOCUMENTS_GENERATED = "documents_generated"


# LLM Schema


class Provider(str, Enum):
    """Supported LLM providers"""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class BaseProviderConfig(BaseModel):
    """Base configuration for all providers"""

    provider: Provider
    model_name: str
    temperature: float = Field(
        default=(os.getenv("DEFAULT_TEMPERATURE") or 0.7), ge=0, le=1
    )
    max_tokens: Optional[int] = Field(default=4096, ge=1)


class OllamaConfig(BaseProviderConfig):
    """Ollama-specific configuration"""

    provider: Literal[Provider.OLLAMA]
    base_url: str = Field(default=os.getenv("OLLAMA_BASE_URL"))

    @field_validator("base_url")
    def validate_base_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v

    @field_validator("model_name")
    def validate_ollama_model(cls, v):
        valid_models = [
            "llama3.1",
            "llama3.2",
            "mistral-nemo:12b",
        ]
        if v not in valid_models:
            raise ValueError(
                f'Ollama model must be one of: {
                             ", ".join(valid_models)}'
            )
        return v


class OpenAIConfig(BaseProviderConfig):
    """OpenAI-specific configuration"""

    provider: Literal[Provider.OPENAI]
    api_key: str
    organization_id: Optional[str] = None

    @field_validator("api_key")
    def validate_api_key(cls, v):
        if not v.startswith("sk-"):
            raise ValueError('OpenAI API key must start with "sk-"')
        return v


class AnthropicConfig(BaseProviderConfig):
    """Anthropic-specific configuration"""

    provider: Literal[Provider.ANTHROPIC]
    api_key: Optional[str] = None

    @field_validator("model_name")
    def validate_model_name(cls, v):
        valid_models = [
            "claude-3-opus-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ]
        if v not in valid_models:
            raise ValueError(
                f'Model must be one of: {
                             ", ".join(valid_models)}'
            )
        return v


class ProviderFactory:
    """Factory for creating provider configurations"""

    @staticmethod
    def create_config(config_dict: Dict[str, Any]) -> BaseProviderConfig:
        provider = config_dict.get("provider")
        if not provider:
            raise ValueError("Provider is required")

        provider_map = {
            Provider.OLLAMA: OllamaConfig,
            Provider.OPENAI: OpenAIConfig,
            Provider.ANTHROPIC: AnthropicConfig,
        }

        config_class = provider_map.get(Provider(provider))
        if not config_class:
            raise ValueError(f"Unsupported provider: {provider}")

        return config_class(**config_dict)


class MessageRole(str, Enum):
    """Valid roles for chat messages"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageFormat(str, Enum):
    """Supported output formats"""

    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class ChatMessage(BaseModel):
    """Schema for a chat message"""

    role: MessageRole
    content: str
    format: Optional[MessageFormat] = Field(default=MessageFormat.TEXT)


class ChatOptions(BaseModel):
    """Options for chat completion"""

    temperature: Optional[float] = Field(default=0.7, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stop_sequences: Optional[List[str]] = None


# CV Parsing Schemas


class ContactInfo(BaseModel):
    """Contact information for a CV"""

    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None


class PersonalInfo(BaseModel):
    """Personal information section of a CV"""

    name: str = Field(..., example="John Doe")
    title: str = Field(..., example="Software Engineer")
    contact: Optional[ContactInfo] = Field(
        default=None, example={"email": "john.doe@example.com", "phone": "123-456-7890"}
    )


class Experience(BaseModel):
    """Work experience entry in a CV"""

    company: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[int] = None
    responsibilities: Optional[List[str]] = Field(default_factory=list)
    achievements: Optional[List[str]] = Field(default_factory=list)


class Education(BaseModel):
    """Education entry in a CV"""

    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[int] = None


class ParsedCVData(BaseModel):
    """Schema for parsed CV data"""

    personal_info: PersonalInfo
    skills: List[str] = Field(..., example=["python", "pandas", "fastapi"])
    experience: List[dict] = Field(
        ...,
        example=[
            {
                "company": "Company A",
                "title": "Developer",
                "duration": 24,
                "responsibilities": ["Developed applications"],
                "achievements": ["Employee of the Month"],
            }
        ],
    )
    education: List[dict] = Field(
        ...,
        example=[
            {
                "degree": "BSc Computer Science",
                "institution": "University X",
                "year": 2020,
            }
        ],
    )
    certifications: List[str] = Field(..., example=["Certified Python Developer"])
    languages: List[str] = Field(..., example=["English", "Spanish"])
    confidence_score: Optional[float] = Field(default=None, example=0.95)

    class Config:
        json_schema_extra = {
            "example": {
                "personal_info": {
                    "name": "John Doe",
                    "title": "Software Engineer",
                    "contact": {
                        "email": "john.doe@example.com",
                        "phone": "123-456-7890",
                    },
                },
                "skills": ["python", "pandas", "fastapi"],
                "experience": [
                    {
                        "company": "Company A",
                        "title": "Developer",
                        "duration": 24,
                        "responsibilities": ["Developed applications"],
                        "achievements": ["Employee of the Month"],
                    }
                ],
                "education": [
                    {
                        "degree": "BSc Computer Science",
                        "institution": "University X",
                        "year": 2020,
                    }
                ],
                "certifications": ["Certified Python Developer"],
                "languages": ["English", "Spanish"],
                "confidence_score": 0.95,
            }
        }


# Job Posting Schemas


class CompanyInfo(BaseModel):
    """Company information from job posting"""

    name: str = Field(default="unknown", example="Tech Corp")
    industry: str = Field(default="unknown", example="Software Development")
    location: str = Field(default="unknown", example="New York, NY")
    company_size: Optional[str] = Field(default=None, example="51-200 employees")


class JobRequirement(BaseModel):
    """Individual requirement with type and importance"""

    skill: str = Field(..., example="Python")
    type: str = Field(default="technical", example="technical")
    required: bool = Field(default=True, example=True)
    experience_years: Optional[int] = Field(default=None, example=3)


class WorkingConditions(BaseModel):
    """Working conditions and benefits"""

    work_mode: Optional[str] = Field(default=None, example="remote")
    schedule: Optional[str] = Field(default=None, example="9 AM - 5 PM")
    benefits: List[str] = Field(
        default_factory=list, example=["Health Insurance", "401(k)"]
    )


class ParsedJobData(BaseModel):
    """Schema for parsed job posting data"""

    company_info: CompanyInfo
    title: str = Field(..., example="Software Engineer")
    experience_level: Optional[str] = Field(default=None, example="Mid-level")
    requirements: List[JobRequirement] = Field(
        ...,
        example=[
            {
                "skill": "Python",
                "type": "technical",
                "required": True,
                "experience_years": 3,
            },
            {
                "skill": "Django",
                "type": "technical",
                "required": True,
                "experience_years": 2,
            },
            {"skill": "Communication", "type": "soft", "required": False},
        ],
    )
    responsibilities: List[str] = Field(
        ...,
        example=[
            "Develop and maintain web applications",
            "Collaborate with cross-functional teams",
            "Participate in code reviews",
        ],
    )
    education: List[str] = Field(..., example=["Bachelor's Degree in Computer Science"])
    certifications: List[str] = Field(..., example=["Certified Python Developer"])
    working_conditions: Optional[WorkingConditions] = Field(default=None)
    salary_range: Optional[Dict[str, Any]] = Field(
        default=None, example={"min": 80000, "max": 120000, "currency": "USD"}
    )
    confidence_score: Optional[float] = Field(default=None, example=0.85)

    class Config:
        json_schema_extra = {
            "example": {
                "company_info": {
                    "name": "Tech Corp",
                    "industry": "Software Development",
                    "location": "New York, NY",
                    "company_size": "51-200 employees",
                },
                "title": "Software Engineer",
                "experience_level": "Mid-level",
                "requirements": [
                    {
                        "skill": "Python",
                        "type": "technical",
                        "required": True,
                        "experience_years": 3,
                    },
                    {
                        "skill": "Django",
                        "type": "technical",
                        "required": True,
                        "experience_years": 2,
                    },
                    {"skill": "Communication", "type": "soft", "required": False},
                ],
                "responsibilities": [
                    "Develop and maintain web applications",
                    "Collaborate with cross-functional teams",
                    "Participate in code reviews",
                ],
                "education": ["Bachelor's Degree in Computer Science"],
                "certifications": ["Certified Python Developer"],
                "working_conditions": {
                    "work_mode": "remote",
                    "schedule": "9 AM - 5 PM",
                    "benefits": ["Health Insurance", "401(k)"],
                },
                "salary_range": {"min": 80000, "max": 120000, "currency": "USD"},
                "confidence_score": 0.85,
            }
        }


# Match Analysis Schemas


class MatchAnalysis(BaseModel):
    """Schema for match analysis results"""

    overall_score: float = Field(..., example=0.75)
    skill_alignment: Dict[str, Any] = Field(
        ...,
        example={
            "score": 0.8,
            "matching_skills": ["Python", "Django"],
            "missing_skills": ["JavaScript"],
            "partial_matches": [
                {
                    "required": "Communication",
                    "similar_found": "Teamwork",
                    "similarity": 0.6,
                }
            ],
        },
    )
    experience_alignment: Dict[str, Any] = Field(
        ...,
        example={
            "score": 0.7,
            "years_required": 3,
            "years_actual": 2,
            "level_match": False,
        },
    )
    education_alignment: Dict[str, Any] = Field(
        ...,
        example={
            "score": 0.9,
            "matches_requirement": True,
            "details": "Bachelor's degree in Computer Science matches the requirement.",
        },
    )


class Recommendation(BaseModel):
    """Schema for recommendations based on match analysis"""

    category: str = Field(..., example="Skill Improvement")
    priority: float = Field(..., example=1.0)
    action: str = Field(..., example="Take an online course in JavaScript")
    expected_impact: str = Field(..., example="Improved job match score by 0.2")


class MatchResult(BaseModel):
    """Schema for the overall match result"""

    match_analysis: MatchAnalysis
    recommendations: List[Recommendation] = Field(
        ...,
        example=[
            {
                "category": "Skill Improvement",
                "priority": 1.0,
                "action": "Take an online course in JavaScript",
                "expected_impact": "Improved job match score by 0.2",
            },
            {
                "category": "Experience Gain",
                "priority": 0.5,
                "action": "Participate in open-source projects",
                "expected_impact": "Gained practical experience in software development.",
            },
        ],
    )
    interview_focus_areas: List[str] = Field(
        ..., example=["Technical skills", "Problem-solving abilities"]
    )
    confidence_score: Optional[float] = Field(default=None, example=0.85)

    class Config:
        json_schema_extra = {
            "example": {
                "match_analysis": {
                    "overall_score": 0.75,
                    "skill_alignment": {
                        "score": 0.8,
                        "matching_skills": ["Python", "Django"],
                        "missing_skills": ["JavaScript"],
                        "partial_matches": [
                            {
                                "required": "Communication",
                                "similar_found": "Teamwork",
                                "similarity": 0.6,
                            }
                        ],
                    },
                    "experience_alignment": {
                        "score": 0.7,
                        "years_required": 3,
                        "years_actual": 2,
                        "level_match": False,
                    },
                    "education_alignment": {
                        "score": 0.9,
                        "matches_requirement": True,
                        "details": "Bachelor's degree in Computer Science matches the requirement.",
                    },
                },
                "recommendations": [
                    {
                        "category": "Skill Improvement",
                        "priority": 1.0,
                        "action": "Take an online course in JavaScript",
                        "expected_impact": "Improved job match score by 0.2",
                    },
                    {
                        "category": "Experience Gain",
                        "priority": 0.5,
                        "action": "Participate in open-source projects",
                        "expected_impact": "Gained practical experience in software development.",
                    },
                ],
                "interview_focus_areas": [
                    "Technical skills",
                    "Problem-solving abilities",
                ],
                "confidence_score": 0.85,
            }
        }


class GeneratedDocuments(BaseModel):
    """
    Represents the documents generated as part of the document generation process.
    """

    updated_resume: str = Field(..., description="The updated resume content")
    cover_letter: str = Field(..., description="The generated cover letter content")
    changes_made: Dict[str, Any] = Field(
        ..., description="A dictionary of changes made to the documents"
    )


class ResumeGenerationResult(BaseModel):
    """
    Represents the result of the resume generation process.
    """

    updated_resume: str = Field(..., description="The updated resume content")
    changes_made: Dict[str, Any] = Field(
        ..., description="A dictionary of changes made to the resume"
    )

class CoverLetterGenerationResult(BaseModel):
    """
    Represents the result of the cover letter generation process.
    """

    cover_letter: str = Field(..., description="The generated cover letter content")
    key_points_addressed: List[str] = Field(
        ..., description="A list of key points addressed in the cover letter"
    )
