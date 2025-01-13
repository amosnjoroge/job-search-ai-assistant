from urllib.parse import urlparse
import aiohttp
import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Type
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
from pydantic import BaseModel

from agents.llms import BaseLLM
from data_handler import DataHandler
from utils.schemas import (
    ChatMessage,
    MessageRole,
    ParsedCVData,
    ParsedJobData,
    MatchResult,
    AgentAction,
    ResumeGenerationResult,
    CoverLetterGenerationResult,
)


@dataclass
class JobData:
    company: str
    title: str
    requirements: List[str]
    raw_text: str


@dataclass
class CVData:
    skills: List[str]
    experience: Dict[str, str]
    raw_text: str


class BaseAgent(ABC):
    # Class-level shared memory for all agents
    _shared_memory = []

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    @classmethod
    def add_to_memory(cls, entry: dict):
        """Add an entry to shared memory"""
        timestamp = datetime.now().isoformat()
        memory_entry = {"timestamp": timestamp, **entry}
        cls._shared_memory.append(memory_entry)

    @classmethod
    def get_memory(cls) -> list:
        """Get all shared memory entries"""
        return cls._shared_memory

    @classmethod
    def clear_memory(cls):
        """Clear shared memory"""
        cls._shared_memory = []

    @abstractmethod
    async def run(self):
        pass

    async def _process_llm_response(
        self, response: str, validator_schema: Type[BaseModel]
    ) -> Dict:
        """Convert LLM response to structured dictionary using provided schema

        Args:
            response (str): Raw response from LLM
            validator_schema (Type[BaseModel]): Pydantic model class to validate against

        Returns:
            Dict: Validated and structured data
        """
        try:
            if not response:
                raise ValueError("Response is empty")
            if isinstance(response, str):
                try:
                    data = json.loads(response)
                except json.JSONDecodeError as e:
                    # If the response contains markdown code blocks, try to extract JSON
                    json_match = re.search(
                        r"```json\s*(.*?)\s*```", response, re.DOTALL
                    )
                    if json_match:
                        data = json.loads(json_match.group(1))
                    else:
                        # Try self-healing with LLM
                        corrected_data = await self._request_correction(
                            response, validator_schema
                        )
                        return corrected_data
            else:
                data = response

            # Validate against schema
            validated_data = validator_schema(**data)
            return validated_data.model_dump()
        except Exception as e:
            # If any validation fails, try self-healing
            corrected_data = await self._request_correction(response, validator_schema)
            return corrected_data

    async def _request_correction(
        self, response: str, validator_schema: Type[BaseModel], retry_counter: int = 0
    ) -> Dict:
        """Request the LLM to correct malformed responses.

        Args:
            response (str): The problematic response
            validator_schema (Type[BaseModel]): Schema to validate against

        Returns:
            Dict: Corrected and validated response
        """
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"""You are an expert data validator and corrector.
                    Your task is to fix malformed JSON responses and ensure they match the required schema.

                    IMPORTANT RULES:
                    1. You MUST respond ONLY in valid JSON format
                    2. Do NOT include any explanations, notes, or text outside the JSON
                    3. All fields must match the exact schema provided below
                    4. Use null for any fields where information is not available
                    5. Ensure all text values are properly escaped strings
                    6. Numbers must be numeric values, not strings
                    7. Boolean values must be true/false, not strings
                    8. Arrays must be properly formatted with square brackets
                    9. Objects must be properly formatted with curly braces

                    Required Schema:
                    {json.dumps(validator_schema.model_json_schema(), indent=2)}""",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"""Fix the following response to match the required schema:

                        Original Response:
                        {response}

                        Common issues to check:
                        1. Missing or malformed JSON structure
                        2. Incorrect data types
                        3. Missing required fields
                        4. Improperly formatted arrays or objects
                        5. Unescaped strings
                        6. Invalid field names

                        Return ONLY the corrected JSON.""",
            ),
        ]

        try:
            corrected_response = await self.llm.chat(messages)

            # Extract content if response is a dict
            if isinstance(corrected_response, dict):
                corrected_response = corrected_response.get("message", {}).get(
                    "content", corrected_response
                )

            # Try to parse the corrected response
            if isinstance(corrected_response, str):
                try:
                    data = json.loads(corrected_response)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code blocks
                    json_match = re.search(
                        r"```json\s*(.*?)\s*```", corrected_response, re.DOTALL
                    )
                    if json_match:
                        data = json.loads(json_match.group(1))
                    else:
                        raise ValueError("Failed to parse corrected response")
            else:
                data = corrected_response

            # Validate against schema
            response = validator_schema(**data)
            return response.model_dump()
        except Exception as e:
            if retry_counter >= 3:
                raise ValueError(
                    f"Failed to correct response after multiple attempts\n{str(e)}"
                )
            print(
                f"Error during response correction: Retrying... ({retry_counter + 1})"
            )
            self._request_correction(response, validator_schema, retry_counter + 1)


class InputAgent(BaseAgent):
    def __init__(self, llm: BaseLLM):
        super().__init__(llm)
        self.dh = DataHandler()

    async def run(
        self, application_id: str, input_data: str, action: AgentAction
    ) -> Dict[str, str]:
        """Process either URL or text input and return structured data"""
        if not input_data:
            raise ValueError("Input data cannot be empty")

        result = None
        if self._is_url(input_data):
            result = await self._handle_url(input_data)
        else:
            result = await self._handle_text(input_data)

        if not result or not result.get("raw_content"):
            raise ValueError("Failed to extract content from input")

        cleaned_text = await self._clean_text(result["raw_content"])
        memory_entry = {
            "agent": {"type": "InputAgent", "llm": self.llm.to_dict()},
            "action": action.value,
            "data": {"content": cleaned_text},
        }
        self.add_to_memory(memory_entry)
        update_data = {
            "update_date": datetime.now().isoformat(),
            "posting_url": result.get("source_url"),
        }
        if action == AgentAction.RESUME_CLEAN:
            update_data["resume"] = cleaned_text
        elif action == AgentAction.JOB_POSTING_CLEAN:
            update_data["posting_text"] = cleaned_text
            update_data["status"] = "new"

        self.dh.update_data(
            table_name=self.dh.TABLE_APPLICATIONS,
            data=update_data,
            condition=f"id = '{application_id}'",
        )
        self.dh.insert_data(
            table_name=self.dh.TABLE_ANALYSIS_STEPS,
            data=(
                application_id,
                json.dumps(memory_entry),
                datetime.now().isoformat(),
            ),
            columns=("application_id", "data", "created_at"),
        )

        return cleaned_text

    def _is_url(self, text: str) -> bool:
        """Validate if input is URL"""
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except:
            return False

    async def _clean_text(self, text: str) -> str:
        """Clean and format input text using LLM"""
        try:
            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="""You are a text cleaning assistant. 
                            Your task is to reformat and clean text from CVs/resumes and job posts and descriptions 
                            into clear, readable formats. Do NOT add or remove any information from the text.
                            Remove unnecessary newlines, normalize spacing, organize sections clearly, 
                            and preserve essential structure and content.
                            Important Note: Respond with only the cleaned text, without any additional comments or explanations.""",
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=f"""Clean and format the following CV text while:
                        1. Maintaining section headers
                        2. Preserving list formatting
                        3. Removing excessive newlines/spaces
                        4. Keeping contact information organized
                        5. Retaining all original content

                        Text to clean: {text}""",
                ),
            ]

            response = await self.llm.chat(messages)

            if isinstance(response, dict):
                cleaned_text = response.get("message", {}).get("content")
            else:
                cleaned_text = response

            if not cleaned_text:
                raise ValueError("Empty response from LLM")

            return cleaned_text

        except Exception as e:
            raise RuntimeError(f"Error during text cleaning: {str(e)}")

    async def _handle_url(self, url: str) -> Dict[str, str]:
        """Process URL input"""
        raw_text = await self._scrape_content(url)
        return {"type": "url", "source_url": url, "raw_content": raw_text}

    async def _handle_text(self, text: str) -> Dict[str, str]:
        """Process text input"""
        return {"type": "text", "source_url": None, "raw_content": text}

    async def _scrape_content(self, url: str) -> str:
        """Scrape job posting content from URL"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")
                        for script in soup(["script", "style"]):
                            script.decompose()
                        return soup.get_text()
                    else:
                        raise ConnectionError(f"Failed to fetch URL: {response.status}")
            except Exception as e:
                raise ConnectionError(f"Error accessing URL: {str(e)}")


class ParserAgent(BaseAgent):
    def __init__(self, llm: BaseLLM):
        super().__init__(llm)
        self.dh = DataHandler()

    async def run(
        self, application_id: str, cv_text: str, job_text: str
    ) -> Dict[str, Any]:
        """Extract structured data from both CV and job posting"""
        cv_data = await self.parse_cv(cv_text) if cv_text else None
        job_data = await self.parse_job_post(job_text) if job_text else None

        result = {
            "cv_data": (
                await self._process_llm_response(
                    cv_data["message"]["content"], ParsedCVData
                )
                if cv_data
                else None
            ),
            "job_data": (
                await self._process_llm_response(
                    job_data["message"]["content"], ParsedJobData
                )
                if job_data
                else None
            ),
        }
        memory_entry = {
            "agent": {"type": "ParserAgent", "llm": self.llm.to_dict()},
            "action": AgentAction.APPLICATION_PARSE.value,
            "data": result,
        }

        self.add_to_memory(memory_entry)

        job_data = result["job_data"]
        self.dh.update_data(
            table_name=self.dh.TABLE_APPLICATIONS,
            data={
                "job_title": job_data["title"],
                "company": job_data["company_info"]["name"],
                "update_date": datetime.now().isoformat(),
                "status": "parsed",
            },
            condition=f"id = '{application_id}'",
        )
        self.dh.insert_data(
            table_name=self.dh.TABLE_ANALYSIS_STEPS,
            data=(
                application_id,
                json.dumps(memory_entry),
                datetime.now().isoformat(),
            ),
            columns=("application_id", "data", "created_at"),
        )

        return result

    async def parse_job_post(self, raw_text: str) -> JobData:
        """Extract structured data from job posting"""
        # Create chat messages
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"""You are an expert job posting analyzer with years of experience in HR and recruitment.
                        Your task is to extract structured information from job postings.

                        IMPORTANT RULES:
                        1. You MUST respond ONLY in valid JSON format
                        2. Do NOT include any explanations, notes, or text outside the JSON
                        3. All fields must match the exact schema provided below
                        4. Use null for any fields where information is not available
                        5. Ensure all text values are properly escaped strings
                        6. Numbers must be numeric values, not strings
                        7. Boolean values must be true/false, not strings
                        8. When calculating durations or dates referring to "current" or "present",
                           use {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} as the reference timestamp

                        Response format must be EXACTLY:
                        {{
                            "company_info": {{
                                "name": string,
                                "industry": string,
                                "location": string,
                                "company_size": string ans Should be capitalized
                            }},
                            "title": string,
                            "experience_level": string,
                            "requirements": [{{
                                "skill": string,
                                "type": string,
                                "required": boolean,
                                "experience_years": number
                            }}],
                            "responsibilities": [string],
                            "education": [string],
                            "certifications": [string],
                            "working_conditions": {{
                                "work_mode": string,
                                "schedule": string,
                                "benefits": [string]
                            }},
                            "salary_range": {{
                                "min": number,
                                "max": number,
                                "currency": string
                            }},
                            "confidence_score": float between 0 and 1
                        }}""",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"""Analyze the following job posting and extract key information in JSON format.
                Be thorough and include both explicit and implicit requirements.

                        Required information:
                        - Company name and any relevant company details
                        - Complete job title and level
                        - Required technical skills and competencies
                        - Preferred/optional skills
                        - Required years of experience
                        - Education requirements
                        - Industry-specific certifications
                        - Key responsibilities
                        - Working conditions (remote/hybrid/onsite)

                        Ensure all skills are in lowercase and multi-word terms use hyphens.
                        Include a confidence score (0-1) for each extracted field.

                        Job posting text:
                        {raw_text}""",
            ),
        ]

        response = await self.llm.chat(messages)
        return response

    async def parse_cv(self, raw_text: str) -> CVData:
        """Extract structured data from CV"""
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"""You are an expert CV/resume analyzer with extensive experience in HR and talent acquisition.
                        Your task is to extract structured information from CVs and resumes.

                        IMPORTANT RULES:
                        1. You MUST respond ONLY in valid JSON format
                        2. Do NOT include any explanations, notes, or text outside the JSON
                        3. All fields must match the exact schema provided below
                        4. Use null for any fields where information is not available
                        5. Ensure all text values are properly escaped strings
                        6. Numbers must be numeric values, not strings
                        7. List all skills in lowercase with hyphens for multi-word terms
                        8. Maintain chronological order in experience and education
                        9. When calculating durations or dates referring to "current" or "present",
                           use {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} as the reference timestamp

                        Response format must be EXACTLY:
                        {{
                            "personal_info": {{
                                "name": string,
                                "title": string,
                                "contact": {{
                                    "email": string,
                                    "phone": string,
                                    "location": string
                                }}
                            }},
                            "skills": [string],
                            "experience": [{{
                                "company": string,
                                "title": string,
                                "duration": number in months,
                                "responsibilities": [string],
                                "achievements": [string]
                            }}],
                            "education": [{{
                                "degree": string,
                                "institution": string,
                                "year": number
                            }}],
                            "certifications": [string],
                            "languages": [string],
                            "confidence_score": float between 0 and 1
                        }}""",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"""Extract the following information in JSON format only:
                        - Personal and contact details
                        - Technical and soft skills
                        - Work experience with dates and achievements
                        - Education and certifications
                        - Language proficiencies

                        CV text:
                        {raw_text}""",
            ),
        ]

        response = await self.llm.chat(messages)
        return response

    def _create_cv_parsing_prompt(self, text: str) -> str:
        return f"""Extract the following from the CV in JSON format:
        - Professional skills
        - Work experience with duration
        Text: {text}"""


class MatcherAgent(BaseAgent):
    def __init__(self, llm: BaseLLM):
        super().__init__(llm)
        self.dh = DataHandler()

    async def run(
        self, application_id: str, parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Main entry point for matching process"""
        if not parsed_data.get("cv_data") or not parsed_data.get("job_data"):
            raise ValueError("Both CV and job data are required for matching")

        cv_data, job_data = parsed_data.values()

        # Calculate various match scores
        skill_match = await self._calculate_skill_match(
            cv_data.get("skills", []), job_data.get("requirements", [])
        )

        experience_match = await self._calculate_experience_match(
            cv_data.get("experience", []), job_data.get("experience_level")
        )

        result = {
            "match_scores": {
                "overall_match": 0.0,  # Will be calculated
                "skill_match": skill_match,
                "experience_match": experience_match,
            },
            "matching_skills": [],
            "missing_skills": [],
            "recommendations": [],
        }

        # Use LLM for detailed analysis
        response = await self._analyze_match(cv_data, job_data, result)
        analysis = await self._process_llm_response(
            response["message"]["content"], MatchResult
        )
        memory_entry = {
            "agent": {"type": "MatcherAgent", "llm": self.llm.to_dict()},
            "action": AgentAction.MATCH_ANALYZE.value,
            "data": {"analysis": analysis},
        }
        self.add_to_memory(memory_entry)

        self.dh.update_data(
            table_name=self.dh.TABLE_APPLICATIONS,
            data={"update_date": datetime.now().isoformat(), "status": "matched"},
            condition=f"id = '{application_id}'",
        )
        self.dh.insert_data(
            table_name=self.dh.TABLE_ANALYSIS_STEPS,
            data=(
                application_id,
                json.dumps(memory_entry),
                datetime.now().isoformat(),
            ),
            columns=("application_id", "data", "created_at"),
        )

        return analysis

    async def _calculate_skill_match(
        self, cv_skills: List[str], job_requirements: List[Dict]
    ) -> float:
        """Calculate match score for skills"""
        if not job_requirements:
            return 0.0

        required_skills = [
            req["skill"].lower()
            for req in job_requirements
            if req.get("required", True)
        ]

        cv_skills_lower = [skill.lower() for skill in cv_skills]

        matched_skills = {
            skill
            for skill in cv_skills_lower
            if any(req_skill in skill for req_skill in required_skills)
        }
        return len(matched_skills) / len(required_skills) if required_skills else 0.0

    async def _calculate_experience_match(
        self, cv_experience: List[Dict], required_level: str
    ) -> float:
        """Calculate match score for experience"""
        if not cv_experience or not required_level:
            return 0.0

        total_years = sum(
            float(exp.get("duration", 0)) / 12
            for exp in cv_experience
            if exp.get("duration")
        )

        # Map experience levels to years
        level_mapping = {"entry": 0, "junior": 2, "mid": 4, "senior": 7, "expert": 10}

        required_years = level_mapping.get(required_level.lower(), 0)
        if required_years == 0:
            return 1.0

        return min(total_years / required_years, 1.0)

    async def _analyze_match(
        self, cv_data: Dict, job_data: Dict, match_scores: Dict
    ) -> Dict:
        """Use LLM to provide detailed analysis and recommendations"""
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=f"""You are an expert job match analyzer with extensive experience in technical recruitment.
                        Your task is to analyze the match between a candidate's CV and job requirements.

                        IMPORTANT RULES:
                        1. You MUST respond ONLY in valid JSON format
                        2. Do NOT include any explanations, notes, or text outside the JSON
                        3. All fields must match the exact schema provided below
                        4. Use null for any fields where information is not available
                        5. Ensure all text values are properly escaped strings
                        6. Numbers must be numeric values, not strings
                        7. All scores must be between 0 and 1
                        8. When referencing skills, use lowercase with hyphens for multi-word terms

                        Response format must be EXACTLY:
                        {{
                            "match_analysis": {{
                                "overall_score": number,
                                "skill_alignment": {{
                                    "score": number,
                                    "matching_skills": [string],
                                    "missing_skills": [string],
                                    "partial_matches": [{{
                                        "required": string,
                                        "similar_found": string,
                                        "similarity": number
                                    }}]
                                }},
                                "experience_alignment": {{
                                    "score": number,
                                    "years_required": number,
                                    "years_actual": number,
                                    "level_match": boolean
                                }},
                                "education_alignment": {{
                                    "score": number,
                                    "matches_requirement": boolean,
                                    "details": string
                                }}
                            }},
                            "recommendations": [{{
                                "category": string,
                                "priority": number,
                                "action": string,
                                "expected_impact": string
                            }}],
                            "interview_focus_areas": [string],
                            "confidence_score": float between 0 and 1
                        }}""",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"""Analyze the match between the candidate's CV and job requirements by:
                        1. Calculating precise match scores for skills, experience, and education
                        2. Identifying exact matching skills and clear skill gaps
                        3. Evaluating experience level alignment
                        4. Providing actionable recommendations
                        5. Suggesting specific interview focus areas

                        Use these data points for analysis:
                        CV Data: {json.dumps(cv_data)}
                        Job Requirements: {json.dumps(job_data)}
                        Initial Match Metrics: {json.dumps(match_scores)}""",
            ),
        ]

        response = await self.llm.chat(messages)
        return response


class DocumentGeneratorAgent(BaseAgent):
    def __init__(self, llm: BaseLLM):
        super().__init__(llm)
        self.dh = DataHandler()

    async def run(
        self,
        application_id: str,
        original_resume: str,
        job_posting: str,
        job_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate optimized documents based on job analysis"""

        # Generate optimized resume
        updated_resume = await self._optimize_resume(original_resume, job_analysis)
        resume_content = updated_resume["message"]["content"]
        validated_resume = await self._process_llm_response(
            resume_content, ResumeGenerationResult
        )

        # Generate cover letter
        cover_letter = await self._generate_cover_letter(
            original_resume, job_analysis, job_posting
        )
        cover_letter_content = cover_letter["message"]["content"]
        validated_cover_letter = await self._process_llm_response(
            cover_letter_content, CoverLetterGenerationResult
        )

        validated_result = {
            "updated_resume": validated_resume["updated_resume"],
            "changes_made": validated_resume.get("changes_made", {}),
            "cover_letter": validated_cover_letter["cover_letter"],
            "key_points_addressed": validated_cover_letter.get(
                "key_points_addressed", []
            ),
        }

        memory_entry = {
            "agent": {"type": "DocumentGeneratorAgent", "llm": self.llm.to_dict()},
            "action": AgentAction.DOCUMENT_GENERATE.value,
            "data": validated_result,
        }
        self.add_to_memory(memory_entry)

        self.dh.update_data(
            table_name=self.dh.TABLE_APPLICATIONS,
            data={
                "update_date": datetime.now().isoformat(),
                "status": "documents_generated",
                "resume_regenerated": validated_result["updated_resume"],
                "cover_letter": validated_result["cover_letter"],
            },
            condition=f"id = '{application_id}'",
        )

        # Log analysis step
        self.dh.insert_data(
            table_name=self.dh.TABLE_ANALYSIS_STEPS,
            data=(application_id, json.dumps(memory_entry), datetime.now().isoformat()),
            columns=("application_id", "data", "created_at"),
        )

        return validated_result

    async def _optimize_resume(
        self, original_resume: str, job_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize resume based on job analysis"""

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="""You are an expert resume optimization specialist. Your task is to make 
                strategic adjustments to resumes to better align with job requirements while maintaining 
                authenticity and truthfulness.

                IMPORTANT RULES:
                1. Never fabricate or exaggerate experiences
                2. Maintain the original structure and format
                3. Only adjust terminology and emphasis
                4. Preserve all dates and quantifiable achievements
                5. Focus on highlighting relevant skills and experiences
                6. Ensure all modifications are factual and verifiable
                7. Very Important: Respond with the tone and style of the original resume
                
                Response format must be JSON with:
                {
                    "updated_resume": string,
                    "changes_made": {
                        "terminology_updates": [string],
                        "emphasized_skills": [string],
                        "reordered_sections": [string]
                    },
                    "confidence_score": float
                }""",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"""Optimize this resume for the analyzed job position. Make strategic adjustments:
                1. Align terminology with job requirements
                2. Emphasize most relevant skills and experiences
                3. Adjust ordering of skills/experiences if needed
                4. Highlight achievements that match job needs
                5. Document all changes made
                
                Original Resume:
                {original_resume}
                
                Job Analysis:
                {json.dumps(job_analysis)}
                
                Return the updated resume with all changes documented.""",
            ),
        ]

        return await self.llm.chat(messages)

    async def _generate_cover_letter(
        self,
        updated_resume: Dict[str, Any],
        job_analysis: Dict[str, Any],
        job_posting: str,
    ) -> Dict[str, Any]:
        """Generate a targeted cover letter based on the job posting and analysis"""

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="""You are an expert cover letter writer with years of experience in professional 
                documentation. Your task is to create compelling, personalized cover letters that 
                effectively connect candidate qualifications with job requirements and company needs.

                IMPORTANT RULES:
                1. Maintain professional tone and format
                2. Focus on specific, relevant achievements
                3. Address key job requirements directly
                4. Include measurable impacts where possible
                5. Keep length to one page
                6. Avoid generic statements
                7. Reference specific company details and job requirements
                8. Show understanding of company culture and values
                9. Demonstrate research and genuine interest
                10. Mirror language and keywords from job posting
                11. Address any specific instructions from job posting
                12. Include proper business letter formatting with date and contact details
                7. Very Important: Respond with the tone and style of the resume
                
                Response format must be JSON with:
                {
                    "cover_letter": string,
                    "key_points_addressed": [string],
                    "confidence_score": float
                }""",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"""Generate a compelling cover letter that:
                1. Opens with a strong, personalized introduction referencing the specific role and company
                2. Demonstrates understanding of company's needs and culture
                3. Highlights 2-3 most relevant achievements that directly address job requirements
                4. Shows enthusiasm and specific interest in the role and company
                5. Addresses any unique requirements or preferences from job posting
                6. Closes with clear next steps and call to action
                
                Use these data points:
                Resume: {updated_resume}
                Job Analysis: {json.dumps(job_analysis)}
                Original Job Posting: {job_posting}
                
                Format as a proper business letter with:
                - Current date
                - Recipient's contact information (if available in job posting)
                - Professional greeting
                - 3-4 focused paragraphs
                - Professional closing
                
                Return a professionally formatted cover letter.""",
            ),
        ]

        return await self.llm.chat(messages)
