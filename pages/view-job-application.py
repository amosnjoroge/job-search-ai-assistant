import json
import asyncio
import streamlit as st
from data_handler import DataHandler
from page_header import page_header
from typing import Dict
from agents.agent import InputAgent, ParserAgent, MatcherAgent, DocumentGeneratorAgent
from agents.llms import LLMFactory
from utils.schemas import AgentAction, Provider, Status

dh = DataHandler()

st.set_page_config(
    page_title="Job Post - Awesome Job Search Assistant",
    page_icon=":mag_right:",
    layout="wide",
)


def back_home():
    st.switch_page("main.py")


def fetch_analysis_step_data(application_id: str, action: AgentAction):

    json_query = f"json_extract(data, '$.action') = '{action.value}'"
    condition = f"application_id='{application_id}' AND {json_query}"
    step_data = dh.query_data(dh.TABLE_ANALYSIS_STEPS, condition=condition)

    return step_data


def display_match_analysis(analysis_data: Dict):
    # 2 columns for the main metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Overall Match Score",
            f"{analysis_data['match_analysis']['overall_score']*100:.0f}%",
            help="""
            This is a quantitative measure that represents the aggregate match quality 
            between the candidate's CV and the job requirements. 
            It is calculated based on various factors, such as skill alignment, 
            experience alignment, and education alignment.
            """,
        )
    with col2:
        st.metric(
            "Confidence Score",
            f"{analysis_data['confidence_score'] is not None and analysis_data['confidence_score']*100:.0f}%",
            help="""
            This is a subjective rating provided by the LLM regarding its 
            own certainty about the accuracy of the overall score and the analysis it has generated.\n
            This score reflects the LLM's internal assessment of how reliable it believes its conclusions are, 
            based on the data it processed and the patterns it recognized.
            """,
        )

    with st.container(border=True):
        left_col, right_col = st.columns(2)

        with left_col:
            # Skills Analysis
            with st.expander("📊 Skills Analysis", expanded=False):
                skill_data = analysis_data["match_analysis"]["skill_alignment"]
                st.metric("Skills Match Score", f"{skill_data['score']*100:.0f}%")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Matching Skills")
                    for skill in skill_data["matching_skills"]:
                        st.success(f"✓ {skill}")

                with col2:
                    st.subheader("Missing Skills")
                    for skill in skill_data["missing_skills"]:
                        st.error(f"✗ {skill}")

                st.subheader("Partial Matches")
                for match in skill_data["partial_matches"]:
                    st.warning(
                        f"Required: {match['required']} → Similar: {match['similar_found']} "
                        f"(Similarity: {match['similarity']*100:.0f}%)"
                    )

            # Education Analysis
            with st.expander("🎓 Education Analysis", expanded=False):
                edu_data = analysis_data["match_analysis"]["education_alignment"]
                st.metric(
                    "Education Match Score", f"{edu_data.get('score', 0)*100:.0f}%"
                )

                if edu_data["matches_requirement"]:
                    st.success(f"✓ Education requirement met: {edu_data['details']}")
                else:
                    st.error(f"✗ Education requirement not met: {edu_data['details']}")

            # Recommendations
            with st.expander("💡 Recommendations", expanded=False):
                for rec in analysis_data["recommendations"]:
                    st.info(
                        f"**{rec['category']}** (Priority: {rec['priority']})\n\n"
                        f"Action: {rec['action']}\n\n"
                        f"Expected Impact: {rec['expected_impact']}"
                    )

        with right_col:
            # Experience Analysis
            with st.expander("👔 Experience Analysis", expanded=False):
                exp_data = analysis_data["match_analysis"]["experience_alignment"]
                st.metric("Experience Match Score", f"{exp_data['score']*100:.0f}%")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Years Required", exp_data["years_required"])
                with col2:
                    st.metric("Years Actual", exp_data["years_actual"])

                if exp_data["level_match"]:
                    st.success("✓ Experience level matches requirements")
                else:
                    st.error("✗ Experience level does not match requirements")

            # Interview Focus Areas
            with st.expander("🎯 Interview Focus Areas", expanded=False):
                for area in analysis_data["interview_focus_areas"]:
                    st.write(f"• {area}")


def handle_error(message: str):
    st.error(message)
    st.empty()
    st.session_state["error_occurred"] = True
    st.stop()


if "current_application" not in st.session_state:
    st.session_state["current_application"] = None

current_app = dh.query_data(dh.TABLE_CURRENT_APPLICATION)
if current_app.empty:
    back_home()

current_app_id = current_app.iloc[0].application_id
application = dh.query_data(
    dh.TABLE_APPLICATIONS, condition=f"id='{current_app_id}'"
).iloc[0]
st.session_state["current_application"] = application

application_id = application.id
job_title = application.job_title
company = application.company
posting_url = application.posting_url
posting_text = application.posting_text
application_date = application.application_date
status = Status(application.status)
resume = application.resume

with st.container(key="application-header"):
    page_header(
        title=f"{"New " if status == "new" else ""}Application Details",
        cta_button_label="Back home",
        cta_button_help="Go back to the home page",
        btn_callback=back_home,
    )

    with st.container(border=True):
        columns = st.columns([2, 4, 2, 4])
        with columns[0]:
            st.markdown("**Job Title**")
            st.markdown("**Company**")
            st.markdown("**Job Post Url**")
        with columns[1]:
            st.markdown(f":briefcase: **{job_title}**")
            st.markdown(f":office: **{company}**")
            st.markdown(f":link: [{posting_url}]({posting_url})")
        with columns[2]:
            st.markdown("**Application Date**")
            st.markdown(f":traffic_light: **{status}**")
        with columns[3]:
            st.markdown(f":calendar: {application_date}")


with st.container(key="application-content", border=True):
    st.markdown(
        """
        <style>
            .st-key-btn-retry-analysis > div > div,
            .st-key-btn-generate-cover-letter > div > div,
            .st-key-btn-save-updates > div > div {
                display: flex;
                justify-content: flex-end;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["error_occurred"] = True
    with st.container():
        cols = st.columns([1, 1])
        with cols[0]:
            st.subheader("Application Analysis")
        with cols[1].container(key="btn-retry-analysis"):
            if st.button(
                "Reanalyze",
                key="retry_analysis",
                icon=":material/refresh:",
                # disabled=st.session_state["error_occurred"],
            ):
                st.session_state["error_occurred"] = False

    # Initialize agents
    llm = LLMFactory.create_llm(Provider.ANTHROPIC)
    # llm = LLMFactory.create_llm(Provider.OLLAMA)
    inputter = InputAgent(llm)
    parser = ParserAgent(llm)
    matcher = MatcherAgent(llm)
    generator = DocumentGeneratorAgent(llm)

    cv_text = resume
    job_text = posting_text
    parsed_data = None
    match_results = None

    with st.spinner("Please be patient while the LLM processes the data..."):
        if status == Status.DRAFT:
            if not (posting_url or posting_text) or not resume:
                handle_error("CV/ Resume or Job Posting not found, please upload them")

            cv_text = asyncio.run(
                inputter.run(application_id, resume, AgentAction.RESUME_CLEAN)
            )
            job_text = asyncio.run(
                inputter.run(
                    application_id,
                    posting_text or posting_url,
                    AgentAction.JOB_POSTING_CLEAN,
                )
            )
            st.rerun()

        if status in [Status.DRAFT, Status.NEW]:
            for text, action in [
                (cv_text, AgentAction.RESUME_CLEAN),
                (job_text, AgentAction.JOB_POSTING_CLEAN),
            ]:
                if not text:
                    text_in_db = fetch_analysis_step_data(application_id, action)
                    if not text_in_db.empty:
                        text = (
                            json.loads(text_in_db.data.values[0])
                            .get("data")
                            .get("content")
                        )
                        if action == AgentAction.RESUME_CLEAN:
                            cv_text = text
                        else:
                            job_text = text
                    else:
                        handle_error(f"Could not find input data in the database")

            parsed_data = asyncio.run(parser.run(application_id, cv_text, job_text))
            st.rerun()

        if status in [Status.DRAFT, Status.NEW, Status.PARSED]:
            if not parsed_data:
                parsed_data_in_db = fetch_analysis_step_data(
                    application_id, AgentAction.APPLICATION_PARSE
                )
                if not parsed_data_in_db.empty:
                    parsed_data = json.loads(parsed_data_in_db.data.values[0]).get(
                        "data"
                    )
                else:
                    handle_error("Could not find parsed data in the database")

            match_results = asyncio.run(matcher.run(application_id, parsed_data))
            st.rerun()

        if status in [
            Status.DRAFT,
            Status.NEW,
            Status.PARSED,
            Status.MATCHED,
            Status.DOCUMENTS_GENERATED,
        ]:
            if not match_results:
                match_results_in_db = fetch_analysis_step_data(
                    application_id, AgentAction.MATCH_ANALYZE
                )
                if not match_results_in_db.empty:
                    match_results = (
                        json.loads(match_results_in_db.data.values[0])
                        .get("data")
                        .get("analysis")
                    )

                else:
                    handle_error("Could not find match results in the database")

    tabs = st.tabs(
        ["🚀 Overall Analysis", "📎 Input Documents", "🚧 Generated Documents"]
    )
    with tabs[0]:
        st.header("Overall Analysis")
        st.write(
            "This section provides a high-level overview of the match analysis results."
        )
        display_match_analysis(match_results)

    with tabs[1]:
        st.header("Input Documents")
        st.write("This section displays the input documents used for the analysis.")
        updated_cv_text = st.text_area("Updated Resume:", cv_text, height=300)
        updated_job_text = st.text_area("Updated Job Posting:", job_text, height=300)

        with st.container(key="btn-save-updates"):
            if st.button(
                "Save Updates",
                key="save_updates",
                icon="💾",
                # Enable button only when text has changed
                disabled=(updated_cv_text == cv_text and updated_job_text == job_text),
            ):
                try:
                    # Update the database with new text
                    # dh.update_application_analysis(application_id, updated_cv_text, updated_job_text)

                    # Update the session state
                    st.session_state["error_occurred"] = False

                    # Show success message
                    st.success("Updates saved successfully!")

                    # Force a rerun to refresh the page with new data
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving updates: {str(e)}")

    with tabs[2]:
        st.header("Professional Documents Viewer")
        st.write(
            "This section displays the documents generated during the analysis process."
        )

        documents = None
        if match_results is not None:
            col1, col2 = st.columns([1, 1])
            with col2.container(key="btn-generate-cover-letter"):
                if st.button(
                    (
                        "Regenerate Documents"
                        if status == Status.DOCUMENTS_GENERATED
                        else "Generate updated Resume and Create Cover Letter"
                    ),
                    key="update_resume",
                    icon="📝",
                ):

                    with st.spinner("Generating documents..."):
                        try:
                            documents = asyncio.run(
                                generator.run(
                                    application_id,
                                    updated_cv_text,
                                    job_text,
                                    match_results,
                                )
                            )

                        except Exception as e:
                            st.error(f"Error generating documents: {str(e)}")

            if documents is None and status == Status.DOCUMENTS_GENERATED:
                documents_in_db = fetch_analysis_step_data(
                    application_id, AgentAction.DOCUMENT_GENERATE
                )
                if not documents_in_db.empty:
                    documents = json.loads(documents_in_db.data.values[0]).get("data")
                else:
                    handle_error("Could not find generated documents in the database")

            def parse_resume_sections(resume_text):
                """Parse resume text into sections based on newline patterns."""
                # Split into main sections (detected by double newlines)
                sections = resume_text.strip().split("\n\n")
                parsed_sections = {}

                current_section = None
                current_content = []

                for section in sections:
                    # Check if this is a section header (no bullet points, shorter than 50 chars)
                    lines = section.strip().split("\n")
                    if len(lines[0]) < 70 and not lines[0].startswith("●"):
                        if current_section:
                            parsed_sections[current_section] = "\n".join(
                                current_content
                            )
                        current_section = lines[0]
                        current_content = lines[1:] if len(lines) > 1 else []
                    else:
                        current_content.extend(lines)

                # Add the last section
                if current_section:
                    parsed_sections[current_section] = "\n".join(current_content)

                return parsed_sections

            def format_contact_info(contact_text):
                """Format contact information into markdown."""
                lines = contact_text.split("\n")
                contacts = [f"📞 {lines[0]}", f"📍 {lines[1]}", f"📧 {lines[2]}"]
                return " | ".join(contacts)

            def format_section_content(content):
                """Format section content into markdown, handling bullet points."""
                if not content:
                    return ""

                # Split content into lines and process each line
                lines = content.split("\n")
                formatted_lines = []

                for line in lines:
                    # Convert bullet points
                    if line.startswith("●"):
                        formatted_lines.append(f"* {line[1:].strip()}")
                    else:
                        formatted_lines.append(line.strip())

                return "\n".join(formatted_lines)

            def create_resume_markdown(sections):
                """Create complete markdown formatted resume."""
                markdown_sections = []

                # Process each section
                for section_title, content in sections.items():
                    markdown_sections.append(f"#### {section_title}")
                    if section_title == list(sections.keys())[0]:
                        markdown_sections.append(format_contact_info(content))
                    else:
                        formatted_content = format_section_content(content)
                        markdown_sections.append(formatted_content)
                    markdown_sections.append("")  # Add spacing between sections

                return "\n".join(markdown_sections)

            def display_cover_letter(cover_letter_text):
                # Parse the cover letter sections
                sections = cover_letter_text.split("\n\n")

                # Display header with date and address
                st.markdown(f"**Date:** {sections[0]}")
                st.markdown(f"**To:** {sections[1]}")

                # Display salutation and main content
                for section in sections[2:]:
                    if section.startswith("Dear"):
                        st.markdown(f"**{section}**")
                    else:
                        st.write(section)

            tab1, tab2 = st.tabs(["Resume", "Cover Letter"])

            with tab1:
                resume_sections = parse_resume_sections(documents["updated_resume"])
                resume_markdown = create_resume_markdown(resume_sections)
                st.markdown(resume_markdown)
                st.download_button(
                    label="Download Resume",
                    data=documents["updated_resume"],
                    file_name="document.pdf",
                    mime="application/pdf",
                )

            with tab2:
                display_cover_letter(documents["cover_letter"])
                st.download_button(
                    label="Download Cover Letter",
                    data=documents["cover_letter"],
                    file_name="cover_letter.txt",
                    mime="text/plain",
                )

        else:
            st.warning("Please complete the analysis to generate documents.")


# # TODO: for debugging purposes only, remove this later
# st.json(st.session_state)
