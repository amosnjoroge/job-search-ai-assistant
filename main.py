import base64
import random
import tempfile
import streamlit as st
import pandas as pd
from page_header import page_header
from data_handler import DataHandler
from utils.readers import read_pdf
from utils.validation import validate_job_application

st.set_page_config(
    page_title="Home - Awesome Job Search Assistant",
    page_icon=":mag_right:",
    layout="wide",
)

HR_LINE = '<hr style="border: 1px solid #3c3c3c;">'


if "current_application" not in st.session_state:
    st.session_state["current_application"] = None


def initial_applications_fetch():
    dh = DataHandler()
    st.session_state["applications"] = dh.query_data(dh.TABLE_APPLICATIONS)


@st.fragment(run_every="5min")
def set_random_quote():
    quotes = [
        "Every 'no' brings you one step closer to your 'yes'. Keep going!",
        "Your dream job is looking for you too, it just needs help with directions.",
        "You haven't failed until you stop trying. And coffee. Coffee helps too.",
        "Success is like a great job - it's 1% inspiration, 99% perseverance, and 100% worth it!",
        "The only difference between you and someone who has their dream job is time and persistence.",
        "Your skills are like fine wine - they get better with time. Keep learning, keep growing!",
        "Remember: Even Thomas Edison had to submit his resume somewhere.",
        "Today's rejections are tomorrow's 'their loss' stories. Keep shining!",
        "You're not job hunting, you're opportunity shopping. And you've got great taste!",
        "Every expert was once a beginner. Your time is coming, just keep moving forward.",
    ]
    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;💫 _{random.choice(quotes)}_")
    st.markdown("<br>", unsafe_allow_html=True)


@st.fragment
def set_active_application(id):
    dh = DataHandler()
    current_app = dh.query_data(dh.TABLE_CURRENT_APPLICATION)
    if current_app.empty:
        dh.insert_data(dh.TABLE_CURRENT_APPLICATION, (1, id), ("id", "application_id"))
    else:
        dh.update_data(dh.TABLE_CURRENT_APPLICATION, {"application_id": id}, "id = 1")

    applications = dh.query_data(dh.TABLE_APPLICATIONS)

    st.session_state["applications"] = applications
    st.session_state["current_application"] = applications[applications["id"] == id]


@st.fragment
def process_application(resume, posting_text, posting_url):
    data = (
        pd.Timestamp.now().isoformat(),
        pd.Timestamp.now().isoformat(),
        resume,
        posting_url,
        posting_text,
        "draft",
    )
    columns = (
        "application_date",
        "update_date",
        "resume",
        "posting_url",
        "posting_text",
        "status",
    )
    dh = DataHandler()
    application_id = dh.insert_data(dh.TABLE_APPLICATIONS, data, columns)
    set_active_application(application_id)
    st.switch_page("pages/view-job-application.py")


@st.dialog("Delete Confirmation!", width="medium")
def delete_application(application_id):
    with st.container(border=True):
        st.write(f":red[Are you sure you want to proceed with deleting application: {application_id}?]")

        no_col, mid_col, yes_col = st.columns([1, 2, 1])
        with no_col:
            if st.button("No"):
                st.rerun()
        with yes_col:
            if st.button("Yes"):
                st.session_state["deletion_confirmed"] = True
                dh = DataHandler()
                dh.delete_data(dh.TABLE_APPLICATIONS, f"id = '{application_id}'")
                st.rerun()


@st.dialog("New Job Application", width="large")
def start_application():
    with st.container(border=True):
        resume = None
        st.file_uploader(label="Choose your resume/CV", type=("pdf"), key="cv_file")

        if st.session_state.get("cv_file"):
            col1, col2 = st.columns([0.9, 0.1])
            resume = read_pdf(st.session_state.get("cv_file"))
            with col1:
                with st.expander("Preview Your CV", expanded=False, icon="📄"):
                    file = st.session_state.cv_file
                    if file.type == "application/pdf":
                        base64_pdf = base64.b64encode(file.getvalue()).decode("utf-8")
                        pdf_display = f'<iframe src="data:application/pdf;base64,{
                            base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    else:
                        st.write("📄 File uploaded:", file.name)
                        st.write("Size:", f"{file.size/1024:.1f} KB")
            with col2:
                if st.button("", help="Upload new CV", icon=":material/close:"):
                    del st.session_state.cv_file

        posting_source = st.toggle(
            label="Use a job posting text.",
            value=False,
            help="Activate to paste job posting text directly from source.",
        )

        posting_text = None
        posting_url = None

        if posting_source:
            posting_text = st.text_area(
                label="Job Posting",
                height=300,
                placeholder="Copy and paste the job posting here...",
                key="job_posting_text",
            )
        else:
            posting_url = st.text_input(
                label="Job Posting Link",
                placeholder="Paste the link to the job posting here...",
                key="job_posting_link",
            )

        if st.button("Submit"):
            is_valid, error_message, validated_data = validate_job_application(
                resume=resume,
                posting_text=posting_text,
                posting_url=posting_url,
            )

            if not is_valid:
                st.error(error_message)
                return

            st.session_state["validated_application_data"] = validated_data
            process_application(resume, posting_text, posting_url)


def table_row(row_df, vertical_alignment="center", header=False):
    title_col, company_col, date_col, status_col, cta_col = st.columns(
        [2, 2, 3, 2, 2], vertical_alignment=vertical_alignment
    )
    with title_col:
        if row_df["job_title"] is None:
            row_df["job_title"] = "Untitled Position"
        if not header and row_df["posting_url"] is not None:
            st.markdown(f"[{row_df['job_title']}]({row_df['posting_url']})")
        else:
            st.write(row_df["job_title"])
    with company_col:
        st.write(row_df["company"])
    with date_col:
        st.write(f"{row_df['application_date']}")
    with status_col:
        st.write(row_df["status"])
    with cta_col:
        if not header:
            view_col, delete_col = st.columns(2, vertical_alignment="center")
            with view_col:
                if st.button(
                    label="View",
                    icon=":material/visibility:",
                    use_container_width=True,
                    key=f"view_{row_df['id']}",
                ):
                    set_active_application(row_df["id"])
                    st.switch_page("pages/view-job-application.py")

            with delete_col:
                st.button(
                    label="",
                    type="tertiary",
                    icon=":material/delete:",
                    use_container_width=True,
                    key=f"delete_{row_df['id']}",
                    on_click=delete_application,
                    kwargs={"application_id": row_df["id"]},
                )

    st.html(HR_LINE)


# Page Content
with st.container(key="home-header"):
    page_header(
        title="Awesome Job Search Assistant",
        cta_button_label="Start New Application",
        cta_button_help="Starts a new job application",
        btn_callback=start_application,
    )
    set_random_quote()


with st.container(key="home-content"):
    # evaluating_tab, submitted_tab = st.tabs(
    #     ["📋 Evaluating Applications", "📤 Submitted Applications"]
    # )
    st.subheader("Applications")
    st.markdown("<p/>", unsafe_allow_html=True)

    initial_applications_fetch()
    st.markdown(
        f"""
            <style>
            [class*='st-key-delete'] button:hover {{
                color: #a61414;
            }}
            </style>
            """,
        unsafe_allow_html=True,
    )

    if st.session_state["deletion_confirmed"]:
        st.toast("Application data deleted successfully", icon="✅")
        st.session_state["deletion_confirmed"] = False

    if st.session_state.applications.empty:
        st.markdown(
            """
                        <p 
                            style='
                                text-align: center; 
                                background-color: #192c40;
                                border-radius: 10px;
                                padding: 20px;'
                        >
                        No applications under evaluation.
                        </p>""",
            unsafe_allow_html=True,
        )

    else:
        table_row(
            {
                "job_title": "##### _Job title_",
                "company": "##### _Company_",
                "application_date": "##### _Application Date_",
                "posting_url": "##### _Job Link_",
                "status": "##### _Status_",
            },
            vertical_alignment="bottom",
            header=True,
        )
        for i in reversed(range(len(st.session_state["applications"]))):
            table_row(st.session_state["applications"].iloc[i])

# with submitted_tab:
#     st.markdown("Applications Submitted")


# TODO: for debugging purposes only, remove this later
# st.json(st.session_state)
