import streamlit as st


def page_header(title, cta_button_label=None, cta_button_help=None, btn_callback=None):
    if "deletion_confirmed" not in st.session_state:
        st.session_state["deletion_confirmed"] = False

    (
        title_col,
        cta_col,
    ) = st.columns([2, 1], vertical_alignment="bottom")

    # Page Header
    with title_col:
        st.title(f"_{title}_")

    with cta_col:
        btn_col, selector_col = st.columns([7, 8], vertical_alignment="bottom")
        st.markdown(
            """<style> .st-key-cta_button button {color: #0e1117;}</style>""",
            unsafe_allow_html=True,
        )
        with btn_col:
            if st.button(
                label=cta_button_label,
                key="cta_button",
                help=cta_button_help,
                type="primary",
                use_container_width=True,
            ):
                btn_callback()

        with selector_col:
            st.selectbox(
                label="LLM Providers",
                key="llm_in_use",
                options=(
                    "OLLAMA llama3.2:latest",
                    "OLLAMA llama3.1:latest",
                    "OLLAMA mistral-nemo:12b",
                    "ANTHROPIC claude-3-opus-latest",
                    "ANTHROPIC claude-3-5-sonnet-latest",
                    "ANTHROPIC claude-3-5-haiku-latest",
                ),
                placeholder="Select the LLM to use...",
                index=1,
            )
