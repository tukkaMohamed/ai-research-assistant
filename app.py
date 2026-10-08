import time
from pathlib import Path

import streamlit as st

from core import chains
from core.agent import INTENTS, run_agent
from core.ingestion import create_chunks, load_pdf
from core.llm import get_llm
from core.retriever import Retriever


PAPERS_DIR = Path("data/papers")
INDEX_PATH = Path("data/index.faiss")

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_retriever():
    retriever = Retriever()

    if INDEX_PATH.exists():
        retriever.load()

    return retriever


@st.cache_resource
def load_llm():
    return get_llm()


retriever = get_retriever()
llm = load_llm()

if "history" not in st.session_state:
    st.session_state.history = []


def index_ready():
    return retriever.index is not None and len(retriever.chunks) > 0


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:

    st.header("Model")
    st.success("Gemini")

    st.header("Papers")

    uploaded = st.file_uploader(
        "Add papers (PDF)",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button(
        "Add to library",
        disabled=not uploaded,
        use_container_width=True,
    ):
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)

        known = set(retriever.sources())
        new = [f for f in uploaded if f.name not in known]

        if not new:
            st.info("These papers are already in the library.")

        else:
            with st.spinner("Embedding new papers..."):

                chunks = []

                for file in new:
                    path = PAPERS_DIR / file.name
                    path.write_bytes(file.getbuffer())

                    chunks.extend(
                        create_chunks(load_pdf(path))
                    )

                retriever.add_chunks(chunks)
                retriever.save()

            st.success(
                f"Added {len(new)} papers ({len(chunks)} chunks)"
            )

    if index_ready():

        with st.expander(
            f"{len(retriever.sources())} papers in the library"
        ):
            for name in retriever.sources():
                st.write(f"- {name}")

    else:
        st.info(
            "The library is empty. Add PDF papers to get started."
        )

    st.header("Settings")

    top_k = st.slider(
        "Excerpts to retrieve",
        3,
        10,
        5,
    )

    mode = st.selectbox(
        "Task",
        ["Auto (agent decides)"] + INTENTS,
    )

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.history = []
        st.rerun()


# ------------------------------------------------------------------
# Main application
# ------------------------------------------------------------------

st.title("📚 AI Research Assistant")

st.caption(
    "Upload research papers, ask questions, summarize papers, "
    "compare studies, and explore research gaps."
)

if not index_ready():
    st.warning("Add papers to the library to start.")
    st.stop()

papers = retriever.sources()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def run_task(label, fn):
    try:
        with st.spinner(label):

            start = time.time()
            result = fn()

        st.caption(
            f"Done in {time.time() - start:.1f}s with {llm.name}"
        )

        return result

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        return None


def render_sources(sources):

    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):

        for i, source in enumerate(sources, start=1):

            score = (
                f" (score {source['score']:.2f})"
                if "score" in source
                else ""
            )

            st.markdown(
                f"**[{i}] {source['source']}** "
                f"- page {source['page']}{score}"
            )

            st.caption(source["text"])


def render_result(out):

    if out.get("intent"):

        names = (
            ", ".join(out["papers"])
            if out["papers"]
            else "whole library"
        )

        st.caption(
            f"Agent chose **{out['intent']}** "
            f"({out['routed_by']}) on: {names}"
        )

    st.markdown(out["answer"])

    if out.get("table"):
        st.dataframe(
            out["table"],
            use_container_width=True,
        )

    render_sources(out.get("sources"))


# ------------------------------------------------------------------
# Navigation
# ------------------------------------------------------------------

page = st.radio(
    "View",
    ["Ask", "Paper tools", "Compare", "Insights"],
    horizontal=True,
    label_visibility="collapsed",
)


# ------------------------------------------------------------------
# Ask
# ------------------------------------------------------------------

if page == "Ask":

    st.caption(
        "Ask questions about the papers. "
        "The agent automatically selects the appropriate task."
    )

    for message in st.session_state.history:

        with st.chat_message(message["role"]):

            if message["role"] == "user":
                st.write(message["content"])

            else:
                render_result(message["out"])

    query = st.chat_input(
        "Ask a question about the papers"
    )

    if query:

        with st.chat_message("user"):
            st.write(query)

        st.session_state.history.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.chat_message("assistant"):

            force = (
                None
                if mode.startswith("Auto")
                else mode
            )

            out = run_task(
                "Thinking...",
                lambda: run_agent(
                    query,
                    retriever,
                    llm,
                    top_k=top_k,
                    force_intent=force,
                ),
            )

            if out:

                render_result(out)

                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "out": out,
                    }
                )


# ------------------------------------------------------------------
# Paper tools
# ------------------------------------------------------------------

elif page == "Paper tools":

    paper = st.selectbox(
        "Paper",
        papers,
    )

    col1, col2 = st.columns(2)

    if col1.button(
        "Summarize",
        use_container_width=True,
    ):

        result = run_task(
            "Summarizing...",
            lambda: chains.summarize_paper(
                paper,
                retriever,
                llm,
            ),
        )

        if result:

            st.markdown(result["summary"])

            render_sources(
                result["sources"]
            )

    refresh = col2.checkbox(
        "Re-extract (ignore cache)"
    )

    if col2.button(
        "Extract structured info",
        use_container_width=True,
    ):

        info = run_task(
            "Extracting...",
            lambda: chains.extract_paper(
                paper,
                retriever,
                llm,
                refresh=refresh,
            ),
        )

        if info:

            st.markdown(
                chains.format_info(info)
            )

            st.download_button(
                "Download JSON",
                info.model_dump_json(indent=2),
                file_name=f"{Path(paper).stem}.json",
            )


# ------------------------------------------------------------------
# Compare
# ------------------------------------------------------------------

elif page == "Compare":

    chosen = st.multiselect(
        "Papers to compare (2-5)",
        papers,
        max_selections=5,
    )

    aspect = st.text_input(
        "Focus (optional)",
        placeholder="e.g. memory efficiency, training data",
    )

    if st.button(
        "Compare",
        disabled=len(chosen) < 2,
    ):

        result = run_task(
            "Extracting and comparing...",
            lambda: chains.compare_papers(
                chosen,
                retriever,
                llm,
                aspect or None,
            ),
        )

        if result:

            st.subheader("Comparison table")

            st.dataframe(
                result["table"],
                use_container_width=True,
            )

            st.subheader("Analysis")

            st.markdown(
                result["analysis"]
            )


# ------------------------------------------------------------------
# Insights
# ------------------------------------------------------------------

else:

    chosen = st.multiselect(
        "Papers (1-6)",
        papers,
        max_selections=6,
    )

    col1, col2 = st.columns(2)

    for label, function, column in [
        (
            "Limitations and research gaps",
            chains.research_gaps,
            col1,
        ),
        (
            "Literature review",
            chains.literature_review,
            col2,
        ),
    ]:

        if column.button(
            label,
            disabled=not chosen,
            use_container_width=True,
        ):

            result = run_task(
                "Analyzing...",
                lambda function=function: function(
                    chosen,
                    retriever,
                    llm,
                ),
            )

            if result:

                st.subheader(label)

                st.markdown(
                    result["analysis"]
                )

                with st.expander(
                    "Extracted data"
                ):
                    st.dataframe(
                        result["table"],
                        use_container_width=True,
                    )