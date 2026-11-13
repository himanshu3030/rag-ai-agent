
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="centered",
)

# --------------------------------------------------
# Inngest configuration
# --------------------------------------------------

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(
        app_id="rag_app",
        is_production=False,
    )


# --------------------------------------------------
# PDF upload and ingestion
# --------------------------------------------------

def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(file.name).name
    file_path = uploads_dir / filename

    file_path.write_bytes(file.getbuffer())

    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()

    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


# --------------------------------------------------
# RAG query
# --------------------------------------------------

async def send_rag_query_event(
    question: str,
    top_k: int,
) -> str:
    client = get_inngest_client()

    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    if not result:
        raise RuntimeError("Inngest did not return an event ID.")

    return result[0]


# --------------------------------------------------
# Inngest API helpers
# --------------------------------------------------

def get_inngest_api_base() -> str:
    return os.getenv(
        "INNGEST_API_BASE",
        "http://127.0.0.1:8288/v1",
    ).rstrip("/")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{get_inngest_api_base()}/events/{event_id}/runs"

    response = requests.get(url, timeout=15)
    response.raise_for_status()

    data = response.json()

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        runs = data.get("data", data.get("runs", []))

        if isinstance(runs, list):
            return runs

        if isinstance(runs, dict):
            return [runs]

    return []


# --------------------------------------------------
# Extract actual RAG result from Inngest response
# --------------------------------------------------

def extract_rag_output(value: Any) -> dict | None:
    """
    Recursively searches the Inngest response for
    the RAG result containing an answer field.
    """

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return extract_rag_output(parsed)
        except (json.JSONDecodeError, TypeError):
            return None

    if isinstance(value, list):
        for item in value:
            result = extract_rag_output(item)

            if result is not None:
                return result

        return None

    if not isinstance(value, dict):
        return None

    # Found the actual RAG response
    if "answer" in value:
        return value

    # Check likely output fields first
    priority_keys = [
        "output",
        "result",
        "data",
        "response",
        "function_output",
        "step_results",
        "steps",
    ]

    for key in priority_keys:
        if key in value:
            result = extract_rag_output(value[key])

            if result is not None:
                return result

    # Check remaining nested fields
    for key, item in value.items():
        if key not in priority_keys:
            result = extract_rag_output(item)

            if result is not None:
                return result

    return None


# --------------------------------------------------
# Wait for Inngest function completion
# --------------------------------------------------

def wait_for_run_output(
    event_id: str,
    timeout_s: float = 180,
    poll_interval_s: float = 1,
) -> dict:

    start_time = time.time()
    last_status = None
    last_run = None

    while time.time() - start_time < timeout_s:

        runs = fetch_runs(event_id)

        if runs:
            run = runs[0]
            last_run = run

            status = str(run.get("status", "")).lower()
            last_status = status or last_status

            if status in (
                "completed",
                "succeeded",
                "success",
                "finished",
            ):

                output = extract_rag_output(run)

                if output is not None:
                    return output

                # Preserve the raw response for troubleshooting
                raise RuntimeError(
                    "Inngest completed the function, but its API "
                    "response did not contain an answer field.\n\n"
                    f"Run response:\n{json.dumps(run, indent=2, default=str)}"
                )

            if status in ("failed", "cancelled"):
                raise RuntimeError(
                    f"Inngest function {status}.\n"
                    f"{json.dumps(run, indent=2, default=str)}"
                )

        time.sleep(poll_interval_s)

    raise TimeoutError(
        f"Timed out waiting for Inngest.\n"
        f"Last status: {last_status}\n"
        f"Last run: {last_run}"
    )


# --------------------------------------------------
# Streamlit UI - PDF ingestion
# --------------------------------------------------

st.title("📄 RAG PDF Assistant")

st.subheader("Upload and Ingest PDF")

with st.form("pdf_upload_form"):

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
    )

    ingest_submitted = st.form_submit_button(
        "Upload and Ingest",
        use_container_width=True,
    )


if ingest_submitted:

    if uploaded_file is None:
        st.warning("Please select a PDF file first.")

    else:
        try:
            with st.spinner("Uploading and triggering ingestion..."):

                pdf_path = save_uploaded_pdf(uploaded_file)

                asyncio.run(
                    send_rag_ingest_event(pdf_path)
                )

            st.success(
                f"Ingestion event triggered: {pdf_path.name}"
            )

            st.info(
                "Check Inngest to confirm that PDF ingestion "
                "has completed successfully."
            )

        except Exception as error:
            st.error(f"PDF ingestion failed: {error}")


# --------------------------------------------------
# Streamlit UI - Ask questions
# --------------------------------------------------

st.divider()

st.subheader("💬 Ask Questions About Your PDFs")

with st.form("rag_query_form"):

    question = st.text_input(
        "Your question",
        placeholder="What are my technical skills?",
    )

    top_k = st.number_input(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    query_submitted = st.form_submit_button(
        "Ask",
        use_container_width=True,
    )


if query_submitted:

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner(
                "Searching documents and generating answer..."
            ):

                # Send query event to Inngest
                event_id = asyncio.run(
                    send_rag_query_event(
                        question.strip(),
                        int(top_k),
                    )
                )

                # Wait for completed function output
                output = wait_for_run_output(event_id)

            # --------------------------------------
            # Display answer
            # --------------------------------------

            st.subheader("Answer")

            answer = output.get("answer")

            if answer:
                st.markdown(answer)
            else:
                st.warning("No answer was returned.")

            # --------------------------------------
            # Display sources
            # --------------------------------------

            sources = output.get("sources", [])

            if sources:
                st.subheader("Sources")

                for source in sources:
                    st.write(f"📄 {source}")

            # --------------------------------------
            # Retrieval information
            # --------------------------------------

            num_contexts = output.get("num_contexts")

            if num_contexts is not None:
                st.caption(
                    f"Retrieved {num_contexts} context chunks."
                )

        except requests.RequestException as error:
            st.error(
                f"Could not communicate with Inngest API: {error}"
            )

        except TimeoutError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"Query failed: {error}")
