
import os
# from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Grok API client (LLM)
# client = OpenAI(
#     api_key=os.getenv("XAI_API_KEY"),
#     base_url="https://api.x.ai/v1"
# )

# LLM_MODEL = "grok-4.7"

# Local HuggingFace embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

embedding_model = SentenceTransformer(EMBED_MODEL)

splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=50
)


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)

    texts = [
        d.text for d in docs
        if getattr(d, "text", None)
    ]

    chunks = []

    for t in texts:
        chunks.extend(splitter.split_text(t))

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings.tolist()


# def generate_answer(question: str, context: str) -> str:
#     response = client.responses.create(
#         model=LLM_MODEL,
#         input=f"""
#         Answer the question using the provided context.

#         Context:
#         {context}

#         Question:
#         {question}

#         If the answer is not available in the context,
#         say you don't know.
#         """
#     )

#     return response.output_text
