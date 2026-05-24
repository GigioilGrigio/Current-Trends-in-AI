from dotenv import load_dotenv
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import VertexAIEmbeddings
import textwrap

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
# Gemini embeddings

## gcloud auth application-default login
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project="68582664605",  # "your project id"
    location="us-central1",
)


def create_db_from_youtube_video_url(video_url: str):

    loader = YoutubeLoader.from_youtube_url(
        video_url,
        add_video_info=False,
    )

    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
    )

    docs = text_splitter.split_documents(transcript)

    # vector database
    db = FAISS.from_documents(docs, embeddings)

    return db


def get_response_from_query(db, query: str, k: int = 4):

    docs = db.similarity_search(query, k=k)

    docs_page_content = "\n\n".join([doc.page_content for doc in docs])

    # Gemini chat model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant that answers questions about a YouTube video
using only the provided transcript context.

Context:
{docs}

Question:
{question}

Rules:
- Only answer from the transcript
- If the transcript does not contain the answer, say "I don't know"
"""
    )

    chain = prompt | model | StrOutputParser()

    response = chain.invoke(
        {
            "docs": docs_page_content,
            "question": query,
        }
    )

    return response, docs


# Example usage
video_url = "https://www.youtube.com/watch?v=2P27Ef-LLuQ"

db = create_db_from_youtube_video_url(video_url)

query = "what are they saying about Impact on Jobs and Agents?"

response, docs = get_response_from_query(db, query)

print(textwrap.fill(response, width=80))
