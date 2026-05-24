# YouTube Video QA with LangChain + Gemini + FAISS

This project lets you ask questions about a YouTube video by:

1. Downloading the video transcript
2. Splitting it into chunks
3. Embedding the chunks using Google Vertex AI embeddings
4. Storing them in a FAISS vector database
5. Querying the most relevant chunks
6. Generating answers using Gemini (Google Generative AI)

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/youtube-qa-bot.git
```

### 2. Create virtual enviroment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Enviroment variables

Create a .env file in the root directory:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

Authenticate to your google account 

```bash
gcloud auth application-default login
```

You will need to set your project id in the code.

project="your-project-id"

to find the project id:

```bash
gcloud projects list
```

### 5. Run

Run the Youtube_bot.ipynb or Youtube_bot.py file.

Feel free to customize and change the video url and prompt

## How It Works

Transcript Loading

- Extracts YouTube captions using YoutubeLoader

Chunking

- Splits transcript into ~2000 character chunks with overlap

Embeddings

- Uses text-embedding-005 via Vertex AI

Vector Store

- Stores embeddings in FAISS for fast similarity search
- Data is stored in memory (RAM)

LLM Answering

- Gemini answers using only retrieved context