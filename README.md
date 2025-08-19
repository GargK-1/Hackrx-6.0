# Intelligent Policy Q&A System: An Advanced RAG Pipeline

This project implements a sophisticated **Retrieval-Augmented Generation (RAG)** pipeline designed to answer complex, multi-part questions about PDF documents, specifically tailored for dense content like insurance policies. It exposes its functionality through a robust FastAPI web server, optimized for performance.

The system intelligently parses user queries, retrieves the most relevant information from a document using a keyword-boosted vector search, and generates accurate, context-aware answers.

## ✨ Key Features

* **Advanced Query Parsing**: Uses a Large Language Model (LLM) to deconstruct complex user questions into distinct sub-queries and identify key phrases.
* **Weighted Vector Retrieval**: Employs a unique weighted search technique that boosts the relevance of key phrases, leading to more accurate context retrieval from the document.
* **Persistent Vector Caching**: Automatically creates and caches a [FAISS](https://github.com/facebookresearch/faiss) vector index for each PDF, ensuring that subsequent requests for the same document are processed instantly without re-indexing.
* **Robust Answer Synthesis**: The final generation step is carefully prompted to synthesize answers *strictly* from the retrieved clauses, handle questions with partial or no available information gracefully, and refuse to answer inappropriate queries.
* **High-Performance API**: Built with **FastAPI**, featuring a model caching mechanism (`lifespan` manager) that loads heavy models (like embeddings and LLMs) only once at startup, drastically reducing response times.
* **Asynchronous Processing**: Leverages Python's `asyncio` to handle multiple questions in a single API call concurrently, improving throughput.

## ⚙️ How It Works (Pipeline Architecture)

The entire process is orchestrated to deliver precise answers by grounding the LLM in the specific content of the provided document.

1.  **API Request**: The `server_test.py` receives a POST request containing a PDF URL and a list of questions.
2.  **PDF Processing & Caching**:
    * The system first checks if a FAISS index for this URL already exists locally.
    * If not, `Reading_PDFBlobURLsIMPROVED.py` downloads the PDF, converts it to structured Markdown, and splits it into manageable chunks.
    * `embedding_search.py` then uses a Hugging Face model (`all-MiniLM-L6-v2`) to create vector embeddings for these chunks and saves them in a local FAISS index for future use.
3.  **Query Parsing**: `llm_parser.py` takes each user question and sends it to an LLM (e.g., GPT-3.5 Turbo) to be broken down into a structured object containing keywords and sub-queries.
4.  **Weighted Retrieval**: For each sub-query, the system performs a vector search on the FAISS index. The identified `key_word` from the parsing step is used to mathematically boost the relevance of chunks containing that term.
5.  **Answer Generation**: The top-ranked document chunks are passed as context to the `logic_evaluation.py` module. It uses a final, robustly prompted LLM call to synthesize a coherent and accurate answer based *only* on the provided information.
6.  **API Response**: The FastAPI server returns a JSON object containing the list of generated answers.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.9+
* An OpenAI API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a file named `.env` in the root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

### Usage

You can run the project in two ways: as a standalone script or as a web server.

#### 1. Running the Local Pipeline Script

To test the full pipeline on a predefined set of questions, you can directly execute the main script. This is useful for debugging and development.

```bash
python main_pipeline.py
```
This will process the default PDF URL and questions defined in the `if __name__ == "__main__":` block of the script.

#### 2. Running the FastAPI Server

To serve the application as an API, use `uvicorn`.

```bash
uvicorn server_test:app --reload
```
The server will start, and you can access it at `http://127.0.0.1:8000`.

You can then send a POST request to the `/api/v1/hackrx/run` endpoint. Here is an example using `curl`:

```bash
curl -X POST "[http://127.0.0.1:8000/api/v1/hackrx/run](http://127.0.0.1:8000/api/v1/hackrx/run)" \
-H "Content-Type: application/json" \
-d '{
    "documents": "[https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A06%3A03Z&se=2026-08-01T17%3A06%3A00Z&sr=b&sp=r&sig=wLlooaThgRx91i2z4WaeggT0qnuUUEzIUKj42GsvMfg%3D](https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A06%3A03Z&se=2026-08-01T17%3A06%3A00Z&sr=b&sp=r&sig=wLlooaThgRx91i2z4WaeggT0qnuUUEzIUKj42GsvMfg%3D)",
    "questions": [
        "If an insured person takes treatment for arthritis at home because no hospital beds are available, under what circumstances would these expenses NOT be covered?",
        "What is the process to add a newly-adopted child as a dependent, and can the insurer refuse cover?"
    ]
}'
```

## 📂 Project Structure

```
.
├── server_test.py                 # FastAPI server entry point and API logic
├── main_pipeline.py               # Main orchestration script for local execution
├── Reading_PDFBlobURLsIMPROVED.py # Handles PDF download, parsing to Markdown, and chunking
├── embedding_search.py            # Manages FAISS vector store creation, loading, and weighted search
├── llm_parser.py                  # Logic for parsing user questions into structured queries
├── logic_evaluation.py            # Generates the final answer from retrieved context
├── requirements.txt               # Project dependencies
└── .env                           # Environment variables (not committed)
```
