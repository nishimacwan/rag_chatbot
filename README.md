# Document Chatbot

A Python-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on a provided PDF document. The app is powered by a local Large Language Model (LLM), ensuring complete privacy.

---

## What It Does 🚀

This project solves the problem of LLM **hallucination** and privacy. Instead of relying on a model's general knowledge, it intelligently retrieves information from a specific document and uses that context to provide accurate, grounded answers. It's a great demonstration of a self-hosted, private AI application.

---

## Technologies Used

* **Python**: The primary programming language.
* **Streamlit**: For creating the user-friendly web interface.
* **LangChain**: To orchestrate the RAG pipeline (document loading, splitting, vectorization, and the retrieval chain).
* **LlamaCpp**: The library used to run the `TinyLlama` LLM locally on the CPU.
* **HuggingFace**: For the sentence embeddings model.

---

## How to Run the App Locally

### Prerequisites

1.  Make sure you have **Python 3.8 or newer** installed.
2.  **Clone this repository** to your local machine: `git clone [Your_GitHub_Repository_URL]`
3.  **Navigate to the project folder**: `cd rag_chatbot`

### Setup

1.  **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
2.  **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

### Model Download

The LLM is too large to be included in the repository. You must download it yourself.

1.  Download the **`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`** model from [HuggingFace](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf).
2.  Place the downloaded file inside the `rag_chatbot` folder.

### Run the App

```bash
python -m streamlit run app.py
