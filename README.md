# Document Loading, Chunking, and Summarization Pipeline

This repository contains a Jupyter Notebook (`chunking_loading_etc.ipynb`) demonstrating a complete pipeline for loading local PDF documents, summarizing text using pretrained Hugging Face transformers, chunking text using LangChain text splitters with overlap, and analyzing the resulting chunk distribution.

## Workspace Structure

- `data/`: Folder containing source documents (e.g., `mamba 3.pdf`).
- `chunking_loading_etc.ipynb`: The main notebook containing the implementation of the pipeline.
- `requirements.txt`: List of required Python packages.
- `README.md`: This document, describing the project and its setup.

---

## Features

### 1. Document Loading
Loads PDF and plain text documents from the `data/` folder into standard LangChain `Document` objects using:
- `PyMuPDFLoader`: For fast processing and rich metadata extraction (title, author, format, total pages, etc.) from PDF headers.
- `DirectoryLoader`: Automated loading of multiple files matching glob patterns (`*.pdf`, `*.txt`).

### 2. Document Summarization
Uses the Hugging Face `transformers` library to run abstractive summarization:
- **Model**: `sshleifer/distilbart-cnn-12-6`
- **Pipeline**: Configured to run on GPU (`device=0`) when CUDA is available.
- Generates abstract summaries of loaded PDF sections with custom length parameters (`max_length=150`, `min_length=40`).

### 3. Text Chunking with Overlap
Prepares large documents for downstream tasks (like Retrieval-Augmented Generation or Vector Database ingestion) by splitting text:
- **Splitter**: `CharacterTextSplitter` (from `langchain_text_splitters`)
- **Parameters**: `chunk_size=256`, `chunk_overlap=50`, `separator="\n"`
- Preserves context across chunk boundaries via overlap.

### 4. Chunk Distribution Analysis
Analyzes chunk count and text lengths using data science tools:
- **Pandas**: Structures chunk metadata and calculates chunk lengths.
- **Matplotlib**: Visualizes the distribution of text chunks per article.

---

## Setup and Installation

### Prerequisites
- Python 3.10+ is recommended.
- A virtual environment (`venv/`) is set up in the workspace.

### Dependencies
Install the required packages using the `requirements.txt` file and additional parsing/data packages:

```bash
pip install -r requirements.txt
pip install pymupdf pandas matplotlib ipykernel
```

---

## Usage

1. **Activate the Virtual Environment**:
   - On Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

2. **Run the Notebook**:
   Open the notebook in your preferred IDE or command line using Jupyter:
   ```bash
   jupyter notebook chunking_loading_etc.ipynb
   ```
   Execute the cells sequentially to observe:
   - Page count and metadata output.
   - Text summarization execution times and results.
   - The generation of overlapping text chunks.
   - Matplotlib visualization of the chunk counts.
