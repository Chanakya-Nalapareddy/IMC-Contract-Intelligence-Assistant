# IMC Contract Intelligence Assistant

Automating Construction Owner Contract Abstraction Using Retrieval-Augmented Generation

Drexel University Capstone Project – 2026  
Authors: **Chanakya Nalapareddy** & **Kriti Sarawgi**  
External Stakeholder: **IMC Construction**

---

# Table of Contents

1. Overview  
2. Problem Statement  
3. Solution Approach  
4. Key Features  
5. System Architecture  
6. End-to-End Pipeline  
7. Repository Structure  
8. Technology Stack  
9. Environment Setup  
10. Installation Guide  
11. Running the Pipeline  
12. Running the Web Interface  
13. Output Artifacts  
14. Dataset Description  
15. Evaluation Results  
16. Limitations  
17. Future Work  
18. Acknowledgements  

---

# Overview

Construction owner contracts are highly complex legal documents that can exceed **hundreds or even thousands of pages**. These contracts contain critical information about:

- Insurance requirements  
- Payment terms  
- Risk allocation  
- Legal obligations  
- Scheduling requirements  
- Dispute resolution mechanisms  

Construction firms must manually review these contracts and populate abstraction forms summarizing key provisions.

This manual process is:

- Time-consuming  
- Labor-intensive  
- Difficult to scale  
- Prone to inconsistencies  

The **IMC Contract Intelligence Assistant** automates this workflow using modern AI techniques including:

- Retrieval-Augmented Generation (RAG)
- Vector embeddings
- Semantic search
- Large Language Models

The system processes large construction contracts, retrieves relevant clauses, generates structured answers for abstraction questions, and evaluates results against manually curated gold answers.

---

# Problem Statement

IMC Construction currently performs **manual contract abstraction** for owner contracts.

Manual abstraction requires:

1. Reading lengthy contracts
2. Finding relevant clauses
3. Extracting information
4. Completing structured abstraction forms

Challenges include:

- Contracts often exceed **500–1000 pages**
- Clauses are distributed across multiple sections
- Terminology varies between contracts
- The process requires **significant domain expertise**
- Manual abstraction does not scale efficiently

The goal of this project is to build an **AI-assisted contract analysis system** that automates large portions of this process.

---

# Solution Approach

The project uses a **Retrieval-Augmented Generation (RAG)** architecture.

Instead of analyzing an entire contract at once, the system:

1. Splits contracts into smaller chunks  
2. Converts chunks into embeddings  
3. Stores embeddings in a vector index  
4. Retrieves relevant clauses for a question  
5. Uses an LLM to generate structured answers  

This approach enables:

- Scalable processing of large documents
- Grounded answers with citations
- Efficient clause retrieval

---

# Key Features

### Automated Contract Abstraction
Automatically answers **120+ abstraction questions** for each contract.

### Semantic Clause Retrieval
Uses vector embeddings to retrieve relevant contract clauses even when wording differs.

### Retrieval-Augmented Generation
Combines semantic search with LLM reasoning.

### Structured JSON Outputs
Ensures answers can populate abstraction forms automatically.

Example:

```json
{
  "value": true,
  "raw_answer": "The contractor must maintain commercial general liability insurance.",
  "citations": ["c00023", "c00045"]
}
```

### Evaluation Framework
Predicted answers are evaluated against gold answers using an LLM judge.

### Interactive Chat Interface
Users can ask natural language questions about the contract.

### Automated PDF Reports
Results are compiled into downloadable reports.

---

# System Architecture

```
Contract Document
        │
        ▼
Text Extraction
        │
        ▼
Chunk Generation
        │
        ▼
Embedding Creation
        │
        ▼
Vector Index (Azure AI Search)
        │
        ▼
Semantic Retrieval
        │
        ▼
LLM Answer Generation
        │
        ▼
Evaluation
        │
        ▼
Results + Web Interface
```

---

# End-to-End Pipeline

The pipeline processes contracts through several stages.

### 1. Document Ingestion

Contracts are uploaded via:

- Web interface
- Raw data directory

Supported formats:

- PDF  
- DOCX  
- TXT  

Outputs:

```
extracted.txt
chunks.jsonl
```

---

### 2. Text Extraction

Libraries used:

- **pypdf**
- **python-docx**

Extraction preserves page references for clause citations.

---

### 3. Chunk Generation

Contracts are split into chunks due to LLM token limits.

Each chunk includes:

- contract_id
- chunk_id
- page_number
- chunk_text

Typical chunk size:

```
target: ~1800 characters
max: ~2400 characters
```

---

### 4. Embedding Generation

Chunks are converted into vector embeddings using **Azure OpenAI**.

---

### 5. Vector Indexing

Embeddings are stored in **Azure AI Search**.

Index fields:

```
contract_id
chunk_id
content
content_vector
```

---

### 6. Retrieval

When a question is asked:

1. The question is embedded
2. Relevant chunks are retrieved
3. Chunks are sent to the LLM

---

### 7. Answer Generation

The LLM produces structured JSON outputs containing:

- predicted value
- raw answer
- clause citations

---

### 8. Batch Abstraction

All abstraction questions are processed automatically.

Results stored in:

```
results.jsonl
```

---

### 9. Evaluation

Predicted answers are compared against manually curated gold answers.

Two evaluation categories:

**Categorical fields**

- boolean
- number
- currency
- percentage
- date

**Descriptive fields**

- text
- clause explanations
- lists

Metrics include semantic similarity, coverage, and contradiction detection.

---

# Web Interface

The project includes a **Shiny for Python application**.

Users can:

1. Upload contracts  
2. Run the abstraction pipeline  
3. Download results reports  
4. Chat with the contract  

The chat interface uses **RAG-based retrieval** to ensure grounded responses.

---

# Repository Structure

```
IMC-Contract-Intelligence-Assistant
│
├── v1
│   │
│   ├── app
│   │   │
│   │   ├── ingest
│   │   │   ├── extract.py
│   │   │   ├── chunk.py
│   │   │   ├── run_ingest.py
│   │   │   └── embed_upsert.py
│   │   │
│   │   ├── rag
│   │   │   ├── retrieve.py
│   │   │   ├── answer.py
│   │   │   ├── batch_run.py
│   │   │   └── rag_chat.py
│   │   │
│   │   ├── pipeline
│   │   │   └── e2e_run.py
│   │   │
│   │   ├── evaluation
│   │   │   └── evaluation.py
│   │   │
│   │   └── reporting
│   │       └── results_pdf.py
│   │
│   ├── data
│   │   ├── raw
│   │   ├── processed
│   │   └── questions.jsonl
|   |   |__ Filled Abstract Forms.xslx
│   │
│   └── shiny_app.py
│
├── scripts
|   |__ create_search_index.py
│
├── requirements.txt
└── README.md
```

---

# Technology Stack

### Programming
Python 3.10+

### AI / NLP
Azure OpenAI  
Azure AI Search  
Embeddings API  
GPT models

### Document Processing
pypdf  
python-docx

### Web Interface
Shiny for Python

### Data Processing
Pandas  
JSONL pipelines

---

# Environment Setup

Create a `.env` file:

```
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=
AZURE_DEPLOYMENT_NAME=

AZURE_OPENAI_EMBEDDINGS_ENDPOINT=
AZURE_OPENAI_EMBEDDINGS_API_KEY=
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=
AZURE_OPENAI_EMBEDDINGS_API_VERSION=

AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_API_KEY=
AZURE_SEARCH_INDEX_NAME=
```

---

# Installation

### Clone Repository

```
git clone <repository_url>
cd IMC-Contract-Intelligence-Assistant
```

### Create Virtual Environment

Windows

```
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux

```
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

---

# Running the Pipeline

Place contract files in:

```
v1/data/raw/
```

Run:

```
py -m v1.app.pipeline.e2e_run
```

Pipeline stages:

1. Ingestion  
2. Chunking  
3. Embedding  
4. Indexing  
5. Abstraction  
6. Evaluation  
7. Report generation  

---

# Running the Web Interface

Start the application:

```
shiny run --reload v1/shiny_app.py
```

Open:

```
http://127.0.0.1:8000
```

---

# Output Files

After processing a contract:

```
v1/data/processed/<contract_id>/
```

Generated artifacts:

```
results.jsonl
evaluation_summary.json
evaluation_metrics_table.csv
evaluation_report.html
<contract_id>.pdf
```

---

# Dataset

The dataset contains **16 real construction owner contracts** provided by IMC Construction.

Each contract includes:

- Owner contract PDF
- Completed abstraction forms

Each form contains approximately **123 questions**.

---

# Evaluation Results

Sample evaluation results:

```
Average gold answers per contract: ~90
Average predictions generated: ~101
Perfect matches: ~25
Overall semantic accuracy: ~74.5%
```

---

# Limitations

- Limited dataset size  
- Contract formatting variability  
- Redacted contract sections  
- OCR for scanned contracts not fully implemented  
- Complex cross-clause reasoning remains difficult  

---

# Future Work

Potential improvements include:

- Expanding dataset size
- Improving clause reasoning
- Adding structured schemas
- Risk detection features
- Explainable clause highlighting
- Conversational memory

---

# Acknowledgements

Drexel University  
IMC Construction  
Capstone Advisors and Faculty