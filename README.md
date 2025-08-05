# RAG System with LM Studio + Qdrant

A simple RAG system using local LLMs via LM Studio and Qdrant vector database.

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Start Qdrant:**
```bash
docker compose up -d
```

Access Qdrant dashboard at: http://localhost:6333/dashboard

3. **Start LM Studio:**
   - Load your model (e.g., Qwen 3 8B)
   - Enable API server (Settings → API Server)

4. **Process documents:**
```bash
python scripts/rag_cli.py process
```

5. **Query the system:**
```bash
python scripts/rag_cli.py query
```

## Supported Files

- PDF, TXT, PNG, JPG, XML, DOCX

## Configuration

Edit `scripts/config.py` for:
- Chunk size and overlap
- Embedding model
- Query settings
- Language settings
- LLM parameters

## CLI Commands

```bash
# Process documents
python scripts/rag_cli.py process
python scripts/rag_cli.py process --folder ./my_docs
python scripts/rag_cli.py process --recreate

# Query
python scripts/rag_cli.py query
python scripts/rag_cli.py query -q "What is RAG?"

# System
python scripts/rag_cli.py stats
python scripts/rag_cli.py reset
```

## Troubleshooting

- **Qdrant not working**: `docker compose restart`
- **LM Studio issues**: Check API server is enabled
- **OCR problems**: Install Tesseract OCR
- **Memory issues**: Reduce batch sizes in config

## Prerequisites

- Python 3.8+
- Docker
- LM Studio
- Tesseract OCR (for images)
