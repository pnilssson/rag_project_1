# Enhanced RAG System with LM Studio + Qdrant

A robust Retrieval-Augmented Generation (RAG) system that combines local LLMs via LM Studio with Qdrant vector database for efficient document processing and querying.

## 🚀 Features

- **Multi-format Document Support**: PDF, TXT, PNG, JPG, XML, DOCX
- **Smart Text Chunking**: Semantic boundary-aware chunking with configurable overlap
- **Advanced OCR**: Automatic OCR for images and PDFs with no text
- **Batch Processing**: Efficient processing of large document collections
- **Interactive Query Interface**: User-friendly CLI with statistics and help
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Configurable System**: Centralized configuration management
- **Error Handling**: Robust error handling with graceful degradation

## 📋 Prerequisites

- Python 3.8+
- Docker and Docker Compose
- LM Studio with API server enabled
- Tesseract OCR (for image processing)

### Install Tesseract OCR

**Windows:**
```bash
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR
```

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

## 🛠️ Installation

1. **Clone and setup:**
```bash
git clone <your-repo>
cd rag_project_1
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start Qdrant:**
```bash
docker compose up -d
```

4. **Start LM Studio:**
   - Open LM Studio
   - Load your preferred model (e.g., Qwen 3 8B)
   - Enable API server (Settings → API Server)
   - Note the API endpoint (usually `http://localhost:1234/v1`)

## 📁 Project Structure

```
rag_project_1/
├── data/                   # Place your documents here
├── db/                     # Qdrant database storage
├── scripts/
│   ├── config.py          # Configuration management
│   ├── utils.py           # Utility functions
│   ├── extract.py         # Document text extraction
│   ├── chunk.py           # Text chunking algorithms
│   ├── embeddings.py      # Vector database operations
│   ├── pipeline.py        # Document processing pipeline
│   ├── query.py           # Query engine
│   └── rag_cli.py         # Command-line interface
├── docker-compose.yml     # Qdrant container setup
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### Method 1: Using the CLI (Recommended)

1. **Process documents:**
```bash
python scripts/rag_cli.py process
```

2. **Query interactively:**
```bash
python scripts/rag_cli.py query
```

3. **Single query:**
```bash
python scripts/rag_cli.py query -q "What is RAG?"
```

4. **View statistics:**
```bash
python scripts/rag_cli.py stats
```

### Method 2: Using Individual Scripts

1. **Process documents:**
```bash
python scripts/pipeline.py
```

2. **Query the system:**
```bash
python scripts/query.py
```

## ⚙️ Configuration

Edit `scripts/config.py` to customize:

- **Chunking**: Size and overlap settings
- **Embeddings**: Model and vector dimensions
- **Query**: Number of results and similarity threshold
- **LLM**: Model name and generation parameters
- **File Processing**: Supported file extensions

```python
# Example configuration
config.chunk_size = 300          # Words per chunk
config.chunk_overlap = 50        # Overlap between chunks
config.top_k = 5                 # Number of chunks to retrieve
config.similarity_threshold = 0.7 # Minimum similarity score
```

## 📊 CLI Commands

### Process Documents
```bash
# Process all documents in data/
python scripts/rag_cli.py process

# Process documents in specific folder
python scripts/rag_cli.py process --folder ./my_docs

# Recreate vector database (delete existing data)
python scripts/rag_cli.py process --recreate
```

### Query System
```bash
# Interactive mode
python scripts/rag_cli.py query

# Single query
python scripts/rag_cli.py query -q "What is machine learning?"

# Interactive commands:
#   q - Quit
#   stats - Show statistics
#   help - Show help
```

### System Management
```bash
# Show statistics
python scripts/rag_cli.py stats

# Reset vector database
python scripts/rag_cli.py reset
```

## 🔧 Advanced Usage

### Custom Document Processing

```python
from scripts.pipeline import RAGPipeline

pipeline = RAGPipeline()
summary = pipeline.process_folder("./my_documents")
print(f"Processed {summary['processed_files']} files")
```

### Programmatic Querying

```python
from scripts.query import RAGQueryEngine

query_engine = RAGQueryEngine()
result = query_engine.query("What is the main topic?")
print(result["answer"])
```

### Custom Chunking

```python
from scripts.chunk import chunk_text, chunk_by_paragraphs

# Semantic chunking
chunks = chunk_text(text, chunk_size=500, overlap=100)

# Paragraph-based chunking
chunks = chunk_by_paragraphs(text, max_chunk_size=400)
```

## 🐛 Troubleshooting

### Common Issues

1. **Qdrant Connection Error:**
   ```bash
   # Check if Qdrant is running
   docker compose ps
   
   # Restart Qdrant
   docker compose restart
   ```

2. **LM Studio Connection Error:**
   - Ensure LM Studio is running
   - Check API server is enabled
   - Verify API endpoint in config

3. **OCR Issues:**
   - Install Tesseract OCR
   - Add to system PATH
   - For Swedish text: `pytesseract.image_to_string(image, lang='swe')`

4. **Memory Issues:**
   - Reduce batch sizes in config
   - Use smaller embedding model
   - Process documents in smaller batches

### Logs

Check `rag_system.log` for detailed error information:
```bash
tail -f rag_system.log
```

## 📈 Performance Tips

1. **Batch Processing**: Process documents in batches for large collections
2. **Chunk Size**: Adjust based on document type (300-500 words works well)
3. **Embedding Model**: Use smaller models for faster processing
4. **Similarity Threshold**: Increase for more relevant results
5. **Top-K**: Adjust based on query complexity

## 🔄 Maintenance

### Regular Tasks

1. **Backup Database:**
   ```bash
   docker compose exec qdrant qdrant snapshot create
   ```

2. **Update Dependencies:**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Clean Logs:**
   ```bash
   rm rag_system.log
   ```

### Reset System

```bash
# Stop services
docker compose down

# Remove all data
docker compose down -v

# Restart
docker compose up -d
python scripts/rag_cli.py process --recreate
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LM Studio](https://lmstudio.ai/) for local LLM hosting
- [Qdrant](https://qdrant.tech/) for vector database
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing