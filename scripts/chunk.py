from typing import List, Dict, Any
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredImageLoader
)
from langchain.schema import Document

from config import config
from utils import clean_text

class LangChainChunker:
    """
    Optimized chunking using LangChain's RecursiveCharacterTextSplitter
    for semantic-aware text splitting across all document types.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        
        # Initialize the recursive character splitter with word-based length function
        def word_length_function(text: str) -> int:
            return len(text.split())
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=word_length_function,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load document using appropriate LangChain loader based on file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects with metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine loader based on file extension
        if file_path.suffix.lower() == '.pdf':
            loader = PyMuPDFLoader(str(file_path))
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            loader = UnstructuredWordDocumentLoader(str(file_path))
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            loader = UnstructuredImageLoader(str(file_path))
        elif file_path.suffix.lower() in ['.txt', '.md']:
            loader = TextLoader(str(file_path), encoding='utf-8')
        else:
            # Fallback to text loader
            loader = TextLoader(str(file_path), encoding='utf-8')
        
        try:
            documents = loader.load()
            
            # Add file metadata
            for doc in documents:
                doc.metadata.update({
                    'source': file_path.name,
                    'file_type': file_path.suffix.lower()
                })
            
            return documents
        except Exception as e:
            raise Exception(f"Error loading document {file_path}: {str(e)}")
    
    def chunk_documents(self, documents: List[Document], source_name: str = None) -> List[Document]:
        """
        Chunk documents using recursive character splitting.
        
        Args:
            documents: List of Document objects to chunk
            source_name: Original filename to use in metadata
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []
        
        try:
            chunks = self.splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                # Clean the chunk content first
                chunk.page_content = clean_text(chunk.page_content)
                
                # Add metadata after cleaning
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk.page_content.split())
                })
                
                # Override source with original name if provided
                if source_name:
                    chunk.metadata['source'] = source_name
            
            # Filter out very short chunks
            chunks = [chunk for chunk in chunks if len(chunk.page_content.split()) >= 10]
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error chunking documents: {str(e)}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None, source_name: str = None) -> List[Document]:
        """
        Chunk raw text using recursive character splitting.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks
            source_name: Original filename to use in metadata
            
        Returns:
            List of Document objects representing chunks
        """
        if not text.strip():
            return []
        
        # Clean the text first
        text = clean_text(text)
        
        # Create a document object
        doc = Document(
            page_content=text,
            metadata=metadata or {}
        )
        
        return self.chunk_documents([doc], source_name)
    
    def chunk_file(self, file_path: str, source_name: str = None) -> List[Document]:
        """
        Load and chunk a file using recursive character splitting.
        
        Args:
            file_path: Path to the file
            source_name: Original filename to use in metadata
            
        Returns:
            List of chunked Document objects
        """
        # Load the document
        documents = self.load_document(file_path)
        return self.chunk_documents(documents, source_name)


