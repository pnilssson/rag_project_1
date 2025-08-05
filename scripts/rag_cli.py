#!/usr/bin/env python3
"""
RAG System CLI - Command Line Interface for RAG operations
"""

import argparse
import sys
import logging
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from pipeline import RAGPipeline
from query import RAGQueryEngine
from embeddings import embedding_manager
from config import config
from utils import setup_logging

logger = setup_logging()

def process_command(args):
    """Process documents and build vector database using LangChain chunking"""
    try:
        print("🤖 RAG System with LangChain Chunking")
        print("=" * 50)
        print(f"Chunking Strategy: RecursiveCharacterTextSplitter")
        print(f"Chunk Size: {config.chunk_size} words")
        print(f"Chunk Overlap: {config.chunk_overlap} words")
        print("=" * 50)
        
        pipeline = RAGPipeline()
        
        if args.recreate:
            logger.info("Recreating vector database collection...")
            print("🗑️  Recreating vector database collection...")
            embedding_manager.init_collection(recreate=True)
            print("✅ Collection recreated!")
        
        folder_path = args.folder or config.data_dir
        print(f"📁 Processing folder: {folder_path}")
        
        summary = pipeline.process_folder(folder_path)
        
        print(f"\n" + "="*50)
        print("📊 PROCESSING SUMMARY")
        print("="*50)
        print(f"Total files: {summary['total_files']}")
        print(f"✅ Successfully processed: {summary['processed_files']}")
        print(f"❌ Failed: {summary['failed_files']}")
        print(f"📈 Success rate: {summary['success_rate']:.2%}")
        
        # Show chunking details for successful files
        if summary['processed']:
            print(f"\n📄 Chunking Details:")
            total_chunks = 0
            for result in summary['processed']:
                chunks = result.get('chunks_count', 0)
                total_chunks += chunks
                print(f"  • {result['file']}: {chunks} chunks")
            print(f"  📊 Total chunks created: {total_chunks}")
        
        if summary['failed_files'] > 0:
            print(f"\n❌ Failed files:")
            for failed in summary['failed']:
                print(f"  • {failed['file']}: {failed['error']}")
        
        print("="*50)
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"❌ Error: {e}")
        return 1

def query_command(args):
    """Query the RAG system"""
    try:
        query_engine = RAGQueryEngine()
        
        if args.question:
            # Single query mode
            result = query_engine.query(args.question)
            
            print(f"\nQ: {result['question']}")
            print(f"A: {result['answer']}")
            
            if result['sources']:
                print(f"\nSources: {', '.join(set(result['sources']))}")
                print(f"Chunks used: {result['chunks_used']}")
        else:
            # Interactive mode
            query_engine.interactive_mode()
        
        return 0
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        print(f"Error: {e}")
        return 1

def stats_command(args):
    """Show system statistics with LangChain chunking info"""
    try:
        stats = embedding_manager.get_collection_info()
        
        print("\n" + "="*50)
        print("🤖 RAG SYSTEM STATISTICS")
        print("="*50)
        print(f"📊 Collection: {stats['name']}")
        print(f"📄 Total chunks: {stats['points_count']}")
        print(f"🔢 Vector dimension: {stats['vector_size']}")
        print(f"📏 Distance metric: {stats['distance']}")
        
        # Show chunking configuration
        print(f"\n🔧 Chunking Configuration:")
        print(f"  • Strategy: RecursiveCharacterTextSplitter")
        print(f"  • Chunk size: {config.chunk_size} words")
        print(f"  • Chunk overlap: {config.chunk_overlap} words")
        print(f"  • Supported formats: PDF, TXT, PNG, JPG, XML, DOCX, MD")
        
        # Show document loaders info
        print(f"\n📚 Document Loaders:")
        print(f"  • PDF: PyMuPDFLoader")
        print(f"  • Word: UnstructuredWordDocumentLoader")
        print(f"  • Images: UnstructuredImageLoader")
        print(f"  • Text/Markdown: TextLoader")
        print(f"  • XML: TextLoader")
        
        print("="*50)
        return 0
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        print(f"❌ Error: {e}")
        return 1

def reset_command(args):
    """Reset the vector database"""
    try:
        print("🗑️  Reset Vector Database")
        print("=" * 30)
        print("⚠️  This will delete ALL chunks and documents!")
        print("   You'll need to reprocess all documents after reset.")
        print("=" * 30)
        
        confirm = input("Are you sure you want to delete all data? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Operation cancelled.")
            return 0
        
        print("🗑️  Deleting collection...")
        embedding_manager.delete_collection()
        print("✅ Vector database reset successfully!")
        print("💡 Run 'python scripts/rag_cli.py process' to reprocess documents")
        
        return 0
        
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        print(f"❌ Error: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="🤖 RAG System CLI with LangChain Chunking - Process documents and query knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/rag_cli.py process                    # Process all documents in data/
  python scripts/rag_cli.py process --folder ./docs    # Process documents in specific folder
  python scripts/rag_cli.py process --recreate         # Recreate vector database
  python scripts/rag_cli.py query                      # Interactive query mode
  python scripts/rag_cli.py query -q "What is RAG?"    # Single query
  python scripts/rag_cli.py stats                      # Show statistics with chunking info
  python scripts/rag_cli.py reset                      # Reset vector database

Supported Formats: PDF, TXT, PNG, JPG, XML, DOCX, MD
Chunking: LangChain RecursiveCharacterTextSplitter
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process documents and build vector database')
    process_parser.add_argument('--folder', '-f', help='Folder containing documents to process')
    process_parser.add_argument('--recreate', '-r', action='store_true', help='Recreate vector database collection')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('--question', '-q', help='Single question to ask')
    
    # Stats command
    subparsers.add_parser('stats', help='Show system statistics')
    
    # Reset command
    subparsers.add_parser('reset', help='Reset vector database (delete all data)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'process':
        return process_command(args)
    elif args.command == 'query':
        return query_command(args)
    elif args.command == 'stats':
        return stats_command(args)
    elif args.command == 'reset':
        return reset_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 