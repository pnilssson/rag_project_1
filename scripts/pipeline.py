import os
import logging
from pathlib import Path
from typing import List, Dict, Any

from scripts.extract import extract_text
from scripts.chunk import chunk_text
from scripts.embeddings import embedding_manager
from scripts.config import config
from scripts.utils import setup_logging, is_supported_file, measure_time

logger = setup_logging()

class RAGPipeline:
    def __init__(self):
        self.data_dir = config.data_dir
        self.processed_files = []
        self.failed_files = []
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file through the RAG pipeline"""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Extract text
            text = extract_text(file_path)
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
                return {"status": "no_text", "file": file_path}
            
            # Chunk text
            chunks = chunk_text(text)
            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return {"status": "no_chunks", "file": file_path}
            
            # Insert chunks
            embedding_manager.insert_chunks(chunks, os.path.basename(file_path))
            
            result = {
                "status": "success",
                "file": file_path,
                "text_length": len(text),
                "chunks_count": len(chunks),
                "file_size": os.path.getsize(file_path)
            }
            
            self.processed_files.append(result)
            logger.info(f"✅ Successfully processed: {os.path.basename(file_path)}")
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "file": file_path,
                "error": str(e)
            }
            self.failed_files.append(error_result)
            logger.error(f"❌ Error processing {file_path}: {e}")
            return error_result
    
    @measure_time
    def process_folder(self, folder_path: str = None) -> Dict[str, Any]:
        """Process all supported files in a folder"""
        if folder_path is None:
            folder_path = self.data_dir
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Data directory not found: {folder_path}")
        
        logger.info(f"Starting to process folder: {folder_path}")
        
        # Get all files
        files = []
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path) and is_supported_file(full_path, config.supported_extensions):
                files.append(full_path)
        
        if not files:
            logger.warning(f"No supported files found in {folder_path}")
            return {"status": "no_files", "folder": folder_path}
        
        logger.info(f"Found {len(files)} supported files to process")
        
        # Process each file
        for file_path in files:
            self.process_file(file_path)
        
        # Summary
        summary = {
            "total_files": len(files),
            "processed_files": len(self.processed_files),
            "failed_files": len(self.failed_files),
            "success_rate": len(self.processed_files) / len(files) if files else 0,
            "processed": self.processed_files,
            "failed": self.failed_files
        }
        
        logger.info(f"Processing complete. Success rate: {summary['success_rate']:.2%}")
        return summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processed data"""
        try:
            collection_info = embedding_manager.get_collection_info()
            return {
                "collection_info": collection_info,
                "processed_files": len(self.processed_files),
                "failed_files": len(self.failed_files),
                "total_chunks": sum(f.get("chunks_count", 0) for f in self.processed_files)
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

def main():
    """Main pipeline execution"""
    try:
        # Initialize collection
        logger.info("Initializing vector database collection...")
        embedding_manager.init_collection()
        
        # Create and run pipeline
        pipeline = RAGPipeline()
        summary = pipeline.process_folder()
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total files: {summary['total_files']}")
        print(f"Successfully processed: {summary['processed_files']}")
        print(f"Failed: {summary['failed_files']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        
        if summary['failed_files'] > 0:
            print("\nFailed files:")
            for failed in summary['failed']:
                print(f"  - {os.path.basename(failed['file'])}: {failed['error']}")
        
        # Get final statistics
        stats = pipeline.get_statistics()
        if "collection_info" in stats:
            print(f"\nVector database: {stats['collection_info']['points_count']} total chunks")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
