from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import uuid
import logging
from typing import List, Dict, Any
import numpy as np

from config import config
from utils import setup_logging, measure_time, batch_process

logger = setup_logging()

class EmbeddingManager:
    def __init__(self):
        self.client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        self.model = SentenceTransformer(config.embedding_model)
        self.collection_name = config.collection_name
    
    def init_collection(self, recreate: bool = False):
        """Initialize or recreate the collection"""
        try:
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if recreate or not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=config.vector_size, 
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    @measure_time
    def insert_chunks(self, chunks: List[str], source_file: str, metadata: Dict[str, Any] = None):
        """Insert chunks with improved batch processing and error handling"""
        if not chunks:
            logger.warning("No chunks to insert")
            return
        
        try:
            # Generate embeddings in batches
            all_vectors = []
            for batch in batch_process(chunks, batch_size=50):
                vectors = self.model.encode(batch, show_progress_bar=True)
                all_vectors.extend(vectors)
            
            # Create points with metadata
            points = []
            for i, (chunk, vector) in enumerate(zip(chunks, all_vectors)):
                point_metadata = {
                    "text": chunk,
                    "source": source_file,
                    "chunk_index": i,
                    "chunk_length": len(chunk.split())
                }
                
                if metadata:
                    point_metadata.update(metadata)
                
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload=point_metadata
                ))
            
            # Insert in batches
            for batch in batch_process(points, batch_size=100):
                self.client.upsert(
                    collection_name=self.collection_name, 
                    points=batch
                )
            
            logger.info(f"Successfully inserted {len(points)} chunks from {source_file}")
            
        except Exception as e:
            logger.error(f"Error inserting chunks from {source_file}: {e}")
            raise
    
    def search_similar(self, query: str, top_k: int = None, score_threshold: float = None) -> List[Dict]:
        """Search for similar chunks with score thresholding"""
        if top_k is None:
            top_k = config.top_k
        if score_threshold is None:
            score_threshold = config.similarity_threshold
        
        try:
            query_vector = self.model.encode(query).tolist()
            
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            results = []
            for hit in hits:
                results.append({
                    "text": hit.payload["text"],
                    "source": hit.payload["source"],
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() 
                               if k not in ["text", "source"]}
                })
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar chunks: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "points_count": info.points_count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise
    
    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

# Global instance
embedding_manager = EmbeddingManager()
