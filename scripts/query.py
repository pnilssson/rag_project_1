import openai
import logging
from typing import List, Dict, Any, Optional
import json

from embeddings import embedding_manager
from config import config
from utils import setup_logging

logger = setup_logging()

class RAGQueryEngine:
    def __init__(self):
        # Configure OpenAI client for LM Studio
        self.client = openai.OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_api_base
        )
        
        self.model = config.llm_model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = None, score_threshold: float = None) -> List[Dict]:
        """Retrieve relevant chunks using semantic search"""
        try:
            results = embedding_manager.search_similar(
                query, 
                top_k=top_k or config.top_k,
                score_threshold=score_threshold or config.similarity_threshold
            )
            
            if not results:
                logger.warning("No relevant chunks found for query")
                return []
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise
    
    def create_context_prompt(self, chunks: List[Dict], query: str) -> str:
        """Create a well-structured prompt with context"""
        if not chunks:
            return f"I'm sorry, but I couldn't find any relevant information in the provided documents to answer your question: '{query}'. Please try rephrasing your question or ask about a different topic that might be covered in the available documents."
        
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "Unknown")
            score = chunk.get("score", 0)
            text = chunk.get("text", "")
            
            context_parts.append(f"Source {i} ({source}, relevance: {score:.3f}):\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # Create language instruction based on system language
        language_instruction = f"Answer in {config.system_language}" if config.system_language != "en" else "Answer in English"
        
        prompt = f"""You are a helpful assistant. Answer only based on the following information. If the information is not sufficient to answer, say so clearly.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer only based on the given context
- If the context doesn't contain sufficient information, say so
- Be specific and use information from the sources
- Cite relevant parts from the sources when appropriate
- {language_instruction}

ANSWER:"""
        
        return prompt
    
    def ask_llm(self, prompt: str) -> str:
        """Send prompt to LLM and get response"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    def query(self, question: str, top_k: int = None, score_threshold: float = None) -> Dict[str, Any]:
        """Complete RAG query process"""
        try:
            logger.info(f"Processing query: {question}")
            
            # Retrieve relevant chunks
            chunks = self.retrieve_relevant_chunks(question, top_k, score_threshold)
            
            # If no chunks found, return early without calling LLM
            if not chunks:
                return {
                    "question": question,
                    "answer": f"I'm sorry, but I couldn't find any relevant information in the provided documents to answer your question: '{question}'. Please try rephrasing your question or ask about a different topic that might be covered in the available documents.",
                    "sources": [],
                    "chunks_used": 0,
                    "chunks": []
                }
            
            # Create prompt
            prompt = self.create_context_prompt(chunks, question)
            logger.info(f"Generated prompt for LLM: {prompt}")
            
            # Get LLM response
            answer = self.ask_llm(prompt)
            
            # Prepare result
            result = {
                "question": question,
                "answer": answer,
                "sources": [chunk.get("source") for chunk in chunks],
                "chunks_used": len(chunks),
                "chunks": chunks
            }
            
            logger.info(f"Query completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "question": question,
                "answer": f"An error occurred: {str(e)}",
                "error": str(e),
                "sources": [],
                "chunks_used": 0
            }
    
    def interactive_mode(self):
        """Run interactive query mode"""
        print("\n" + "="*60)
        print("RAG QUERY SYSTEM")
        print("="*60)
        print("Type 'q' to quit")
        print("Type 'stats' to see statistics")
        print("Type 'help' for help")
        print("="*60)
        
        while True:
            try:
                question = input("\nAsk a question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() == 'q':
                    print("Exiting...")
                    break
                
                if question.lower() == 'stats':
                    self.show_statistics()
                    continue
                
                if question.lower() == 'help':
                    self.show_help()
                    continue
                
                # Process query
                logger.info(f"User question in interactive mode: {question}")
                result = self.query(question)
                
                # Display result
                print("\n" + "-"*40)
                print("ANSWER:")
                print("-"*40)
                print(result["answer"])
                
                if result["sources"]:
                    print(f"\nSources: {', '.join(set(result['sources']))}")
                    print(f"Chunks used: {result['chunks_used']}")
                
                print("-"*40)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"An error occurred: {e}")
    
    def show_statistics(self):
        """Show system statistics"""
        try:
            stats = embedding_manager.get_collection_info()
            print("\n" + "="*40)
            print("SYSTEM STATISTICS")
            print("="*40)
            print(f"Vector database: {stats['points_count']} chunks")
            print(f"Vector dimension: {stats['vector_size']}")
            print(f"Distance metric: {stats['distance']}")
            print("="*40)
        except Exception as e:
            print(f"Could not retrieve statistics: {e}")
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*40)
        print("HELP")
        print("="*40)
        print("Commands:")
        print("  q - Quit")
        print("  stats - Show system statistics")
        print("  help - Show this help")
        print("\nTips:")
        print("  - Ask specific questions for better answers")
        print("  - The system searches in your documents")
        print("  - Answers are based only on the documents")
        print("="*40)

def main():
    """Main query execution"""
    try:
        # Test connection to vector database
        logger.info("Testing vector database connection...")
        embedding_manager.get_collection_info()
        
        # Create query engine
        query_engine = RAGQueryEngine()
        
        # Start interactive mode
        query_engine.interactive_mode()
        
    except Exception as e:
        logger.error(f"Query system failed: {e}")
        print(f"System error: {e}")
        print("Check that:")
        print("1. Qdrant is running (docker compose up -d)")
        print("2. LM Studio is running with API server enabled")
        print("3. Documents have been processed (python scripts/pipeline.py)")

if __name__ == "__main__":
    main()
