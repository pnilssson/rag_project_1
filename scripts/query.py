import openai
import logging
from typing import List, Dict, Any, Optional
import json

from scripts.embeddings import embedding_manager
from scripts.config import config
from scripts.utils import setup_logging

logger = setup_logging()

class RAGQueryEngine:
    def __init__(self):
        # Configure OpenAI client for LM Studio
        openai.api_base = config.openai_api_base
        openai.api_key = config.openai_api_key
        
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
            return f"Du är en hjälpsam assistent. Svara på följande fråga baserat på din kunskap:\n\nFråga: {query}"
        
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "Unknown")
            score = chunk.get("score", 0)
            text = chunk.get("text", "")
            
            context_parts.append(f"Källa {i} ({source}, relevans: {score:.3f}):\n{text}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Du är en hjälpsam assistent. Svara endast baserat på följande information. Om informationen inte räcker för att svara, säg det tydligt.

KONTEXT:
{context}

FRÅGA: {query}

INSTRUKTIONER:
- Svara endast baserat på den givna kontexten
- Om kontexten inte innehåller tillräcklig information, säg det
- Var konkret och använd information från källorna
- Citera relevanta delar från källorna när det är lämpligt
- Svara på svenska om frågan är på svenska, annars på engelska

SVAR:"""
        
        return prompt
    
    def ask_llm(self, prompt: str) -> str:
        """Send prompt to LLM and get response"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=30
            )
            
            return response.choices[0].message["content"]
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    def query(self, question: str, top_k: int = None, score_threshold: float = None) -> Dict[str, Any]:
        """Complete RAG query process"""
        try:
            logger.info(f"Processing query: {question}")
            
            # Retrieve relevant chunks
            chunks = self.retrieve_relevant_chunks(question, top_k, score_threshold)
            
            # Create prompt
            prompt = self.create_context_prompt(chunks, question)
            
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
                "answer": f"Ett fel uppstod: {str(e)}",
                "error": str(e),
                "sources": [],
                "chunks_used": 0
            }
    
    def interactive_mode(self):
        """Run interactive query mode"""
        print("\n" + "="*60)
        print("RAG QUERY SYSTEM")
        print("="*60)
        print("Skriv 'q' för att avsluta")
        print("Skriv 'stats' för att se statistik")
        print("Skriv 'help' för hjälp")
        print("="*60)
        
        while True:
            try:
                question = input("\nStäll en fråga: ").strip()
                
                if not question:
                    continue
                
                if question.lower() == 'q':
                    print("Avslutar...")
                    break
                
                if question.lower() == 'stats':
                    self.show_statistics()
                    continue
                
                if question.lower() == 'help':
                    self.show_help()
                    continue
                
                # Process query
                result = self.query(question)
                
                # Display result
                print("\n" + "-"*40)
                print("SVAR:")
                print("-"*40)
                print(result["answer"])
                
                if result["sources"]:
                    print(f"\nKällor: {', '.join(set(result['sources']))}")
                    print(f"Antal chunks använda: {result['chunks_used']}")
                
                print("-"*40)
                
            except KeyboardInterrupt:
                print("\nAvslutar...")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"Ett fel uppstod: {e}")
    
    def show_statistics(self):
        """Show system statistics"""
        try:
            stats = embedding_manager.get_collection_info()
            print("\n" + "="*40)
            print("SYSTEM STATISTIK")
            print("="*40)
            print(f"Vector database: {stats['points_count']} chunks")
            print(f"Vector dimension: {stats['vector_size']}")
            print(f"Distance metric: {stats['distance']}")
            print("="*40)
        except Exception as e:
            print(f"Kunde inte hämta statistik: {e}")
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*40)
        print("HJÄLP")
        print("="*40)
        print("Kommandon:")
        print("  q - Avsluta")
        print("  stats - Visa systemstatistik")
        print("  help - Visa denna hjälp")
        print("\nTips:")
        print("  - Ställ specifika frågor för bättre svar")
        print("  - Systemet söker i dina dokument")
        print("  - Svaret baseras endast på dokumenten")
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
        print(f"Systemfel: {e}")
        print("Kontrollera att:")
        print("1. Qdrant är igång (docker compose up -d)")
        print("2. LM Studio är igång med API-server aktiverad")
        print("3. Dokument har bearbetats (python scripts/pipeline.py)")

if __name__ == "__main__":
    main()
