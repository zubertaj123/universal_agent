"""
Embedding service for RAG
"""
from typing import List, Dict, Any
import openai
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import hashlib
from pathlib import Path

from app.core.config import settings
from app.utils.text_processing import chunk_text
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingService:
    """Service for generating and storing embeddings"""
    
    def __init__(self):
        self.use_openai = settings.OPENAI_API_KEY is not None
        
        if self.use_openai:
            openai.api_key = settings.OPENAI_API_KEY
        else:
            # Use local model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(settings.DATA_DIR / "chromadb"),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="call_center_knowledge",
            metadata={"description": "Call center knowledge base"}
        )
        
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.use_openai:
            response = await openai.Embedding.acreate(
                model=settings.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        else:
            # Use local model
            embedding = self.model.encode(text)
            return embedding.tolist()
            
    async def process_document(
        self,
        filename: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process and store document"""
        try:
            # Generate document ID
            doc_id = hashlib.md5(f"{filename}-{content[:100]}".encode()).hexdigest()
            
            # Chunk document
            chunks = chunk_text(content, chunk_size=500, overlap=50)
            
            # Process each chunk
            chunk_ids = []
            embeddings = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}-{i}"
                chunk_ids.append(chunk_id)
                
                # Generate embedding
                embedding = await self.embed_text(chunk)
                embeddings.append(embedding)
                
                # Prepare metadata
                chunk_metadata = {
                    "document_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **(metadata or {})
                }
                metadatas.append(chunk_metadata)
                
            # Store in ChromaDB
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            
            return {
                "document_id": doc_id,
                "chunks": len(chunks),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise
            
    async def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = await self.embed_text(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
            
    async def delete_document(self, document_id: str):
        """Delete document and its chunks"""
        try:
            # Get all chunks for document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results['ids']:
                # Delete chunks
                self.collection.delete(ids=results['ids'])
                
            return {"status": "success", "deleted": len(results['ids'])}
            
        except Exception as e:
            logger.error(f"Delete error: {e}")
            raise