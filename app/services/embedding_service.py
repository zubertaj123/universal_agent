"""
Embedding service for RAG
"""
from typing import List, Dict, Any, Optional
import asyncio
import openai
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import hashlib
from pathlib import Path
import json

from app.core.config import settings
from app.utils.text_processing import chunk_text
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingService:
    """Service for generating and storing embeddings"""
    
    def __init__(self):
        self.use_openai = settings.OPENAI_API_KEY is not None
        self.model = None
        self.client = None
        self.collection = None
        self._initialize_embeddings()
        self._initialize_chromadb()
        
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if self.use_openai:
            openai.api_key = settings.OPENAI_API_KEY
            logger.info("Using OpenAI embeddings")
        else:
            # Use local sentence transformer model
            logger.info("Loading local embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Local embedding model loaded")
            
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Ensure data directory exists
            chroma_path = settings.DATA_DIR / "chromadb"
            chroma_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="call_center_knowledge",
                metadata={"description": "Call center knowledge base"}
            )
            
            logger.info(f"ChromaDB initialized with {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
        
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            if self.use_openai:
                # Use OpenAI API
                response = await openai.Embedding.acreate(
                    model=getattr(settings, 'EMBEDDING_MODEL', 'text-embedding-3-small'),
                    input=text.replace('\n', ' ')
                )
                return response.data[0].embedding
            else:
                # Use local model
                if self.model is None:
                    raise ValueError("Local embedding model not initialized")
                
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None, 
                    lambda: self.model.encode(text, convert_to_tensor=False)
                )
                return embedding.tolist()
                
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise
            
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
            
            # Check if document already exists
            existing = self.collection.get(
                where={"document_id": doc_id}
            )
            
            if existing['ids']:
                logger.info(f"Document {filename} already exists, updating...")
                # Delete existing chunks
                self.collection.delete(ids=existing['ids'])
            
            # Clean and chunk document
            cleaned_content = self._clean_content(content)
            chunks = chunk_text(cleaned_content, chunk_size=500, overlap=50)
            
            if not chunks:
                return {
                    "document_id": doc_id,
                    "chunks": 0,
                    "status": "error",
                    "message": "No content to process"
                }
            
            # Process each chunk
            chunk_ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
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
                    "content_type": self._detect_content_type(content),
                    "processed_at": asyncio.get_event_loop().time(),
                    **(metadata or {})
                }
                metadatas.append(chunk_metadata)
                documents.append(chunk)
                
            # Store in ChromaDB
            if chunk_ids:
                self.collection.add(
                    ids=chunk_ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                
                logger.info(f"Processed document {filename}: {len(chunk_ids)} chunks")
            
            return {
                "document_id": doc_id,
                "chunks": len(chunk_ids),
                "status": "success",
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"Document processing error for {filename}: {e}")
            return {
                "document_id": None,
                "chunks": 0,
                "status": "error",
                "error": str(e)
            }
            
    async def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if not query.strip():
                return []
                
            # Generate query embedding
            query_embedding = await self.embed_text(query)
            
            # Prepare search parameters
            search_params = {
                "query_embeddings": [query_embedding],
                "n_results": min(n_results, 20),  # Limit to reasonable number
            }
            
            if filter_metadata:
                search_params["where"] = filter_metadata
                
            # Search in ChromaDB
            results = self.collection.query(**search_params)
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = {
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if results.get('distances') else 0.0
                    }
                    
                    # Calculate similarity score (1 - distance)
                    result["similarity"] = max(0.0, 1.0 - result["distance"])
                    
                    formatted_results.append(result)
                    
            # Sort by similarity (highest first)
            formatted_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.debug(f"Search for '{query}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
            
    async def search_by_category(
        self,
        query: str,
        category: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within a specific category"""
        return await self.search(
            query=query,
            n_results=n_results,
            filter_metadata={"category": category}
        )
        
    async def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document"""
        try:
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if not results['ids']:
                return None
                
            # Aggregate chunk information
            chunks = []
            metadata = None
            
            for i in range(len(results['ids'])):
                chunk_meta = results['metadatas'][i]
                if metadata is None:
                    metadata = {
                        "document_id": document_id,
                        "filename": chunk_meta.get("filename"),
                        "content_type": chunk_meta.get("content_type"),
                        "total_chunks": chunk_meta.get("total_chunks", 0)
                    }
                    
                chunks.append({
                    "id": results['ids'][i],
                    "index": chunk_meta.get("chunk_index", i),
                    "content_preview": results['documents'][i][:100] + "..." if len(results['documents'][i]) > 100 else results['documents'][i]
                })
                
            return {
                "metadata": metadata,
                "chunks": sorted(chunks, key=lambda x: x.get("index", 0)),
                "total_chunks": len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Document info error: {e}")
            return None
            
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete document and its chunks"""
        try:
            # Get all chunks for document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if not results['ids']:
                return {
                    "status": "error", 
                    "message": "Document not found",
                    "deleted": 0
                }
                
            # Delete chunks
            self.collection.delete(ids=results['ids'])
            
            logger.info(f"Deleted document {document_id} with {len(results['ids'])} chunks")
            
            return {
                "status": "success", 
                "deleted": len(results['ids']),
                "document_id": document_id
            }
            
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "deleted": 0
            }
            
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the knowledge base"""
        try:
            # Get all unique document IDs
            all_results = self.collection.get()
            
            if not all_results['metadatas']:
                return []
                
            # Group by document_id
            documents = {}
            for metadata in all_results['metadatas']:
                doc_id = metadata.get('document_id')
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get('filename', 'Unknown'),
                        "content_type": metadata.get('content_type', 'text'),
                        "total_chunks": metadata.get('total_chunks', 0),
                        "processed_at": metadata.get('processed_at')
                    }
                    
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"List documents error: {e}")
            return []
            
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            count = self.collection.count()
            documents = await self.list_documents()
            
            return {
                "total_chunks": count,
                "total_documents": len(documents),
                "collection_name": self.collection.name,
                "embedding_model": "OpenAI" if self.use_openai else "SentenceTransformer"
            }
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "error": str(e)
            }
            
    def _clean_content(self, content: str) -> str:
        """Clean content before processing"""
        # Remove excessive whitespace
        content = ' '.join(content.split())
        
        # Remove very short lines (likely noise)
        lines = content.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(cleaned_lines)
        
    def _detect_content_type(self, content: str) -> str:
        """Detect content type"""
        content_lower = content.lower()
        
        if 'q:' in content_lower and 'a:' in content_lower:
            return 'faq'
        elif 'policy' in content_lower or 'coverage' in content_lower:
            return 'policy'
        elif 'claim' in content_lower or 'process' in content_lower:
            return 'procedure'
        elif 'phone' in content_lower or 'contact' in content_lower:
            return 'contact_info'
        else:
            return 'general'
            
    async def bulk_process_directory(
        self, 
        directory_path: str, 
        file_pattern: str = "*.txt"
    ) -> Dict[str, Any]:
        """Process all files in a directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            return {
                "status": "error",
                "message": f"Directory {directory_path} does not exist"
            }
            
        files = list(directory.glob(file_pattern))
        results = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                result = await self.process_document(
                    filename=file_path.name,
                    content=content,
                    metadata={
                        "source_path": str(file_path),
                        "file_size": file_path.stat().st_size,
                        "file_modified": file_path.stat().st_mtime
                    }
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    "filename": file_path.name,
                    "status": "error",
                    "error": str(e)
                })
                
        successful = sum(1 for r in results if r.get("status") == "success")
        total_chunks = sum(r.get("chunks", 0) for r in results)
        
        return {
            "status": "completed",
            "files_processed": len(files),
            "successful": successful,
            "failed": len(files) - successful,
            "total_chunks": total_chunks,
            "results": results
        }