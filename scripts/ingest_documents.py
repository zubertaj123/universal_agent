#!/usr/bin/env python3
"""
Ingest documents into the knowledge base
"""
import asyncio
import sys
from pathlib import Path
sys.path.append('.')

from app.services.embedding_service import EmbeddingService
from app.utils.logger import setup_logger
import argparse

logger = setup_logger(__name__)

async def ingest_file(embedding_service: EmbeddingService, file_path: Path):
    """Ingest a single file"""
    logger.info(f"Processing {file_path.name}...")
    
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Process document
        result = await embedding_service.process_document(
            filename=file_path.name,
            content=content,
            metadata={
                "source": str(file_path),
                "type": file_path.suffix[1:],  # Remove the dot
            }
        )
        
        logger.info(f"Successfully processed {file_path.name}: {result['chunks']} chunks created")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return False

async def ingest_directory(directory_path: str, file_pattern: str = "*.txt"):
    """Ingest all files in a directory"""
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        logger.error(f"Directory {directory_path} does not exist")
        return
        
    # Initialize embedding service
    embedding_service = EmbeddingService()
    
    # Find all matching files
    files = list(dir_path.glob(file_pattern))
    logger.info(f"Found {len(files)} files to process")
    
    # Process each file
    success_count = 0
    for file_path in files:
        if await ingest_file(embedding_service, file_path):
            success_count += 1
            
    logger.info(f"Completed: {success_count}/{len(files)} files processed successfully")

async def main():
    parser = argparse.ArgumentParser(description="Ingest documents into knowledge base")
    parser.add_argument("path", help="File or directory path")
    parser.add_argument("--pattern", default="*.txt", help="File pattern for directory ingestion")
    parser.add_argument("--single", action="store_true", help="Ingest single file")
    
    args = parser.parse_args()
    
    if args.single:
        # Ingest single file
        embedding_service = EmbeddingService()
        await ingest_file(embedding_service, Path(args.path))
    else:
        # Ingest directory
        await ingest_directory(args.path, args.pattern)

if __name__ == "__main__":
    asyncio.run(main())