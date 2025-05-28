import os
from pathlib import Path
from typing import List
from omegaconf import DictConfig
import glob

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
from langchain.schema import Document

class DatabaseBuilder:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.paths.project_root)
        self.data_dir = Path(cfg.paths.data_dir)
        self.database_dir = Path(cfg.paths.database_dir)
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(
            api_key=cfg.llm.api_key,
            model=cfg.llm.embeddings.model,
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.llm.embeddings.chunk_size,
            chunk_overlap=cfg.llm.embeddings.chunk_overlap,
        )
    
    def build_database(self) -> Chroma:
        """
        The only public method to build vector database.
        Returns a Chroma vector database.
        """
        # Check if database already exists
        if self._database_exists():
            print(f"Database already exists at {self.database_dir}, using existing database.")
            return self._load_existing_database()
        
        print(f"Initializing new database at {self.database_dir}")
        return self._create_new_database()
    
    def _database_exists(self) -> bool:
        """Check if database directory exists and has content"""
        if not self.database_dir.exists():
            return False
        
        # Check if there are any files in database directory
        return any(self.database_dir.iterdir())
    
    def _load_existing_database(self) -> Chroma:
        """Load existing database"""
        return Chroma(
            persist_directory=str(self.database_dir),
            embedding_function=self.embeddings
        )
    
    def _create_new_database(self) -> Chroma:
        """Create new database"""
        # Create database directory if not exists
        self.database_dir.mkdir(parents=True, exist_ok=True)
        
        # Check data directory
        if not self.data_dir.exists():
            print(f"Data directory {self.data_dir} does not exist, initializing empty database.")
            documents = []
        else:
            print(f"Loading documents from {self.data_dir}")
            documents = self._load_documents()
        
        # Create database with documents (can be empty)
        if documents:
            print(f"Processing {len(documents)} documents...")
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            print(f"Split into {len(chunks)} chunks for embedding...")
            
            # Create vector database with batch processing
            vectordb = self._create_database_with_batches(chunks)
        else:
            # Create empty database
            vectordb = Chroma(
                persist_directory=str(self.database_dir),
                embedding_function=self.embeddings
            )
        
        print("Database created successfully!")
        return vectordb
    
    def _load_documents(self) -> List[Document]:
        """Load all documents from data directory"""
        documents = []
        
        # Define file patterns and corresponding loaders
        file_patterns = {
            "*.txt": TextLoader,
            "*.md": UnstructuredMarkdownLoader,
            "*.pdf": PyPDFLoader,
            "*.csv": CSVLoader,
            "*.json": JSONLoader,
            "*.docx": UnstructuredWordDocumentLoader,
            "*.doc": UnstructuredWordDocumentLoader
        }
        
        # Iterate through all file patterns
        for pattern, loader_class in file_patterns.items():
            # Use rglob to find all files (including root directory and subdirectories)
            files = list(self.data_dir.rglob(pattern))
            # Remove duplicate files by converting to set then back to list
            unique_files = list(set(files))
            
            for file_path in unique_files:
                try:
                    print(f"Loading: {file_path}")
                    
                    # Special handling for JSON files
                    if pattern == "*.json":
                        loader = loader_class(
                            file_path=str(file_path),
                            jq_schema='.', 
                            text_content=False
                        )
                    # Special handling for CSV files
                    elif pattern == "*.csv":
                        loader = loader_class(
                            file_path=str(file_path),
                            encoding='utf-8'
                        )
                    else:
                        loader = loader_class(str(file_path))
                    
                    # Load document
                    docs = loader.load()
                    
                    # Add metadata about source file
                    for doc in docs:
                        doc.metadata['source_file'] = str(file_path)
                        doc.metadata['file_type'] = file_path.suffix
                    
                    documents.extend(docs)
                    
                except Exception as e:
                    print(f"Error loading file {file_path}: {str(e)}")
                    continue
        
        print(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _create_database_with_batches(self, chunks: List[Document]) -> Chroma:
        """Create database by processing chunks in batches to avoid token limits"""
        # Calculate batch size based on estimated tokens per chunk
        # Assuming average ~500 tokens per chunk, use batch size of 500 chunks (~250k tokens)
        batch_size = 500
        
        # Create empty database first
        vectordb = Chroma(
            persist_directory=str(self.database_dir),
            embedding_function=self.embeddings
        )
        
        # Process chunks in batches
        total_batches = (len(chunks) + batch_size - 1) // batch_size  # Ceiling division
        
        for i in range(0, len(chunks), batch_size):
            batch_num = (i // batch_size) + 1
            batch_chunks = chunks[i:i + batch_size]
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
            
            try:
                # Add batch to existing database
                vectordb.add_documents(batch_chunks)
                print(f"Successfully added batch {batch_num}")
                
            except Exception as e:
                if "max_tokens_per_request" in str(e):
                    print(f"Batch {batch_num} too large, splitting further...")
                    # Split this batch into smaller sub-batches
                    sub_batch_size = batch_size // 2
                    for j in range(0, len(batch_chunks), sub_batch_size):
                        sub_batch = batch_chunks[j:j + sub_batch_size]
                        try:
                            vectordb.add_documents(sub_batch)
                            print(f"Successfully added sub-batch ({len(sub_batch)} chunks)")
                        except Exception as sub_e:
                            print(f"Error processing sub-batch: {str(sub_e)}")
                            # Process chunks one by one as last resort
                            self._add_chunks_individually(vectordb, sub_batch)
                else:
                    print(f"Error processing batch {batch_num}: {str(e)}")
                    continue
        
        return vectordb
    
    def _add_chunks_individually(self, vectordb: Chroma, chunks: List[Document]):
        """Add chunks one by one as fallback for very large chunks"""
        for idx, chunk in enumerate(chunks):
            try:
                vectordb.add_documents([chunk])
                if (idx + 1) % 10 == 0:
                    print(f"Added {idx + 1}/{len(chunks)} individual chunks...")
            except Exception as e:
                print(f"Failed to add individual chunk {idx + 1}: {str(e)}")
                continue