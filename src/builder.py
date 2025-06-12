import os
from pathlib import Path
from typing import List, Set
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
        
        # File to track processed documents
        self.processed_files_path = self.database_dir / "processed_files.txt"
        
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
            print(f"Database already exists at {self.database_dir}")
            return self._update_database()
        
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
    
    def _get_processed_files(self) -> Set[str]:
        """Get set of already processed files from tracking file"""
        processed_files = set()
        
        if self.processed_files_path.exists():
            try:
                with open(self.processed_files_path, 'r', encoding='utf-8') as f:
                    processed_files = set(line.strip() for line in f if line.strip())
            except Exception as e:
                print(f"Error reading processed files list: {str(e)}")
                
        return processed_files
    
    def _save_processed_files(self, processed_files: Set[str]):
        """Save the list of processed files to tracking file"""
        try:
            # Ensure database directory exists
            self.database_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.processed_files_path, 'w', encoding='utf-8') as f:
                for file_path in sorted(processed_files):
                    f.write(f"{file_path}\n")
        except Exception as e:
            print(f"Error saving processed files list: {str(e)}")
    
    def _get_all_data_files(self) -> List[Path]:
        """Get all data files from data directory"""
        all_files = []
        
        # Define file patterns
        file_patterns = ["*.txt", "*.md", "*.pdf", "*.csv", "*.json", "*.docx", "*.doc"]
        
        if not self.data_dir.exists():
            return all_files
        
        # Iterate through all file patterns
        for pattern in file_patterns:
            # Use rglob to find all files (including root directory and subdirectories)
            files = list(self.data_dir.rglob(pattern))
            all_files.extend(files)
        
        # Remove duplicates and return sorted list
        return list(set(all_files))
    
    def _update_database(self) -> Chroma:
        """Update existing database with new documents"""
        # Load existing database
        vectordb = self._load_existing_database()
        
        # Get processed files
        processed_files = self._get_processed_files()
        
        # Get all current files
        all_files = self._get_all_data_files()
        
        # Find new files (files not in processed list)
        new_files = []
        for file_path in all_files:
            file_str = str(file_path)
            if file_str not in processed_files:
                new_files.append(file_path)
        
        if not new_files:
            print("No new files to process. Database is up to date.")
            return vectordb
        
        print(f"Found {len(new_files)} new files to process:")
        for file_path in new_files:
            print(f"  - {file_path}")
        
        # Load new documents
        new_documents = self._load_specific_documents(new_files)
        
        if new_documents:
            print(f"Processing {len(new_documents)} new documents...")
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(new_documents)
            print(f"Split into {len(chunks)} chunks for embedding...")
            
            # Add new chunks to existing database
            self._add_chunks_to_database(vectordb, chunks)
            
            # Update processed files list
            processed_files.update(str(f) for f in new_files)
            self._save_processed_files(processed_files)
            
            print(f"Successfully added {len(new_documents)} new documents to database!")
        else:
            print("No valid documents found in new files.")
            
            # Still update processed files list to avoid re-checking failed files
            processed_files.update(str(f) for f in new_files)
            self._save_processed_files(processed_files)
        
        return vectordb
    
    def _create_new_database(self) -> Chroma:
        """Create new database"""
        # Create database directory if not exists
        self.database_dir.mkdir(parents=True, exist_ok=True)
        
        # Check data directory
        if not self.data_dir.exists():
            print(f"Data directory {self.data_dir} does not exist, initializing empty database.")
            documents = []
            all_files = []
        else:
            print(f"Loading documents from {self.data_dir}")
            all_files = self._get_all_data_files()
            documents = self._load_specific_documents(all_files)
        
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
        
        # Save processed files list
        processed_files = set(str(f) for f in all_files)
        self._save_processed_files(processed_files)
        
        print("Database created successfully!")
        return vectordb
    
    def _load_documents(self) -> List[Document]:
        """Load all documents from data directory (kept for compatibility)"""
        all_files = self._get_all_data_files()
        return self._load_specific_documents(all_files)
    
    def _load_specific_documents(self, file_paths: List[Path]) -> List[Document]:
        """Load documents from specific file paths"""
        documents = []
        
        # Define file patterns and corresponding loaders
        loader_mapping = {
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".pdf": PyPDFLoader,
            ".csv": CSVLoader,
            ".json": JSONLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".doc": UnstructuredWordDocumentLoader
        }
        
        for file_path in file_paths:
            try:
                file_extension = file_path.suffix.lower()
                loader_class = loader_mapping.get(file_extension)
                
                if not loader_class:
                    print(f"Skipping unsupported file type: {file_path}")
                    continue
                
                print(f"Loading: {file_path}")
                
                # Special handling for JSON files
                if file_extension == ".json":
                    loader = loader_class(
                        file_path=str(file_path),
                        jq_schema='.', 
                        text_content=False
                    )
                # Special handling for CSV files
                elif file_extension == ".csv":
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
        
        print(f"Successfully loaded {len(documents)} documents from {len(file_paths)} files")
        return documents
    
    def _add_chunks_to_database(self, vectordb: Chroma, chunks: List[Document]):
        """Add chunks to existing database with batch processing"""
        if not chunks:
            return
        
        # Calculate batch size based on estimated tokens per chunk
        batch_size = 500
        
        # Process chunks in batches
        total_batches = (len(chunks) + batch_size - 1) // batch_size  # Ceiling division
        
        for i in range(0, len(chunks), batch_size):
            batch_num = (i // batch_size) + 1
            batch_chunks = chunks[i:i + batch_size]
            
            print(f"Adding batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
            
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
    
    def _create_database_with_batches(self, chunks: List[Document]) -> Chroma:
        """Create database by processing chunks in batches to avoid token limits"""
        # Create empty database first
        vectordb = Chroma(
            persist_directory=str(self.database_dir),
            embedding_function=self.embeddings
        )
        
        # Add chunks to database
        self._add_chunks_to_database(vectordb, chunks)
        
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