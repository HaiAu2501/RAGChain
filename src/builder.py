import os
import re
import pickle
from pathlib import Path
from typing import List, Set, Dict, Any
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

# BM25 implementation
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Cảnh báo: rank_bm25 không tìm thấy. Cài đặt với: pip install rank-bm25")
    BM25Okapi = None


class HybridSearchIndex:
    """Container for BM25 index and document mapping"""
    
    def __init__(self, bm25_index=None, doc_mapping=None, tokenized_corpus=None):
        self.bm25_index = bm25_index
        self.doc_mapping = doc_mapping or {}  # index -> document mapping
        self.tokenized_corpus = tokenized_corpus or []
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search BM25 index and return results with scores"""
        if not self.bm25_index:
            return []
        
        query_tokens = self.tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top k results
        top_indices = sorted(range(len(scores)), 
                           key=lambda i: scores[i], 
                           reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            if idx in self.doc_mapping:
                results.append({
                    'document': self.doc_mapping[idx],
                    'score': scores[idx],
                    'index': idx
                })
        
        return results
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization for BM25"""
        if not text:
            return []
        
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split and filter empty tokens
        tokens = [token for token in text.split() if token.strip()]
        
        return tokens


class DatabaseBuilder:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.paths.project_root)
        self.data_dir = Path(cfg.paths.data_dir)
        self.database_dir = Path(cfg.paths.database_dir)
        
        # File to track processed documents
        self.processed_files_path = self.database_dir / "processed_files.txt"
        self.bm25_index_path = self.database_dir / "bm25_index.pkl"
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(
            api_key=cfg.llm.api_key,
            model=cfg.llm.embeddings.model,
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.llm.embeddings.chunk_size,
            chunk_overlap=cfg.llm.embeddings.chunk_overlap,
        )
        
        # BM25 settings
        self.use_hybrid = cfg.tree.retrieval.use_hybrid
        self.bm25_k1 = cfg.tree.retrieval.bm25_k1
        self.bm25_b = cfg.tree.retrieval.bm25_b
    
    def build_database(self) -> tuple:
        """
        Build both vector database and BM25 index.
        Always returns tuple: (vectordb, bm25_index)
        bm25_index will be None if hybrid is disabled or BM25Okapi is not available
        """
        # Check if database already exists
        if self._database_exists():
            print(f"Cơ sở dữ liệu đã tồn tại tại {self.database_dir}")
            return self._update_database()
        
        print(f"Khởi tạo cơ sở dữ liệu mới tại {self.database_dir}")
        return self._create_new_database()
    
    def _database_exists(self) -> bool:
        """Check if database directory exists and has content"""
        if not self.database_dir.exists():
            return False
        
        # Check if there are any files in database directory
        return any(self.database_dir.iterdir())
    
    def _load_existing_database(self) -> tuple:
        """Load existing vector database and BM25 index. Always returns tuple."""
        vectordb = Chroma(
            persist_directory=str(self.database_dir),
            embedding_function=self.embeddings
        )
        
        bm25_index = self._load_bm25_index()
        
        return vectordb, bm25_index
    
    def _load_bm25_index(self) -> HybridSearchIndex:
        """Load BM25 index from disk"""
        if not self.use_hybrid or BM25Okapi is None:
            return HybridSearchIndex()
        
        if self.bm25_index_path.exists():
            try:
                with open(self.bm25_index_path, 'rb') as f:
                    data = pickle.load(f)
                
                print(f"Đã tải chỉ mục BM25 với {len(data.get('doc_mapping', {}))} tài liệu")
                
                # Reconstruct BM25 index
                tokenized_corpus = data.get('tokenized_corpus', [])
                if tokenized_corpus:
                    bm25_index = BM25Okapi(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)
                else:
                    bm25_index = None
                
                return HybridSearchIndex(
                    bm25_index=bm25_index,
                    doc_mapping=data.get('doc_mapping', {}),
                    tokenized_corpus=tokenized_corpus
                )
                
            except Exception as e:
                print(f"Lỗi tải chỉ mục BM25: {str(e)}")
                return HybridSearchIndex()
        
        return HybridSearchIndex()
    
    def _save_bm25_index(self, bm25_index: HybridSearchIndex):
        """Save BM25 index to disk"""
        if not self.use_hybrid or not bm25_index.bm25_index:
            return
        
        try:
            # Prepare data for serialization
            data = {
                'doc_mapping': bm25_index.doc_mapping,
                'tokenized_corpus': bm25_index.tokenized_corpus
            }
            
            # Ensure database directory exists
            self.database_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Đã lưu chỉ mục BM25 với {len(bm25_index.doc_mapping)} tài liệu")
            
        except Exception as e:
            print(f"Lỗi lưu chỉ mục BM25: {str(e)}")
    
    def _create_bm25_index(self, documents: List[Document]) -> HybridSearchIndex:
        """Create BM25 index from documents"""
        if not self.use_hybrid or BM25Okapi is None or not documents:
            return HybridSearchIndex()
        
        print("Đang tạo chỉ mục BM25...")
        
        tokenized_corpus = []
        doc_mapping = {}
        
        for idx, doc in enumerate(documents):
            # Tokenize document content
            tokens = HybridSearchIndex.tokenize(doc.page_content)
            tokenized_corpus.append(tokens)
            doc_mapping[idx] = doc
        
        # Create BM25 index
        bm25_index = BM25Okapi(tokenized_corpus, k1=self.bm25_k1, b=self.bm25_b)
        
        hybrid_index = HybridSearchIndex(
            bm25_index=bm25_index,
            doc_mapping=doc_mapping,
            tokenized_corpus=tokenized_corpus
        )
        
        print(f"Đã tạo chỉ mục BM25 với {len(doc_mapping)} tài liệu")
        
        return hybrid_index
    
    def _update_bm25_index(self, existing_index: HybridSearchIndex, new_documents: List[Document]) -> HybridSearchIndex:
        """Update BM25 index with new documents"""
        if not self.use_hybrid or BM25Okapi is None:
            return existing_index
        
        if not new_documents:
            return existing_index
        
        print("Đang cập nhật chỉ mục BM25...")
        
        # Get existing data
        existing_corpus = existing_index.tokenized_corpus.copy()
        existing_mapping = existing_index.doc_mapping.copy()
        
        # Add new documents
        start_idx = len(existing_mapping)
        
        for idx, doc in enumerate(new_documents):
            tokens = HybridSearchIndex.tokenize(doc.page_content)
            existing_corpus.append(tokens)
            existing_mapping[start_idx + idx] = doc
        
        # Rebuild BM25 index with all documents
        bm25_index = BM25Okapi(existing_corpus, k1=self.bm25_k1, b=self.bm25_b)
        
        updated_index = HybridSearchIndex(
            bm25_index=bm25_index,
            doc_mapping=existing_mapping,
            tokenized_corpus=existing_corpus
        )
        
        print(f"Đã cập nhật chỉ mục BM25: {len(existing_mapping)} tổng số tài liệu")
        
        return updated_index
    
    def _get_processed_files(self) -> Set[str]:
        """Get set of already processed files from tracking file"""
        processed_files = set()
        
        if self.processed_files_path.exists():
            try:
                with open(self.processed_files_path, 'r', encoding='utf-8') as f:
                    processed_files = set(line.strip() for line in f if line.strip())
            except Exception as e:
                print(f"Lỗi đọc danh sách tệp đã xử lý: {str(e)}")
                
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
            print(f"Lỗi lưu danh sách tệp đã xử lý: {str(e)}")
    
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
    
    def _update_database(self) -> tuple:
        """Update existing database with new documents"""
        # Load existing databases
        vectordb, bm25_index = self._load_existing_database()
        
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
            print("Không có tệp mới để xử lý. Cơ sở dữ liệu đã cập nhật.")
            return vectordb, bm25_index
        
        print(f"Tìm thấy {len(new_files)} tệp mới để xử lý:")
        for file_path in new_files:
            print(f"  - {file_path}")
        
        # Load new documents
        new_documents = self._load_specific_documents(new_files)
        
        if new_documents:
            print(f"Đang xử lý {len(new_documents)} tài liệu mới...")
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(new_documents)
            print(f"Đã chia thành {len(chunks)} đoạn để embedding...")
            
            # Add new chunks to existing vector database
            self._add_chunks_to_database(vectordb, chunks)
            
            # Update BM25 index
            updated_bm25_index = self._update_bm25_index(bm25_index, chunks)
            self._save_bm25_index(updated_bm25_index)
            
            # Update processed files list
            processed_files.update(str(f) for f in new_files)
            self._save_processed_files(processed_files)
            
            print(f"Đã thêm thành công {len(new_documents)} tài liệu mới vào cơ sở dữ liệu!")
            
            return vectordb, updated_bm25_index
        else:
            print("Không tìm thấy tài liệu hợp lệ trong các tệp mới.")
            
            # Still update processed files list to avoid re-checking failed files
            processed_files.update(str(f) for f in new_files)
            self._save_processed_files(processed_files)
        
        return vectordb, bm25_index
    
    def _create_new_database(self) -> tuple:
        """Create new database. Always returns tuple (vectordb, bm25_index)."""
        # Create database directory if not exists
        self.database_dir.mkdir(parents=True, exist_ok=True)
        
        # Check data directory
        if not self.data_dir.exists():
            print(f"Thư mục dữ liệu {self.data_dir} không tồn tại, khởi tạo cơ sở dữ liệu trống.")
            documents = []
            all_files = []
        else:
            print(f"Đang tải tài liệu từ {self.data_dir}")
            all_files = self._get_all_data_files()
            documents = self._load_specific_documents(all_files)
        
        # Create databases with documents (can be empty)
        if documents:
            print(f"Đang xử lý {len(documents)} tài liệu...")
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            print(f"Đã chia thành {len(chunks)} đoạn để embedding...")
            
            # Create vector database with batch processing
            vectordb = self._create_database_with_batches(chunks)
            
            # Create BM25 index
            bm25_index = self._create_bm25_index(chunks)
            self._save_bm25_index(bm25_index)
        else:
            # Create empty databases
            vectordb = Chroma(
                persist_directory=str(self.database_dir),
                embedding_function=self.embeddings
            )
            bm25_index = HybridSearchIndex()  # Empty index
        
        # Save processed files list
        processed_files = set(str(f) for f in all_files)
        self._save_processed_files(processed_files)
        
        print("Cơ sở dữ liệu đã được tạo thành công!")
        return vectordb, bm25_index
    
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
                    print(f"Bỏ qua loại tệp không được hỗ trợ: {file_path}")
                    continue
                
                print(f"Đang tải: {file_path}")
                
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
                print(f"Lỗi tải tệp {file_path}: {str(e)}")
                continue
        
        print(f"Đã tải thành công {len(documents)} tài liệu từ {len(file_paths)} tệp")
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
            
            print(f"Đang thêm lô {batch_num}/{total_batches} ({len(batch_chunks)} đoạn)...")
            
            try:
                # Add batch to existing database
                vectordb.add_documents(batch_chunks)
                print(f"Đã thêm thành công lô {batch_num}")
                
            except Exception as e:
                if "max_tokens_per_request" in str(e):
                    print(f"Lô {batch_num} quá lớn, đang chia nhỏ hơn...")
                    # Split this batch into smaller sub-batches
                    sub_batch_size = batch_size // 2
                    for j in range(0, len(batch_chunks), sub_batch_size):
                        sub_batch = batch_chunks[j:j + sub_batch_size]
                        try:
                            vectordb.add_documents(sub_batch)
                            print(f"Đã thêm thành công lô con ({len(sub_batch)} đoạn)")
                        except Exception as sub_e:
                            print(f"Lỗi xử lý lô con: {str(sub_e)}")
                            # Process chunks one by one as last resort
                            self._add_chunks_individually(vectordb, sub_batch)
                else:
                    print(f"Lỗi xử lý lô {batch_num}: {str(e)}")
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
                    print(f"Đã thêm {idx + 1}/{len(chunks)} đoạn riêng lẻ...")
            except Exception as e:
                print(f"Thất bại khi thêm đoạn riêng lẻ {idx + 1}: {str(e)}")
                continue