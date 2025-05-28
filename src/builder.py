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
        
        # Create database directory
        self.database_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(
            api_key=cfg.llm.api_key,
            model=cfg.llm.embeddings.model,
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.llm.embeddings.chunk_size,
            chunk_overlap=cfg.llm.embeddings.chunk_overlap,
        )
        