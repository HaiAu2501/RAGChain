import os
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from src.builder import DatabaseBuilder
from src.workflow import create_workflow, initialize_state

load_dotenv()

# Initialize workflow at module level for langgraph dev
def _initialize_workflow():
    """Initialize workflow with default config for langgraph dev"""
    # Load config manually without hydra decorator
    from hydra import initialize, compose
    
    try:
        with initialize(config_path="cfg", version_base="1.2"):
            cfg = compose(config_name="config")
            
            # Build vector database
            print("Setting up vector database...")
            builder = DatabaseBuilder(cfg)
            result = builder.build_database()

            if isinstance(result, tuple):
                vectordb, bm25_index = result
            else:
                vectordb = result
                bm25_index = None
            
            # Create workflow
            print("Initializing Multi-hop RAG workflow...")
            workflow = create_workflow(cfg, vectordb, bm25_index)
            
            return workflow
    except Exception as e:
        print(f"Error initializing workflow: {e}")
        return None

# Create workflow instance for langgraph dev
WORKFLOW = _initialize_workflow()