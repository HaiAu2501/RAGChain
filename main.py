import hydra
from dotenv import load_dotenv
from src.builder import DatabaseBuilder
from src.workflow import create_workflow, initialize_state

load_dotenv()


@hydra.main(config_path="cfg", config_name="config", version_base="1.2")
def main(cfg):
    print("=== Multi-hop RAG System with Hybrid Retrieval ===")
    
    # Check if hybrid retrieval is enabled
    use_hybrid = cfg.tree.retrieval.use_hybrid
    if use_hybrid:
        print("🔍 Hybrid Retrieval Mode: Dense + Sparse (BM25)")
        print(f"   Fusion Method: {cfg.tree.retrieval.fusion_method}")
        print(f"   Dense Weight: {cfg.tree.retrieval.dense_weight}")
        print(f"   Sparse Weight: {cfg.tree.retrieval.sparse_weight}")
    else:
        print("🔍 Dense Retrieval Mode Only")
    
    # Step 1: Build or load vector database and BM25 index
    print("\n1. Setting up databases...")
    builder = DatabaseBuilder(cfg)
    
    try:
        result = builder.build_database()
        
        # Handle both tuple and single return values
        if isinstance(result, tuple):
            vectordb, bm25_index = result
        else:
            vectordb = result
            bm25_index = None
        
        if use_hybrid:
            # Check BM25 index status
            if bm25_index and bm25_index.bm25_index:
                print(f"✅ BM25 index ready with {len(bm25_index.doc_mapping)} documents")
            else:
                print("⚠️  BM25 index not available - falling back to dense retrieval only")
                bm25_index = None
        else:
            print("🔍 Dense retrieval mode only (hybrid disabled in config)")
            bm25_index = None
                
    except ImportError as e:
        print(f"⚠️  BM25 dependency error: {e}")
        print("⚠️  Install with: pip install rank-bm25")
        print("⚠️  Falling back to dense retrieval only")
        
        # Try to get just the vector database
        try:
            result = builder.build_database()
            vectordb = result[0] if isinstance(result, tuple) else result
            bm25_index = None
        except Exception as e2:
            print(f"❌ Error building vector database: {e2}")
            raise e2
            
    except Exception as e:
        print(f"❌ Error building databases: {e}")
        raise e
    
    # Step 2: Create workflow
    print("\n2. Initializing Multi-hop RAG workflow...")
    workflow = create_workflow(cfg, vectordb, bm25_index)
    
    # Step 3: Interactive question-answering loop
    print("\n3. Ready for questions!")
    print("Enter your questions (type 'quit' to exit):")
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not question:
                print("Please enter a valid question.")
                continue
            
            print(f"\nProcessing question: {question}")
            print("=" * 50)
            
            # Initialize state for this question
            initial_state = initialize_state(
                question=question,
                max_iterations=cfg.tree.hyperparams.n_iterations
            )
            
            # Run the workflow
            try:
                result = workflow.invoke(initial_state)
                
                # Display final answer
                print(f"\n🎯 FINAL ANSWER:")
                print("=" * 50)
                print(result.get("final_answer", "No answer generated"))
                print("=" * 50)
                
            except Exception as e:
                print(f"❌ Error processing question: {str(e)}")
                print("Please try a different question or check your configuration.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            continue

# What are the latest developments in artificial intelligence and how do they impact software development?

if __name__ == "__main__":
    main()