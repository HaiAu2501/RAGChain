import hydra
from dotenv import load_dotenv
from src.builder import DatabaseBuilder
from src.workflow import create_workflow, initialize_state

load_dotenv()


@hydra.main(config_path="cfg", config_name="config", version_base="1.2")
def main(cfg):
    print("=== Multi-hop RAG System ===")
    
    # Step 1: Build or load vector database
    print("\n1. Setting up vector database...")
    builder = DatabaseBuilder(cfg)
    vectordb = builder.build_database()
    
    # Step 2: Create workflow
    print("\n2. Initializing Multi-hop RAG workflow...")
    workflow = create_workflow(cfg, vectordb)
    
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