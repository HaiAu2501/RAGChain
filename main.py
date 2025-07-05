import hydra
from dotenv import load_dotenv
from src.builder import DatabaseBuilder
from src.workflow import create_workflow, initialize_state

load_dotenv()


@hydra.main(config_path="cfg", config_name="config", version_base="1.2")
def main(cfg):
    print("=== Hệ thống RAG đa bước với Tìm kiếm Hybrid ===")
    
    # Check if hybrid retrieval is enabled
    use_hybrid = cfg.tree.retrieval.use_hybrid
    if use_hybrid:
        print("🔍 Chế độ Tìm kiếm Hybrid: Dense + Sparse (BM25)")
        print(f"   Phương pháp Fusion: {cfg.tree.retrieval.fusion_method}")
        print(f"   Trọng số Dense: {cfg.tree.retrieval.dense_weight}")
        print(f"   Trọng số Sparse: {cfg.tree.retrieval.sparse_weight}")
    else:
        print("🔍 Chỉ sử dụng Chế độ Tìm kiếm Dense")
    
    # Step 1: Build or load vector database and BM25 index
    print("\n1. Đang thiết lập cơ sở dữ liệu...")
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
                print(f"✅ Chỉ mục BM25 sẵn sàng với {len(bm25_index.doc_mapping)} tài liệu")
            else:
                print("⚠️  Chỉ mục BM25 không khả dụng - quay lại chỉ sử dụng tìm kiếm dense")
                bm25_index = None
        else:
            print("🔍 Chỉ sử dụng chế độ tìm kiếm dense (hybrid đã tắt trong cấu hình)")
            bm25_index = None
                
    except ImportError as e:
        print(f"⚠️  Lỗi phụ thuộc BM25: {e}")
        print("⚠️  Cài đặt với: pip install rank-bm25")
        print("⚠️  Quay lại chỉ sử dụng tìm kiếm dense")
        
        # Try to get just the vector database
        try:
            result = builder.build_database()
            vectordb = result[0] if isinstance(result, tuple) else result
            bm25_index = None
        except Exception as e2:
            print(f"❌ Lỗi xây dựng cơ sở dữ liệu vector: {e2}")
            raise e2
            
    except Exception as e:
        print(f"❌ Lỗi xây dựng cơ sở dữ liệu: {e}")
        raise e
    
    # Step 2: Create workflow
    print("\n2. Đang khởi tạo quy trình RAG đa bước...")
    workflow = create_workflow(cfg, vectordb, bm25_index)
    
    # Step 3: Interactive question-answering loop
    print("\n3. Sẵn sàng cho các câu hỏi!")
    print("Nhập câu hỏi của bạn (gõ 'quit' để thoát):")
    
    while True:
        try:
            question = input("\nCâu hỏi: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', 'thoát', 'kết thúc']:
                print("Tạm biệt!")
                break
                
            if not question:
                print("Vui lòng nhập một câu hỏi hợp lệ.")
                continue
            
            print(f"\nĐang xử lý câu hỏi: {question}")
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
                print(f"\n🎯 CÂU TRẢ LỜI CUỐI CÙNG:")
                print("=" * 50)
                print(result.get("final_answer", "Không có câu trả lời được tạo"))
                print("=" * 50)
                
            except Exception as e:
                print(f"❌ Lỗi xử lý câu hỏi: {str(e)}")
                print("Vui lòng thử một câu hỏi khác hoặc kiểm tra cấu hình của bạn.")
        
        except KeyboardInterrupt:
            print("\n\nBị gián đoạn bởi người dùng. Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi không mong muốn: {str(e)}")
            continue

# Những phát triển mới nhất trong trí tuệ nhân tạo là gì và chúng tác động như thế nào đến phát triển phần mềm?

if __name__ == "__main__":
    main()