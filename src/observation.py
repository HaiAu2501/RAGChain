from src.state import OverallState, QueryResult
from typing import Dict, Any, List, Tuple
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import Document


# Global variables (will be set in main.py)
VECTOR_DB = None
BM25_INDEX = None
N_RESULTS = None
SEARCH_WRAPPER = None
USE_HYBRID = None
DENSE_WEIGHT = None
SPARSE_WEIGHT = None
FUSION_METHOD = None
RRF_K = None


def initialize_observation(cfg, vectordb, bm25_index=None):
    """Initialize observation node with configuration, vector database and BM25 index"""
    global VECTOR_DB, BM25_INDEX, N_RESULTS, SEARCH_WRAPPER
    global USE_HYBRID, DENSE_WEIGHT, SPARSE_WEIGHT, FUSION_METHOD, RRF_K
    
    VECTOR_DB = vectordb
    BM25_INDEX = bm25_index
    N_RESULTS = cfg.tree.hyperparams.n_results
    
    # Initialize DuckDuckGo search with rate limit protection
    try:
        import random
        import time
        
        # Add small random delay to avoid synchronized requests
        time.sleep(random.uniform(0.5, 2.0))
        
        # Try with minimal settings first to avoid rate limits
        SEARCH_WRAPPER = DuckDuckGoSearchAPIWrapper(
            max_results=min(N_RESULTS, 3),  # Limit results to reduce load
            backend="lite"  # Use lighter backend
        )
        
        print("✅ Đã khởi tạo DuckDuckGo search wrapper (chế độ lite)")
        print("💡 Tip: Để tránh rate limit, hệ thống sẽ:")
        print("   - Sử dụng backend lite")
        print("   - Giới hạn số kết quả web")
        print("   - Có delay ngẫu nhiên giữa các requests")
        print("   - Ưu tiên cơ sở dữ liệu nội bộ")
        
    except Exception as e:
        print(f"⚠️ Lỗi khởi tạo DuckDuckGo search wrapper: {e}")
        print("⚠️ Tìm kiếm web có thể không hoạt động ổn định")
        try:
            # Fallback to basic initialization
            SEARCH_WRAPPER = DuckDuckGoSearchAPIWrapper()
            print("🔄 Đã fallback sang cấu hình cơ bản")
        except:
            SEARCH_WRAPPER = None
            print("❌ Không thể khởi tạo web search, chỉ sử dụng cơ sở dữ liệu")
    
    # Hybrid retrieval settings
    USE_HYBRID = cfg.tree.retrieval.use_hybrid
    DENSE_WEIGHT = cfg.tree.retrieval.dense_weight
    SPARSE_WEIGHT = cfg.tree.retrieval.sparse_weight
    FUSION_METHOD = cfg.tree.retrieval.fusion_method
    RRF_K = cfg.tree.retrieval.rrf_k


def reciprocal_rank_fusion(dense_results: List[Document], sparse_results: List[Dict], k: int = 60) -> List[Document]:
    """Combine dense and sparse results using Reciprocal Rank Fusion"""
    score_dict = {}
    doc_dict = {}
    
    # Process dense results (vector similarity search)
    for rank, doc in enumerate(dense_results):
        # Create unique document ID based on content hash
        doc_id = str(hash(doc.page_content))
        score_dict[doc_id] = score_dict.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_dict[doc_id] = doc
    
    # Process sparse results (BM25)
    for rank, result in enumerate(sparse_results):
        doc = result['document']
        doc_id = str(hash(doc.page_content))
        score_dict[doc_id] = score_dict.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_dict[doc_id] = doc
    
    # Sort by combined score and return documents
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N_RESULTS documents
    result_docs = []
    for doc_id, score in sorted_items[:N_RESULTS]:
        doc = doc_dict[doc_id]
        # Add fusion score to metadata
        doc.metadata['fusion_score'] = score
        result_docs.append(doc)
    
    return result_docs


def weighted_fusion(dense_results: List[Document], sparse_results: List[Dict], 
                   dense_weight: float, sparse_weight: float) -> List[Document]:
    """Combine dense and sparse results using weighted scoring"""
    score_dict = {}
    doc_dict = {}
    
    # Normalize dense scores (similarity scores are already 0-1)
    max_dense_score = 1.0
    for rank, doc in enumerate(dense_results):
        doc_id = str(hash(doc.page_content))
        # Use inverse rank as score proxy (higher rank = lower score)
        dense_score = (len(dense_results) - rank) / len(dense_results)
        score_dict[doc_id] = dense_weight * dense_score
        doc_dict[doc_id] = doc
    
    # Normalize sparse scores
    if sparse_results:
        max_sparse_score = max(result['score'] for result in sparse_results)
        if max_sparse_score > 0:
            for result in sparse_results:
                doc = result['document']
                doc_id = str(hash(doc.page_content))
                sparse_score = result['score'] / max_sparse_score
                score_dict[doc_id] = score_dict.get(doc_id, 0) + sparse_weight * sparse_score
                doc_dict[doc_id] = doc
    
    # Sort by combined score and return documents
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N_RESULTS documents
    result_docs = []
    for doc_id, score in sorted_items[:N_RESULTS]:
        doc = doc_dict[doc_id]
        # Add fusion score to metadata
        doc.metadata['fusion_score'] = score
        result_docs.append(doc)
    
    return result_docs


def parallel_fusion(dense_results: List[Document], sparse_results: List[Dict]) -> List[Document]:
    """Simple parallel fusion - take top results from both methods"""
    result_docs = []
    seen_docs = set()
    
    # Take half from dense results
    dense_count = N_RESULTS // 2
    for doc in dense_results[:dense_count]:
        doc_id = str(hash(doc.page_content))
        if doc_id not in seen_docs:
            doc.metadata['retrieval_method'] = 'dense'
            result_docs.append(doc)
            seen_docs.add(doc_id)
    
    # Take remaining from sparse results
    sparse_count = N_RESULTS - len(result_docs)
    for result in sparse_results[:sparse_count * 2]:  # Check more to fill remaining slots
        doc = result['document']
        doc_id = str(hash(doc.page_content))
        if doc_id not in seen_docs and len(result_docs) < N_RESULTS:
            doc.metadata['retrieval_method'] = 'sparse'
            doc.metadata['bm25_score'] = result['score']
            result_docs.append(doc)
            seen_docs.add(doc_id)
    
    return result_docs


def search_vector_database_hybrid(query: str) -> List[str]:
    """Hybrid search combining dense and sparse retrieval"""
    try:
        if VECTOR_DB is None:
            return ["Cơ sở dữ liệu vector không khả dụng"]
        
        # Always perform dense retrieval
        dense_results = VECTOR_DB.similarity_search(query, k=N_RESULTS)
        
        # Perform sparse retrieval if hybrid is enabled and BM25 index is available
        sparse_results = []
        if USE_HYBRID and BM25_INDEX and BM25_INDEX.bm25_index:
            sparse_results = BM25_INDEX.search(query, k=N_RESULTS)
        
        # Combine results based on fusion method
        if USE_HYBRID and sparse_results:
            if FUSION_METHOD == "rrf":
                final_results = reciprocal_rank_fusion(dense_results, sparse_results, RRF_K)
            elif FUSION_METHOD == "weighted":
                final_results = weighted_fusion(dense_results, sparse_results, DENSE_WEIGHT, SPARSE_WEIGHT)
            elif FUSION_METHOD == "parallel":
                final_results = parallel_fusion(dense_results, sparse_results)
            else:
                # Fallback to dense only
                final_results = dense_results[:N_RESULTS]
        else:
            # Use dense results only
            final_results = dense_results[:N_RESULTS]
        
        # Format results for output
        content_results = []
        for doc in final_results:
            content = doc.page_content.strip()
            if content:
                # Add source information if available
                source = doc.metadata.get('source_file', 'Nguồn không xác định')
                
                # Add retrieval information
                retrieval_info = []
                if 'fusion_score' in doc.metadata:
                    retrieval_info.append(f"Điểm Fusion: {doc.metadata['fusion_score']:.3f}")
                if 'retrieval_method' in doc.metadata:
                    retrieval_info.append(f"Phương pháp: {doc.metadata['retrieval_method']}")
                if 'bm25_score' in doc.metadata:
                    retrieval_info.append(f"Điểm BM25: {doc.metadata['bm25_score']:.3f}")
                
                retrieval_str = f" ({', '.join(retrieval_info)})" if retrieval_info else ""
                content_results.append(f"Nguồn: {source}{retrieval_str}\nNội dung: {content}")
        
        if not content_results:
            return ["Không tìm thấy thông tin liên quan trong cơ sở dữ liệu"]
        
        return content_results
        
    except Exception as e:
        return [f"Lỗi tìm kiếm cơ sở dữ liệu: {str(e)}"]


def search_vector_database(query: str) -> List[str]:
    """Legacy function for backward compatibility - now uses hybrid search"""
    return search_vector_database_hybrid(query)


def search_web(query: str) -> List[str]:
    """Search the web using DuckDuckGo through LangChain with comprehensive error handling"""
    import time
    import random
    
    try:
        if SEARCH_WRAPPER is None:
            return ["Tìm kiếm web không khả dụng - trình tìm kiếm chưa được khởi tạo"]
        
        # Add timeout handling with retry mechanism
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                print(f"Đang thử tìm kiếm web (lần {attempt + 1}/{max_retries}): {query}")
                
                # Use DuckDuckGo search with specified max_results
                search_results = SEARCH_WRAPPER.results(query, max_results=N_RESULTS)
                
                formatted_results = []
                
                for result in search_results:
                    # Extract information from each result dict
                    title = result.get('title', 'Không có tiêu đề')
                    snippet = result.get('snippet', 'Không có tóm tắt')
                    link = result.get('link', 'Không có liên kết')
                    
                    # Format the result for better readability
                    formatted_result = f"Tiêu đề: {title}\nTóm tắt: {snippet}\nLiên kết: {link}"
                    formatted_results.append(formatted_result)
                
                if not formatted_results:
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (attempt + 1) + random.randint(1, 3)
                        print(f"Không có kết quả, thử lại sau {wait_time} giây...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return [f"Không tìm thấy kết quả tìm kiếm web cho truy vấn: '{query}' sau {max_retries} lần thử"]
                
                print(f"Tìm kiếm web thành công: {len(formatted_results)} kết quả")
                return formatted_results
                
            except Exception as attempt_error:
                error_msg = str(attempt_error).lower()
                
                # Handle different types of errors
                if "ratelimit" in error_msg or "rate limit" in error_msg or "202" in str(attempt_error):
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter for rate limits
                        wait_time = (base_delay * (2 ** attempt)) + random.randint(0, 3)
                        print(f"⚠️ Rate limit detected (lần {attempt + 1}): {attempt_error}")
                        print(f"Chờ {wait_time} giây để tránh rate limit...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return [
                            f"🚫 DuckDuckGo đã giới hạn tần suất truy cập cho truy vấn: '{query}'",
                            "Điều này thường xảy ra khi có quá nhiều requests trong thời gian ngắn.",
                            "💡 Giải pháp:",
                            "- Hệ thống sẽ dựa vào cơ sở dữ liệu nội bộ",
                            "- Thử lại sau vài phút",
                            "- Hoặc sử dụng từ khóa khác ngắn gọn hơn",
                            f"Chi tiết: {attempt_error}"
                        ]
                
                elif any(keyword in error_msg for keyword in ['timeout', 'timed out', 'connection', 'network']):
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (attempt + 1) + random.randint(2, 5)
                        print(f"🌐 Lỗi kết nối (lần {attempt + 1}): {attempt_error}")
                        print(f"Thử lại sau {wait_time} giây...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return [
                            f"🌐 Lỗi kết nối mạng cho truy vấn: '{query}' sau {max_retries} lần thử.",
                            "Có thể do:",
                            "- Mạng internet không ổn định",
                            "- DuckDuckGo tạm thời không khả dụng",
                            "- Firewall/proxy chặn kết nối",
                            "💡 Hệ thống sẽ chỉ sử dụng thông tin từ cơ sở dữ liệu nội bộ."
                        ]
                
                elif "blocked" in error_msg or "forbidden" in error_msg or "403" in str(attempt_error):
                    return [
                        f"🚫 Truy cập bị chặn cho truy vấn: '{query}'",
                        "DuckDuckGo có thể đã chặn IP này do quá nhiều requests.",
                        "💡 Giải pháp:",
                        "- Đợi 15-30 phút trước khi thử lại",
                        "- Sử dụng VPN nếu cần",
                        "- Hệ thống sẽ dựa vào cơ sở dữ liệu nội bộ"
                    ]
                
                else:
                    # Other errors, don't retry immediately
                    if attempt < max_retries - 1:
                        wait_time = base_delay + random.randint(2, 5)
                        print(f"❌ Lỗi khác (lần {attempt + 1}): {attempt_error}")
                        print(f"Thử lại sau {wait_time} giây...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return [f"❌ Lỗi tìm kiếm web cho '{query}': {attempt_error}"]
        
        # If we get here, all retries failed
        return [
            f"❌ Không thể thực hiện tìm kiếm web cho '{query}' sau {max_retries} lần thử.",
            "💡 Hệ thống sẽ tiếp tục với thông tin từ cơ sở dữ liệu nội bộ.",
            "Thử lại sau ít phút hoặc sử dụng từ khóa khác."
        ]
        
    except Exception as e:
        return [
            f"💥 Lỗi nghiêm trọng khi tìm kiếm web cho '{query}': {str(e)}",
            "🔄 Hệ thống sẽ chỉ sử dụng thông tin từ cơ sở dữ liệu nội bộ."
        ]


def observation_node(state: OverallState) -> Dict[str, Any]:
    """
    Execute database and web search queries to collect information
    This node does not use LLM - it only executes searches
    """
    import time
    import random
    
    database_queries = state.get("current_database_queries", [])
    web_queries = state.get("current_web_queries", [])
    
    query_results = []
    
    # Execute database queries (now using hybrid search)
    for query in database_queries:
        if query.strip():  # Skip empty queries
            try:
                results = search_vector_database_hybrid(query)
                query_results.append(QueryResult(
                    query=query,
                    results=results,
                    source="database"
                ))
            except Exception as e:
                query_results.append(QueryResult(
                    query=query,
                    results=[f"Truy vấn cơ sở dữ liệu thất bại: {str(e)}"],
                    source="database"
                ))
    
    # Execute web search queries with rate limit awareness
    web_search_count = 0
    max_web_searches = 3  # Limit concurrent web searches
    
    for query in web_queries:
        if query.strip():  # Skip empty queries
            web_search_count += 1
            
            # Add delay between web searches to avoid rate limits
            if web_search_count > 1:
                delay = random.uniform(3, 8)  # Random delay 3-8 seconds
                print(f"⏱️ Chờ {delay:.1f}s giữa các web search để tránh rate limit...")
                time.sleep(delay)
            
            # Skip excessive web searches to prevent rate limiting
            if web_search_count > max_web_searches:
                query_results.append(QueryResult(
                    query=query,
                    results=[
                        f"⏭️ Bỏ qua web search cho '{query}' để tránh rate limit",
                        f"Đã thực hiện {max_web_searches} web searches, ưu tiên cơ sở dữ liệu nội bộ",
                        "💡 Thông tin từ database thường đủ để trả lời câu hỏi"
                    ],
                    source="web"
                ))
                continue
            
            try:
                results = search_web(query)
                query_results.append(QueryResult(
                    query=query,
                    results=results,
                    source="web"
                ))
            except Exception as e:
                error_msg = f"Tìm kiếm web thất bại: {str(e)}"
                if any(keyword in str(e).lower() for keyword in ["ratelimit", "rate limit", "202", "blocked", "forbidden"]):
                    error_msg = f"🚫 Web search bị giới hạn: {query}. Chỉ sử dụng kết quả từ cơ sở dữ liệu."
                elif "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    error_msg = f"⏱️ Web search bị timeout: {query}. Chỉ sử dụng kết quả từ cơ sở dữ liệu."
                
                query_results.append(QueryResult(
                    query=query,
                    results=[error_msg],
                    source="web"
                ))

    print("\n[QUAN SÁT] Đang thực hiện truy vấn cơ sở dữ liệu và tìm kiếm web...")
    
    # Print retrieval method info
    if USE_HYBRID and BM25_INDEX and BM25_INDEX.bm25_index:
        print(f"🔍 Sử dụng tìm kiếm hybrid: fusion {FUSION_METHOD} (Dense: {DENSE_WEIGHT}, Sparse: {SPARSE_WEIGHT})")
    else:
        print("🔍 Chỉ sử dụng tìm kiếm dense")

    # Count successful vs failed searches
    db_success = sum(1 for r in query_results if r.source == "database" and not any("thất bại" in res for res in r.results))
    web_success = sum(1 for r in query_results if r.source == "web" and not any(keyword in res for res in r.results for keyword in ["thất bại", "timeout", "rate limit", "Lỗi", "🚫", "⏱️", "❌"]))
    
    print(f"📊 Kết quả: {len(query_results)} truy vấn (DB: {db_success}/{len(database_queries)} ✅, Web: {web_success}/{len(web_queries)} ✅)")
    
    for i, result in enumerate(query_results, 1):
        source_vietnamese = "CƠ SỞ DỮ LIỆU" if result.source.upper() == "DATABASE" else "WEB"
        
        # Determine status based on result content
        has_error = any(keyword in str(result.results).lower() for keyword in ["thất bại", "timeout", "rate limit", "lỗi", "🚫", "⏱️", "❌", "⏭️"])
        status = "⚠️" if has_error else "✅"
        
        print(f"{i}. {status} TRUY VẤN {source_vietnamese}: {result.query}")
        for j, content in enumerate(result.results, 1):
            # Truncate long content for display
            display_content = content[:200] + "..." if len(content) > 200 else content
            print(f"   {j}. {display_content}")
    
    return {
        "current_query_results": query_results
    }