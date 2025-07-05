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
    SEARCH_WRAPPER = DuckDuckGoSearchAPIWrapper()
    
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
            return ["Vector database not available"]
        
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
                source = doc.metadata.get('source_file', 'Unknown source')
                
                # Add retrieval information
                retrieval_info = []
                if 'fusion_score' in doc.metadata:
                    retrieval_info.append(f"Fusion Score: {doc.metadata['fusion_score']:.3f}")
                if 'retrieval_method' in doc.metadata:
                    retrieval_info.append(f"Method: {doc.metadata['retrieval_method']}")
                if 'bm25_score' in doc.metadata:
                    retrieval_info.append(f"BM25 Score: {doc.metadata['bm25_score']:.3f}")
                
                retrieval_str = f" ({', '.join(retrieval_info)})" if retrieval_info else ""
                content_results.append(f"Source: {source}{retrieval_str}\nContent: {content}")
        
        if not content_results:
            return ["No relevant information found in database"]
        
        return content_results
        
    except Exception as e:
        return [f"Database search error: {str(e)}"]


def search_vector_database(query: str) -> List[str]:
    """Legacy function for backward compatibility - now uses hybrid search"""
    return search_vector_database_hybrid(query)


def search_web(query: str) -> List[str]:
    """Search the web using DuckDuckGo through LangChain"""
    try:
        if SEARCH_WRAPPER is None:
            return ["Web search not available - search wrapper not initialized"]
        
        # Use DuckDuckGo search with specified max_results
        search_results = SEARCH_WRAPPER.results(query, max_results=N_RESULTS)
        
        formatted_results = []
        
        for result in search_results:
            # Extract information from each result dict
            title = result.get('title', 'No title')
            snippet = result.get('snippet', 'No snippet available')
            link = result.get('link', 'No link')
            
            # Format the result for better readability
            formatted_result = f"Title: {title}\nSnippet: {snippet}\nLink: {link}"
            formatted_results.append(formatted_result)
        
        if not formatted_results:
            return [f"No web search results found for query: '{query}'"]
        
        return formatted_results
        
    except Exception as e:
        return [f"Web search error for '{query}': {str(e)}"]


def observation_node(state: OverallState) -> Dict[str, Any]:
    """
    Execute database and web search queries to collect information
    This node does not use LLM - it only executes searches
    """
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
                    results=[f"Database query failed: {str(e)}"],
                    source="database"
                ))
    
    # Execute web search queries
    for query in web_queries:
        if query.strip():  # Skip empty queries
            try:
                results = search_web(query)
                query_results.append(QueryResult(
                    query=query,
                    results=results,
                    source="web"
                ))
            except Exception as e:
                query_results.append(QueryResult(
                    query=query,
                    results=[f"Web search failed: {str(e)}"],
                    source="web"
                ))

    print("\n[OBSERVATION] Executing database and web search queries...")
    
    # Print retrieval method info
    if USE_HYBRID and BM25_INDEX and BM25_INDEX.bm25_index:
        print(f"Using hybrid retrieval: {FUSION_METHOD} fusion (Dense: {DENSE_WEIGHT}, Sparse: {SPARSE_WEIGHT})")
    else:
        print("Using dense retrieval only")

    print(f"Collected {len(query_results)} query results:")
    for i, result in enumerate(query_results, 1):
        print(f"{i}. {result.source.upper()} QUERY: {result.query}")
        for j, content in enumerate(result.results, 1):
            # Truncate long content for display
            display_content = content[:200] + "..." if len(content) > 200 else content
            print(f"   {j}. {display_content}")
    
    return {
        "current_query_results": query_results
    }