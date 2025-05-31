from src.state import OverallState, QueryResult
from typing import Dict, Any, List
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


# Global variables (will be set in main.py)
VECTOR_DB = None
N_RESULTS = None
SEARCH_WRAPPER = None


def initialize_observation(cfg, vectordb):
    """Initialize observation node with configuration and vector database"""
    global VECTOR_DB, N_RESULTS, SEARCH_WRAPPER
    VECTOR_DB = vectordb
    N_RESULTS = cfg.tree.hyperparams.n_results
    SEARCH_WRAPPER = DuckDuckGoSearchAPIWrapper()


def search_vector_database(query: str) -> List[str]:
    """Search the vector database and return results"""
    try:
        if VECTOR_DB is None:
            return ["Vector database not available"]
        
        # Search for similar documents
        results = VECTOR_DB.similarity_search(query, k=N_RESULTS)
        
        # Extract content from results
        content_results = []
        for doc in results:
            content = doc.page_content.strip()
            if content:
                # Add source information if available
                source = doc.metadata.get('source_file', 'Unknown source')
                content_results.append(f"Source: {source}\nContent: {content}")
        
        if not content_results:
            return ["No relevant information found in database"]
        
        return content_results
        
    except Exception as e:
        return [f"Database search error: {str(e)}"]


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
    
    # Execute database queries
    for query in database_queries:
        if query.strip():  # Skip empty queries
            try:
                results = search_vector_database(query)
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

    print(f"Collected {len(query_results)} query results:")
    for i, result in enumerate(query_results, 1):
        print(f"{i}. {result.source.upper()} QUERY: {result.query}")
        for j, content in enumerate(result.results, 1):
            print(f"   {j}. {content}")
    
    return {
        "current_query_results": query_results
    }