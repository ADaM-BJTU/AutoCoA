"""
Retrieval Module

This module provides functionality to retrieve documents from a search API endpoint.
It supports single queries with options for specifying the number of results and
whether to include relevance scores.
"""

import time
import requests


class RetrievalClient:
    """Client for interacting with the document retrieval service."""

    def __init__(self, base_url="YOUR_RETRIEVAL_API_URL"):
        """
        Initialize the retrieval client.

        Args:
            base_url (str): The base URL for the retrieval service.
        """
        self.base_url = base_url

    def retrieve(self, query, top_k=3, return_score=True, timeout=15):
        """
        Retrieve documents based on a query.

        Args:
            query (str): The search query text.
            top_k (int): Number of top results to return.
            return_score (bool): Whether to include relevance scores in results.
            timeout (int): Request timeout in seconds.

        Returns:
            dict: The retrieval results or None if the request failed.
        """
        payload = {
            "query": query,
            "tok_k": top_k,  # Note: This matches the original param name, consider renaming to "top_k" in the future
            "return_score": return_score
        }

        try:
            start_time = time.time()
            response = requests.post(self.base_url, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            # Validate response structure
            self._validate_response(result, return_score)
            
            elapsed_time = time.time() - start_time
            # Uncomment for debugging:
            # print(f"Query completed in {elapsed_time:.2f} seconds")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Retrieval failed: {e}")
            return None
    
    def _validate_response(self, result, return_score):
        """
        Validate the structure of the API response.
        
        Args:
            result (dict): The API response to validate.
            return_score (bool): Whether scores were requested.
            
        Raises:
            AssertionError: If the response structure is invalid.
        """
        assert isinstance(result, dict), "Response should be a dictionary"
        if return_score:
            assert "documents" in result and "scores" in result, "Expected 'documents' and 'scores' in response"
        else:
            assert "documents" in result, "Expected 'documents' in response"


def format_results(result, top_k=1):
    """
    Format the retrieval results into a readable string.

    Args:
        result (dict): The retrieval results from the API.
        top_k (int): Number of results to include.

    Returns:
        str: Formatted string of results.
    """
    if not result or "documents" not in result:
        return "No results found."
        
    if top_k == 1:
        return result["documents"][0]["contents"]
    
    content_list = []
    for index, doc in enumerate(result["documents"][:top_k], start=1):
        content_list.append(f"result {index}: {doc['contents']}")

    return "\n".join(content_list)


def get_search_results(query, top_k=2):
    """
    Get search results formatted with XML tags.

    Args:
        query (str): The search query.
        top_k (int): Number of results to retrieve.

    Returns:
        str: Formatted search results with XML tags.
    """
    client = RetrievalClient()
    retrieval_result = client.retrieve(query, top_k)
    
    if not retrieval_result:
        return "<search_result>No results available</search_result>"
    
    search_result = format_results(retrieval_result, top_k)
    return f"<search_result>{search_result}</search_result>"


if __name__ == "__main__":
    query = "where was Goo Goo Dolls formed?"
    search_results = get_search_results(query)
    print(search_results)