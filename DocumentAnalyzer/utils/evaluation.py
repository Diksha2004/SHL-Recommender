import json
import os
from typing import Dict, List, Any
import numpy as np

def load_test_data() -> List[Dict[str, Any]]:
    """Load test dataset for evaluation."""
    if os.path.exists('data/test_set.json'):
        with open('data/test_set.json', 'r') as f:
            return json.load(f)
    else:
        # If file doesn't exist, import the scraper and create it
        from utils.scraper import load_test_dataset
        return load_test_dataset()

def recall_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Compute Recall@K.
    
    Args:
        recommended: List of recommended assessment names
        relevant: List of relevant assessment names
        k: Number of top results to consider
    
    Returns:
        Recall@K score
    """
    if not relevant:  # Avoid division by zero
        return 0.0
    
    # Consider only top k recommendations
    top_k = recommended[:k]
    
    # Count relevant items in top k
    relevant_in_top_k = sum(1 for item in top_k if item in relevant)
    
    # Compute recall
    return relevant_in_top_k / len(relevant)

def precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Compute Precision@K.
    
    Args:
        recommended: List of recommended assessment names
        relevant: List of relevant assessment names
        k: Number of top results to consider
    
    Returns:
        Precision@K score
    """
    if k == 0:  # Avoid division by zero
        return 0.0
    
    # Consider only top k recommendations
    top_k = recommended[:k]
    
    # Count relevant items in top k
    relevant_in_top_k = sum(1 for item in top_k if item in relevant)
    
    # Compute precision
    return relevant_in_top_k / k

def average_precision_at_k(recommended: List[str], relevant: List[str], k: int) -> float:
    """
    Compute Average Precision@K.
    
    Args:
        recommended: List of recommended assessment names
        relevant: List of relevant assessment names
        k: Number of top results to consider
    
    Returns:
        AP@K score
    """
    if not relevant:  # Avoid division by zero
        return 0.0
    
    # Consider only top k recommendations
    top_k = recommended[:k]
    
    ap_sum = 0.0
    hits = 0
    
    for i, item in enumerate(top_k):
        if item in relevant:
            hits += 1
            # Precision at position i+1
            precision_at_i = hits / (i + 1)
            ap_sum += precision_at_i
    
    # AP is the sum of precision at each relevant item divided by min(k, number of relevant items)
    return ap_sum / min(k, len(relevant))

def evaluate_model(recommendation_engine, k=3):
    """
    Evaluate the recommendation model using test data.
    
    Args:
        recommendation_engine: The recommendation engine to evaluate
        k: Number of top results to consider for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load test data
    test_data = load_test_data()
    
    total_queries = len(test_data)
    total_recall = 0.0
    total_map = 0.0
    
    query_results = []
    
    for test_case in test_data:
        query = test_case["query"]
        relevant_assessments = test_case["relevant_assessments"]
        
        # Get recommendations
        recommendations = recommendation_engine.get_recommendations(query, max_results=10)
        recommended_names = [rec["name"] for rec in recommendations]
        
        # Calculate metrics
        recall = recall_at_k(recommended_names, relevant_assessments, k)
        ap = average_precision_at_k(recommended_names, relevant_assessments, k)
        
        total_recall += recall
        total_map += ap
        
        # Store results for this query
        query_results.append({
            "query": query,
            "recall_at_k": recall,
            "ap_at_k": ap,
            "recommended": recommended_names[:k],
            "relevant": relevant_assessments
        })
    
    # Calculate mean metrics
    mean_recall_at_k = total_recall / total_queries if total_queries > 0 else 0
    mean_ap_at_k = total_map / total_queries if total_queries > 0 else 0
    
    # Return evaluation results
    return {
        "mean_recall_at_k": mean_recall_at_k,
        "map_at_k": mean_ap_at_k,
        "k": k,
        "query_results": query_results
    }

def format_evaluation_results(results):
    """Format evaluation results for display."""
    output = f"## Evaluation Results (k={results['k']})\n\n"
    output += f"Mean Recall@{results['k']}: {results['mean_recall_at_k']:.4f}\n"
    output += f"MAP@{results['k']}: {results['map_at_k']:.4f}\n\n"
    
    output += "### Detailed Results by Query\n\n"
    
    for i, query_result in enumerate(results['query_results']):
        output += f"**Query {i+1}:** {query_result['query']}\n"
        output += f"Recall@{results['k']}: {query_result['recall_at_k']:.4f}\n"
        output += f"AP@{results['k']}: {query_result['ap_at_k']:.4f}\n"
        output += f"Recommended: {', '.join(query_result['recommended'])}\n"
        output += f"Relevant: {', '.join(query_result['relevant'])[:100]}{'...' if len(', '.join(query_result['relevant'])) > 100 else ''}\n\n"
    
    return output
