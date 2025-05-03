from flask import Flask, request, jsonify
import validators
from utils.recommendation import RecommendationEngine
from utils.evaluation import evaluate_model
import os

app = Flask(__name__)

# Initialize recommendation engine
recommendation_engine = RecommendationEngine()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({
        "status": "ok",
        "message": "API is operational"
    }), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Assessment recommendation endpoint.
    
    Accepts a job description or natural language query and returns
    recommended relevant assessments.
    
    Expected JSON payload:
    {
        "query": "string",  // Required: Job description or natural language query
        "is_url": boolean,  // Optional: Indicates if query is a URL (default: false)
        "limit": integer    // Optional: Maximum number of recommendations (default: 10)
    }
    """
    # Get request payload
    data = request.json
    
    # Validate input
    if not data or 'query' not in data or not data['query']:
        return jsonify({
            "status": "error",
            "message": "Missing required field: query"
        }), 400
    
    query = data['query']
    is_url = data.get('is_url', False)
    limit = min(int(data.get('limit', 10)), 10)  # Cap at 10 max
    
    # Validate URL if is_url is true
    if is_url and not validators.url(query):
        return jsonify({
            "status": "error",
            "message": "Invalid URL provided"
        }), 400
    
    try:
        # Get recommendations
        recommendations = recommendation_engine.get_recommendations(
            query,
            is_url=is_url,
            max_results=limit
        )
        
        # Format the response
        response_data = []
        for rec in recommendations:
            response_data.append({
                "name": rec["name"],
                "url": rec["url"],
                "remote_testing": rec.get("remote_testing", "Yes"),
                "adaptive_irt": rec.get("adaptive_irt", "No"),
                "duration": rec.get("duration", "N/A"),
                "test_type": rec.get("test_type", "N/A")
            })
        
        # Return response
        return jsonify({
            "status": "success",
            "query": query,
            "recommendations": response_data
        }), 200
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to process recommendation: {str(e)}"
        }), 500

@app.route('/evaluate', methods=['GET'])
def evaluate():
    """
    Evaluation endpoint to get model performance metrics.
    """
    try:
        k = int(request.args.get('k', 3))
        eval_results = evaluate_model(recommendation_engine, k=k)
        
        return jsonify({
            "status": "success",
            "mean_recall_at_k": eval_results['mean_recall_at_k'],
            "map_at_k": eval_results['map_at_k'],
            "k": eval_results['k'],
            "details": [
                {
                    "query": result['query'],
                    "recall_at_k": result['recall_at_k'],
                    "ap_at_k": result['ap_at_k']
                }
                for result in eval_results['query_results']
            ]
        }), 200
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Evaluation failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
