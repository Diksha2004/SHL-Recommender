import streamlit as st
import pandas as pd
from utils.recommendation import RecommendationEngine
from utils.evaluation import evaluate_model, format_evaluation_results
import validators

# Set page config
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📋",
    layout="wide"
)

# Initialize the recommendation engine
@st.cache_resource
def load_recommendation_engine():
    with st.spinner("Loading recommendation engine..."):
        engine = RecommendationEngine()
    return engine

recommendation_engine = load_recommendation_engine()

# Page title and description
st.title("SHL Assessment Recommendation System")
st.markdown("""
This application helps hiring managers find the right SHL assessments for their job roles. 
Enter a job description, query, or paste a job posting URL below to get started.
""")

# Tabs for different functionality
tab1, tab2 = st.tabs(["Get Recommendations", "Evaluation"])

with tab1:
    # Input options
    input_type = st.radio(
        "Choose input type:",
        ["Natural Language Query", "Job Description Text", "Job Description URL"]
    )
    
    if input_type == "Natural Language Query":
        query = st.text_area(
            "Enter your query:",
            placeholder="Example: I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
            height=100
        )
        is_url = False
    
    elif input_type == "Job Description Text":
        query = st.text_area(
            "Enter job description:",
            placeholder="Paste the full job description here...",
            height=200
        )
        is_url = False
    
    else:  # Job Description URL
        query = st.text_input(
            "Enter job posting URL:",
            placeholder="https://example.com/job-posting"
        )
        is_url = True
    
    # Query validation for URL
    url_valid = True
    if input_type == "Job Description URL" and query:
        url_valid = validators.url(query)
        if not url_valid:
            st.error("Please enter a valid URL")
    
    # Number of recommendations selector
    num_recommendations = st.slider(
        "Number of recommendations:", 
        min_value=1,
        max_value=10,
        value=5
    )
    
    # Get recommendations button
    if st.button("Get Recommendations", disabled=(not query or not url_valid)):
        with st.spinner("Finding the best assessments for you..."):
            # Get recommendations
            recommendations = recommendation_engine.get_recommendations(
                query,
                is_url=is_url,
                max_results=num_recommendations
            )
            
            if recommendations:
                # Convert to DataFrame for display
                data = []
                for rec in recommendations:
                    data.append({
                        "Assessment Name": rec["name"],
                        "Remote Testing": "Yes" if rec.get("remote_testing_support", False) else "No",
                        "Adaptive/IRT": "Yes" if rec.get("adaptive_irt_support", False) else "No",
                        "Duration": rec.get("duration", "N/A"),
                        "Test Type": rec.get("test_type", "N/A"),
                        "Relevance Score": int(rec.get("similarity", 0)),
                        "URL": rec.get("url", "#")
                    })
                
                df = pd.DataFrame(data)
                
                # Make the assessment name a clickable link
                def make_clickable(name, url):
                    if url and url != "#":
                        return f'<a href="{url}" target="_blank">{name}</a>'
                    return name
                
                df["Assessment Name"] = df.apply(
                    lambda x: make_clickable(x["Assessment Name"], x["URL"]), 
                    axis=1
                )
                
                # Drop URL column as it's now embedded in the name
                df = df.drop(columns=["URL"])
                
                # Show the results
                st.subheader("Recommended Assessments")
                st.write(df.to_html(escape=False), unsafe_allow_html=True)
                
                # Display explanations if available from OpenAI
                has_explanations = any(rec.get('explanation') for rec in recommendations)
                if has_explanations:
                    st.markdown("### Why these recommendations?")
                    for rec in recommendations:
                        if rec.get('explanation'):
                            st.markdown(f"**{rec['name']}**: {rec['explanation']}")
                else:
                    # Generic explanation if no specific ones available
                    st.markdown("### Why these recommendations?")
                    st.write("""
                    The recommendations above are based on semantic matching between your query and the SHL assessment catalog.
                    Key factors that influenced the recommendations include:
                    - Keywords and skills mentioned in your query
                    - Duration constraints (if specified)
                    - Role or position requirements
                    - AI-based analysis of job requirements
                    """)
            else:
                st.error("No recommendations found. Please try a different query or input.")

with tab2:
    st.subheader("Model Evaluation")
    st.write("""
    This section evaluates the recommendation system's performance using a test dataset with known relevant assessments for each query.
    The evaluation metrics are:
    - **Mean Recall@3**: Measures how many of the relevant assessments are in the top 3 recommendations.
    - **Mean Average Precision@3 (MAP@3)**: Measures both relevance and ranking quality of the top 3 recommendations.
    """)
    
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating recommendation system..."):
            # Run evaluation
            eval_results = evaluate_model(recommendation_engine, k=3)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Recall@3", f"{eval_results['mean_recall_at_k']:.4f}")
            with col2:
                st.metric("MAP@3", f"{eval_results['map_at_k']:.4f}")
            
            # Display detailed results
            st.markdown("### Detailed Evaluation Results")
            
            for i, query_result in enumerate(eval_results['query_results']):
                with st.expander(f"Query {i+1}"):
                    st.write(f"**Query:** {query_result['query']}")
                    st.write(f"**Recall@3:** {query_result['recall_at_k']:.4f}")
                    st.write(f"**AP@3:** {query_result['ap_at_k']:.4f}")
                    
                    st.write("**Recommended (Top 3):**")
                    for j, rec in enumerate(query_result['recommended']):
                        is_relevant = rec in query_result['relevant']
                        st.write(f"{j+1}. {rec} {'✓' if is_relevant else ''}")
                    
                    st.write("**Relevant Assessments:**")
                    for rel in query_result['relevant']:
                        st.write(f"- {rel}")

# Sidebar with additional information
with st.sidebar:
    st.subheader("About")
    st.write("""
    This application helps hiring managers find the right SHL assessments for their needs. It uses natural language processing to match job requirements with suitable assessments.
    """)
    
    st.subheader("Features")
    st.write("""
    - Natural language query processing
    - Job description analysis
    - Web URL content extraction
    - AI-powered matching with SHL assessments
    - Duration and skill-based filtering
    - Detailed explanations for recommendations
    """)
    
    st.subheader("How It Works")
    st.write("""
    1. Your input (query/job description) is processed using NLP
    2. The system analyzes requirements using AI-powered matching
    3. Results are ranked by relevance to your requirements
    4. Each recommendation includes explanations of why it's relevant
    5. Duration constraints and other preferences are applied
    """)
