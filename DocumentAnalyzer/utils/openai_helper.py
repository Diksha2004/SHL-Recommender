"""OpenAI integration for enhanced assessment recommendations."""
import os
import json
from typing import List, Dict, Any, Optional

from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_enhanced_recommendations(assessments: List[Dict[str, Any]], query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Generate enhanced recommendations using OpenAI.
    
    Args:
        assessments: List of assessment dictionaries
        query: User query or processed job description
        max_results: Maximum number of recommendations to return
        
    Returns:
        List of recommended assessments with scores
    """
    # Format assessments for prompt
    assessments_text = format_assessments_for_prompt(assessments)
    
    # Create the prompt for GPT
    prompt = f"""
You are an expert SHL assessment specialist. A recruiter needs to select appropriate SHL assessments for a position.

Here is the job description or query:
"{query}"

Below is a list of available SHL assessments with their properties:
{assessments_text}

Based on the job description/query, recommend up to {max_results} most relevant assessments that would help evaluate candidates for this position.
For each recommendation, provide:
1. Assessment name
2. A score from 0 to 100 indicating relevance (higher is more relevant)
3. A brief explanation for why this assessment is relevant (max 2 sentences)

Format your response as a JSON array of objects with the following structure:
[
  {{
    "name": "Exact name of the assessment from the list",
    "score": score as an integer,
    "explanation": "Brief explanation"
  }}
]

Ensure the assessment names EXACTLY match those in the provided list.
"""

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        # Parse response
        result = json.loads(response.choices[0].message.content)
        
        # Process recommendations
        recommendations = []
        if "recommendations" in result:
            raw_recommendations = result["recommendations"]
        else:
            # If the model didn't use the "recommendations" key, assume the response itself is the array
            raw_recommendations = result
            if not isinstance(raw_recommendations, list):
                raw_recommendations = []
        
        # Match recommended assessments to actual assessment objects
        assessment_dict = {assessment['name']: assessment for assessment in assessments}
        
        for rec in raw_recommendations:
            if not isinstance(rec, dict) or 'name' not in rec:
                continue
                
            name = rec.get('name')
            if name in assessment_dict:
                # Create a copy of the assessment and add OpenAI-specific scoring
                assessment = assessment_dict[name].copy()
                assessment['similarity'] = rec.get('score', 0)
                assessment['explanation'] = rec.get('explanation', '')
                recommendations.append(assessment)
        
        # If no valid recommendations were found, return empty list
        if not recommendations:
            return []
            
        # Sort by score and return top results
        recommendations.sort(key=lambda x: x['similarity'], reverse=True)
        return recommendations[:max_results]
    
    except Exception as e:
        print(f"Error using OpenAI for recommendations: {str(e)}")
        return []  # Return empty list if there's an error


def format_assessments_for_prompt(assessments: List[Dict[str, Any]]) -> str:
    """Format assessments into a string for the prompt."""
    formatted_list = []
    
    for assessment in assessments:
        # Extract key properties
        name = assessment.get('name', 'Unknown')
        test_type = assessment.get('test_type', 'Not specified')
        duration = assessment.get('duration', 'Not specified')
        description = assessment.get('description', '')
        
        # Format assessment info
        formatted = f"Assessment: {name}\n"
        formatted += f"Type: {test_type}\n"
        formatted += f"Duration: {duration}\n"
        
        # Add remote testing support if available
        if 'remote_testing_support' in assessment:
            formatted += f"Remote Testing: {'Yes' if assessment['remote_testing_support'] else 'No'}\n"
        
        # Add adaptive/IRT support if available
        if 'adaptive_irt_support' in assessment:
            formatted += f"Adaptive/IRT: {'Yes' if assessment['adaptive_irt_support'] else 'No'}\n"
        
        # Add description if available
        if description:
            formatted += f"Description: {description}\n"
        
        formatted_list.append(formatted)
    
    return "\n".join(formatted_list)


def extract_role_skills(job_description: str) -> Dict[str, Any]:
    """
    Extract key role information and required skills from a job description.
    
    Args:
        job_description: The job description text
        
    Returns:
        Dictionary with role information and skills
    """
    prompt = f"""
Analyze the following job description and extract:
1. The job title
2. The key technical skills required (programming languages, tools, etc.)
3. The key soft skills required (communication, leadership, etc.)
4. The level of experience required (entry, mid, senior)

Format your response as a JSON object:
{{
  "job_title": "string",
  "technical_skills": ["skill1", "skill2", ...],
  "soft_skills": ["skill1", "skill2", ...],
  "experience_level": "string"
}}

Job Description:
{job_description}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error extracting role skills: {str(e)}")
        return {
            "job_title": "",
            "technical_skills": [],
            "soft_skills": [],
            "experience_level": ""
        }