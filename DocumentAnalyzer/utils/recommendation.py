import os
import json
import re
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
# Using basic text similarity instead of sentence_transformers
# from sentence_transformers import SentenceTransformer
import requests
from trafilatura import fetch_url, extract

# Import OpenAI helper
from utils.openai_helper import get_enhanced_recommendations, extract_role_skills

class RecommendationEngine:
    """Class to recommend SHL assessments based on query or job description."""
    
    def __init__(self):
        """Initialize the recommendation engine with keyword matching."""
        # Load assessment data
        self.assessments = self._load_assessments()
        
        # We're using keyword matching instead of embeddings due to dependency constraints
        print("Using keyword-based matching for recommendations")
        self.model = None
    
    def _load_assessments(self) -> List[Dict[str, Any]]:
        """Load assessments from the JSON file."""
        try:
            if os.path.exists('data/assessments.json'):
                with open('data/assessments.json', 'r') as f:
                    return json.load(f)
            else:
                # If file doesn't exist, import the scraper and run it
                from utils.scraper import get_all_assessments
                return get_all_assessments()
        except Exception as e:
            print(f"Error loading assessments: {str(e)}")
            return []
    
    def _generate_embeddings(self):
        """Generate and store embeddings for all assessments."""
        if not self.model:
            print("Model not available, skipping embeddings generation")
            return
        
        # Create texts to embed - combine name, description, and test_type for better matching
        texts = [
            f"{assessment['name']} {assessment.get('description', '')} {assessment.get('test_type', '')}"
            for assessment in self.assessments
        ]
        
        # Generate embeddings
        self.assessment_embeddings = self.model.encode(texts)
    
    def extract_text_from_url(self, url: str) -> str:
        """Extract text content from a URL."""
        try:
            downloaded = fetch_url(url)
            text = extract(downloaded)
            if text:
                return text
            # Fallback if trafilatura fails
            response = requests.get(url)
            if response.status_code == 200:
                return response.text
            return ""
        except Exception as e:
            print(f"Error extracting text from URL: {str(e)}")
            return ""
    
    def preprocess_query(self, query: str, is_url: bool = False) -> str:
        """Preprocess the query text or extract text from URL with enhanced processing."""
        if is_url:
            query = self.extract_text_from_url(query)
        
        # Clean up and normalize text
        query = query.strip()
        
        # Extract key information if this is a job description
        if len(query.split()) > 50:  # Likely a job description
            # Extract important sentences that might contain requirements
            important_sentences = []
            
            # Split into sentences more accurately
            sentences = re.split(r'(?<=[.!?])\s+', query)
            
            # Expanded keywords for better extraction
            primary_keywords = [
                'skills', 'requirements', 'qualifications', 'experience', 
                'proficiency', 'knowledge', 'familiar', 'understanding',
                'assessment', 'test', 'evaluation', 'competencies'
            ]
            
            secondary_keywords = [
                'job description', 'responsibilities', 'duties', 'objectives',
                'technical', 'programming', 'development', 'management',
                'proficient in', 'expertise in', 'ability to', 'capable of'
            ]
            
            # Capture role-specific terms
            role_keywords = [
                'developer', 'engineer', 'analyst', 'manager', 'leader', 'director',
                'specialist', 'consultant', 'coordinator', 'administrator', 'designer'
            ]
            
            # Process each sentence
            for sentence in sentences:
                sentence = sentence.strip()
                sentence_lower = sentence.lower()
                
                # Check for primary keywords (high importance)
                if any(keyword in sentence_lower for keyword in primary_keywords) and len(sentence) > 15:
                    important_sentences.append(sentence)
                    continue
                
                # Check for secondary keywords (medium importance)
                if any(keyword in sentence_lower for keyword in secondary_keywords) and len(sentence) > 20:
                    important_sentences.append(sentence)
                    continue
                
                # Check for role specification (also important)
                if any(role in sentence_lower for role in role_keywords) and len(sentence) > 20:
                    important_sentences.append(sentence)
                
            # If we found important sentences, use them. Otherwise, extract key phrases
            if important_sentences:
                processed_query = ' '.join(important_sentences)
            else:
                # Extract key phrases if no important sentences found
                processed_query = query
                
                # Extract phrases like "X years experience in [technology]"
                experience_phrases = re.findall(r'\d+\+?\s+years?\s+(?:of\s+)?(?:experience|expertise)\s+(?:in|with)\s+[\w\s]+', query.lower())
                
                # Extract phrases with technical requirements
                tech_phrases = re.findall(r'(?:knowledge|proficiency|skills|experience)\s+(?:of|in|with)\s+[\w\s,]+', query.lower())
                
                # Extract required/preferred skills sections
                skill_sections = re.findall(r'(?:required|preferred|essential|desired|key)\s+skills[:\s]+[\w\s,]+', query.lower())
                
                # Combine all extracted phrases
                all_phrases = experience_phrases + tech_phrases + skill_sections
                if all_phrases:
                    processed_query = ' '.join(all_phrases)
            
            return processed_query
        
        return query
    
    def get_duration_constraint(self, query: str) -> Optional[int]:
        """Extract duration constraint from query if present."""
        # Look for patterns like "40 minutes", "under 30 mins", "less than 45 minutes"
        patterns = [
            r'(\d+)\s*(?:minute|minutes|mins|min)',
            r'(?:less than|under|maximum|max|up to)\s*(\d+)\s*(?:minute|minutes|mins|min)'
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, query, re.IGNORECASE)
            if matches:
                try:
                    return int(matches.group(1))
                except:
                    pass
        
        return None
    
    def get_recommendations_embedding(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get recommendations using embedding similarity."""
        if not self.model:
            # Fall back to keyword-based matching if model not available
            return self.get_recommendations_keywords(query, max_results)
        
        # Encode the query
        query_embedding = self.model.encode(query)
        
        # Calculate cosine similarity with all assessments
        similarities = np.dot(self.assessment_embeddings, query_embedding) / (
            np.linalg.norm(self.assessment_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get indices of top matches
        top_indices = np.argsort(similarities)[::-1][:max_results]
        
        # Extract duration constraint if present
        duration_constraint = self.get_duration_constraint(query)
        
        # Apply duration filter if needed
        if duration_constraint:
            filtered_assessments = []
            for idx in top_indices:
                assessment = self.assessments[idx]
                assessment_duration = self._extract_duration(assessment.get('duration', ''))
                
                if assessment_duration and assessment_duration <= duration_constraint:
                    # Add similarity score for ranking
                    assessment = assessment.copy()
                    assessment['similarity'] = float(similarities[idx])
                    filtered_assessments.append(assessment)
                
                # If we have enough results, stop
                if len(filtered_assessments) >= max_results:
                    break
            
            # If we don't have enough results after filtering, add more without duration filter
            if len(filtered_assessments) < max_results and len(filtered_assessments) < len(top_indices):
                for idx in top_indices:
                    if len(filtered_assessments) >= max_results:
                        break
                    
                    assessment = self.assessments[idx]
                    
                    # Check if this assessment is already in the filtered list
                    if not any(a['name'] == assessment['name'] for a in filtered_assessments):
                        assessment = assessment.copy()
                        assessment['similarity'] = float(similarities[idx])
                        filtered_assessments.append(assessment)
            
            return filtered_assessments
        else:
            # No duration constraint, just return top matches
            return [
                {**self.assessments[idx], 'similarity': float(similarities[idx])}
                for idx in top_indices
            ]
    
    def _extract_duration(self, duration_str: str) -> Optional[int]:
        """Extract duration in minutes from duration string."""
        if not duration_str:
            return None
        
        match = re.search(r'(\d+)', duration_str)
        if match:
            try:
                return int(match.group(1))
            except:
                pass
        
        return None
    
    def get_recommendations_keywords(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Enhanced keyword-based recommendation method with more sophisticated matching."""
        # Extract keywords from query
        query_lower = query.lower()
        
        # Define expanded keywords for different categories with weights
        tech_keywords = {
            'high': ['java', 'python', 'sql', 'javascript', 'react', 'angular', 'node', 'coding', 
                    '.net', 'c#', 'c++', 'software development', 'developer', 'programming'],
            'medium': ['database', 'cloud', 'aws', 'azure', 'tech', 'it', 'computer', 'web', 
                      'mobile', 'frontend', 'backend', 'fullstack', 'api', 'application'],
            'low': ['technology', 'technical', 'development', 'code', 'algorithm', 'solution']
        }
        
        sales_keywords = {
            'high': ['sales', 'customer service', 'account management', 'client relationship', 
                    'business development', 'sales representative', 'retail'],
            'medium': ['marketing', 'customer', 'service', 'account', 'representative', 
                      'business', 'client', 'relationship', 'store', 'sales support'],
            'low': ['communication', 'negotiation', 'persuasion', 'presentation', 'commercial']
        }
        
        language_keywords = {
            'high': ['english proficiency', 'communication skills', 'verbal communication',
                   'written communication', 'language test'],
            'medium': ['english', 'language', 'communication', 'verbal', 'written', 
                     'speak', 'writing', 'reading', 'comprehension'],
            'low': ['fluent', 'proficient', 'expression', 'clarity', 'articulate']
        }
        
        management_keywords = {
            'high': ['leadership', 'team management', 'project management', 'manager',
                   'executive', 'director', 'supervisor'],
            'medium': ['manage', 'team', 'leader', 'administration', 'project', 
                      'coordination', 'supervise', 'organize'],
            'low': ['planning', 'delegation', 'strategy', 'objectives', 'vision']
        }
        
        # Add specific skill keywords
        skill_keywords = {
            'high': ['agile', 'scrum', 'teamwork', 'collaboration', 'problem-solving', 
                    'analytical', 'critical thinking'],
            'medium': ['skill', 'ability', 'competence', 'expertise', 'proficiency',
                      'experience', 'knowledge'],
            'low': ['familiar', 'understanding', 'aware', 'exposure']
        }
        
        # Define keyword weights
        weights = {'high': 8, 'medium': 5, 'low': 2}
        
        # Extract exact phrases from query for better matching
        query_phrases = [phrase.strip() for phrase in re.findall(r'\b[\w\s]{5,}\b', query_lower)]
        
        # Score each assessment based on enhanced keyword matches
        scored_assessments = []
        
        for assessment in self.assessments:
            score = 0
            assessment_text = f"{assessment['name']} {assessment.get('description', '')} {assessment.get('test_type', '')}".lower()
            
            # Direct test type matching (high weight)
            if assessment.get('test_type', '').lower() in query_lower:
                score += 10
                
            # Check for category matches with weighted scoring
            for category in [tech_keywords, sales_keywords, language_keywords, management_keywords, skill_keywords]:
                for weight_level, keywords in category.items():
                    # Check if both query and assessment contain keywords from this category
                    query_matches = any(keyword in query_lower for keyword in keywords)
                    assessment_matches = any(keyword in assessment_text for keyword in keywords)
                    
                    # If both have matches, add weighted score
                    if query_matches and assessment_matches:
                        score += weights[weight_level]
                        
                        # For high importance keywords, add extra points for direct matches
                        if weight_level == 'high':
                            for keyword in keywords:
                                if keyword in query_lower and keyword in assessment_text:
                                    score += 5  # Bonus for exact matches of important terms
            
            # Phrase matching (important for context)
            for phrase in query_phrases:
                if phrase in assessment_text:
                    # Longer phrase matches are more significant
                    phrase_length_bonus = min(len(phrase.split()), 5)  # Cap at 5 words
                    score += 3 * phrase_length_bonus
            
            # Individual word matching with improved weighting
            query_words = set(re.findall(r'\b\w{4,}\b', query_lower))  # Only words with 4+ chars
            assessment_words = set(re.findall(r'\b\w{4,}\b', assessment_text))
            
            # Score based on word overlap
            for word in query_words:
                if word in assessment_words:
                    # Weight by word length (longer words usually carry more meaning)
                    word_weight = min(len(word) / 4, 2)  # Normalize with cap
                    score += word_weight
            
            # Duration constraint handling with improved weighting
            duration_constraint = self.get_duration_constraint(query)
            if duration_constraint:
                assessment_duration = self._extract_duration(assessment.get('duration', ''))
                if assessment_duration:
                    # Exact match
                    if assessment_duration == duration_constraint:
                        score += 8
                    # Within 5 minutes
                    elif abs(assessment_duration - duration_constraint) <= 5:
                        score += 6
                    # Within 10 minutes
                    elif abs(assessment_duration - duration_constraint) <= 10:
                        score += 4
                    # Within duration constraint
                    elif assessment_duration <= duration_constraint:
                        score += 3
                    # Exceeds by less than 10 minutes
                    elif assessment_duration <= duration_constraint + 10:
                        score += 1
                    # Exceeds by too much
                    else:
                        score -= 3  # Stronger penalty for significantly exceeding time constraint
            
            # Add to scored list with a minimum threshold
            if score >= 2:  # Only include reasonably relevant matches
                assessment_copy = assessment.copy()
                assessment_copy['similarity'] = score
                scored_assessments.append(assessment_copy)
        
        # Sort by score and return top results
        scored_assessments.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_assessments[:max_results]
    
    def get_recommendations_openai(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get recommendations using OpenAI."""
        try:
            # Use OpenAI to generate enhanced recommendations
            return get_enhanced_recommendations(self.assessments, query, max_results)
        except Exception as e:
            print(f"Error getting OpenAI recommendations: {str(e)}")
            # Fall back to keyword-based recommendations
            return self.get_recommendations_keywords(query, max_results)

    def get_recommendations(self, query: str, is_url: bool = False, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get assessment recommendations based on query or job description URL."""
        # Preprocess query
        processed_query = self.preprocess_query(query, is_url)
        
        if not processed_query:
            return []
        
        # Try to use OpenAI for enhanced recommendations
        try:
            # Only use OpenAI for longer queries or job descriptions to avoid unnecessary API calls
            if len(processed_query.split()) > 10:
                openai_recommendations = self.get_recommendations_openai(processed_query, max_results)
                if openai_recommendations:
                    return openai_recommendations
        except Exception as e:
            print(f"Error using OpenAI recommendations, falling back to basic methods: {str(e)}")
        
        # Fall back to basic methods if OpenAI fails or for very simple queries
        if self.model:
            recommendations = self.get_recommendations_embedding(processed_query, max_results)
        else:
            recommendations = self.get_recommendations_keywords(processed_query, max_results)
        
        # Ensure we have at least one result
        if not recommendations and self.assessments:
            recommendations = [self.assessments[0]]
        
        # Cap at max_results
        return recommendations[:max_results]
