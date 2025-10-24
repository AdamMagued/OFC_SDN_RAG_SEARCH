# RAG API
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag import OfacSDNRAG
from typing import Optional
import os
from config import config
import re
import requests
import json

app = FastAPI(title="OFAC SDN RAG API with Ollama Translation and Comparison")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

class QueryRequest(BaseModel):
    query: str
    dob: Optional[str] = None
    birthplace: Optional[str] = None

# ---- Load RAG with Ollama configuration ----
rag = OfacSDNRAG(
    ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"), 
    ollama_model=os.getenv("OLLAMA_MODEL", "mistral:latest")
)
SDN_CSV_PATH = os.getenv("SDN_CSV_PATH", "./data/sdn.csv")
ALT_CSV_PATH = os.getenv("ALT_CSV_PATH", "./data/alt.csv")
rag.load_data(SDN_CSV_PATH, ALT_CSV_PATH)
rag.build_search_index()

def extract_dob_from_remarks(remarks: str) -> str:
    """Extract date of birth information from remarks field"""
    if not remarks or remarks == "-0-":
        return ""
    
    # Look for DOB patterns
    dob_patterns = [
        r'DOB\s+(\d{1,2}\s+\w+\s+\d{4})',  # DOB 25 Jan 1970
        r'DOB\s+(\d{1,2}/\d{1,2}/\d{4})',   # DOB 01/25/1970
        r'DOB\s+(\d{4}-\d{1,2}-\d{1,2})',   # DOB 1970-01-25
        r'alt\.\s*DOB\s+(\d{1,2}\s+\w+\s+\d{4})',  # alt. DOB 29 May 1966
    ]
    
    dobs = []
    for pattern in dob_patterns:
        matches = re.findall(pattern, remarks, re.IGNORECASE)
        dobs.extend(matches)
    
    return "; ".join(dobs) if dobs else ""

def extract_birthplace_from_remarks(remarks: str) -> str:
    """Extract birthplace information from remarks field"""
    if not remarks or remarks == "-0-":
        return ""
    
    # Look for POB (Place of Birth) patterns
    pob_patterns = [
        r'POB\s+([^;]+?)(?:;|nationality|DOB|Secondary|$)',  # POB location
        r'POB\s+(.+?)(?:;)',  # More specific POB extraction
    ]
    
    for pattern in pob_patterns:
        match = re.search(pattern, remarks, re.IGNORECASE)
        if match:
            birthplace = match.group(1).strip()
            # Clean up common trailing patterns
            birthplace = re.sub(r'\s*nationality.*$', '', birthplace, flags=re.IGNORECASE)
            return birthplace
    
    return ""

def extract_nationality_from_remarks(remarks: str) -> str:
    """Extract nationality information from remarks field"""
    if not remarks or remarks == "-0-":
        return ""
    
    # Look for nationality patterns
    nat_pattern = r'nationality\s+([^;]+?)(?:;|Secondary|$)'
    match = re.search(nat_pattern, remarks, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return ""

def extract_program_from_remarks(remarks: str) -> str:
    """Extract program information from remarks field"""
    if not remarks or remarks == "-0-":
        return ""
    
    # Look for program patterns
    prog_patterns = [
        r'Secondary sanctions risk:\s*([^;]+)',
        r'Executive Order\s+(\d+)',
    ]
    
    for pattern in prog_patterns:
        match = re.search(pattern, remarks, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ""

def enhance_candidate_details(candidate):
    """Add DOB and birthplace details to candidate"""
    details = candidate.get('details', {})
    
    # Try to get row data to extract DOB and birthplace
    if 'row_data' in candidate:
        row_data = candidate['row_data']
        remarks = str(row_data.get('remarks_split', ''))
        
        details['dob_info'] = extract_dob_from_remarks(remarks)
        details['birthplace_info'] = extract_birthplace_from_remarks(remarks)
        details['nationality'] = extract_nationality_from_remarks(remarks)
        details['program'] = row_data.get('program', '') or extract_program_from_remarks(remarks)
        
        candidate['details'] = details
    
    return candidate

def call_ollama_for_comparison(query_name: str, query_dob: str, query_birthplace: str, 
                              candidate_name: str, candidate_dob: str, candidate_birthplace: str, 
                              match_score: float) -> dict:
    """Use Ollama to intelligently compare query with candidate and make a decision"""
    
    # Construct a detailed prompt for Ollama
    prompt = f"""You are an expert in identity matching for OFAC sanctions screening. Your task is to determine if a search query matches a candidate from the sanctions database.

SEARCH QUERY:
- Name: {query_name}
- Date of Birth: {query_dob if query_dob else "Not provided"}
- Birthplace: {query_birthplace if query_birthplace else "Not provided"}

CANDIDATE MATCH:
- Name: {candidate_name}
- Date of Birth: {candidate_dob if candidate_dob else "Not available"}
- Birthplace: {candidate_birthplace if candidate_birthplace else "Not available"}
- Name Similarity Score: {match_score:.3f}

ANALYSIS INSTRUCTIONS:
1. Consider name similarity, date of birth matching, and birthplace matching
2. Account for variations in name spelling, transliteration, and cultural differences
3. Consider partial matches in dates (same year, similar dates)
4. Consider geographical proximity or alternative names for places
5. If DOB or birthplace is missing on either side, focus on available information
6. Remember that sanctions targets may use aliases or have data variations

Provide your response in this exact JSON format:
{{
    "decision": "MATCH" or "NO_MATCH" or "POSSIBLE_MATCH",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your decision",
    "recommendation": "Action recommendation"
}}

Decision criteria:
- MATCH: High confidence this is the same person (consider strong name match + DOB/birthplace confirmation)
- POSSIBLE_MATCH: Reasonable chance this could be the same person but requires human review
- NO_MATCH: Low confidence this is the same person

Respond only with the JSON, no additional text."""

    try:
        response = requests.post(
            f"{rag.ollama_url}/api/generate",
            json={
                "model": rag.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower temperature for more consistent responses
                    "num_predict": 200
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            ollama_response = result.get('response', '').strip()
            
            # Try to extract JSON from the response
            try:
                # Find JSON in the response
                start_idx = ollama_response.find('{')
                end_idx = ollama_response.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = ollama_response[start_idx:end_idx]
                    parsed_result = json.loads(json_str)
                    return parsed_result
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError):
                # Fallback to simple text analysis if JSON parsing fails
                response_lower = ollama_response.lower()
                if "match" in response_lower and "no" not in response_lower:
                    decision = "POSSIBLE_MATCH"
                    confidence = 0.6
                elif "no" in response_lower and "match" in response_lower:
                    decision = "NO_MATCH" 
                    confidence = 0.3
                else:
                    decision = "POSSIBLE_MATCH"
                    confidence = 0.5
                
                return {
                    "decision": decision,
                    "confidence": confidence,
                    "reasoning": "Ollama response parsing failed, using fallback analysis",
                    "recommendation": "Human review recommended due to parsing error"
                }
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
            
    except Exception as e:
        # Fallback logic if Ollama fails
        fallback_confidence = match_score
        
        # Simple fallback decision logic
        if match_score > 0.8:
            if (query_dob and candidate_dob and query_dob in candidate_dob) or \
               (query_birthplace and candidate_birthplace and query_birthplace.lower() in candidate_birthplace.lower()):
                decision = "MATCH"
                confidence = 0.8
            else:
                decision = "POSSIBLE_MATCH"
                confidence = 0.6
        elif match_score > 0.6:
            decision = "POSSIBLE_MATCH"
            confidence = 0.5
        else:
            decision = "NO_MATCH"
            confidence = 0.3
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": f"Ollama unavailable ({str(e)}), using fallback logic based on name similarity score",
            "recommendation": "Human review recommended due to system limitation"
        }

def calculate_dob_similarity(query_dob: str, candidate_dob: str) -> float:
    """Calculate similarity between query DOB and candidate DOB"""
    if not query_dob or not candidate_dob:
        return 0.0
    
    # Simple date matching - convert to comparable format
    query_parts = re.findall(r'\d+', query_dob)
    candidate_parts = re.findall(r'\d+', candidate_dob)
    
    if len(query_parts) >= 3 and len(candidate_parts) >= 3:
        # Compare year, month, day
        year_match = query_parts[-1] == candidate_parts[-1]  # Year usually last
        month_day_matches = sum(1 for q, c in zip(query_parts[:2], candidate_parts[:2]) if q == c)
        
        if year_match and month_day_matches == 2:
            return 1.0  # Perfect match
        elif year_match and month_day_matches == 1:
            return 0.7  # Partial match
        elif year_match:
            return 0.4  # Year match only
    
    return 0.0

def calculate_birthplace_similarity(query_place: str, candidate_place: str) -> float:
    """Calculate similarity between query birthplace and candidate birthplace"""
    if not query_place or not candidate_place:
        return 0.0
    
    query_lower = query_place.lower()
    candidate_lower = candidate_place.lower()
    
    # Exact match
    if query_lower == candidate_lower:
        return 1.0
    
    # Partial matches
    if query_lower in candidate_lower or candidate_lower in query_lower:
        return 0.8
    
    # Word-level matching
    query_words = set(query_lower.split())
    candidate_words = set(candidate_lower.split())
    
    if query_words & candidate_words:  # Any common words
        overlap = len(query_words & candidate_words)
        total = len(query_words | candidate_words)
        return 0.5 * (overlap / total) if total > 0 else 0.0
    
    return 0.0

@app.post("/query")
def query_api(request: QueryRequest, top_k: int = Query(10)):
    # Search using the name (existing functionality preserved)
    candidates = rag.search(request.query, top_k=top_k)
    
    # Enhance candidates with DOB and birthplace details
    enhanced_candidates = []
    for candidate in candidates:
        enhanced_candidate = enhance_candidate_details(candidate)
        enhanced_candidates.append(enhanced_candidate)
    
    # Apply additional filtering based on DOB and birthplace if provided
    if request.dob or request.birthplace:
        filtered_candidates = []
        for candidate in enhanced_candidates:
            candidate_details = candidate.get('details', {})
            match_score_adjustment = 0
            
            # Check DOB match if provided
            if request.dob and candidate_details.get('dob_info'):
                dob_similarity = calculate_dob_similarity(request.dob, candidate_details['dob_info'])
                match_score_adjustment += dob_similarity * 0.3  # Max 0.3 boost for perfect DOB match
            
            # Check birthplace match if provided  
            if request.birthplace and candidate_details.get('birthplace_info'):
                place_similarity = calculate_birthplace_similarity(request.birthplace, candidate_details['birthplace_info'])
                match_score_adjustment += place_similarity * 0.2  # Max 0.2 boost for perfect birthplace match
            
            # Adjust the score if we have DOB or birthplace matches
            if match_score_adjustment > 0:
                candidate['score'] = min(1.0, candidate['score'] + match_score_adjustment)
                candidate['details']['score_adjustment'] = match_score_adjustment
                candidate['details']['dob_similarity'] = calculate_dob_similarity(request.dob or "", candidate_details.get('dob_info', ''))
                candidate['details']['birthplace_similarity'] = calculate_birthplace_similarity(request.birthplace or "", candidate_details.get('birthplace_info', ''))
            
            filtered_candidates.append(candidate)
        
        # Re-sort by adjusted scores
        enhanced_candidates = sorted(filtered_candidates, key=lambda x: x['score'], reverse=True)
    
    best = enhanced_candidates[0] if enhanced_candidates else None
    
    # Use Ollama for intelligent decision making if we have a best candidate
    ollama_analysis = None
    final_decision = "NO"
    confidence = 0.0
    
    if best:
        best_details = best.get('details', {})
        ollama_analysis = call_ollama_for_comparison(
            query_name=request.query,
            query_dob=request.dob or "",
            query_birthplace=request.birthplace or "",
            candidate_name=best['name'],
            candidate_dob=best_details.get('dob_info', ''),
            candidate_birthplace=best_details.get('birthplace_info', ''),
            match_score=best['score']
        )
        
        # Map Ollama decision to YES/NO format
        if ollama_analysis['decision'] == "MATCH":
            final_decision = "YES"
        elif ollama_analysis['decision'] == "POSSIBLE_MATCH":
            final_decision = "POSSIBLE"  # New category for uncertain matches
        else:
            final_decision = "NO"
            
        confidence = ollama_analysis.get('confidence', best['score'])

    # Include translation info in response
    translation_info = {}
    if best and 'details' in best:
        translation_info = {
            'original_query': best['details'].get('original_query', request.query),
            'translated_query': best['details'].get('translated_query', request.query)
        }

    # Prepare best match details
    best_match_details = None
    if best and 'details' in best:
        details = best['details']
        best_match_details = {
            'dob_info': details.get('dob_info', ''),
            'birthplace_info': details.get('birthplace_info', ''),
            'nationality': details.get('nationality', ''),
            'program': details.get('program', ''),
            'score_adjustment': details.get('score_adjustment', 0),
            'dob_similarity': details.get('dob_similarity', 0),
            'birthplace_similarity': details.get('birthplace_similarity', 0)
        }

    return {
        "query": request.query,
        "dob_query": request.dob,
        "birthplace_query": request.birthplace,
        "decision": final_decision,
        "best_match_name": best["name"] if best else None,
        "best_match_score": best["score"] if best else None,
        "best_match_details": best_match_details,
        "confidence": confidence,
        "ollama_analysis": ollama_analysis,  # Include Ollama's detailed analysis
        "translation_info": translation_info,
        "candidates_debug": enhanced_candidates[:10]  # Return top 10 for display
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "ollama_url": rag.ollama_url, "ollama_model": rag.ollama_model}

@app.get("/test-translation/{name}")
def test_translation(name: str):
    """Test endpoint to see what Ollama translates a name to"""
    translated = rag.translate_name_with_ollama(name)
    return {
        "original": name,
        "translated": translated,
        "preprocessed": rag.preprocess_name(translated)
    }

@app.get("/search-names/{pattern}")
def search_names_in_db(pattern: str):
    """Search for names in the database that contain a pattern"""
    matches = []
    pattern_lower = pattern.lower()

    for item in rag.combined_names[:1000]:  # Limit to first 1000 to avoid huge response
        if pattern_lower in item['original_name'].lower():
            matches.append({
                "name": item['original_name'],
                "source": item['source'],
                "type": item['type'],
                "id": item['id']
            })

    return {"pattern": pattern, "matches": matches[:20]}  # Return top 20 matches

@app.get("/extract-details/{entity_id}")
def extract_details(entity_id: str):
    """Extract DOB and birthplace details for a specific entity"""
    for item in rag.combined_names:
        if item['id'] == entity_id:
            enhanced_item = enhance_candidate_details({
                'row_data': item['row_data'],
                'details': {}
            })
            return {
                "entity_id": entity_id,
                "name": item['original_name'],
                "details": enhanced_item['details']
            }
    
    return {"error": "Entity not found"}