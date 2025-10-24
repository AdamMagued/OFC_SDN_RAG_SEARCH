# RAG PY 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import os
from config import config
import re, logging, requests, json
from dataclasses import dataclass
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchCandidate:
    id: str
    name: str
    type: str
    similarity_score: float
    source: str
    additional_info: Dict

import os  # Add at top of file

class OfacSDNRAG:
    def __init__(self, ollama_url: str = None, ollama_model: str = None):
        self.sdn_df = None
        self.alt_df = None
        self.combined_names = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        # Use environment variables with fallback to localhost
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "mistral:latest")
        self.sdn_df = None
        self.alt_df = None
        self.combined_names = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def load_data(self, sdn_path: str, alt_path: str):
        try:
            sdn_cols = ["id", "sdn_name", "sdn_type", "program", "title",
                        "call_sign", "vess_type", "tonnage", "grt",
                        "vessel_flag", "vessel_owner", "remarks_split"]
            self.sdn_df = pd.read_csv(sdn_path, header=None, names=sdn_cols)

            alt_cols = ["alt_id", "entity_id", "type", "alt_name", "note"]
            self.alt_df = pd.read_csv(alt_path, header=None, names=alt_cols)
        except Exception as e:
            logger.error(f"Error loading CSV files: {e}")
            raise

    def preprocess_name(self, name: str) -> str:
        if not isinstance(name, str) or pd.isna(name):
            return ""
        name = name.lower().strip()
        name = re.sub(r'[^\w\s\-]', ' ', name)
        name = re.sub(r'\s+', ' ', name)
        prefixes = ['mr','mrs','ms','dr','prof','sir','ltd','inc','corp']
        words = [w for w in name.split() if w not in prefixes]
        return ' '.join(words)

    def translate_name_with_ollama(self, name: str) -> str:
        """
        Translate/transliterate name to English using Ollama API
        """
        try:
            prompt = f"""You are a name translator. Translate or transliterate the following name to English. 
Return ONLY the English name, nothing else. Do not add explanations, quotes, or additional text.
If the name is already in English, return it as is.

Name: {name}

English name:"""

            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 50
                }
            }

            response = requests.post(f"{self.ollama_url}/api/generate", 
                                   json=payload, 
                                   timeout=2000)
            
            if response.status_code == 200:
                result = response.json()
                translated_name = result.get('response', '').strip()
                
                # Clean up the response - remove any extra formatting
                translated_name = re.sub(r'^["\'\s]+|["\'\s]+$', '', translated_name)
                translated_name = translated_name.split('\n')[0]  # Take only first line
                
                logger.info(f"Translated '{name}' to '{translated_name}'")
                return translated_name if translated_name else name
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return name
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return name
        except Exception as e:
            logger.error(f"Unexpected error in translation: {e}")
            return name

    def build_search_index(self):
        sdn_names = []
        alt_names = []
        for idx, row in self.sdn_df.iterrows():
            name = self.preprocess_name(row.get('sdn_name',''))
            if name:
                sdn_names.append({'name': name,
                                  'original_name': row['sdn_name'],
                                  'id': str(row['id']),
                                  'type': row['sdn_type'],
                                  'source': 'sdn',
                                  'row_data': row.to_dict()})
        for idx, row in self.alt_df.iterrows():
            name = self.preprocess_name(row.get('alt_name',''))
            if name:
                alt_names.append({'name': name,
                                  'original_name': row['alt_name'],
                                  'id': str(row['entity_id']),
                                  'type': row['type'],
                                  'source': 'alt',
                                  'row_data': row.to_dict()})

        self.combined_names = sdn_names + alt_names
        if self.combined_names:
            names_list = [c['name'] for c in self.combined_names]
            self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3),
                                                    analyzer='char_wb',
                                                    min_df=1,
                                                    max_features=10000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(names_list)
        else:
            logger.warning("No names found to build TF-IDF index.")

    def calculate_similarity_scores(self, query, candidate):
        q = self.preprocess_name(query)
        c = self.preprocess_name(candidate)
        scores = {
            'fuzz_ratio': fuzz.ratio(q,c)/100,
            'fuzz_partial': fuzz.partial_ratio(q,c)/100,
            'fuzz_token_sort': fuzz.token_sort_ratio(q,c)/100,
            'fuzz_token_set': fuzz.token_set_ratio(q,c)/100
        }
        weights = {'fuzz_ratio':0.3,'fuzz_partial':0.2,'fuzz_token_sort':0.3,'fuzz_token_set':0.2}
        scores['weighted_average'] = sum(scores[k]*weights[k] for k in scores)
        return scores

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        # Translate the query to English using Ollama
        original_query = query
        translated_query = self.translate_name_with_ollama(query)
        
        logger.info(f"Searching with translated query: '{translated_query}' (original: '{original_query}')")

        if not self.combined_names or self.tfidf_matrix is None:
            logger.warning("TF-IDF index not built")
            return []

        query_vec = self.tfidf_vectorizer.transform([self.preprocess_name(translated_query)])
        cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(cosine_scores)[-top_k*3:][::-1]

        results = []
        seen = set()
        for idx in top_indices:
            if len(results) >= top_k:
                break
            cdata = self.combined_names[idx]
            if cdata['original_name'].lower() in seen:
                continue
            seen.add(cdata['original_name'].lower())
            scores = self.calculate_similarity_scores(translated_query, cdata['original_name'])
            
            # Include row_data in the result for DOB/birthplace extraction
            result_item = {
                'id': cdata['id'],
                'name': cdata['original_name'],
                'type': cdata['type'],
                'score': scores['weighted_average'],
                'source': cdata['source'],
                'row_data': cdata['row_data'],  # Add this line to include full row data
                'details': {
                    'tfidf_score': cosine_scores[idx], 
                    'scores': scores,
                    'original_query': original_query,
                    'translated_query': translated_query
                }
            }
            results.append(result_item)
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]