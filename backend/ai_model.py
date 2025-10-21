import numpy as np
from sentence_transformers import SentenceTransformer
import re

class SubtitleAI:
    def __init__(self):
        # Load multilingual model for English-Chinese
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Critical phrase pairs
        self.critical_phrases = {
            'yes': ['是', '对的', '好的'],
            'no': ['不', '不是', '不要'],
            'look': ['看', '瞧', '看看'],
            'ship': ['船', '帆船'],
            'big': ['大', '巨大'],
            'why': ['为什么', '為何'],
            'because': ['因为', '由於'],
            'magic': ['神', '魔', '魔法'],
            'lamp': ['燈', '灯'],
            'hello': ['你好', '您好'],
            'thank you': ['谢谢', '感谢']
        }
        
        # Semantic meaning groups
        self.semantic_groups = {
            "wish_better": {
                "english": ["wish ours was that fancy", "i'd be so happy if", "want our ship to be better"],
                "chinese": ["我們的船好破爛", "船如果那麼漂亮就好了", "希望我們的船更好"],
                "confidence": 0.9
            },
            "question_why": {
                "english": ["why is that", "because it looks better", "why"],
                "chinese": ["為什麼", "就因為它比較新", "原因是"],
                "confidence": 0.95
            }
        }

    def time_to_seconds(self, time_str):
        """Convert SRT time to seconds"""
        try:
            time_str = str(time_str)
            if ',' in time_str:
                time_str = time_str.replace(',', '.')
            parts = time_str.split(':')
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
        except:
            pass
        return 0

    def calculate_match_score(self, eng_sub, chi_sub):
        """Calculate how well English and Chinese subtitles match"""
        score = 0
        
        # 1. Timing similarity (30%)
        eng_time = self.time_to_seconds(eng_sub['start'])
        chi_time = self.time_to_seconds(chi_sub['start'])
        time_diff = abs(eng_time - chi_time)
        timing_score = max(0, 1 - (time_diff / 5))  # 5 second window
        score += timing_score * 0.3
        
        # 2. Content similarity (50%)
        content_score = self.calculate_content_similarity(eng_sub['text'], chi_sub['text'])
        score += content_score * 0.5
        
        # 3. Length ratio (20%)
        eng_words = len(eng_sub['text'].split())
        chi_chars = len(chi_sub['text'])
        length_ratio = eng_words / max(chi_chars * 0.3, 1)  # Chinese is more dense
        length_score = max(0, 1 - abs(length_ratio - 1))
        score += length_score * 0.2
        
        return min(1.0, score)

    def calculate_content_similarity(self, eng_text, chi_text):
        """Calculate content similarity using multiple methods"""
        eng_text_clean = self.clean_text(eng_text)
        chi_text_clean = chi_text
        
        # Method 1: Critical phrase matching
        critical_score = self.calculate_critical_score(eng_text_clean, chi_text_clean)
        
        # Method 2: Semantic similarity (using sentence transformers)
        semantic_score = self.calculate_semantic_similarity(eng_text_clean, chi_text_clean)
        
        # Method 3: Keyword matching
        keyword_score = self.calculate_keyword_score(eng_text_clean, chi_text_clean)
        
        # Combine scores
        return (critical_score * 0.4 + semantic_score * 0.4 + keyword_score * 0.2)

    def clean_text(self, text):
        """Clean text by removing CC markers, speaker names, etc."""
        # Remove speaker names (OMAR: )
        text = re.sub(r'^[A-Z]+:\s*', '', text)
        # Remove sound effects in parentheses
        text = re.sub(r'\([^)]*\)', '', text)
        # Remove music markers
        text = re.sub(r'♪', '', text)
        return text.strip()

    def calculate_critical_score(self, eng_text, chi_text):
        """Score based on critical phrase matches"""
        eng_lower = eng_text.lower()
        matches = 0
        possible = 0
        
        for eng_phrase, chi_phrases in self.critical_phrases.items():
            has_eng = eng_phrase in eng_lower
            has_chi = any(chi_phrase in chi_text for chi_phrase in chi_phrases)
            
            if has_eng or has_chi:
                possible += 1
                if has_eng and has_chi:
                    matches += 1
        
        return matches / max(possible, 1)

    def calculate_semantic_similarity(self, eng_text, chi_text):
        """Calculate semantic similarity using AI model"""
        try:
            embeddings = self.model.encode([eng_text, chi_text])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return max(0, float(similarity))
        except:
            return 0.3

    def calculate_keyword_score(self, eng_text, chi_text):
        """Simple keyword matching score"""
        # This can be enhanced with more sophisticated NLP
        return self.calculate_critical_score(eng_text, chi_text)

    def align_subtitles(self, english_subs, chinese_subs):
        """Align English and Chinese subtitles"""
        aligned_pairs = []
        
        for eng_sub in english_subs:
            best_match = None
            best_score = 0
            
            for chi_sub in chinese_subs:
                score = self.calculate_match_score(eng_sub, chi_sub)
                if score > best_score:
                    best_score = score
                    best_match = chi_sub
            
            aligned_pairs.append({
                'sequence': eng_sub['id'],
                'eng_time': eng_sub['start'],
                'chi_time': best_match['start'] if best_match else 'NO MATCH',
                'english': eng_sub['text'],
                'chinese': best_match['text'] if best_match else 'NO MATCH',
                'confidence': best_score,
                'status': 'ALIGNED' if best_score > 0.7 else 'REVIEW' if best_score > 0.4 else 'MISALIGNED'
            })
        
        return aligned_pairs
