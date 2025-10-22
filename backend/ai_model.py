import re
import math
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

class SmartSubtitleAI:
    def __init__(self):
        # Load multilingual model for English-Chinese semantic matching
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        
        # Learning system
        self.learned_pairs = []
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.3
        }
        
        # Critical phrases as fallback
        self.critical_phrases = {
            'yes': ['是', '对的', '好的'],
            'no': ['不', '不是', '不要'],
            'look': ['看', '瞧', '看看'],
            'ship': ['船', '帆船', '船只'],
            'big': ['大', '巨大', '很大'],
            'why': ['为什么', '為何', '为啥'],
            'because': ['因为', '由於', '所以'],
            'magic': ['神', '魔', '魔法'],
            'lamp': ['燈', '灯', '灯笼'],
            'family': ['家庭', '家人', '家规'],
            'parents': ['父母', '爸妈', '家长'],
            'honor': ['孝顺', '尊敬', '尊重']
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

    def semantic_similarity(self, eng_text, chi_text):
        """Calculate semantic similarity using multilingual embeddings"""
        try:
            # Encode both texts
            embeddings = self.model.encode([eng_text, chi_text], convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            
            return max(0, similarity)  # Ensure non-negative
        except Exception as e:
            # Fallback to keyword matching if embedding fails
            return self.keyword_similarity(eng_text, chi_text)

    def keyword_similarity(self, eng_text, chi_text):
        """Fallback keyword-based similarity"""
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

    def combined_scoring(self, eng_sub, chi_sub, semantic_score):
        """Combine semantic meaning with timing and learned patterns"""
        # Timing proximity (30% weight)
        eng_time = self.time_to_seconds(eng_sub['start'])
        chi_time = self.time_to_seconds(chi_sub['start'])
        time_diff = abs(eng_time - chi_time)
        timing_score = max(0, 1 - (time_diff / 5.0))  # 5-second window
        
        # Learned pattern boost (20% weight)
        learned_boost = self.get_learned_boost(eng_sub['text'], chi_sub['text'])
        
        # Combined score
        final_score = (
            semantic_score * 0.5 +      # Semantic meaning (50%)
            timing_score * 0.3 +        # Timing (30%)
            learned_boost * 0.2         # Learned patterns (20%)
        )
        
        return min(1.0, final_score)

    def get_learned_boost(self, eng_text, chi_text):
        """Boost score based on learned successful pairs"""
        for learned_pair in self.learned_pairs:
            if (learned_pair['english'].lower() in eng_text.lower() and
                learned_pair['chinese'] in chi_text):
                return learned_pair['confidence_boost']
        return 0.0

    def context_aware_matching(self, english_subs, chinese_subs, eng_index, chi_index, window=2):
        """Check surrounding context for better matching"""
        context_score = 0
        context_pairs = 0
        
        # Check previous and next subtitles
        for offset in range(-window, window + 1):
            if offset == 0:
                continue  # Skip current pair
                
            eng_ctx_idx = eng_index + offset
            chi_ctx_idx = chi_index + offset
            
            if (0 <= eng_ctx_idx < len(english_subs) and 
                0 <= chi_ctx_idx < len(chinese_subs)):
                
                eng_ctx = english_subs[eng_ctx_idx]['text']
                chi_ctx = chinese_subs[chi_ctx_idx]['text']
                
                # If surrounding context also matches well
                ctx_similarity = self.semantic_similarity(eng_ctx, chi_ctx)
                if ctx_similarity > 0.6:
                    context_score += ctx_similarity
                    context_pairs += 1
        
        return context_score / max(context_pairs, 1) if context_pairs > 0 else 0

    def align_subtitles(self, english_subs, chinese_subs):
        """Smart alignment with semantic understanding"""
        aligned_pairs = []
        
        for eng_index, eng_sub in enumerate(english_subs):
            best_match = None
            best_score = 0
            best_chi_index = -1
            
            for chi_index, chi_sub in enumerate(chinese_subs):
                # Calculate semantic similarity
                semantic_score = self.semantic_similarity(eng_sub['text'], chi_sub['text'])
                
                # Combined scoring with timing and learned patterns
                combined_score = self.combined_scoring(eng_sub, chi_sub, semantic_score)
                
                # Context awareness boost
                context_boost = self.context_aware_matching(english_subs, chinese_subs, eng_index, chi_index)
                final_score = combined_score * 0.9 + context_boost * 0.1
                
                if final_score > best_score:
                    best_score = final_score
                    best_match = chi_sub
                    best_chi_index = chi_index
            
            # Determine status with adaptive thresholds
            status = self.determine_status(best_score, eng_sub['text'])
            match_quality = self.assess_match_quality(best_score)
            
            aligned_pairs.append({
                'sequence': eng_sub['id'],
                'eng_time': eng_sub['start'],
                'chi_time': best_match['start'] if best_match else 'NO MATCH',
                'english': eng_sub['text'],
                'chinese': best_match['text'] if best_match else 'NO MATCH',
                'confidence': round(best_score, 3),
                'status': status,
                'match_quality': match_quality,
                'semantic_score': round(best_score, 3)
            })
        
        return aligned_pairs

    def determine_status(self, score, eng_text):
        """Adaptive status determination"""
        eng_text_lower = eng_text.lower()
        
        # Simple phrases need lower threshold
        if any(simple in eng_text_lower for simple in ['yes', 'no', 'okay', 'hello', 'thank you']):
            threshold_aligned = 0.6
        # Complex sentences need higher threshold
        elif len(eng_text_lower.split()) > 8:
            threshold_aligned = 0.75
        # Default thresholds
        else:
            threshold_aligned = 0.7
        
        if score > threshold_aligned:
            return 'ALIGNED'
        elif score > 0.4:
            return 'REVIEW'
        else:
            return 'MISALIGNED'

    def assess_match_quality(self, score):
        """Detailed match quality assessment"""
        if score > 0.9:
            return "EXCELLENT - Near perfect semantic match"
        elif score > 0.8:
            return "VERY_GOOD - Strong meaning alignment"
        elif score > 0.7:
            return "GOOD - Reliable semantic match"
        elif score > 0.6:
            return "FAIR - Moderate meaning similarity"
        elif score > 0.4:
            return "WEAK - Low semantic similarity"
        else:
            return "POOR - Little meaningful connection"

    def learn_from_feedback(self, english_text, chinese_text, was_correct):
        """Learn from user corrections"""
        if was_correct:
            self.learned_pairs.append({
                'english': english_text,
                'chinese': chinese_text,
                'confidence_boost': 0.3,
                'usage_count': 1
            })
        # Keep only recent learning
        if len(self.learned_pairs) > 100:
            self.learned_pairs.pop(0)
