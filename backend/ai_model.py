import re
import math
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import gc

class SmartSubtitleAI:
    def __init__(self):
        # Use a smaller, more efficient multilingual model
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Memory optimization
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        self.learned_pairs = []

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

    def encode_batch_memory_safe(self, texts, batch_size=8):
        """Encode texts in batches to save memory"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
            all_embeddings.extend(embeddings)
            # Clear memory
            if i % 32 == 0:
                gc.collect()
        return np.array(all_embeddings)

    def semantic_similarity_batch(self, eng_embeddings, chi_embeddings, eng_idx, chi_idx):
        """Calculate similarity between precomputed embeddings"""
        eng_emb = eng_embeddings[eng_idx]
        chi_emb = chi_embeddings[chi_idx]
        
        # Manual cosine similarity to avoid tensor operations
        dot_product = np.dot(eng_emb, chi_emb)
        norm_eng = np.linalg.norm(eng_emb)
        norm_chi = np.linalg.norm(chi_emb)
        
        if norm_eng > 0 and norm_chi > 0:
            return dot_product / (norm_eng * norm_chi)
        return 0.0

    def combined_scoring(self, eng_sub, chi_sub, semantic_score):
        """Combine semantic meaning with timing"""
        eng_time = self.time_to_seconds(eng_sub['start'])
        chi_time = self.time_to_seconds(chi_sub['start'])
        time_diff = abs(eng_time - chi_time)
        timing_score = max(0, 1 - (time_diff / 5.0))
        
        final_score = (semantic_score * 0.7 + timing_score * 0.3)
        return min(1.0, final_score)

    def align_subtitles(self, english_subs, chinese_subs):
        """Memory-optimized alignment with proper BERT semantics"""
        print(f"Aligning {len(english_subs)} English and {len(chinese_subs)} Chinese subtitles")
        
        # Precompute all embeddings first (memory efficient)
        eng_texts = [sub['text'] for sub in english_subs]
        chi_texts = [sub['text'] for sub in chinese_subs]
        
        print("Computing English embeddings...")
        eng_embeddings = self.encode_batch_memory_safe(eng_texts, batch_size=4)
        
        print("Computing Chinese embeddings...")
        chi_embeddings = self.encode_batch_memory_safe(chi_texts, batch_size=4)
        
        aligned_pairs = []
        
        # Process in small chunks
        chunk_size = min(20, len(english_subs))
        for chunk_start in range(0, len(english_subs), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(english_subs))
            
            for eng_idx in range(chunk_start, chunk_end):
                eng_sub = english_subs[eng_idx]
                best_match = None
                best_score = 0
                best_chi_idx = -1
                
                # Search in reasonable time window
                search_start = max(0, eng_idx - 15)
                search_end = min(len(chinese_subs), eng_idx + 15)
                
                for chi_idx in range(search_start, search_end):
                    chi_sub = chinese_subs[chi_idx]
                    
                    # Calculate semantic similarity
                    semantic_score = self.semantic_similarity_batch(
                        eng_embeddings, chi_embeddings, eng_idx, chi_idx
                    )
                    
                    # Combined scoring
                    combined_score = self.combined_scoring(eng_sub, chi_sub, semantic_score)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = chi_sub
                        best_chi_idx = chi_idx
                
                # Determine status
                if best_score > 0.7:
                    status = 'ALIGNED'
                elif best_score > 0.4:
                    status = 'REVIEW'
                else:
                    status = 'MISALIGNED'
                
                aligned_pairs.append({
                    'sequence': eng_sub['id'],
                    'eng_time': eng_sub['start'],
                    'chi_time': best_match['start'] if best_match else 'NO MATCH',
                    'english': eng_sub['text'],
                    'chinese': best_match['text'] if best_match else 'NO MATCH',
                    'confidence': round(best_score, 3),
                    'status': status,
                    'match_quality': self.assess_match_quality(best_score)
                })
            
            # Clear memory between chunks
            gc.collect()
        
        return aligned_pairs

    def assess_match_quality(self, score):
        """Match quality assessment"""
        if score > 0.9:
            return "EXCELLENT"
        elif score > 0.8:
            return "VERY_GOOD"
        elif score > 0.7:
            return "GOOD"
        elif score > 0.6:
            return "FAIR"
        elif score > 0.4:
            return "WEAK"
        else:
            return "POOR"

    def learn_from_feedback(self, english_text, chinese_text, was_correct):
        """Learn from user corrections"""
        if was_correct:
            self.learned_pairs.append({
                'english': english_text,
                'chinese': chinese_text,
                'confidence_boost': 0.3
            })
        if len(self.learned_pairs) > 50:
            self.learned_pairs.pop(0)
