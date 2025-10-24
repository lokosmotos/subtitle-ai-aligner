import re
import math
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import gc

class SmartSubtitleAI:
    def __init__(self):
        # Use lazy loading to save memory
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Enhanced keyword matching for English-Chinese
        self.critical_phrases = {
            'yes': ['是', '对的', '好的', '没错', '是的'],
            'no': ['不', '不是', '不要', '没有', '别'],
            'hello': ['你好', '您好', '嗨', '哈喽'],
            'thank': ['谢谢', '感谢', '多谢', '谢谢您'],
            'sorry': ['对不起', '抱歉', '不好意思', '抱歉了'],
            'goodbye': ['再见', '拜拜', '再会', '下次见'],
            'please': ['请', '拜托', '求你了'],
            'what': ['什么', '何事', '干嘛', '干啥'],
            'why': ['为什么', '為何', '为啥', '为何'],
            'how': ['怎么', '如何', '怎样', '怎么样'],
            'where': ['哪里', '何处', '哪儿', '什么地方'],
            'when': ['什么时候', '何时', '啥时候'],
            'who': ['谁', '何人', '什么人'],
            'look': ['看', '瞧', '看看', '观看', '瞅'],
            'listen': ['听', '听着', '听听', '听我说'],
            'come': ['来', '过来', '来到', '来吧'],
            'go': ['去', '走', '离开', '走吧'],
            'big': ['大', '巨大', '很大', '大型', '大大的'],
            'small': ['小', '小小', '小型', '微小', '小小的'],
            'good': ['好', '很好', '不错', '优秀', '好的'],
            'bad': ['坏', '不好', '糟糕', '差劲', '坏的'],
            'love': ['爱', '爱情', '恋爱', '爱着', '爱上'],
            'family': ['家庭', '家人', '家', '家人'],
            'friend': ['朋友', '好友', '友人', '好朋友'],
            'school': ['学校', '上学', '校园', '学生'],
            'time': ['时间', '时候', '时光', '时刻'],
            'dream': ['梦想', '梦', '幻想', '做梦'],
            'home': ['家', '家里', '家庭', '家园'],
            'work': ['工作', '上班', '做事', '干活'],
            'water': ['水', '喝水', '水面', '水流'],
            'food': ['食物', '吃的', '食品', '饭菜'],
            'money': ['钱', '金钱', '货币', '钱财'],
            'day': ['天', '日子', '白天', '一天'],
            'night': ['晚上', '夜晚', '夜里', '黑夜'],
            'mother': ['妈妈', '母亲', '妈', '娘'],
            'father': ['爸爸', '父亲', '爸', '爹'],
            'child': ['孩子', '儿童', '小孩', '小朋友'],
            'man': ['男人', '男子', '男士', '汉子'],
            'woman': ['女人', '女子', '女士', '妇女'],
            'happy': ['开心', '快乐', '高兴', '幸福'],
            'sad': ['伤心', '悲伤', '难过', '悲哀'],
            'beautiful': ['美丽', '漂亮', '美', '好看'],
            'ugly': ['丑', '丑陋', '难看', '丑恶'],
        }
        self.learned_pairs = []

    def lazy_load_model(self):
        """Lazy load the model to save memory"""
        if not self.is_loaded:
            print("Loading ultra-lightweight AI model...")
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Memory optimization
            torch.set_grad_enabled(False)
            self.is_loaded = True
            print("Tiny model loaded successfully!")

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts):
        """Get embeddings for texts with memory optimization"""
        self.lazy_load_model()
        
        # Use smaller max_length to save memory
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, 
                                     return_tensors='pt', max_length=128)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()

    def time_to_seconds(self, time_str):
        """Convert SRT time to seconds"""
        try:
            time_str = str(time_str).replace(',', '.')
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
        """Calculate similarity with keyword fallback"""
        try:
            # Try keyword matching first (fast and memory efficient)
            keyword_score = self.keyword_similarity(eng_text, chi_text)
            
            # If keyword score is high, use it directly
            if keyword_score > 0.7:
                return keyword_score
            
            # Otherwise use the tiny model
            embeddings = self.get_embeddings([eng_text, chi_text])
            
            # Manual cosine similarity
            dot_product = np.dot(embeddings[0], embeddings[1])
            norm_a = np.linalg.norm(embeddings[0])
            norm_b = np.linalg.norm(embeddings[1])
            
            if norm_a > 0 and norm_b > 0:
                similarity = dot_product / (norm_a * norm_b)
                # Combine with keyword score for better accuracy
                combined_score = max(keyword_score, similarity)
                return min(1.0, combined_score)
            
            return keyword_score
            
        except Exception as e:
            print(f"Embedding error: {e}")
            # Fallback to keyword matching if anything fails
            return self.keyword_similarity(eng_text, chi_text)

    def keyword_similarity(self, eng_text, chi_text):
        """Enhanced keyword-based similarity"""
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
        
        return matches / max(possible, 1) if possible > 0 else 0.0

    def combined_scoring(self, eng_sub, chi_sub, semantic_score):
        """Combine semantic meaning with timing"""
        eng_time = self.time_to_seconds(eng_sub['start'])
        chi_time = self.time_to_seconds(chi_sub['start'])
        time_diff = abs(eng_time - chi_time)
        timing_score = max(0, 1 - (time_diff / 5.0))
        
        final_score = (semantic_score * 0.7 + timing_score * 0.3)
        return min(1.0, final_score)

    def align_subtitles(self, english_subs, chinese_subs):
        """Memory-safe alignment for large files"""
        print(f"Aligning {len(english_subs)} English and {len(chinese_subs)} Chinese subtitles")
        
        aligned_pairs = []
        
        # Process in very small batches for memory safety
        batch_size = 10
        for i in range(0, len(english_subs), batch_size):
            batch_end = min(i + batch_size, len(english_subs))
            
            for eng_idx in range(i, batch_end):
                eng_sub = english_subs[eng_idx]
                best_match = None
                best_score = 0
                
                # Search in reasonable window
                search_start = max(0, eng_idx - 15)
                search_end = min(len(chinese_subs), eng_idx + 15)
                
                for chi_idx in range(search_start, search_end):
                    chi_sub = chinese_subs[chi_idx]
                    
                    # Calculate similarity
                    semantic_score = self.semantic_similarity(eng_sub['text'], chi_sub['text'])
                    combined_score = self.combined_scoring(eng_sub, chi_sub, semantic_score)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = chi_sub
                
                # Determine status
                if best_score > 0.6:
                    status = 'ALIGNED'
                elif best_score > 0.3:
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
            
            # Clear memory after each batch
            gc.collect()
        
        print("Alignment completed successfully!")
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
        if was_correct and len(self.learned_pairs) < 50:
            self.learned_pairs.append({
                'english': english_text,
                'chinese': chinese_text
            })

    def clear_memory(self):
        """Clear model from memory when not needed"""
        if self.is_loaded:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            print("AI model cleared from memory")
