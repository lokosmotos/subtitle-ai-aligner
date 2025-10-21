import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SubtitleAI:
    def __init__(self):
        # Critical phrase pairs - expanded for better matching
        self.critical_phrases = {
            'yes': ['是', '对的', '好的', '可以'],
            'no': ['不', '不是', '不要', '不行'],
            'look': ['看', '瞧', '看看', '看一下'],
            'ship': ['船', '帆船', '船只', '船舶'],
            'big': ['大', '巨大', '很大', '大的'],
            'why': ['为什么', '為何', '为啥', '为何'],
            'because': ['因为', '由於', '所以'],
            'magic': ['神', '魔', '魔法', '神奇'],
            'lamp': ['燈', '灯', '灯笼'],
            'hello': ['你好', '您好', '嗨'],
            'thank you': ['谢谢', '感谢', '多谢'],
            'sorry': ['对不起', '抱歉'],
            'okay': ['好', '可以', '行', '没问题'],
            'please': ['请', '拜托'],
            'what': ['什么', '啥', '何事'],
            'where': ['哪里', '哪儿', '何处'],
            'when': ['什么时候', '何时'],
            'how': ['怎么', '如何', '怎样'],
            'who': ['谁', '何人'],
            'good': ['好', '良好', '不错'],
            'bad': ['坏', '不好', '糟糕'],
            'happy': ['开心', '高兴', '快乐'],
            'sad': ['伤心', '难过', '悲伤'],
            'beautiful': ['漂亮', '美丽', '好看'],
            'ugly': ['丑', '难看', '丑陋']
        }
        
        # Semantic meaning groups
        self.semantic_groups = {
            "wish_better": {
                "english": ["wish ours was that fancy", "i'd be so happy if", "want our ship to be better", "if only ours was"],
                "chinese": ["我們的船好破爛", "船如果那麼漂亮就好了", "希望我們的船更好", "要是我們的船也這樣"],
                "confidence": 0.9
            },
            "question_why": {
                "english": ["why is that", "because it looks better", "why", "what's the reason"],
                "chinese": ["為什麼", "就因為它比較新", "原因是", "為何這樣"],
                "confidence": 0.95
            },
            "ship_history": {
                "english": ["this boat has seen us through many storms", "it has been through storms", "survived many storms"],
                "chinese": ["這艘船帶我們渡過很多暴風雨", "經歷過很多風雨", "闖過很多暴風雨"],
                "confidence": 0.85
            },
            "look_direction": {
                "english": ["hey look", "over there", "look at that", "check that out"],
                "chinese": ["妳看", "看那裡", "快看", "瞧那邊"],
                "confidence": 0.95
            },
            "comparison": {
                "english": ["better than", "worse than", "bigger than", "smaller than"],
                "chinese": ["比...好", "比...差", "比...大", "比...小"],
                "confidence": 0.8
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
        
        # 1. Timing similarity (40%)
        eng_time = self.time_to_seconds(eng_sub['start'])
        chi_time = self.time_to_seconds(chi_sub['start'])
        time_diff = abs(eng_time - chi_time)
        timing_score = max(0, 1 - (time_diff / 5))  # 5 second window
        score += timing_score * 0.4
        
        # 2. Content similarity (60%)
        content_score = self.calculate_content_similarity(eng_sub['text'], chi_sub['text'])
        score += content_score * 0.6
        
        return min(1.0, score)

    def calculate_content_similarity(self, eng_text, chi_text):
        """Calculate content similarity using multiple lightweight methods"""
        eng_text_clean = self.clean_text(eng_text)
        chi_text_clean = chi_text
        
        # Method 1: Critical phrase matching (40%)
        critical_score = self.calculate_critical_score(eng_text_clean, chi_text_clean)
        
        # Method 2: Semantic group matching (30%)
        semantic_score = self.calculate_semantic_score(eng_text_clean, chi_text_clean)
        
        # Method 3: Length-based similarity (20%)
        length_score = self.calculate_length_score(eng_text_clean, chi_text_clean)
        
        # Method 4: Structural similarity (10%)
        structure_score = self.calculate_structure_score(eng_text_clean, chi_text_clean)
        
        # Combine scores
        return (critical_score * 0.4 + semantic_score * 0.3 + 
                length_score * 0.2 + structure_score * 0.1)

    def clean_text(self, text):
        """Clean text by removing CC markers, speaker names, etc."""
        if not text:
            return ""
            
        # Remove speaker names (OMAR: )
        text = re.sub(r'^[A-Z]+:\s*', '', text)
        # Remove sound effects in parentheses
        text = re.sub(r'\([^)]*\)', '', text)
        # Remove music markers
        text = re.sub(r'♪', '', text)
        # Remove anything in brackets
        text = re.sub(r'\[[^\]]*\]', '', text)
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

    def calculate_semantic_score(self, eng_text, chi_text):
        """Score based on semantic group matches"""
        eng_lower = eng_text.lower()
        
        for group_name, group_data in self.semantic_groups.items():
            has_english = any(phrase in eng_lower for phrase in group_data["english"])
            has_chinese = any(phrase in chi_text for phrase in group_data["chinese"])
            
            if has_english and has_chinese:
                return group_data["confidence"]
        
        return 0.2

    def calculate_length_score(self, eng_text, chi_text):
        """Score based on text length similarity"""
        eng_words = len(eng_text.split())
        chi_chars = len(chi_text)
        
        # Chinese is more dense, so adjust ratio
        ideal_ratio = 0.3  # 1 English word ≈ 3 Chinese characters
        actual_ratio = eng_words / max(chi_chars, 1)
        
        return max(0, 1 - abs(actual_ratio - ideal_ratio))

    def calculate_structure_score(self, eng_text, chi_text):
        """Score based on structural similarity"""
        score = 0
        
        # Check if both are questions
        eng_is_question = '?' in eng_text
        chi_is_question = '？' in chi_text or '?' in chi_text
        if eng_is_question and chi_is_question:
            score += 0.5
        
        # Check if both are exclamations
        eng_is_exclamation = '!' in eng_text
        chi_is_exclamation = '！' in chi_text or '!' in chi_text
        if eng_is_exclamation and chi_is_exclamation:
            score += 0.5
        
        return score

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
            
            # Determine status based on confidence
            if best_score > 0.7:
                status = 'ALIGNED'
            elif best_score > 0.4:
                status = 'REVIEW'
            else:
                status = 'MISALIGNED'
                best_match = None
            
            aligned_pairs.append({
                'sequence': eng_sub['id'],
                'eng_time': eng_sub['start'],
                'chi_time': best_match['start'] if best_match else 'NO MATCH',
                'english': eng_sub['text'],
                'chinese': best_match['text'] if best_match else 'NO MATCH',
                'confidence': best_score,
                'status': status
            })
        
        return aligned_pairs
