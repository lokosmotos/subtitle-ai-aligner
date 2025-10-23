import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SmartSubtitleAI:
    def __init__(self):
        # Lightweight TF-IDF vectorizer with limited features
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Critical phrases for English-Chinese matching
        self.critical_phrases = {
            'yes': ['是', '对的', '好的', '没错'],
            'no': ['不', '不是', '不要', '没有'],
            'hello': ['你好', '您好', '嗨'],
            'thank': ['谢谢', '感谢', '多谢'],
            'sorry': ['对不起', '抱歉', '不好意思'],
            'goodbye': ['再见', '拜拜', '再会'],
            'please': ['请', '拜托'],
            'what': ['什么', '何事', '干嘛'],
            'why': ['为什么', '為何', '为啥'],
            'how': ['怎么', '如何', '怎样'],
            'where': ['哪里', '何处', '哪儿'],
            'when': ['什么时候', '何时'],
            'who': ['谁', '何人'],
            'which': ['哪个', '哪一种'],
            'look': ['看', '瞧', '看看', '观看'],
            'listen': ['听', '听着', '听听'],
            'come': ['来', '过来', '来到'],
            'go': ['去', '走', '离开'],
            'big': ['大', '巨大', '很大', '大型'],
            'small': ['小', '小小', '小型', '微小'],
            'good': ['好', '很好', '不错', '优秀'],
            'bad': ['坏', '不好', '糟糕', '差劲'],
            'love': ['爱', '喜欢', '热爱', '爱情'],
            'hate': ['恨', '讨厌', '憎恨'],
            'family': ['家庭', '家人', '家规', '家族'],
            'friend': ['朋友', '友人', '好友'],
            'time': ['时间', '时候', '时光'],
            'day': ['天', '日子', '白天'],
            'night': ['晚上', '夜晚', '夜间'],
            'water': ['水', '水分', '水域'],
            'food': ['食物', '食品', '吃的'],
            'house': ['房子', '房屋', '家'],
            'car': ['车', '汽车', '车子'],
            'money': ['钱', '金钱', '货币'],
            'work': ['工作', '干活', '做事'],
            'school': ['学校', '上学', '校园'],
            'book': ['书', '书籍', '书本'],
            'movie': ['电影', '影片', '片子'],
            'music': ['音乐', '歌曲', '乐曲'],
            'man': ['男人', '男士', '男性'],
            'woman': ['女人', '女士', '女性'],
            'child': ['孩子', '儿童', '小孩'],
            'father': ['父亲', '爸爸', '爹'],
            'mother': ['母亲', '妈妈', '娘'],
            'son': ['儿子', '小子'],
            'daughter': ['女儿', '闺女'],
            'brother': ['兄弟', '哥哥', '弟弟'],
            'sister': ['姐妹', '姐姐', '妹妹'],
            'city': ['城市', '都市', '城区'],
            'country': ['国家', '乡村', '农村'],
            'world': ['世界', '地球', '天下'],
            'life': ['生活', '生命', '人生'],
            'death': ['死亡', '死', '去世'],
            'happy': ['快乐', '开心', '高兴'],
            'sad': ['悲伤', '伤心', '难过'],
            'angry': ['生气', '愤怒', '发怒'],
            'beautiful': ['美丽', '漂亮', '美好'],
            'ugly': ['丑陋', '难看', '丑'],
            'new': ['新', '新的', '新鲜'],
            'old': ['老', '旧', '古老'],
            'young': ['年轻', '年青', '幼小'],
            'hot': ['热', '炎热', '热门'],
            'cold': ['冷', '寒冷', '冷淡'],
            'fast': ['快', '快速', '迅速'],
            'slow': ['慢', '缓慢', '迟钝'],
            'strong': ['强', '强大', '强壮'],
            'weak': ['弱', '弱小', '虚弱'],
            'rich': ['富', '富有', '富裕'],
            'poor': ['穷', '贫穷', '贫困'],
            'right': ['对', '正确', '右边'],
            'wrong': ['错', '错误', '不对'],
            'true': ['真', '真实', '真正'],
            'false': ['假', '虚假', '伪造']
        }
        self.learned_pairs = []

    def time_to_seconds(self, time_str):
        """Convert SRT time to seconds - optimized"""
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
        """Calculate similarity using lightweight TF-IDF"""
        try:
            # Use keyword matching first (faster)
            keyword_score = self.keyword_similarity(eng_text, chi_text)
            if keyword_score > 0.8:
                return keyword_score
            
            # Fall back to TF-IDF if needed
            texts = [eng_text.lower(), chi_text]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return max(keyword_score, similarity)
        except:
            return self.keyword_similarity(eng_text, chi_text)

    def keyword_similarity(self, eng_text, chi_text):
        """Optimized keyword-based similarity"""
        eng_lower = eng_text.lower()
        matches = 0
        
        for eng_phrase, chi_phrases in self.critical_phrases.items():
            if eng_phrase in eng_lower:
                for chi_phrase in chi_phrases:
                    if chi_phrase in chi_text:
                        matches += 1
                        break
        
        # Normalize by number of English keywords found
        eng_keywords_found = sum(1 for phrase in self.critical_phrases if phrase in eng_lower)
        if eng_keywords_found > 0:
            return matches / eng_keywords_found
        return 0.0

    def combined_scoring(self, eng_sub, chi_sub, semantic_score):
        """Optimized combined scoring"""
        # Timing proximity (40% weight)
        eng_time = self.time_to_seconds(eng_sub['start'])
        chi_time = self.time_to_seconds(chi_sub['start'])
        time_diff = abs(eng_time - chi_time)
        timing_score = max(0, 1 - (time_diff / 3.0))  # 3-second window
        
        # Combined score
        final_score = (semantic_score * 0.6 + timing_score * 0.4)
        return min(1.0, final_score)

    def align_subtitles(self, english_subs, chinese_subs):
        """Memory-optimized alignment"""
        aligned_pairs = []
        
        # Process in smaller batches to save memory
        batch_size = min(50, len(english_subs))
        
        for i in range(0, len(english_subs), batch_size):
            batch_eng = english_subs[i:i + batch_size]
            
            for eng_sub in batch_eng:
                best_match = None
                best_score = 0
                
                # Search in reasonable window around English timestamp
                eng_time = self.time_to_seconds(eng_sub['start'])
                search_range = min(20, len(chinese_subs))
                
                start_idx = max(0, i - search_range)
                end_idx = min(len(chinese_subs), i + search_range)
                
                for chi_index in range(start_idx, end_idx):
                    chi_sub = chinese_subs[chi_index]
                    
                    # Quick timing check
                    chi_time = self.time_to_seconds(chi_sub['start'])
                    if abs(eng_time - chi_time) > 10:  # 10-second threshold
                        continue
                    
                    semantic_score = self.semantic_similarity(eng_sub['text'], chi_sub['text'])
                    combined_score = self.combined_scoring(eng_sub, chi_sub, semantic_score)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = chi_sub
                
                status = 'ALIGNED' if best_score > 0.6 else 'REVIEW' if best_score > 0.3 else 'MISALIGNED'
                
                aligned_pairs.append({
                    'sequence': eng_sub['id'],
                    'eng_time': eng_sub['start'],
                    'chi_time': best_match['start'] if best_match else 'NO MATCH',
                    'english': eng_sub['text'],
                    'chinese': best_match['text'] if best_match else 'NO MATCH',
                    'confidence': round(best_score, 3),
                    'status': status
                })
        
        return aligned_pairs

    def learn_from_feedback(self, english_text, chinese_text, was_correct):
        """Lightweight learning"""
        if was_correct and len(self.learned_pairs) < 50:  # Limit learned pairs
            self.learned_pairs.append({
                'english': english_text,
                'chinese': chinese_text
            })
