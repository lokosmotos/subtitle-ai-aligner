import re
import math

class SubtitleAI:
    def __init__(self):
        # Expanded critical phrase pairs for better matching
        self.critical_phrases = {
            'yes': ['是', '对的', '好的', '可以', '没错'],
            'no': ['不', '不是', '不要', '不行', '没有'],
            'look': ['看', '瞧', '看看', '看一下', '注视'],
            'ship': ['船', '帆船', '船只', '船舶', '舰船'],
            'big': ['大', '巨大', '很大', '大的', '庞大'],
            'why': ['为什么', '為何', '为啥', '为何', '何故'],
            'because': ['因为', '由於', '所以', '原因是'],
            'magic': ['神', '魔', '魔法', '神奇', '魔术'],
            'lamp': ['燈', '灯', '灯笼', '油灯'],
            'hello': ['你好', '您好', '嗨', '哈喽'],
            'thank you': ['谢谢', '感谢', '多谢', '谢谢你'],
            'sorry': ['对不起', '抱歉', '不好意思'],
            'okay': ['好', '可以', '行', '没问题', '好吧'],
            'please': ['请', '拜托', '求你了'],
            'what': ['什么', '啥', '何事', '干什么'],
            'where': ['哪里', '哪儿', '何处', '什么地方'],
            'when': ['什么时候', '何时', '几时'],
            'how': ['怎么', '如何', '怎样', '怎么样'],
            'who': ['谁', '何人', '什么人'],
            'good': ['好', '良好', '不错', '优秀', '棒'],
            'bad': ['坏', '不好', '糟糕', '差劲'],
            'happy': ['开心', '高兴', '快乐', '愉快'],
            'sad': ['伤心', '难过', '悲伤', '悲哀'],
            'beautiful': ['漂亮', '美丽', '好看', '优美'],
            'ugly': ['丑', '难看', '丑陋', '丑恶'],
            'love': ['爱', '喜欢', '爱情', '恋爱'],
            'hate': ['恨', '讨厌', '厌恶', '憎恨'],
            'want': ['要', '想要', '需要', '希望'],
            'have': ['有', '拥有', '具有', '具备'],
            'can': ['可以', '能', '能够', '会'],
            'cannot': ['不能', '不可以', '不会', '无法'],
            'will': ['会', '将要', '愿意', '打算'],
            'go': ['去', '走', '前往', '出发'],
            'come': ['来', '过来', '来到', '来临'],
            'see': ['看见', '看到', '见面', '见到'],
            'hear': ['听见', '听到', '听闻', '听说'],
            'say': ['说', '讲', '告诉', '说道'],
            'think': ['想', '思考', '认为', '觉得'],
            'know': ['知道', '了解', '认识', '明白'],
            'time': ['时间', '时候', '时刻', '时光'],
            'day': ['天', '日', '白天', '日子'],
            'night': ['晚上', '夜晚', '夜里', '晚间'],
            'water': ['水', '水分', '水域', '海水'],
            'fire': ['火', '火焰', '火灾', '火光'],
            'earth': ['土', '土地', '地球', '大地'],
            'air': ['空气', '天空', '大气', '空中'],
            'man': ['男人', '男子', '男士', '男性'],
            'woman': ['女人', '女子', '女士', '女性'],
            'child': ['孩子', '儿童', '小孩', '小朋友'],
            'friend': ['朋友', '友人', '好友', '伙伴'],
            'family': ['家庭', '家人', '家属', '亲人'],
            'home': ['家', '家庭', '家园', '家里'],
            'food': ['食物', '食品', '吃的', '饭菜'],
            'money': ['钱', '金钱', '货币', '资金'],
            'work': ['工作', '干活', '做事', '职业'],
            'life': ['生活', '生命', '人生', '生存'],
            'death': ['死亡', '死', '去世', '逝世']
            'children': ['孩子', '儿童', '小朋友'],
            'learning': ['学', '学习', '学到'],
            'dear': ['亲爱的', '亲爱'],
            'unclear': ['不确定', '不清楚', '不明'],
            'aladdin': ['阿拉丁'],
            'princess': ['公主'],
            'story': ['故事', '说说', '讲述'],
        }
        
        # Semantic meaning groups
        self.semantic_groups = {
            "wish_better": {
                "english": ["wish ours was that fancy", "i'd be so happy if", "want our ship to be better", "if only ours was", "hope ours was"],
                "chinese": ["我們的船好破爛", "船如果那麼漂亮就好了", "希望我們的船更好", "要是我們的船也這樣", "真想我們的船也這麼好"],
                "confidence": 0.9
            },
            "question_why": {
                "english": ["why is that", "because it looks better", "why", "what's the reason", "how come"],
                "chinese": ["為什麼", "就因為它比較新", "原因是", "為何這樣", "怎麼回事"],
                "confidence": 0.95
            },
            "ship_history": {
                "english": ["this boat has seen us through many storms", "it has been through storms", "survived many storms", "weather many storms"],
                "chinese": ["這艘船帶我們渡過很多暴風雨", "經歷過很多風雨", "闖過很多暴風雨", "經歷無數風雨"],
                "confidence": 0.85
            },
            "look_direction": {
                "english": ["hey look", "over there", "look at that", "check that out", "see there"],
                "chinese": ["妳看", "看那裡", "快看", "瞧那邊", "看那邊"],
                "confidence": 0.95
            },
            "comparison": {
                "english": ["better than", "worse than", "bigger than", "smaller than", "more than", "less than"],
                "chinese": ["比...好", "比...差", "比...大", "比...小", "比...多", "比...少"],
                "confidence": 0.8
            },
            "agreement": {
                "english": ["i agree", "that's right", "exactly", "you're right", "true"],
                "chinese": ["我同意", "沒錯", "確實", "你說得對", "是的"],
                "confidence": 0.9
            },
            "disagreement": {
                "english": ["i disagree", "that's wrong", "no way", "not true", "incorrect"],
                "chinese": ["我不同意", "不對", "不可能", "不是真的", "錯誤"],
                "confidence": 0.9
            }
        }

    def time_to_seconds(self, time_str):
        """Convert SRT time to seconds - simple version"""
        try:
            time_str = str(time_str)
            # Handle both , and . as decimal separator
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
        
        # 1. Timing similarity (50% weight)
        eng_time = self.time_to_seconds(eng_sub['start'])
        chi_time = self.time_to_seconds(chi_sub['start'])
        time_diff = abs(eng_time - chi_time)
        timing_score = max(0, 1 - (time_diff / 5))  # 5 second window
        score += timing_score * 0.5
        
        # 2. Content similarity (50% weight)
        content_score = self.calculate_content_similarity(eng_sub['text'], chi_sub['text'])
        score += content_score * 0.5
        
        return min(1.0, max(0, score))

    def calculate_content_similarity(self, eng_text, chi_text):
        """Calculate content similarity using rule-based methods"""
        eng_text_clean = self.clean_text(eng_text)
        chi_text_clean = chi_text
        
        # Method 1: Critical phrase matching (60%)
        critical_score = self.calculate_critical_score(eng_text_clean, chi_text_clean)
        
        # Method 2: Semantic group matching (30%)
        semantic_score = self.calculate_semantic_score(eng_text_clean, chi_text_clean)
        
        # Method 3: Structural similarity (10%)
        structure_score = self.calculate_structure_score(eng_text_clean, chi_text_clean)
        
        # Combine scores
        return (critical_score * 0.6 + semantic_score * 0.3 + structure_score * 0.1)

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
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
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
        
        return 0.1  # Low score if no semantic match

    def calculate_structure_score(self, eng_text, chi_text):
        """Score based on structural similarity"""
        score = 0
        
        # Check if both are questions
        eng_is_question = '?' in eng_text
        chi_is_question = '？' in chi_text or '?' in chi_text
        if eng_is_question == chi_is_question:
            score += 0.3
        
        # Check if both are exclamations
        eng_is_exclamation = '!' in eng_text
        chi_is_exclamation = '！' in chi_text or '!' in chi_text
        if eng_is_exclamation == chi_is_exclamation:
            score += 0.3
        
        # Check if both have similar length ratio
        eng_words = len(eng_text.split())
        chi_chars = len(chi_text)
        if eng_words > 0 and chi_chars > 0:
            # Rough ratio: English words to Chinese characters
            ratio = eng_words / chi_chars
            if 0.2 <= ratio <= 0.6:  # Reasonable range
                score += 0.4
        
        return min(1.0, score)

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
                'confidence': round(best_score, 2),
                'status': status
            })
        
        return aligned_pairs
