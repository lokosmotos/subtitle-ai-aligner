from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import re
import math
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import gc

app = Flask(__name__)
CORS(app)

# Memory-optimized AI Model
class SmartSubtitleAI:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
    
    def lazy_load_model(self):
        if not self.is_loaded:
            print("Loading lightweight AI model...")
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            torch.set_grad_enabled(False)
            self.is_loaded = True
            print("AI model loaded successfully!")
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embeddings(self, texts):
        self.lazy_load_model()
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()
    
    def semantic_similarity(self, text1, text2):
        try:
            embeddings = self.get_embeddings([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1])
            return float(similarity)
        except Exception as e:
            print(f"AI similarity error: {e}")
            return self.keyword_similarity(text1, text2)
    
    def keyword_similarity(self, eng_text, chi_text):
        # Enhanced keyword matching for common phrases
        keyword_pairs = {
            'hello': ['你好', '您好', '嗨'],
            'thank': ['谢谢', '感谢', '多谢'],
            'sorry': ['对不起', '抱歉'],
            'yes': ['是', '是的', '对啊'],
            'no': ['不', '不是', '没有'],
            'good': ['好', '很好', '不错'],
            'bad': ['坏', '不好', '糟糕'],
            'big': ['大', '很大', '大型'],
            'small': ['小', '很小', '小型'],
            'love': ['爱', '喜欢', '爱着'],
            'family': ['家庭', '家人', '家'],
            'friend': ['朋友', '好友', '友人'],
            'school': ['学校', '上学', '校园'],
            'time': ['时间', '时候', '时光'],
            'dream': ['梦想', '梦', '幻想']
        }
        
        eng_lower = eng_text.lower()
        matches = 0
        total_keywords = 0
        
        for eng_word, chi_words in keyword_pairs.items():
            has_eng = eng_word in eng_lower
            has_chi = any(chi_word in chi_text for chi_word in chi_words)
            
            if has_eng or has_chi:
                total_keywords += 1
                if has_eng and has_chi:
                    matches += 1
        
        return matches / max(total_keywords, 1)

ai_model = SmartSubtitleAI()

def parse_srt(content):
    subs = []
    lines = content.strip().split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            # Next line should be timestamp
            if i + 1 < len(lines):
                time_line = lines[i + 1].strip()
                if '-->' in time_line:
                    # Next lines should be text
                    text_lines = []
                    j = i + 2
                    while j < len(lines) and lines[j].strip() != '':
                        text_lines.append(lines[j].strip())
                        j += 1
                    
                    if text_lines:
                        start_time = time_line.split(' --> ')[0]
                        text = ' '.join(text_lines)
                        subs.append({
                            'id': len(subs) + 1,
                            'start': start_time,
                            'text': text
                        })
                    
                    i = j
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    return subs

def detect_language(text):
    # Check for Chinese characters
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    chinese_ratio = len(chinese_chars) / max(len(text), 1)
    
    # Check for English (Latin characters)
    english_chars = re.findall(r'[a-zA-Z]', text)
    english_ratio = len(english_chars) / max(len(text), 1)
    
    if chinese_ratio > 0.3:
        return 'chinese', round(chinese_ratio * 100)
    elif english_ratio > 0.6:
        return 'english', round(english_ratio * 100)
    else:
        return 'unknown', 0

def time_to_seconds(time_str):
    try:
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

@app.route('/')
def home():
    return jsonify({
        "message": "Smart Subtitle AI Aligner API",
        "status": "running",
        "version": "1.0"
    })

@app.route('/api/detect-language', methods=['POST'])
def api_detect_language():
    try:
        data = request.get_json()
        content = data.get('content', '')
        
        if not content:
            return jsonify({"error": "No content provided"}), 400
        
        language, confidence = detect_language(content[:1000])  # Sample first 1000 chars
        
        return jsonify({
            "language": language,
            "confidence": confidence,
            "sample": content[:200] + "..." if len(content) > 200 else content
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/align', methods=['POST'])
def api_align_subtitles():
    try:
        data = request.get_json()
        english_srt = data.get('english_srt', '')
        chinese_srt = data.get('chinese_srt', '')
        
        if not english_srt or not chinese_srt:
            return jsonify({"error": "Both English and Chinese SRT content required"}), 400
        
        # Parse SRT files
        english_subs = parse_srt(english_srt)
        chinese_subs = parse_srt(chinese_srt)
        
        if not english_subs or not chinese_subs:
            return jsonify({"error": "Could not parse subtitles from provided content"}), 400
        
        print(f"Processing {len(english_subs)} English and {len(chinese_subs)} Chinese subtitles")
        
        # Memory-safe alignment
        results = []
        batch_size = 20  # Small batches for memory safety
        
        for i in range(0, len(english_subs), batch_size):
            batch_end = min(i + batch_size, len(english_subs))
            
            for eng_idx in range(i, batch_end):
                eng_sub = english_subs[eng_idx]
                best_match = None
                best_score = 0
                
                # Search in reasonable window around the same position
                search_start = max(0, eng_idx - 15)
                search_end = min(len(chinese_subs), eng_idx + 15)
                
                for chi_idx in range(search_start, search_end):
                    chi_sub = chinese_subs[chi_idx]
                    
                    # Calculate semantic similarity
                    semantic_score = ai_model.semantic_similarity(eng_sub['text'], chi_sub['text'])
                    
                    # Calculate timing score
                    eng_time = time_to_seconds(eng_sub['start'])
                    chi_time = time_to_seconds(chi_sub['start'])
                    time_diff = abs(eng_time - chi_time)
                    timing_score = max(0, 1 - (time_diff / 8.0))  # 8-second window
                    
                    # Combined score
                    combined_score = (semantic_score * 0.7) + (timing_score * 0.3)
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = chi_sub
                
                # Determine status and quality
                if best_score > 0.7:
                    status = 'ALIGNED'
                    quality = 'EXCELLENT' if best_score > 0.9 else 'VERY_GOOD' if best_score > 0.8 else 'GOOD'
                elif best_score > 0.5:
                    status = 'REVIEW'
                    quality = 'FAIR'
                else:
                    status = 'MISALIGNED'
                    quality = 'POOR'
                
                results.append({
                    'sequence': eng_sub['id'],
                    'eng_time': eng_sub['start'],
                    'chi_time': best_match['start'] if best_match else 'NO MATCH',
                    'english': eng_sub['text'],
                    'chinese': best_match['text'] if best_match else 'NO MATCH',
                    'confidence': round(best_score, 3),
                    'status': status,
                    'quality': quality
                })
            
            # Clear memory after each batch
            gc.collect()
        
        # Generate summary
        aligned_count = len([r for r in results if r['status'] == 'ALIGNED'])
        review_count = len([r for r in results if r['status'] == 'REVIEW'])
        misaligned_count = len([r for r in results if r['status'] == 'MISALIGNED'])
        
        return jsonify({
            "success": True,
            "results": results,
            "summary": {
                "total_english": len(english_subs),
                "total_chinese": len(chinese_subs),
                "aligned": aligned_count,
                "needs_review": review_count,
                "misaligned": misaligned_count,
                "alignment_rate": round(aligned_count / len(english_subs) * 100, 1),
                "ai_model": "multilingual-semantic"
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Alignment failed: {str(e)}"}), 500

@app.route('/api/generate-srt', methods=['POST'])
def api_generate_srt():
    try:
        data = request.get_json()
        aligned_pairs = data.get('aligned_pairs', [])
        
        srt_lines = []
        sequence = 1
        
        for pair in aligned_pairs:
            if pair['status'] != 'MISALIGNED' and pair['chinese'] != 'NO MATCH':
                srt_lines.append(str(sequence))
                srt_lines.append(f"{pair['eng_time']} --> {increment_time(pair['eng_time'], 3000)}")
                srt_lines.append(pair['chinese'])
                srt_lines.append(pair['english'])
                srt_lines.append('')
                sequence += 1
        
        if sequence == 1:
            return jsonify({"error": "No valid aligned pairs to generate SRT"}), 400
        
        return jsonify({
            "success": True,
            "srt_content": "\n".join(srt_lines),
            "total_pairs": sequence - 1
        })
        
    except Exception as e:
        return jsonify({"error": f"SRT generation failed: {str(e)}"}), 500

def increment_time(time_str, ms_to_add):
    try:
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            time_parts = parts[2].split('.')
            seconds = int(time_parts[0])
            milliseconds = int(time_parts[1]) if len(time_parts) > 1 else 0
            
            milliseconds += ms_to_add
            if milliseconds >= 1000:
                seconds += milliseconds // 1000
                milliseconds = milliseconds % 1000
            
            if seconds >= 60:
                minutes += seconds // 60
                seconds = seconds % 60
            
            if minutes >= 60:
                hours += minutes // 60
                minutes = minutes % 60
            
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    except:
        pass
    return time_str

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
