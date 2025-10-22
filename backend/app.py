from flask import Flask, request, jsonify
from flask_cors import CORS
from srt_parser import parse_srt
from ai_model import SmartSubtitleAI
import os
import json

app = Flask(__name__)
CORS(app)

# Initialize SMART AI model
ai_model = SmartSubtitleAI()

@app.route('/')
def home():
    return jsonify({
        "message": "Smart Subtitle AI Aligner API",
        "status": "running",
        "features": [
            "Semantic matching with multilingual BERT",
            "Context-aware alignment",
            "Adaptive confidence scoring",
            "Learning from user feedback"
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Smart AI Subtitle Aligner is running"})

@app.route('/api/align', methods=['POST'])
def align_subtitles():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        english_srt = data.get('english_srt', '')
        chinese_srt = data.get('chinese_srt', '')
        
        if not english_srt:
            return jsonify({"error": "English SRT content is required"}), 400
        if not chinese_srt:
            return jsonify({"error": "Chinese SRT content is required"}), 400
        
        # Parse SRT files
        english_subs = parse_srt(english_srt)
        chinese_subs = parse_srt(chinese_srt)
        
        if len(english_subs) == 0:
            return jsonify({"error": "Could not parse any English subtitles"}), 400
        if len(chinese_subs) == 0:
            return jsonify({"error": "Could not parse any Chinese subtitles"}), 400
        
        # Smart alignment with AI
        results = ai_model.align_subtitles(english_subs, chinese_subs)
        
        return jsonify({
            "success": True,
            "results": results,
            "summary": {
                "total_english": len(english_subs),
                "total_chinese": len(chinese_subs),
                "aligned": len([r for r in results if r['status'] == 'ALIGNED']),
                "needs_review": len([r for r in results if r['status'] == 'REVIEW']),
                "misaligned": len([r for r in results if r['status'] == 'MISALIGNED']),
                "ai_model": "multilingual-bert-semantic"
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Alignment failed: {str(e)}"}), 500

@app.route('/api/learn', methods=['POST'])
def learn_from_feedback():
    """Endpoint for AI learning from user corrections"""
    try:
        data = request.get_json()
        english_text = data.get('english_text', '')
        chinese_text = data.get('chinese_text', '')
        was_correct = data.get('was_correct', False)
        
        ai_model.learn_from_feedback(english_text, chinese_text, was_correct)
        
        return jsonify({
            "success": True,
            "message": "AI learned from feedback",
            "learned_pairs_count": len(ai_model.learned_pairs)
        })
        
    except Exception as e:
        return jsonify({"error": f"Learning failed: {str(e)}"}), 500

@app.route('/api/generate-srt', methods=['POST'])
def generate_srt():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        aligned_pairs = data.get('aligned_pairs', [])
        
        srt_content = []
        sequence = 1
        
        for pair in aligned_pairs:
            if pair['status'] != 'MISALIGNED' and pair['chinese'] != 'NO MATCH':
                # Chinese on top, English on bottom
                srt_content.append(str(sequence))
                srt_content.append(f"{pair['eng_time']} --> {increment_time(pair['eng_time'], 3000)}")
                srt_content.append(pair['chinese'])
                srt_content.append(pair['english'])
                srt_content.append('')
                sequence += 1
        
        if sequence == 1:
            return jsonify({"error": "No aligned pairs to generate SRT"}), 400
        
        return jsonify({
            "success": True,
            "srt_content": "\n".join(srt_content),
            "message": f"Generated SRT with {sequence-1} aligned pairs"
        })
        
    except Exception as e:
        return jsonify({"error": f"SRT generation failed: {str(e)}"}), 500

def increment_time(time_str, ms_to_add):
    """Helper function to increment time"""
    try:
        time_str = time_str.replace('.', ',')
        
        parts = time_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            time_parts = parts[2].split(',')
            seconds = int(time_parts[0])
            milliseconds = int(time_parts[1]) if len(time_parts) > 1 else 0
            
            # Add milliseconds
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
