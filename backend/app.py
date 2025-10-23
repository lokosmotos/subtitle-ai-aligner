from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import os

app = Flask(__name__)
CORS(app)

# Import after Flask app creation to manage memory better
from ai_model import SmartSubtitleAI
ai_model = SmartSubtitleAI()

def parse_srt(srt_content):
    """Lightweight SRT parser"""
    subtitles = []
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                seq_num = lines[0].strip()
                timestamp = lines[1].strip()
                time_parts = timestamp.split(' --> ')
                if len(time_parts) == 2:
                    start_time = time_parts[0].strip()
                    text = ' '.join(line.strip() for line in lines[2:])
                    
                    subtitles.append({
                        'id': seq_num,
                        'start': start_time,
                        'text': text
                    })
            except:
                continue
                
    return subtitles

@app.route('/')
def home():
    return jsonify({"message": "Lightweight Subtitle AI Aligner", "status": "running"})

@app.route('/api/align', methods=['POST'])
def align_subtitles():
    try:
        data = request.get_json()
        english_srt = data.get('english_srt', '')
        chinese_srt = data.get('chinese_srt', '')
        
        if not english_srt or not chinese_srt:
            return jsonify({"error": "Both English and Chinese SRT content required"}), 400
        
        # Parse SRT files
        english_subs = parse_srt(english_srt)
        chinese_subs = parse_srt(chinese_srt)
        
        if len(english_subs) == 0 or len(chinese_subs) == 0:
            return jsonify({"error": "Could not parse subtitles"}), 400
        
        # Smart alignment
        results = ai_model.align_subtitles(english_subs, chinese_subs)
        
        return jsonify({
            "success": True,
            "results": results,
            "summary": {
                "total_english": len(english_subs),
                "total_chinese": len(chinese_subs),
                "aligned": len([r for r in results if r['status'] == 'ALIGNED']),
                "needs_review": len([r for r in results if r['status'] == 'REVIEW']),
                "misaligned": len([r for r in results if r['status'] == 'MISALIGNED'])
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Alignment failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
