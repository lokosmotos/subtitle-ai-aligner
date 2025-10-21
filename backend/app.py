from flask import Flask, request, jsonify
from flask_cors import CORS
from srt_parser import parse_srt
from ai_model import SubtitleAI
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize AI model
ai_model = SubtitleAI()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Subtitle AI Aligner is running"})

@app.route('/api/align', methods=['POST'])
def align_subtitles():
    try:
        data = request.json
        english_srt = data.get('english_srt', '')
        chinese_srt = data.get('chinese_srt', '')
        
        if not english_srt or not chinese_srt:
            return jsonify({"error": "Both English and Chinese SRT content required"}), 400
        
        # Parse SRT files
        english_subs = parse_srt(english_srt)
        chinese_subs = parse_srt(chinese_srt)
        
        print(f"Parsed {len(english_subs)} English and {len(chinese_subs)} Chinese subtitles")
        
        # Align subtitles using AI
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
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-srt', methods=['POST'])
def generate_srt():
    try:
        data = request.json
        aligned_pairs = data.get('aligned_pairs', [])
        
        srt_content = []
        sequence = 1
        
        for pair in aligned_pairs:
            if pair['status'] == 'ALIGNED' and pair['chinese'] != 'NO MATCH':
                # Chinese on top, English on bottom
                srt_content.append(str(sequence))
                srt_content.append(f"{pair['eng_time']} --> {pair['eng_time']}")
                srt_content.append(pair['chinese'])
                srt_content.append(pair['english'])
                srt_content.append('')
                sequence += 1
        
        return jsonify({
            "success": True,
            "srt_content": "\n".join(srt_content)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
