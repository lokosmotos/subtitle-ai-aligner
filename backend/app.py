from flask import Flask, request, jsonify
from flask_cors import CORS
from srt_parser import parse_srt
from ai_model import SubtitleAI
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        english_srt = data.get('english_srt', '')
        chinese_srt = data.get('chinese_srt', '')
        
        if not english_srt or not chinese_srt:
            return jsonify({"error": "Both English and Chinese SRT content required"}), 400
        
        logger.info(f"Received SRT content: English {len(english_srt)} chars, Chinese {len(chinese_srt)} chars")
        
        # Parse SRT files
        english_subs = parse_srt(english_srt)
        chinese_subs = parse_srt(chinese_srt)
        
        logger.info(f"Parsed {len(english_subs)} English and {len(chinese_subs)} Chinese subtitles")
        
        if len(english_subs) == 0 or len(chinese_subs) == 0:
            return jsonify({"error": "Could not parse subtitles from provided content"}), 400
        
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
        logger.error(f"Alignment error: {str(e)}")
        return jsonify({"error": f"Alignment failed: {str(e)}"}), 500

@app.route('/api/generate-srt', methods=['POST'])
def generate_srt():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        aligned_pairs = data.get('aligned_pairs', [])
        
        srt_content = []
        sequence = 1
        
        for pair in aligned_pairs:
            if pair['status'] == 'ALIGNED' and pair['chinese'] != 'NO MATCH':
                # Chinese on top, English on bottom
                srt_content.append(str(sequence))
                srt_content.append(f"{pair['eng_time']} --> {increment_time(pair['eng_time'], 3000)}")
                srt_content.append(pair['chinese'])
                srt_content.append(pair['english'])
                srt_content.append('')
                sequence += 1
        
        return jsonify({
            "success": True,
            "srt_content": "\n".join(srt_content)
        })
        
    except Exception as e:
        logger.error(f"SRT generation error: {str(e)}")
        return jsonify({"error": f"SRT generation failed: {str(e)}"}), 500

def increment_time(time_str, ms_to_add):
    """Helper function to increment time"""
    try:
        if ',' in time_str:
            time_str = time_str.replace(',', '.')
        
        parts = time_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds_parts = parts[2].split('.')
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
            
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
