import pysrt
import re

def parse_srt(content):
    """Parse SRT content from string"""
    try:
        # Clean content first
        cleaned = clean_srt_content(content)
        subs = pysrt.from_string(cleaned)
        return [{
            'id': sub.index,
            'start': str(sub.start),
            'end': str(sub.end),
            'text': sub.text
        } for sub in subs]
    except Exception as e:
        # Fallback manual parsing
        return parse_srt_manual(content)

def clean_srt_content(content):
    """Clean messy SRT content"""
    # Fix common formatting issues
    cleaned = re.sub(r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3})\s+(\d{2}:\d{2}:\d{2},\d{3})', 
                    r'\1\n\2 --> \3', content)
    # Ensure proper line breaks
    cleaned = re.sub(r'\n\n+', '\n\n', cleaned)
    return cleaned.strip()

def parse_srt_manual(content):
    """Manual parsing for problematic SRT files"""
    subtitles = []
    lines = content.strip().split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():  # ID line
            try:
                sub_id = int(line)
                i += 1
                if i < len(lines):
                    # Time line
                    time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', lines[i])
                    if time_match:
                        start_time = time_match.group(1)
                        end_time = time_match.group(2)
                        i += 1
                        # Text lines
                        text_lines = []
                        while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                            text_lines.append(lines[i].strip())
                            i += 1
                        
                        if text_lines:
                            subtitles.append({
                                'id': sub_id,
                                'start': start_time,
                                'end': end_time,
                                'text': ' '.join(text_lines)
                            })
            except:
                i += 1
        else:
            i += 1
    
    return subtitles
