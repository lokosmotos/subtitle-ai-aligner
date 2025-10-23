# srt_parser.py
import re

def parse_srt(srt_content):
    """
    Parse SRT content and return list of subtitle dictionaries
    """
    subtitles = []
    
    # Split by double newlines to get individual subtitle blocks
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                # First line is sequence number
                seq_num = lines[0].strip()
                
                # Second line is timestamp
                timestamp = lines[1].strip()
                time_parts = timestamp.split(' --> ')
                if len(time_parts) == 2:
                    start_time = time_parts[0].strip()
                    
                    # Remaining lines are the subtitle text
                    text = ' '.join(line.strip() for line in lines[2:])
                    
                    subtitles.append({
                        'id': seq_num,
                        'start': start_time,
                        'text': text
                    })
            except Exception as e:
                # Skip malformed blocks
                continue
                
    return subtitles
