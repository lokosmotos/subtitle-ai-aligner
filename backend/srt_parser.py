import re

def parse_srt(content):
    """Parse SRT content from string - simple and robust"""
    if not content or not isinstance(content, str):
        return []
    
    subtitles = []
    
    # Normalize line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    # Split into blocks
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        
        if len(lines) >= 3:
            try:
                # First line should be ID
                sub_id = lines[0]
                
                # Second line should be timestamp
                time_line = lines[1]
                
                # Extract times - handle different formats
                time_match = re.search(r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})', time_line)
                if time_match:
                    start_time = time_match.group(1).replace('.', ',')
                    end_time = time_match.group(2).replace('.', ',')
                    
                    # Remaining lines are text
                    text = ' '.join(lines[2:])
                    
                    subtitles.append({
                        'id': sub_id,
                        'start': start_time,
                        'end': end_time,
                        'text': text
                    })
            except Exception as e:
                # Skip problematic blocks
                continue
    
    return subtitles
