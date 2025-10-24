import re

def parse_srt(content):
    """
    Parse SRT file content into list of subtitle dictionaries
    """
    subs = []
    lines = content.strip().split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this line is a sequence number
        if line.isdigit():
            # Next line should be timestamp
            if i + 1 < len(lines):
                time_line = lines[i + 1].strip()
                
                # Verify it's a timestamp line
                if '-->' in time_line:
                    # Extract start and end times
                    time_parts = time_line.split(' --> ')
                    if len(time_parts) == 2:
                        start_time = time_parts[0].strip()
                        end_time = time_parts[1].strip()
                        
                        # Collect text lines until empty line or next sequence
                        text_lines = []
                        j = i + 2
                        while j < len(lines) and lines[j].strip() != '':
                            # Skip lines that are numbers (could be next sequence)
                            if not lines[j].strip().isdigit():
                                text_lines.append(lines[j].strip())
                            j += 1
                        
                        if text_lines:
                            text = ' '.join(text_lines)
                            subs.append({
                                'id': int(line),
                                'start': start_time,
                                'end': end_time,
                                'text': text
                            })
                        
                        i = j  # Move to next subtitle
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    return subs

def validate_srt_content(content):
    """
    Basic validation of SRT content
    """
    if not content or not content.strip():
        return False, "Empty SRT content"
    
    lines = content.strip().split('\n')
    
    # Check for basic SRT structure
    has_sequence = any(line.strip().isdigit() for line in lines)
    has_timestamps = any('-->' in line for line in lines)
    
    if not has_sequence or not has_timestamps:
        return False, "Invalid SRT format - missing sequence numbers or timestamps"
    
    return True, "Valid SRT content"

def detect_encoding_issues(content):
    """
    Detect common encoding issues in SRT files
    """
    issues = []
    
    # Check for common encoding problems
    if 'Ã' in content or '©' in content or 'Â' in content:
        issues.append("Possible encoding issues detected")
    
    # Check for BOM (Byte Order Mark)
    if content.startswith('\ufeff'):
        issues.append("UTF-8 BOM detected")
    
    return issues
