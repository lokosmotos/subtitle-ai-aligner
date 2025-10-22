const API_BASE_URL = 'https://your-backend-url.onrender.com';

// Color-coded confidence display
function getConfidenceColor(confidence) {
    if (confidence > 0.8) return '#10b981'; // Green
    if (confidence > 0.6) return '#f59e0b'; // Yellow
    if (confidence > 0.4) return '#f97316'; // Orange
    return '#ef4444'; // Red
}

function getStatusIcon(status) {
    const icons = {
        'ALIGNED': '✅',
        'REVIEW': '⚠️', 
        'MISALIGNED': '❌'
    };
    return icons[status] || '🔍';
}

async function alignSubtitles() {
    const englishSRT = document.getElementById('english-srt').value.trim();
    const chineseSRT = document.getElementById('chinese-srt').value.trim();
    
    if (!englishSRT || !chineseSRT) {
        showError('Please provide both English and Chinese SRT content');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/align`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({english_srt: englishSRT, chinese_srt: chineseSRT})
        });
        
        const data = await response.json();
        
        if (data.success) {
            displaySmartResults(data);
            enableDownload();
        } else {
            showError(data.error);
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        hideLoading();
    }
}

function displaySmartResults(data) {
    const resultsDiv = document.getElementById('results');
    const summaryDiv = document.getElementById('summary');
    const resultsBody = document.getElementById('results-body');
    
    // Smart summary
    const alignedCount = data.summary.aligned;
    const totalCount = data.summary.total_english;
    const accuracy = ((alignedCount / totalCount) * 100).toFixed(1);
    
    summaryDiv.innerHTML = `
        <div class="smart-summary">
            <h3>🧠 AI Analysis Complete</h3>
            <div class="metrics">
                <div class="metric">
                    <span class="value">${accuracy}%</span>
                    <span class="label">Alignment Accuracy</span>
                </div>
                <div class="metric">
                    <span class="value">${data.summary.aligned}</span>
                    <span class="label">High Confidence</span>
                </div>
                <div class="metric">
                    <span class="value">${data.summary.needs_review}</span>
                    <span class="label">Needs Review</span>
                </div>
                <div class="metric">
                    <span class="value">${data.summary.misaligned}</span>
                    <span class="label">Misaligned</span>
                </div>
            </div>
            <div class="ai-info">
                <small>Powered by Multilingual BERT • Semantic Matching • Context-Aware</small>
            </div>
        </div>
    `;
    
    // Smart results table
    resultsBody.innerHTML = '';
    data.results.forEach(result => {
        const row = document.createElement('tr');
        const confidenceColor = getConfidenceColor(result.confidence);
        const statusIcon = getStatusIcon(result.status);
        
        row.innerHTML = `
            <td>${result.sequence}</td>
            <td>${result.eng_time}</td>
            <td>${result.chi_time}</td>
            <td class="text-content">${escapeHtml(result.english)}</td>
            <td class="text-content">${escapeHtml(result.chinese)}</td>
            <td style="color: ${confidenceColor}; font-weight: bold">
                ${(result.confidence * 100).toFixed(1)}%
            </td>
            <td>
                <span class="status-badge status-${result.status.toLowerCase()}">
                    ${statusIcon} ${result.status}
                </span>
            </td>
            <td class="match-quality">${result.match_quality}</td>
            <td>
                <button onclick="provideFeedback('${escapeHtml(result.english)}', '${escapeHtml(result.chinese)}', true)" 
                        class="btn-success" title="Mark as correct">✓</button>
                <button onclick="provideFeedback('${escapeHtml(result.english)}', '${escapeHtml(result.chinese)}', false)" 
                        class="btn-danger" title="Mark as incorrect">✗</button>
            </td>
        `;
        
        resultsBody.appendChild(row);
    });
    
    showResults();
}

async function provideFeedback(englishText, chineseText, isCorrect) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/learn`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                english_text: englishText,
                chinese_text: chineseText,
                was_correct: isCorrect
            })
        });
        
        const data = await response.json();
        if (data.success) {
            showMessage('✅ Feedback recorded - AI is learning!', 'success');
        }
    } catch (error) {
        showMessage('❌ Failed to send feedback', 'error');
    }
}

// ... (rest of your existing frontend code)
