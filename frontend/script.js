// Configuration - Update this with your Render URL later
const API_BASE_URL = 'http://localhost:5000'; // Change to your Render URL after deployment

let currentResults = null;

async function alignSubtitles() {
    const englishSRT = document.getElementById('english-srt').value.trim();
    const chineseSRT = document.getElementById('chinese-srt').value.trim();
    
    if (!englishSRT || !chineseSRT) {
        showError('Please provide both English and Chinese SRT content');
        return;
    }
    
    showLoading();
    hideError();
    hideResults();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/align`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                english_srt: englishSRT,
                chinese_srt: chineseSRT
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Alignment failed');
        }
        
        currentResults = data;
        displayResults(data);
        enableDownload();
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    const summaryDiv = document.getElementById('summary');
    const resultsBody = document.getElementById('results-body');
    
    // Display summary
    summaryDiv.innerHTML = `
        <div class="summary-item summary-total">
            <div>Total English</div>
            <div>${data.summary.total_english}</div>
        </div>
        <div class="summary-item summary-aligned">
            <div>Aligned</div>
            <div>${data.summary.aligned}</div>
        </div>
        <div class="summary-item summary-review">
            <div>Needs Review</div>
            <div>${data.summary.needs_review}</div>
        </div>
        <div class="summary-item summary-misaligned">
            <div>Misaligned</div>
            <div>${data.summary.misaligned}</div>
        </div>
    `;
    
    // Display results table
    resultsBody.innerHTML = '';
    data.results.forEach(result => {
        const row = document.createElement('tr');
        
        const confidenceClass = result.confidence > 0.7 ? 'confidence-high' : 
                              result.confidence > 0.4 ? 'confidence-medium' : 'confidence-low';
        
        row.innerHTML = `
            <td>${result.sequence}</td>
            <td>${result.eng_time}</td>
            <td>${result.chi_time}</td>
            <td>${escapeHtml(result.english)}</td>
            <td>${escapeHtml(result.chinese)}</td>
            <td class="${confidenceClass}">${Math.round(result.confidence * 100)}%</td>
            <td class="status-${result.status.toLowerCase()}">${result.status}</td>
        `;
        
        resultsBody.appendChild(row);
    });
    
    showResults();
}

async function downloadSRT() {
    if (!currentResults) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/generate-srt`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                aligned_pairs: currentResults.results
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'SRT generation failed');
        }
        
        // Create and download SRT file
        const blob = new Blob([data.srt_content], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'aligned_subtitles.srt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
    } catch (error) {
        showError('Failed to download SRT: ' + error.message);
    }
}

// Utility functions
function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

function showResults() {
    document.getElementById('results').classList.remove('hidden');
}

function hideResults() {
    document.getElementById('results').classList.add('hidden');
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    errorDiv.classList.remove('hidden');
}

function hideError() {
    document.getElementById('error').classList.add('hidden');
}

function enableDownload() {
    document.getElementById('download-btn').disabled = false;
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    console.log('AI Subtitle Aligner loaded');
});
