/**
 * Cyberbullying Detection System - Client Application JavaScript
 */

document.addEventListener('DOMContentLoaded', function () {
    initTextareaCounter();
    initSamplePrompts();
    initDetectionForm();
    
    // Page specific initializers
    if (document.getElementById('historyTableBody')) {
        loadHistory();
        initHistorySearch();
    }

    if (document.getElementById('chartCyberbullyingDonut')) {
        loadStatistics();
    }

    if (document.getElementById('modelComparisonTable')) {
        loadModelMetrics();
    }
});

/**
 * Real-time character and word counter for text input.
 */
function initTextareaCounter() {
    const textarea = document.getElementById('inputText');
    const charCount = document.getElementById('charCount');
    
    if (textarea && charCount) {
        textarea.addEventListener('input', function () {
            const length = this.value.length;
            charCount.textContent = `${length} characters`;
        });
    }
}

/**
 * Sample prompt filler buttons for easy demonstration.
 */
function initSamplePrompts() {
    const sampleButtons = document.querySelectorAll('.btn-sample');
    const textarea = document.getElementById('inputText');

    sampleButtons.forEach(button => {
        button.addEventListener('click', function () {
            if (textarea) {
                textarea.value = this.getAttribute('data-text');
                textarea.dispatchEvent(new Event('input'));
                textarea.focus();
            }
        });
    });
}

/**
 * Handles message prediction form submission via AJAX.
 */
function initDetectionForm() {
    const form = document.getElementById('detectionForm');
    const textarea = document.getElementById('inputText');
    const btnDetect = document.getElementById('btnDetect');
    const btnSpinner = document.getElementById('btnSpinner');
    const btnText = document.getElementById('btnText');
    const resultArea = document.getElementById('resultArea');

    if (!form) return;

    form.addEventListener('submit', async function (e) {
        e.preventDefault();
        const text = textarea.value.trim();

        if (!text) {
            alert('Please enter a message or comment to analyze.');
            return;
        }

        // Set Loading UI state
        btnDetect.disabled = true;
        btnSpinner.classList.remove('d-none');
        btnText.textContent = 'Analyzing with Model...';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (data.status === 'success') {
                renderResult(data);
            } else {
                renderError(data.message || 'An error occurred during detection.');
            }
        } catch (error) {
            console.error('Error classifying text:', error);
            renderError('Unable to connect to Flask server. Please make sure python app.py is running.');
        } finally {
            // Reset Button State
            btnDetect.disabled = false;
            btnSpinner.classList.add('d-none');
            btnText.textContent = 'Detect Cyberbullying';
        }
    });
}

/**
 * Renders the classification prediction result.
 */
function renderResult(data) {
    const resultArea = document.getElementById('resultArea');
    if (!resultArea) return;

    const isCyberbullying = data.is_cyberbullying;
    const confidence = data.confidence;
    const severity = data.severity;
    const category = data.category;
    const modelName = data.model_name;

    let html = '';

    if (isCyberbullying) {
        html = `
            <div class="result-card-danger shadow-sm">
                <div class="d-flex align-items-center justify-content-between flex-wrap gap-2 mb-3">
                    <div class="d-flex align-items-center gap-2">
                        <span class="result-badge-cyberbullying">⚠️ Cyberbullying Detected</span>
                        <span class="badge bg-dark text-light px-3 py-2 rounded-pill">${category}</span>
                        <span class="badge bg-danger text-light px-3 py-2 rounded-pill">Severity: ${severity}</span>
                    </div>
                    <span class="text-muted small">Classifier: <strong>${modelName}</strong></span>
                </div>

                <div class="alert alert-danger border-0 mb-3" role="alert">
                    <h5 class="alert-heading fw-bold mb-2">🚨 Harmful Content Warning!</h5>
                    <p class="mb-0">Our machine learning model analyzed this text and flagged it as harmful or abusive cyberbullying language. Cyberbullying causes severe emotional distress. Please refrain from spreading harassment or toxicity online.</p>
                </div>

                <div class="mb-3">
                    <div class="d-flex justify-content-between mb-1">
                        <span class="fw-semibold">Model Confidence Score:</span>
                        <span class="fw-bold text-danger">${confidence}%</span>
                    </div>
                    <div class="progress progress-confidence">
                        <div class="progress-bar bg-danger" role="progressbar" style="width: ${confidence}%" aria-valuenow="${confidence}" aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                </div>

                <div class="bg-white p-3 rounded-3 border border-danger-subtle">
                    <div class="row align-items-center">
                        <div class="col-md-8">
                            <strong class="d-block text-dark mb-1">💬 Analyzed Text Snippet:</strong>
                            <p class="text-muted mb-0 fst-italic">"${escapeHtml(data.raw_text)}"</p>
                        </div>
                        <div class="col-md-4 text-md-end mt-2 mt-md-0">
                            <a href="/about" class="btn btn-outline-danger btn-sm">Support & Helplines</a>
                        </div>
                    </div>
                </div>
            </div>
        `;
    } else {
        html = `
            <div class="result-card-safe shadow-sm">
                <div class="d-flex align-items-center justify-content-between flex-wrap gap-2 mb-3">
                    <div class="d-flex align-items-center gap-2">
                        <span class="result-badge-safe">✅ Not Cyberbullying</span>
                        <span class="badge bg-success-subtle text-success border border-success-subtle px-3 py-2 rounded-pill">Clean Content</span>
                    </div>
                    <span class="text-muted small">Classifier: <strong>${modelName}</strong></span>
                </div>

                <div class="alert alert-success border-0 mb-3" role="alert">
                    <h5 class="alert-heading fw-bold mb-1">✨ Content Verified Safe</h5>
                    <p class="mb-0">No harmful cyberbullying, harassment, or abusive patterns were detected in the submitted text.</p>
                </div>

                <div class="mb-3">
                    <div class="d-flex justify-content-between mb-1">
                        <span class="fw-semibold">Model Safety Confidence:</span>
                        <span class="fw-bold text-success">${confidence}%</span>
                    </div>
                    <div class="progress progress-confidence">
                        <div class="progress-bar bg-success" role="progressbar" style="width: ${confidence}%" aria-valuenow="${confidence}" aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                </div>

                <div class="bg-white p-3 rounded-3 border border-success-subtle">
                    <strong class="d-block text-dark mb-1">💬 Analyzed Text Snippet:</strong>
                    <p class="text-muted mb-0 fst-italic">"${escapeHtml(data.raw_text)}"</p>
                </div>
            </div>
        `;
    }

    resultArea.innerHTML = html;
}

function renderError(message) {
    const resultArea = document.getElementById('resultArea');
    if (resultArea) {
        resultArea.innerHTML = `
            <div class="alert alert-warning border-0 shadow-sm p-3 mb-0" role="alert">
                <div class="d-flex align-items-center">
                    <i class="bi bi-exclamation-triangle-fill me-2 fs-4 text-warning"></i>
                    <div>
                        <strong class="d-block">Detection Notice</strong>
                        <span>${escapeHtml(message)}</span>
                    </div>
                </div>
            </div>
        `;
    }
}

/**
 * Loads detection history table records from SQLite DB.
 */
async function loadHistory(searchQuery = '') {
    const tableBody = document.getElementById('historyTableBody');
    const emptyState = document.getElementById('emptyState');
    const countBadge = document.getElementById('historyCountBadge');

    if (!tableBody) return;

    try {
        const url = searchQuery ? `/api/history?q=${encodeURIComponent(searchQuery)}` : '/api/history';
        const response = await fetch(url);
        const data = await response.json();

        if (data.status === 'success') {
            const records = data.history;
            if (countBadge) countBadge.textContent = records.length;

            if (records.length === 0) {
                tableBody.innerHTML = '';
                if (emptyState) emptyState.classList.remove('d-none');
                return;
            }

            if (emptyState) emptyState.classList.add('d-none');

            tableBody.innerHTML = records.map(rec => {
                const isCyber = rec.prediction === 'Cyberbullying';
                const badgeClass = isCyber ? 'bg-danger' : 'bg-success';
                const confColor = isCyber ? 'text-danger' : 'text-success';

                return `
                    <tr>
                        <td class="fw-semibold">#${rec.id}</td>
                        <td style="max-width: 320px;">
                            <div class="text-truncate" title="${escapeHtml(rec.text)}">${escapeHtml(rec.text)}</div>
                            <small class="text-muted d-block mt-1">${rec.cleaned_text ? 'Cleaned: ' + escapeHtml(rec.cleaned_text) : ''}</small>
                        </td>
                        <td>
                            <span class="badge ${badgeClass} px-2.5 py-1.5 rounded-pill">${escapeHtml(rec.prediction)}</span>
                        </td>
                        <td>
                            <span class="fw-bold ${confColor}">${rec.confidence}%</span>
                        </td>
                        <td>
                            <span class="badge bg-secondary-subtle text-secondary rounded-pill px-2.5">${escapeHtml(rec.category || 'General')}</span>
                        </td>
                        <td class="text-muted small">${escapeHtml(rec.timestamp)}</td>
                        <td class="text-end">
                            <button class="btn btn-outline-danger btn-sm rounded-circle p-1 px-2" onclick="deleteRecord(${rec.id})" title="Delete Record">
                                🗑️
                            </button>
                        </td>
                    </tr>
                `;
            }).join('');
        }
    } catch (error) {
        console.error('Error fetching history:', error);
    }
}

function initHistorySearch() {
    const searchInput = document.getElementById('historySearch');
    if (searchInput) {
        let timer;
        searchInput.addEventListener('input', function () {
            clearTimeout(timer);
            timer = setTimeout(() => {
                loadHistory(this.value.trim());
            }, 300);
        });
    }

    const btnClearAll = document.getElementById('btnClearHistory');
    if (btnClearAll) {
        btnClearAll.addEventListener('click', async function () {
            if (confirm('Are you sure you want to delete ALL detection history records? This cannot be undone.')) {
                try {
                    const response = await fetch('/api/history/clear', { method: 'POST' });
                    const res = await response.json();
                    if (res.status === 'success') {
                        loadHistory();
                    }
                } catch (err) {
                    console.error('Error clearing history:', err);
                }
            }
        });
    }
}

async function deleteRecord(id) {
    if (confirm(`Delete history entry #${id}?`)) {
        try {
            const response = await fetch(`/api/history/${id}`, { method: 'DELETE' });
            const res = await response.json();
            if (res.status === 'success') {
                loadHistory();
            }
        } catch (err) {
            console.error('Error deleting record:', err);
        }
    }
}

/**
 * Loads aggregate statistics and renders Chart.js visualizations.
 */
async function loadStatistics() {
    try {
        const response = await fetch('/api/stats');
        const res = await response.json();

        if (res.status === 'success') {
            const stats = res.data;

            // Update Metric Cards
            document.getElementById('statTotal').textContent = stats.total_messages;
            document.getElementById('statCyberbullying').textContent = stats.cyberbullying_count;
            document.getElementById('statClean').textContent = stats.clean_count;
            document.getElementById('statPercentage').textContent = stats.cyberbullying_percentage + '%';

            // Donut Chart: Cyberbullying vs Clean
            const ctxDonut = document.getElementById('chartCyberbullyingDonut').getContext('2d');
            new Chart(ctxDonut, {
                type: 'doughnut',
                data: {
                    labels: ['Cyberbullying', 'Clean Content'],
                    datasets: [{
                        data: [stats.cyberbullying_count, stats.clean_count],
                        backgroundColor: ['#dc2626', '#16a34a'],
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'bottom' }
                    }
                }
            });

            // Bar Chart: Category Distribution
            const catLabels = Object.keys(stats.category_distribution);
            const catValues = Object.values(stats.category_distribution);

            const ctxBar = document.getElementById('chartCategoryBar').getContext('2d');
            new Chart(ctxBar, {
                type: 'bar',
                data: {
                    labels: catLabels.length ? catLabels : ['No Data'],
                    datasets: [{
                        label: 'Messages per Category',
                        data: catValues.length ? catValues : [0],
                        backgroundColor: '#4f46e5',
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: { beginAtZero: true, precision: 0 }
                    }
                }
            });
        }
    } catch (err) {
        console.error('Error loading statistics:', err);
    }
}

/**
 * Loads machine learning model evaluation metrics and confusion matrix.
 */
async function loadModelMetrics() {
    try {
        const response = await fetch('/api/model-metrics');
        const res = await response.json();

        if (res.status === 'success') {
            const m = res.metrics;

            // Header summary values
            document.getElementById('metricBestModel').textContent = m.best_model || 'Classifier';
            document.getElementById('metricTotalSamples').textContent = m.total_samples || 0;
            document.getElementById('metricTrainSamples').textContent = m.train_samples || 0;
            document.getElementById('metricTestSamples').textContent = m.test_samples || 0;

            const bestModelMetrics = m.models_performance[m.best_model] || {};
            document.getElementById('metricAccuracy').textContent = (bestModelMetrics.accuracy * 100).toFixed(2) + '%';
            document.getElementById('metricPrecision').textContent = (bestModelMetrics.precision * 100).toFixed(2) + '%';
            document.getElementById('metricRecall').textContent = (bestModelMetrics.recall * 100).toFixed(2) + '%';
            document.getElementById('metricF1').textContent = (bestModelMetrics.f1_score * 100).toFixed(2) + '%';

            // Populate Model Comparison Table
            const tableBody = document.getElementById('modelComparisonTable');
            tableBody.innerHTML = Object.keys(m.models_performance).map(modelName => {
                const perf = m.models_performance[modelName];
                const isBest = (modelName === m.best_model);
                const rowClass = isBest ? 'table-success fw-semibold' : '';

                return `
                    <tr class="${rowClass}">
                        <td>
                            ${escapeHtml(modelName)}
                            ${isBest ? '<span class="badge bg-success ms-2">Best Selected</span>' : ''}
                        </td>
                        <td>${(perf.accuracy * 100).toFixed(2)}%</td>
                        <td>${(perf.precision * 100).toFixed(2)}%</td>
                        <td>${(perf.recall * 100).toFixed(2)}%</td>
                        <td><strong>${(perf.f1_score * 100).toFixed(2)}%</strong></td>
                    </tr>
                `;
            }).join('');

            // Confusion Matrix Grid
            const cm = bestModelMetrics.confusion_matrix;
            if (cm && cm.length === 2) {
                document.getElementById('cellTN').textContent = cm[0][0]; // True Negative
                document.getElementById('cellFP').textContent = cm[0][1]; // False Positive
                document.getElementById('cellFN').textContent = cm[1][0]; // False Negative
                document.getElementById('cellTP').textContent = cm[1][1]; // True Positive
            }
        }
    } catch (err) {
        console.error('Error loading model metrics:', err);
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
