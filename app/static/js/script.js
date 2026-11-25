// AQI Prediction Dashboard — JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('AQI Prediction Dashboard loaded.');

    // Smooth scroll for nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            // Update active state
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });

    // Intersection observer for active nav link
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === '#' + entry.target.id) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }, { rootMargin: '-20% 0px -70% 0px' });

    sections.forEach(section => observer.observe(section));

    // Animate metric bars on scroll
    const metricBars = document.querySelectorAll('.metric-fill');
    const barObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.width = entry.target.style.width;
                barObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    metricBars.forEach(bar => {
        const targetWidth = bar.style.width;
        bar.style.width = '0%';
        bar.dataset.targetWidth = targetWidth;
        barObserver.observe(bar);
    });

    // Trigger bar animations
    const barAnimObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const bars = entry.target.querySelectorAll('.metric-fill');
                bars.forEach((bar, i) => {
                    setTimeout(() => {
                        bar.style.width = bar.dataset.targetWidth || bar.style.width;
                    }, i * 150);
                });
                barAnimObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.3 });

    document.querySelectorAll('.model-result-card').forEach(card => barAnimObserver.observe(card));

    // Add hover glow effect to cards
    document.querySelectorAll('.card, .model-result-card, .category-card, .meteo-card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.boxShadow = '0 8px 40px rgba(58, 123, 213, 0.15)';
        });
        card.addEventListener('mouseleave', () => {
            card.style.boxShadow = '';
        });
    });

    // Staggered fade-in for grid children
    const staggerContainers = document.querySelectorAll('.category-grid, .model-cards, .meteo-comparison, .training-stats');
    const staggerObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const children = entry.target.children;
                Array.from(children).forEach((child, i) => {
                    child.style.opacity = '0';
                    child.style.transform = 'translateY(20px)';
                    setTimeout(() => {
                        child.style.transition = 'all 0.5s ease';
                        child.style.opacity = '1';
                        child.style.transform = 'translateY(0)';
                    }, i * 100);
                });
                staggerObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.2 });

    staggerContainers.forEach(container => staggerObserver.observe(container));

    // ============================================================
    // Dynamic Live Prediction Feed & Terminal Poller
    // ============================================================
    const AQI_COLORS = {
        0: '#009966', // Good
        1: '#FFDE33', // Satisfactory
        2: '#FF9933', // Moderate
        3: '#CC0033', // Poor
        4: '#660099', // Very Poor
        5: '#7E0023'  // Severe
    };

    let lastKnownTimestamp = null;
    const initialUpdated = document.getElementById('live-updated-text');
    if (initialUpdated && initialUpdated.textContent.includes('Updated:')) {
        lastKnownTimestamp = initialUpdated.textContent.replace('Updated:', '').trim();
    }

    // Create Toast Container
    const toast = document.createElement('div');
    toast.className = 'live-toast';
    toast.id = 'live-toast';
    document.body.appendChild(toast);

    function showToast(message) {
        toast.textContent = message;
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3800);
    }

    function renderDynamicPrediction(p) {
        const bodyEl = document.getElementById('live-prediction-body');
        const cmdBanner = document.getElementById('live-command-banner');
        const updatedText = document.getElementById('live-updated-text');
        const card = document.getElementById('live-prediction');

        if (!bodyEl || !p) return;

        // Flash update animation
        if (card) {
            card.classList.remove('flash-update');
            void card.offsetWidth; // force reflow
            card.classList.add('flash-update');
        }

        // Update timestamps & banner
        if (updatedText) {
            updatedText.textContent = `Updated: ${p.updated_at} (Live Sync)`;
        }
        if (cmdBanner) {
            cmdBanner.innerHTML = `<span class="cmd-label">Source:</span> <code>${p.command || 'terminal command'}</code>`;
        }

        const s = p.summary || {};
        const catColor = AQI_COLORS[s.dominant_code] || '#3a7bd5';

        let html = `
            <div class="live-hero-card" style="--aqi-theme: ${catColor};">
                <div class="live-hero-left">
                    <div class="live-category-badge" style="background: ${catColor};">
                        ${(s.dominant_category || 'AQI').toUpperCase()}
                    </div>
                    <div class="live-hero-details">
                        <h3>${s.title || 'Latest Air Quality Prediction'}</h3>
                        <p class="live-hero-sub">
                            Target: <strong>${s.target_datetime || ''}</strong> (${s.mode || 'Historical / Forecast'}) — ${s.risk_level || ''} Risk
                        </p>
                    </div>
                </div>
                <div class="live-hero-right">
                    <div class="live-stat-pill">
                        <span class="pill-label">Confidence</span>
                        <span class="pill-val">${s.avg_confidence || 0}%</span>
                    </div>
                    <div class="live-stat-pill">
                        <span class="pill-label">Risk Level</span>
                        <span class="pill-val">${s.risk_level || 'Normal'}</span>
                    </div>
                </div>
            </div>
        `;

        // Render Single Datetime Prediction View
        if (p.data) {
            const d = p.data;
            html += `
                <div class="live-sub-section">
                    <div class="single-pred-grid">
                        <div class="single-pred-card">
                            <h4>Health Impact and Guidance</h4>
                            <div class="risk-badge-large" style="border-color: ${(d.risk_info && d.risk_info.color) || '#ff9800'}; color: ${(d.risk_info && d.risk_info.color) || '#ff9800'};">
                                ${(d.risk_info && d.risk_info.risk_level) || 'Moderate'} Risk
                            </div>
                            <p class="single-advisory-p"><strong>Guidance:</strong> ${(d.risk_info && d.risk_info.advisory) || 'No specific advisory'}</p>
                            <p class="single-impact-p"><strong>Health Effect:</strong> ${(d.risk_info && d.risk_info.impact) || 'Air quality is acceptable.'}</p>
                        </div>
                        <div class="single-pred-card">
                            <h4>Category Probability Distribution</h4>
                            <div class="prob-bars-list">
            `;
            if (d.probabilities) {
                const AQI_ORDER = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'];
                Object.entries(d.probabilities).forEach(([catName, prob], idx) => {
                    const probPct = prob <= 1.0 ? (prob * 100) : prob;
                    const catIndex = AQI_ORDER.indexOf(catName) !== -1 ? AQI_ORDER.indexOf(catName) : idx;
                    const catColor = AQI_COLORS[catIndex] || '#3a7bd5';
                    const isSelected = catName === d.predicted_category;
                    html += `
                        <div class="prob-row-item ${isSelected ? 'prob-row-highlight' : ''}">
                            <span class="prob-cat-name">${catName}</span>
                            <div class="prob-progress-track">
                                <div class="prob-progress-fill" style="width: ${probPct}%; background: ${catColor};"></div>
                            </div>
                            <span class="prob-val-text">${Number(probPct).toFixed(1)}%</span>
                        </div>
                    `;
                });
            }
            html += `</div></div></div></div>`;
        }

        bodyEl.innerHTML = html;
    }

    // ============================================================
    // Interactive Date & Time Picker Controls
    // ============================================================
    const dtInput = document.getElementById('pred-datetime-input');
    const predictDtBtn = document.getElementById('btn-predict-datetime');
    const loadingBar = document.getElementById('pred-loading-bar');
    const loadingMsg = document.getElementById('pred-loading-msg');
    const presetButtons = document.querySelectorAll('.preset-chip');

    // Helper: format Date object to local YYYY-MM-DDTHH:MM for <input type="datetime-local">
    function formatDateTimeLocal(dateObj) {
        const yyyy = dateObj.getFullYear();
        const mm = String(dateObj.getMonth() + 1).padStart(2, '0');
        const dd = String(dateObj.getDate()).padStart(2, '0');
        const hh = String(dateObj.getHours()).padStart(2, '0');
        return `${yyyy}-${mm}-${dd}T${hh}:00`;
    }

    // Initialize DateTime input to current hour
    if (dtInput) {
        const initialDate = new Date();
        initialDate.setMinutes(0, 0, 0);
        dtInput.value = formatDateTimeLocal(initialDate);
    }

    // Quick Preset Chips
    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            presetButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const preset = btn.dataset.preset;
            const targetDate = new Date();
            targetDate.setMinutes(0, 0, 0);

            if (preset === 'now') {
                dtInput.value = formatDateTimeLocal(targetDate);
            } else if (preset === 'tomorrow') {
                targetDate.setDate(targetDate.getDate() + 1);
                targetDate.setHours(10, 0, 0, 0);
                dtInput.value = formatDateTimeLocal(targetDate);
            } else if (preset === 'day_after') {
                targetDate.setDate(targetDate.getDate() + 2);
                targetDate.setHours(14, 0, 0, 0);
                dtInput.value = formatDateTimeLocal(targetDate);
            } else if (preset === 'past_monsoon') {
                dtInput.value = '2025-07-15T12:00';
            } else if (preset === 'past_winter') {
                dtInput.value = '2024-11-15T20:00';
            }
        });
    });

    // Remove active highlight from preset chips if user manually changes the datetime input
    if (dtInput) {
        dtInput.addEventListener('input', () => {
            presetButtons.forEach(b => b.classList.remove('active'));
        });
    }

    // Predict AQI Button Handler (Date & Time Inference)
    if (predictDtBtn && dtInput) {
        predictDtBtn.addEventListener('click', async () => {
            const rawVal = dtInput.value;
            if (!rawVal) {
                showToast('Please select a valid date and time first.');
                dtInput.focus();
                return;
            }

            // Convert to clean "YYYY-MM-DD HH:MM"
            const formattedDt = rawVal.replace('T', ' ');

            // UI Loading State
            const originalBtnHTML = predictDtBtn.innerHTML;
            predictDtBtn.disabled = true;
            predictDtBtn.innerHTML = '<span class="loading-spinner" style="display:inline-block; vertical-align:middle; margin-right:6px;"></span> <span class="btn-text">Calculating...</span>';

            if (loadingBar && loadingMsg) {
                loadingMsg.textContent = `Extracting features and predicting for ${formattedDt}...`;
                loadingBar.style.display = 'block';
            }

            try {
                const response = await fetch('/api/run-prediction', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        type: 'datetime',
                        datetime: formattedDt
                    })
                });

                const data = await response.json();

                if (data.status === 'success' && data.prediction) {
                    lastKnownTimestamp = data.prediction.updated_at;
                    renderDynamicPrediction(data.prediction);

                    const predSummary = data.prediction.summary || {};
                    const cat = predSummary.dominant_category || 'Predicted';
                    const conf = predSummary.avg_confidence || '';
                    showToast(`Air quality predicted for ${formattedDt}: ${cat} (${conf}% confidence)`);

                    // Smooth scroll to top of live prediction card
                    const liveSection = document.getElementById('live-prediction');
                    if (liveSection) {
                        liveSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                } else {
                    showToast(data.message || 'Prediction failed. Please try another date or time.');
                }
            } catch (err) {
                console.error('Error during prediction:', err);
                showToast('Prediction request failed: ' + err.message);
            } finally {
                predictDtBtn.disabled = false;
                predictDtBtn.innerHTML = originalBtnHTML;
                if (loadingBar) {
                    loadingBar.style.display = 'none';
                }
            }
        });
    }

    async function checkLatestPrediction(manual = false) {
        try {
            const res = await fetch('/api/latest-prediction');
            const data = await res.json();
            if (data.status === 'success' && data.prediction) {
                const p = data.prediction;
                if (manual || p.updated_at !== lastKnownTimestamp) {
                    const isNew = lastKnownTimestamp !== null && p.updated_at !== lastKnownTimestamp;
                    lastKnownTimestamp = p.updated_at;
                    renderDynamicPrediction(p);
                    if (isNew) {
                        const cat = (p.summary && p.summary.dominant_category) || 'Result';
                        showToast(`New prediction synced: ${cat}`);
                    } else if (manual) {
                        showToast('Live predictions are up to date.');
                    }
                }
            }
        } catch (err) {
            console.error('Error fetching latest prediction:', err);
        }
    }

    // Auto-poll every 2.5 seconds for real-time terminal synchronization
    setInterval(() => checkLatestPrediction(false), 2500);

    // Refresh Button Handler
    const refreshBtn = document.getElementById('btn-refresh-live');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            refreshBtn.textContent = 'Checking...';
            checkLatestPrediction(true).finally(() => {
                setTimeout(() => { refreshBtn.textContent = 'Refresh'; }, 600);
            });
        });
    }
});
