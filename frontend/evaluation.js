const API_URL = 'http://localhost:8000/api';

// Track per-question chart instances to destroy on re-run
const _charts = {};

// ── Load test cases ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch(`${API_URL}/test-cases`);
        if (!res.ok) throw new Error('Failed to fetch test cases');
        renderTestCases(await res.json());
    } catch (e) {
        document.getElementById('testCasesList').innerHTML =
            `<div style="padding:20px;color:var(--red);text-align:center;font-size:12px;">Gagal memuat data. Pastikan backend berjalan.</div>`;
    }

    // ── Theme Management ──────────────────────────────────────────────────────
    const themeToggle = document.getElementById("themeToggle");
    if (themeToggle) {
        const isDark = localStorage.getItem("theme") === "dark";
        if (isDark) {
            document.documentElement.setAttribute("data-theme", "dark");
            themeToggle.checked = true;
        }
        themeToggle.addEventListener("change", (e) => {
            if (e.target.checked) {
                document.documentElement.setAttribute("data-theme", "dark");
                localStorage.setItem("theme", "dark");
            } else {
                document.documentElement.removeAttribute("data-theme");
                localStorage.setItem("theme", "light");
            }
        });
    }

    // ── Scenario Description Updater ──────────────────────────────────────────
    const scenarioSelect = document.getElementById('scenarioSelect');
    const scenarioDesc = document.getElementById('scenarioDesc');
    const SCENARIO_DESCS = {
        conservative: 'Sampling near-greedy — akurasi maksimum, minim variasi.',
        balanced: 'Konfigurasi produksi — keseimbangan presisi dan keterbacaan.',
        explorative: 'Sampling lebih luas — diversitas jawaban lebih tinggi.',
        creative: 'Sampling warm — ekspresi maksimal, uji robustness model.',
    };
    if (scenarioSelect && scenarioDesc) {
        scenarioSelect.addEventListener('change', () => {
            scenarioDesc.textContent = SCENARIO_DESCS[scenarioSelect.value] || '';
        });
    }
});

// ── Render sidebar list ────────────────────────────────────────────
function renderTestCases(testCases) {
    const list = document.getElementById('testCasesList');
    list.innerHTML = '';
    testCases.forEach((tc, i) => {
        const item = document.createElement('label');
        item.className = 'test-case-item';

        const cb = document.createElement('input');
        cb.type = 'checkbox'; cb.className = 'tc-checkbox'; cb.value = tc.instruction;
        cb.addEventListener('change', updateSelectionCount);

        const idx = document.createElement('span');
        idx.className = 'tc-index'; idx.textContent = `Q${i + 1}`;

        const txt = document.createElement('span');
        txt.className = 'test-case-text'; txt.title = tc.instruction;
        txt.textContent = tc.instruction;

        item.append(cb, idx, txt);
        list.appendChild(item);
    });

    document.getElementById('selectAll').addEventListener('change', e => {
        document.querySelectorAll('.tc-checkbox').forEach(cb => cb.checked = e.target.checked);
        updateSelectionCount();
    });
}

function updateSelectionCount() {
    const n = document.querySelectorAll('.tc-checkbox:checked').length;
    document.getElementById('selCount').textContent = n;
    document.getElementById('btnEvaluate').disabled = n === 0;
}

// ── Evaluate ────────────────────────────────────────────────────────
document.getElementById('btnEvaluate').addEventListener('click', async () => {
    const selected = [...document.querySelectorAll('.tc-checkbox:checked')].map(cb => cb.value);
    if (!selected.length) return;

    // Destroy old charts
    Object.values(_charts).forEach(c => c?.destroy());
    Object.keys(_charts).forEach(k => delete _charts[k]);

    document.getElementById('loadingOverlay').style.display = 'flex';
    document.getElementById('resultsContainer').innerHTML = '';
    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('vizSection').style.display = 'none';

    const t0 = Date.now();

    try {
        const scenario = document.getElementById('scenarioSelect')?.value || 'balanced';
        // 5-hour timeout — explorative/creative scenarios with 50 questions can take 3+ hours
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 18000000);
        const res = await fetch(`${API_URL}/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ instructions: selected, scenario: scenario }),
            signal: controller.signal,
        });
        clearTimeout(timeoutId);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

        renderSummary(data.results, data.total_time_sec, elapsed);
        if (data.global_viz) renderGlobalViz(data.global_viz);
        renderResults(data.results);

    } catch (e) {
        // If aborted or connection lost, the backend may have finished and cached results.
        // Auto-attempt loading cached results before showing the error.
        if (e.name === 'AbortError' || e.message?.includes('abort')) {
            document.getElementById('resultsContainer').innerHTML =
                `<div style="padding:30px;color:var(--gold);text-align:center;">
                    Koneksi timeout — evaluasi mungkin masih berjalan di backend.<br>
                    Mencoba memuat hasil tersimpan...
                </div>`;
            // Wait a moment then try loading cached results
            setTimeout(() => loadLastResults(), 2000);
        } else {
            document.getElementById('resultsContainer').innerHTML =
                `<div style="padding:30px;color:var(--red);text-align:center;">Evaluasi gagal: ${e.message}<br><br>
                 <button onclick="loadLastResults()" style="padding:10px 20px;background:var(--gold);color:#000;border:none;border-radius:var(--rs);font-weight:700;cursor:pointer;">
                     Muat Hasil Terakhir
                 </button></div>`;
        }
    } finally {
        document.getElementById('loadingOverlay').style.display = 'none';
    }
});

// ── Load Last Cached Results ─────────────────────────────────────────
async function loadLastResults() {
    // Destroy old charts
    Object.values(_charts).forEach(c => c?.destroy());
    Object.keys(_charts).forEach(k => delete _charts[k]);

    document.getElementById('resultsContainer').innerHTML = '';
    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('vizSection').style.display = 'none';

    try {
        const res = await fetch(`${API_URL}/evaluate/last`);
        if (!res.ok) throw new Error(`HTTP ${res.status} — Tidak ada hasil tersimpan.`);
        const data = await res.json();

        renderSummary(data.results, data.total_time_sec, data.total_time_sec);
        if (data.global_viz) renderGlobalViz(data.global_viz);
        renderResults(data.results);
    } catch (e) {
        document.getElementById('resultsContainer').innerHTML =
            `<div style="padding:30px;color:var(--red);text-align:center;">Gagal memuat hasil: ${e.message}</div>`;
    }
}

// ══════════════════════════════════════════════════════════════════════
// SUMMARY SECTION
// ══════════════════════════════════════════════════════════════════════

function avg(results, path) {
    const vals = results.map(r => {
        const parts = path.split('.');
        let v = r;
        for (const p of parts) v = v?.[p];
        return typeof v === 'number' ? v : null;
    }).filter(x => x !== null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
}

const METRICS_DEF = [
    // [label, pathVanilla, pathFinetuned, lowerIsBetter]
    ['SacreBLEU', 'vanilla.metrics.Semantic.SacreBLEU', 'finetuned.metrics.Semantic.SacreBLEU', false],
    ['ROUGE-L', 'vanilla.metrics.Semantic.ROUGE-L', 'finetuned.metrics.Semantic.ROUGE-L', false],
    ['METEOR', 'vanilla.metrics.Semantic.METEOR', 'finetuned.metrics.Semantic.METEOR', false],
    ['BERTScore (F1)', 'vanilla.metrics.Sequential.BERTScore (F1)', 'finetuned.metrics.Sequential.BERTScore (F1)', false],
    ['Sent Sim', 'vanilla.metrics.Sequential.Sentence Similarity', 'finetuned.metrics.Sequential.Sentence Similarity', false],
    ['NLI Entail', 'vanilla.metrics.Sequential.NLI Entailment', 'finetuned.metrics.Sequential.NLI Entailment', false],
    ['Perplexity', 'vanilla.metrics.Sequential.Perplexity', 'finetuned.metrics.Sequential.Perplexity', true],
    ['NLaw Score', 'vanilla.metrics.Latent.NLaw Score (Cosine)', 'finetuned.metrics.Latent.NLaw Score (Cosine)', false],
    ['L2 Dist', 'vanilla.metrics.Latent.L2 Latent Space', 'finetuned.metrics.Latent.L2 Latent Space', true],
];

function renderSummary(results, serverTime, clientTime) {
    const container = document.getElementById('resultsContainer');

    let vanillaWins = 0, ftWins = 0;

    const rows = METRICS_DEF.map(([label, vPath, fPath, lowerIsBetter]) => {
        const vAvg = avg(results, vPath);
        const fAvg = avg(results, fPath);
        let vClass = '', fClass = '';
        if (vAvg !== null && fAvg !== null) {
            const vWins = lowerIsBetter ? vAvg <= fAvg : vAvg >= fAvg;
            vClass = vWins ? 'winner' : '';
            fClass = !vWins ? 'winner' : '';
            if (vWins) vanillaWins++; else ftWins++;
        }
        const isPPPL = label === 'Perplexity';
        return `
        <tr>
            <td class="metric-row-label">${label}</td>
            <td class="metric-row-val ${vClass}">${vAvg !== null ? fmt(vAvg, isPPPL) : 'N/A'}</td>
            <td class="metric-row-val ${fClass}">${fAvg !== null ? fmt(fAvg, isPPPL) : 'N/A'}</td>
        </tr>`;
    }).join('');

    const overallWinner = ftWins > vanillaWins
        ? `<span style="color:var(--gold)">Fine-Tuned menang ${ftWins}/${METRICS_DEF.length} metrik</span>`
        : `<span style="color:var(--t3)">Vanilla menang ${vanillaWins}/${METRICS_DEF.length} metrik</span>`;

    const summary = document.createElement('div');
    summary.innerHTML = `
    <div class="summary-card">
        <div class="summary-header">
            <div>
                <div class="summary-title">Ringkasan Evaluasi</div>
                <div class="summary-sub">${results.length} kasus &nbsp;·&nbsp; Server: ${serverTime}s &nbsp;·&nbsp; Total: ${clientTime}s</div>
            </div>
            <div class="summary-winner">${overallWinner}</div>
        </div>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Metrik</th>
                    <th><span class="th-badge vanilla-badge">Vanilla</span></th>
                    <th><span class="th-badge ft-badge">Fine-Tuned</span></th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    </div>`;
    container.appendChild(summary);
}

// ══════════════════════════════════════════════════════════════════════
// RESULT CARDS
// ══════════════════════════════════════════════════════════════════════

function fmt(v, isPPPL = false) {
    if (v === 'N/A' || v === null || v === undefined) return 'N/A';
    if (v === Infinity || v === 'Infinity') return '∞';
    if (typeof v === 'number') return isPPPL ? v.toFixed(5) : v.toFixed(2);
    return String(v);
}

function winClass(a, b, lowerBetter = false) {
    if (typeof a !== 'number' || typeof b !== 'number') return '';
    return lowerBetter ? (a <= b ? 'winner' : '') : (a >= b ? 'winner' : '');
}

function renderResults(results) {
    const container = document.getElementById('resultsContainer');

    results.forEach((res, idx) => {
        const vm = res.vanilla.metrics;
        const fm = res.finetuned.metrics;
        const chartId = `chart-pca-${idx}`;
        const chartIdT = `chart-tsne-${idx}`;

        const card = document.createElement('div');
        card.className = 'result-card';

        card.innerHTML = `
        <div class="result-card-header">
            <h3>Test Case #${idx + 1}
                <span style="margin-left:10px;font-size:10px;font-weight:400;color:var(--t3);">
                    Vanilla ${res.vanilla.gen_time_sec}s &nbsp;|&nbsp; Fine-Tuned ${res.finetuned.gen_time_sec}s &nbsp;|&nbsp; Metrik ${res.eval_time_sec}s
                </span>
            </h3>
            <div class="result-instruction">${escHtml(res.instruction)}</div>
            <div class="result-gt"><strong>Ground Truth</strong>${escHtml(res.ground_truth)}</div>
            ${res.context_used ? `<div class="result-gt" style="border-left-color:var(--b1);margin-top:6px;"><strong style="color:var(--t3);">Konteks RAG</strong>${escHtml(res.context_used)}</div>` : ''}
        </div>

        <div class="result-body">
            <!-- Vanilla -->
            <div class="model-col vanilla">
                <div class="model-col-title">Vanilla (qwen3.5:9b)</div>
                <div class="response-box">${escHtml(res.vanilla.response)}</div>
                <div class="layer-label">Semantic</div>
                <div class="metrics-grid">
                    ${metricCell('SacreBLEU', vm.Semantic?.SacreBLEU, fm.Semantic?.SacreBLEU)}
                    ${metricCell('ROUGE-L', vm.Semantic?.['ROUGE-L'], fm.Semantic?.['ROUGE-L'])}
                    ${metricCell('METEOR', vm.Semantic?.METEOR, fm.Semantic?.METEOR)}
                </div>
                <div class="layer-label">Sequential</div>
                <div class="metrics-grid two">
                    ${metricCell('BERTScore', vm.Sequential?.['BERTScore (F1)'], fm.Sequential?.['BERTScore (F1)'])}
                    ${metricCell('Sent Sim', vm.Sequential?.['Sentence Similarity'], fm.Sequential?.['Sentence Similarity'])}
                    ${metricCell('NLI Entail', vm.Sequential?.['NLI Entailment'], fm.Sequential?.['NLI Entailment'])}
                    ${metricCell('Perplexity', vm.Sequential?.Perplexity, fm.Sequential?.Perplexity, true)}
                </div>
                <div class="layer-label">Latent Space</div>
                <div class="metrics-grid two">
                    ${metricCell('NLaw Score', vm.Latent?.['NLaw Score (Cosine)'], fm.Latent?.['NLaw Score (Cosine)'])}
                    ${metricCell('L2 Dist', vm.Latent?.['L2 Latent Space'], fm.Latent?.['L2 Latent Space'], true)}
                </div>
            </div>

            <!-- Fine-Tuned -->
            <div class="model-col finetuned">
                <div class="model-col-title">Fine-Tuned (qwen3.5-9b-nlaw)</div>
                <div class="response-box">${escHtml(res.finetuned.response)}</div>
                <div class="layer-label">Semantic</div>
                <div class="metrics-grid">
                    ${metricCell('SacreBLEU', fm.Semantic?.SacreBLEU, vm.Semantic?.SacreBLEU)}
                    ${metricCell('ROUGE-L', fm.Semantic?.['ROUGE-L'], vm.Semantic?.['ROUGE-L'])}
                    ${metricCell('METEOR', fm.Semantic?.METEOR, vm.Semantic?.METEOR)}
                </div>
                <div class="layer-label">Sequential</div>
                <div class="metrics-grid two">
                    ${metricCell('BERTScore', fm.Sequential?.['BERTScore (F1)'], vm.Sequential?.['BERTScore (F1)'])}
                    ${metricCell('Sent Sim', fm.Sequential?.['Sentence Similarity'], vm.Sequential?.['Sentence Similarity'])}
                    ${metricCell('NLI Entail', fm.Sequential?.['NLI Entailment'], vm.Sequential?.['NLI Entailment'])}
                    ${metricCell('Perplexity', fm.Sequential?.Perplexity, vm.Sequential?.Perplexity, true)}
                </div>
                <div class="layer-label">Latent Space</div>
                <div class="metrics-grid two">
                    ${metricCell('NLaw Score', fm.Latent?.['NLaw Score (Cosine)'], vm.Latent?.['NLaw Score (Cosine)'])}
                    ${metricCell('L2 Dist', fm.Latent?.['L2 Latent Space'], vm.Latent?.['L2 Latent Space'], true)}
                </div>
            </div>
        </div>

        <!-- Per-question latent space charts -->
        ${res.combined_viz ? `
        <div class="per-q-viz">
            <div class="per-q-viz-title">Latent Space — Test Case #${idx + 1}</div>
            <div class="per-q-viz-grid">
                <div class="chart-wrap"><canvas id="${chartId}"></canvas></div>
                <div class="chart-wrap"><canvas id="${chartIdT}"></canvas></div>
            </div>
        </div>` : ''}
        `;

        container.appendChild(card);

        // Render charts after card is in DOM
        if (res.combined_viz) {
            renderPerQuestionCharts(res.combined_viz, chartId, chartIdT, idx + 1);
        }
    });
}

function metricCell(label, ownVal, otherVal, lowerBetter = false) {
    const cls = winClass(ownVal, otherVal, lowerBetter);
    const isPPPL = label === 'Perplexity';
    return `<div class="metric-cell ${cls}">
        <div class="metric-lbl">${label}</div>
        <div class="metric-val">${fmt(ownVal, isPPPL)}</div>
    </div>`;
}

// ══════════════════════════════════════════════════════════════════════
// PER-QUESTION CHARTS
// ══════════════════════════════════════════════════════════════════════

const CHART_OPTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: { labels: { color: '#8fa3c0', font: { family: 'Inter', size: 11 }, padding: 10 } },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: (${ctx.parsed.x.toFixed(3)}, ${ctx.parsed.y.toFixed(3)})` } }
    },
    scales: {
        x: { ticks: { color: '#4e6280', font: { size: 10 } }, grid: { color: 'rgba(30,45,72,0.8)' } },
        y: { ticks: { color: '#4e6280', font: { size: 10 } }, grid: { color: 'rgba(30,45,72,0.8)' } }
    }
};

function makeScatterData(pca) {
    if (!pca) return null;
    return {
        datasets: [
            { label: 'Vanilla', data: [{ x: pca.vanilla[0], y: pca.vanilla[1] }], backgroundColor: '#94a3b8', pointRadius: 11, pointHoverRadius: 14 },
            { label: 'Fine-Tuned', data: [{ x: pca.finetuned[0], y: pca.finetuned[1] }], backgroundColor: '#c9a227', pointRadius: 11, pointHoverRadius: 14 },
            { label: 'Ground Truth', data: [{ x: pca.ground_truth[0], y: pca.ground_truth[1] }], backgroundColor: '#22c55e', pointRadius: 11, pointHoverRadius: 14, pointStyle: 'rect' },
        ]
    };
}

function renderPerQuestionCharts(viz, pcaId, tsneId, qNum) {
    const pcaData = makeScatterData(viz.PCA);
    const tsneData = makeScatterData(viz.tSNE);

    if (pcaData) {
        const c = new Chart(document.getElementById(pcaId).getContext('2d'), {
            type: 'scatter',
            data: pcaData,
            options: { ...CHART_OPTS, plugins: { ...CHART_OPTS.plugins, title: { display: true, text: `PCA — Q${qNum}`, color: '#8fa3c0', font: { size: 12, family: 'Inter' } } } }
        });
        _charts[pcaId] = c;
    }

    if (tsneData) {
        const c = new Chart(document.getElementById(tsneId).getContext('2d'), {
            type: 'scatter',
            data: tsneData,
            options: { ...CHART_OPTS, plugins: { ...CHART_OPTS.plugins, title: { display: true, text: `t-SNE — Q${qNum}`, color: '#8fa3c0', font: { size: 12, family: 'Inter' } } } }
        });
        _charts[tsneId] = c;
    }
}

// ── Helpers ─────────────────────────────────────────────────────────
function escHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/\n/g, '<br>');
}

// ══════════════════════════════════════════════════════════════════════
// GLOBAL LATENT SPACE COMPILATION CHART
// ══════════════════════════════════════════════════════════════════════

const Q_COLORS = [
    '#e07b54', '#54aae0', '#7be07b', '#e0c554', '#c554e0',
    '#54e0c5', '#e05484', '#8454e0', '#e0a854', '#54e084',
];

function _buildGlobalDatasets(points, coordKey) {
    // Group by question, then split into 3 model-type datasets per q
    const byQ = {};
    points.forEach(p => {
        const q = p.q;
        if (!byQ[q]) byQ[q] = { vanilla: null, finetuned: null, ground_truth: null };
        byQ[q][p.type] = p[coordKey];
    });

    const datasets = [];
    Object.keys(byQ).sort((a, b) => +a - +b).forEach(q => {
        const color = Q_COLORS[(+q - 1) % Q_COLORS.length];
        const pts = byQ[q];
        if (pts.vanilla)
            datasets.push({
                label: `Q${q} Vanilla`,
                data: [{ x: pts.vanilla[0], y: pts.vanilla[1] }],
                backgroundColor: color + 'aa',
                borderColor: color,
                pointStyle: 'circle',
                pointRadius: 10,
                pointHoverRadius: 13,
            });
        if (pts.finetuned)
            datasets.push({
                label: `Q${q} Fine-Tuned`,
                data: [{ x: pts.finetuned[0], y: pts.finetuned[1] }],
                backgroundColor: color,
                borderColor: color,
                pointStyle: 'triangle',
                pointRadius: 11,
                pointHoverRadius: 14,
            });
        if (pts.ground_truth)
            datasets.push({
                label: `Q${q} GT`,
                data: [{ x: pts.ground_truth[0], y: pts.ground_truth[1] }],
                backgroundColor: color + '66',
                borderColor: color,
                pointStyle: 'rect',
                pointRadius: 9,
                pointHoverRadius: 12,
            });
    });
    return datasets;
}

function renderGlobalViz(globalViz) {
    const container = document.getElementById('resultsContainer');
    const points = globalViz.points || [];
    if (!points.length) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'global-viz-card';
    wrapper.innerHTML = `
    <div class="global-viz-header" style="display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; gap: 10px;">
        <div>
            <div class="global-viz-title">Kompilasi Latent Space</div>
            <div class="global-viz-sub">
                ○ Lingkaran = Vanilla &nbsp; △ Segitiga = Fine-Tuned &nbsp; □ Persegi = Ground Truth
            </div>
        </div>
        <select id="globalVizFilter" style="background:var(--s2); border:1px solid var(--b1); color:var(--t2); padding:6px 12px; border-radius:var(--rs); font-size:11.5px; outline:none; cursor:pointer;">
            <option value="all">Semua Pertanyaan</option>
        </select>
    </div>
    <div class="global-viz-charts">
        <div class="chart-wrap"><canvas id="chart-global-pca"></canvas></div>
        <div class="chart-wrap"><canvas id="chart-global-tsne"></canvas></div>
    </div>`;
    container.appendChild(wrapper);

    const sharedOpts = (title) => ({
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            title: { display: true, text: title, color: '#c9a227', font: { size: 12, family: 'Inter', weight: 700 } },
            tooltip: {
                callbacks: {
                    label: ctx => `${ctx.dataset.label}: (${ctx.parsed.x.toFixed(3)}, ${ctx.parsed.y.toFixed(3)})`
                }
            }
        },
        scales: {
            x: { ticks: { color: '#4e6280', font: { size: 9 } }, grid: { color: 'rgba(30,45,72,0.7)' } },
            y: { ticks: { color: '#4e6280', font: { size: 9 } }, grid: { color: 'rgba(30,45,72,0.7)' } }
        }
    });

    const pcaData = _buildGlobalDatasets(points, 'pca');
    const tsneData = _buildGlobalDatasets(points, 'tsne');

    const cPCA = new Chart(document.getElementById('chart-global-pca').getContext('2d'), {
        type: 'scatter', data: { datasets: pcaData }, options: sharedOpts('PCA Global — Semua Pertanyaan')
    });
    const cTSNE = new Chart(document.getElementById('chart-global-tsne').getContext('2d'), {
        type: 'scatter', data: { datasets: tsneData }, options: sharedOpts('t-SNE Global — Semua Pertanyaan')
    });
    _charts['chart-global-pca'] = cPCA;
    _charts['chart-global-tsne'] = cTSNE;

    // Populate dropdown
    const filterSelect = document.getElementById('globalVizFilter');
    const uniqueQs = [...new Set(points.map(p => p.q))].sort((a, b) => +a - +b);
    uniqueQs.forEach(q => {
        const opt = document.createElement('option');
        opt.value = q;
        opt.textContent = `Test Case #${q}`;
        filterSelect.appendChild(opt);
    });

    filterSelect.addEventListener('change', (e) => {
        const selected = e.target.value;
        [cPCA, cTSNE].forEach(chart => {
            chart.data.datasets.forEach(ds => {
                if (selected === 'all') {
                    ds.hidden = false;
                } else {
                    ds.hidden = !ds.label.startsWith(`Q${selected} `);
                }
            });
            chart.update();
        });
    });
}
