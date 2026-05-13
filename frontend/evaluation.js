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
        const res = await fetch(`${API_URL}/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ instructions: selected })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

        renderSummary(data.results, data.total_time_sec, elapsed);
        if (data.global_viz) renderGlobalViz(data.global_viz);
        renderResults(data.results);

    } catch (e) {
        document.getElementById('resultsContainer').innerHTML =
            `<div style="padding:30px;color:var(--red);text-align:center;">Evaluasi gagal: ${e.message}</div>`;
    } finally {
        document.getElementById('loadingOverlay').style.display = 'none';
    }
});

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
        return `
        <tr>
            <td class="metric-row-label">${label}</td>
            <td class="metric-row-val ${vClass}">${vAvg !== null ? vAvg.toFixed(4) : 'N/A'}</td>
            <td class="metric-row-val ${fClass}">${fAvg !== null ? fAvg.toFixed(4) : 'N/A'}</td>
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

function fmt(v) {
    if (v === 'N/A' || v === null || v === undefined) return 'N/A';
    if (v === Infinity || v === 'Infinity') return '∞';
    if (typeof v === 'number') return v.toFixed(4);
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
    return `<div class="metric-cell ${cls}">
        <div class="metric-lbl">${label}</div>
        <div class="metric-val">${fmt(ownVal)}</div>
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
    <div class="global-viz-header">
        <div class="global-viz-title">Kompilasi Latent Space — Semua Pertanyaan</div>
        <div class="global-viz-sub">
            Seluruh embedding diprojeksikan ke ruang koordinat bersama (shared PCA/t-SNE).
            ○ Lingkaran = Vanilla &nbsp; △ Segitiga = Fine-Tuned &nbsp; □ Persegi = Ground Truth
        </div>
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
            legend: {
                display: true,
                labels: { color: '#8fa3c0', font: { family: 'Inter', size: 9 }, padding: 6, boxWidth: 10 }
            },
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
}
