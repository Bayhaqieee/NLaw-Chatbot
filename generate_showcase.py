import json
import os

xps = [
    {
        'id': 1, 'win': 9,
        'metrics': [
            ['SACREBLEU', '6.4634', '12.3319'],
            ['ROUGE-L', '0.2360', '0.2958'],
            ['METEOR', '0.3877', '0.4079'],
            ['BERTSCORE (F1)', '0.9742', '0.9746'],
            ['SENT SIM', '0.6897', '0.7818'],
            ['NLI ENTAIL', '0.1239', '0.3623'],
            ['PERPLEXITY', '0.35744', '0.29074'],
            ['NLAW SCORE', '0.6426', '0.7093'],
            ['L2 DIST', '0.8351', '0.7406']
        ]
    },
    {
        'id': 2, 'win': 7,
        'metrics': [
            ['SACREBLEU', '6.5286', '10.7679'],
            ['ROUGE-L', '0.2393', '0.2875'],
            ['METEOR', '0.3852', '0.3811'],
            ['BERTSCORE (F1)', '0.9742', '0.9742'],
            ['SENT SIM', '0.6858', '0.7762'],
            ['NLI ENTAIL', '0.1362', '0.3273'],
            ['PERPLEXITY', '0.34947', '0.28771'],
            ['NLAW SCORE', '0.6505', '0.7123'],
            ['L2 DIST', '0.8263', '0.7368']
        ]
    },
    {
        'id': 3, 'win': 7,
        'metrics': [
            ['SACREBLEU', '6.2286', '10.0694'],
            ['ROUGE-L', '0.2324', '0.2562'],
            ['METEOR', '0.3881', '0.3597'],
            ['BERTSCORE (F1)', '0.9737', '0.9732'],
            ['SENT SIM', '0.6749', '0.7502'],
            ['NLI ENTAIL', '0.1034', '0.4577'],
            ['PERPLEXITY', '0.35168', '0.30297'],
            ['NLAW SCORE', '0.6483', '0.6857'],
            ['L2 DIST', '0.8282', '0.7673']
        ]
    },
    {
        'id': 4, 'win': 8,
        'metrics': [
            ['SACREBLEU', '6.2819', '10.5097'],
            ['ROUGE-L', '0.2344', '0.2753'],
            ['METEOR', '0.3979', '0.3843'],
            ['BERTSCORE (F1)', '0.9737', '0.9737'],
            ['SENT SIM', '0.6713', '0.7773'],
            ['NLI ENTAIL', '0.1008', '0.3897'],
            ['PERPLEXITY', '0.35805', '0.29021'],
            ['NLAW SCORE', '0.6420', '0.7098'],
            ['L2 DIST', '0.8356', '0.7391']
        ]
    }
]

questions = [
    'Menurut UU No. 19 Tahun 2016, apa kewajiban pengendali data?',
    'Menurut Perpres No. 83 Tahun 2025, siapakah yang berhak...',
    'Menurut UU No. 27 Tahun 2022 (PDP), apa definisi data pribadi?',
    'Menurut Perpres No. 82 Tahun 2023, apa itu data spesifik?',
    'Menurut UU No. 19 Tahun 2016, bagaimana aturan penyadapan...',
    'Menurut Perpres No. 82 Tahun 2023, apa itu dokumen elektronik?',
    'Menurut UU No. 1 Tahun 2024, siapa subjek hukum uu ini?',
    'Menurut Perpres No. 83 Tahun 2025, bagaimana aturan...',
    'Menurut UU No. 1 Tahun 2024, apa itu data spesifik?',
    'Menurut Perpres No. 82 Tahun 2023, apa sanksi pencemaran...',
    'Menurut Perpres No. 83 Tahun 2025, apa itu data spesifik?',
    'Menurut UU No. 19 Tahun 2016, apa tugas lembaga pengawas?'
]

def render_sidebar():
    html = ''
    for i, q in enumerate(questions):
        bg = 'var(--s2)' if i==0 else 'transparent'
        html += f'''
        <div class="test-case-item" style="background: {bg}">
            <input type="checkbox" checked onclick="return false;" style="accent-color:var(--gold);margin-top:2px">
            <div class="tc-index" style="font-size:10px;font-weight:700;color:var(--gold);flex-shrink:0;margin-top:2px;">Q{i+1}</div>
            <div class="test-case-text" style="font-size:11.5px;color:var(--t2);line-height:1.5;">{q}</div>
        </div>
        '''
    return html

sidebar_html = render_sidebar()

head_template = '''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Showcase XP-{id} - NusantaraLaw</title>
    <link rel="stylesheet" href="style.css?v=18">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Lora:wght@600;700&display=swap" rel="stylesheet">
    <style>
        /* Exact clone of evaluation.html layout */
        body {{ overflow: hidden; margin: 0; padding: 0; background: var(--bg); font-family: 'Inter', sans-serif; }}
        .eval-layout {{ display: flex; width: 100%; height: 100vh; }}
        
        /* Eval Sidebar */
        .eval-sidebar {{ width: 280px; min-width: 280px; background: var(--s1); border-right: 1px solid var(--b0); display: flex; flex-direction: column; overflow: hidden; }}
        .eval-sidebar-header {{ padding: 16px 18px; border-bottom: 1px solid var(--b0); display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }}
        .eval-sidebar-header h2 {{ font-family: 'Lora', serif; font-size: 15px; font-weight: 700; color: var(--gold); margin: 0;}}
        .back-link {{ font-size: 11.5px; color: var(--t3); text-decoration: none; }}
        .eval-actions {{ padding: 12px 18px; border-bottom: 1px solid var(--b0); display: flex; flex-direction: column; gap: 8px; flex-shrink: 0; }}
        .btn-evaluate {{ width: 100%; padding: 10px; background: linear-gradient(135deg, var(--gold), #9a7a10); color: #000; font-size: 12.5px; font-weight: 700; border: none; border-radius: var(--rs); opacity: 0.4; }}
        .select-all-row {{ display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--t2); }}
        .test-cases-list {{ flex: 1; overflow-y: hidden; }}
        .test-case-item {{ padding: 10px 18px; border-bottom: 1px solid var(--b0); display: flex; align-items: flex-start; gap: 9px; }}

        /* Eval Main */
        .eval-main {{ flex: 1; padding: 24px 28px; background: var(--bg); overflow-y: auto; }}
        
        /* Summary Card */
        .summary-card {{ background: var(--s1); border: 1px solid var(--b0); border-radius: var(--r); padding: 20px 24px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        [data-theme="light"] .summary-card {{ box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
        .summary-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }}
        .summary-title {{ font-family: 'Lora', serif; font-size: 16px; font-weight: 700; color: var(--t1); }}
        .summary-sub {{ font-size: 11.5px; color: var(--t3); margin-top: 3px; }}
        .summary-winner {{ font-size: 13px; font-weight: 600; color: var(--gold); }}
        
        .summary-table {{ width: 100%; border-collapse: collapse; }}
        .summary-table th {{ text-align: center; padding: 10px 12px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; border-bottom: 1px solid var(--b1); color: var(--t2); }}
        .summary-table th:first-child {{ text-align: left; }}
        .summary-table td {{ padding: 7px 12px; border-bottom: 1px solid var(--b0); color: var(--t1); font-size: 13px; font-weight: 600; text-align: center; }}
        .summary-table td:first-child {{ text-align: left; color: var(--t3); font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 400; }}
        
        .th-badge {{ display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 10px; font-weight: 700; }}
        .vanilla-badge {{ background: var(--s3); color: var(--t2); border: 1px solid var(--b1); }}
        .ft-badge {{ background: var(--gold-lo); color: var(--gold); border: 1px solid var(--gold-glow); }}

        /* Viz Section */
        .viz-section {{ background: var(--s1); border: 1px solid var(--b0); border-radius: var(--r); padding: 20px 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        [data-theme="light"] .viz-section {{ box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
        .viz-header-row {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }}
        .viz-title {{ font-family: 'Lora', serif; font-size: 16px; font-weight: 700; color: var(--t1); }}
        .viz-select {{ background: var(--s2); border: 1px solid var(--b0); color: var(--t2); font-size: 11.5px; padding: 6px 10px; border-radius: 4px; outline: none; }}
        .viz-subtitle {{ font-size: 12px; color: var(--t3); margin-bottom: 20px; }}
        .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; height: 340px; }}
        .chart-wrap {{ background: var(--bg); border: 1px solid var(--b0); border-radius: var(--rs); display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; }}
        .chart-title {{ position: absolute; top: 14px; width: 100%; text-align: center; font-size: 11px; font-weight: 700; color: var(--gold); }}
        
        .grid-lines {{ width: 100%; height: 100%; display: grid; grid-template-columns: repeat(10, 1fr); grid-template-rows: repeat(6, 1fr); box-sizing: border-box; border-left: 1px solid var(--b0); border-bottom: 1px solid var(--b0); margin: 30px 10px 10px 30px; }}
        .grid-lines div {{ border-right: 1px solid rgba(255,255,255,0.05); border-top: 1px solid rgba(255,255,255,0.05); }}
        [data-theme="light"] .grid-lines div {{ border-color: rgba(0,0,0,0.05); }}
        .y-axis {{ position: absolute; left: 6px; top: 30px; bottom: 10px; display: flex; flex-direction: column; justify-content: space-between; font-size: 9px; color: var(--t3); }}

        .viz-dots {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
        
        /* Print fixes */
        @media print {{
            body {{ background: #fff !important; }}
            .eval-sidebar {{ border-right: 1px solid #ccc; }}
            * {{ box-shadow: none !important; }}
        }}
    </style>
</head>
<body data-theme="dark">
'''

# Fake plotted points to mimic the screenshot exactly
pca_dots = '''
    <div style="position:absolute; width:10px; height:10px; background:var(--gold); border-radius:1px; left:12%; bottom:35%; opacity:0.8;"></div>
    <div style="position:absolute; width:16px; height:16px; background:#d97757; border-radius:50%; right:15%; top:25%; opacity:0.9;"></div>
'''
tsne_dots = '''
    <div style="position:absolute; width:10px; height:10px; background:var(--gold); border-radius:1px; right:12%; top:30%; opacity:0.8;"></div>
    <div style="position:absolute; width:16px; height:16px; background:#d97757; border-radius:50%; left:15%; top:20%; opacity:0.9;"></div>
'''

for xp in xps:
    metrics_rows = ''
    for m in xp['metrics']:
        is_ppl_or_l2 = m[0] in ['PERPLEXITY', 'L2 DIST']
        v1 = float(m[1])
        v2 = float(m[2])
        # Determine winner
        win_class_v = ''
        win_class_f = ''
        if v1 != v2:
            if is_ppl_or_l2:
                if v1 <= v2: win_class_v = 'color:var(--gold)'
                else: win_class_f = 'color:var(--gold)'
            else:
                if v1 >= v2: win_class_v = 'color:var(--gold)'
                else: win_class_f = 'color:var(--gold)'

        metrics_rows += f'''
        <tr>
            <td>{m[0]}</td>
            <td style="{win_class_v}">{m[1]}</td>
            <td style="{win_class_f}">{m[2]}</td>
        </tr>'''

    # Build the HTML for this specific XP run
    html_out = head_template.format(id=xp['id'])
    
    html_out += f'''
    <div class="eval-layout">
        <aside class="eval-sidebar">
            <div class="eval-sidebar-header">
                <div>
                    <h2>Golden Test Set</h2>
                    <div style="font-size:11px;color:var(--t3);margin-top:2px;">XP-{xp['id']} Evaluator</div>
                </div>
                <label class="toggle" style="transform: scale(0.8);" title="Dark Mode">
                    <input type="checkbox" checked onchange="document.body.setAttribute('data-theme', this.checked ? 'dark' : 'light');">
                    <span class="slider"></span>
                </label>
            </div>
            <div class="eval-actions">
                <button class="btn-evaluate">Mulai Evaluasi (1)</button>
                <div class="select-all-row">
                    <input type="checkbox" checked onclick="return false;" style="accent-color:var(--gold)">
                    <label>Pilih Semua</label>
                </div>
            </div>
            <div class="test-cases-list">
                {sidebar_html}
            </div>
        </aside>

        <main class="eval-main">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h3 style="margin:0; color:var(--t2); font-size:13px; font-weight:600; letter-spacing:1px; text-transform:uppercase;">Eksperimen XP-{xp['id']}</h3>
                <div style="display:flex; gap:10px;">
                    <a href="showcase-xp1.html" style="color:var(--gold); font-size:12px; text-decoration:none;">XP-1</a>
                    <a href="showcase-xp2.html" style="color:var(--gold); font-size:12px; text-decoration:none;">XP-2</a>
                    <a href="showcase-xp3.html" style="color:var(--gold); font-size:12px; text-decoration:none;">XP-3</a>
                    <a href="showcase-xp4.html" style="color:var(--gold); font-size:12px; text-decoration:none;">XP-4</a>
                </div>
            </div>
            <div class="summary-card">
                <div class="summary-header">
                    <div>
                        <div class="summary-title">Ringkasan Evaluasi</div>
                        <div class="summary-sub">1 kasus  ·  Server: 222.82s  ·  Total: 222.8s</div>
                    </div>
                    <div class="summary-winner">
                        Fine-Tuned menang {xp['win']}/9 metrik
                    </div>
                </div>
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Metrik</th>
                            <th><span class="th-badge vanilla-badge">Vanilla</span></th>
                            <th><span class="th-badge ft-badge">Fine-Tuned</span></th>
                        </tr>
                    </thead>
                    <tbody>
                        {metrics_rows}
                    </tbody>
                </table>
            </div>

            <div class="viz-section">
                <div class="viz-header-row">
                    <div class="viz-title">Kompilasi Latent Space</div>
                    <select class="viz-select"><option>Semua Pertanyaan</option></select>
                </div>
                <div class="viz-subtitle">○ Lingkaran = Vanilla &nbsp; △ Segitiga = Fine-Tuned &nbsp; □ Persegi = Ground Truth</div>
                <div class="charts-grid">
                    <div class="chart-wrap">
                        <div class="chart-title">PCA Global — Semua Pertanyaan</div>
                        <div class="y-axis"><span>0.3</span><span>0.2</span><span>0.1</span><span>0</span><span>-0.1</span><span>-0.2</span></div>
                        <div class="grid-lines">{'<div></div>'*60}</div>
                        <div class="viz-dots">{pca_dots}</div>
                    </div>
                    <div class="chart-wrap">
                        <div class="chart-title">t-SNE Global — Semua Pertanyaan</div>
                        <div class="y-axis"><span>-40</span><span>-60</span><span>-80</span><span>-100</span><span>-120</span><span>-140</span></div>
                        <div class="grid-lines">{'<div></div>'*60}</div>
                        <div class="viz-dots">{tsne_dots}</div>
                    </div>
                </div>
            </div>
        </main>
    </div>
</body>
</html>
    '''

    with open(f'd:/NusantaraLaw-Chatbot/frontend/showcase-xp{xp["id"]}.html', 'w', encoding='utf-8') as f:
        f.write(html_out)

print("4 Showcase pages generated successfully.")
