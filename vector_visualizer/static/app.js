// Global Variables
let allVectors = [];          // Original list of vectors loaded from backend
let milvusStats = {};         // Collection metadata
let currentPlotData = [];     // Traces currently rendered
let currentLayout = {};       // Plotly layout config
let queryVectorTrace = null;  // Custom projected query trace
let queryLinesTraces = [];    // Nearest neighbor connection lines traces
let selectedPointIndex = -1;  // Pointer for info panel

const PALETTE = [
    "#D4AF37", // Gold
    "#10B981", // Emerald Green
    "#3B82F6", // Royal Blue
    "#EC4899", // Vivid Pink
    "#8B5CF6", // Deep Purple
    "#F59E0B", // Amber Orange
    "#06B6D4", // Bright Cyan
    "#EF4444", // Crimson Red
    "#14B8A6"  // Teal
];

// Initialize UI Elements
document.addEventListener("DOMContentLoaded", () => {
    // Icons
    lucide.createIcons();

    // Elements
    const limitSlider = document.getElementById("limit-slider");
    const limitInput = document.getElementById("limit-input");
    const btnRefresh = document.getElementById("btn-refresh");
    const btnProject = document.getElementById("btn-project");
    const btnClearProject = document.getElementById("btn-clear-project");
    const colorBySelect = document.getElementById("color-by");
    const filterDocSelect = document.getElementById("filter-doc");
    const searchHighlight = document.getElementById("search-highlight");
    const searchResultsCount = document.getElementById("search-results-count");
    const queryText = document.getElementById("query-text");

    // Synchronize inputs
    limitSlider.addEventListener("input", () => {
        limitInput.value = limitSlider.value;
    });
    limitInput.addEventListener("input", () => {
        let val = parseInt(limitInput.value);
        if (isNaN(val)) val = 100;
        if (val < 100) val = 100;
        if (val > 2000) val = 2000;
        limitSlider.value = val;
    });

    // Refresh Space Action
    btnRefresh.addEventListener("click", () => {
        loadVectorSpace();
    });

    // Color Theme / Grouping Change
    colorBySelect.addEventListener("change", () => {
        rebuildPlot();
    });

    // Document Filter Change
    filterDocSelect.addEventListener("change", () => {
        rebuildPlot();
    });

    // Search / Highlight Text Change (Realtime)
    searchHighlight.addEventListener("input", () => {
        applySearchHighlight();
    });

    // Project Custom Query
    btnProject.addEventListener("click", () => {
        projectCustomQuery();
    });

    // Clear Custom Query
    btnClearProject.addEventListener("click", () => {
        clearCustomQuery();
    });

    // Load initial data
    loadVectorSpace();
});

// Loading Helper
function showLoading(show, message = "") {
    const overlay = document.getElementById("loading-overlay");
    const loadingText = document.getElementById("loading-text");
    if (show) {
        loadingText.textContent = message;
        overlay.style.display = "flex";
        overlay.style.opacity = "1";
    } else {
        overlay.style.opacity = "0";
        setTimeout(() => {
            overlay.style.display = "none";
        }, 300);
    }
}

// Fetch vector space and fit PCA on server
async function loadVectorSpace() {
    const limit = document.getElementById("limit-input").value;
    showLoading(true, `Menghubungkan ke Milvus & menghitung PCA (Batas: ${limit})...`);

    try {
        const response = await fetch(`/api/vectors?limit=${limit}`);
        const data = await response.json();

        if (data.status === "error") {
            alert(`Error: ${data.message}`);
            showLoading(false);
            return;
        }

        allVectors = data.vectors || [];
        
        // Update stats
        document.getElementById("stat-total").textContent = data.total_rows.toLocaleString();
        document.getElementById("stat-loaded").textContent = data.loaded_rows.toLocaleString();
        
        if (data.explained_variance_ratio) {
            const sumVar = data.explained_variance_ratio.reduce((a, b) => a + b, 0) * 100;
            document.getElementById("stat-variance").textContent = 
                `PC1 + PC2 + PC3 = ${sumVar.toFixed(2)}% dari varians`;
        } else {
            document.getElementById("stat-variance").textContent = "N/A";
        }

        // Reset query states
        queryVectorTrace = null;
        queryLinesTraces = [];
        document.getElementById("btn-clear-project").disabled = true;

        // Populate doc filters
        populateDocFilter();

        // Render the Plot
        rebuildPlot();

    } catch (err) {
        console.error("Gagal memuat ruang vektor:", err);
        alert("Gagal terhubung ke visualizer backend. Pastikan server online.");
    } finally {
        showLoading(false);
    }
}

// Populate Document Filter Select options
function populateDocFilter() {
    const filterSelect = document.getElementById("filter-doc");
    // Preserve default first option
    filterSelect.innerHTML = '<option value="all">Semua Dokumen</option>';

    const distinctDocs = [...new Set(allVectors.map(v => v.doc_name))].sort();
    distinctDocs.forEach(doc => {
        const opt = document.createElement("option");
        opt.value = doc;
        opt.textContent = doc;
        filterSelect.appendChild(opt);
    });
}

// Group Vectors and Build Plotly Traces
function rebuildPlot() {
    const colorBy = document.getElementById("color-by").value;
    const filterDoc = document.getElementById("filter-doc").value;

    // Filter local vectors
    let filteredVectors = allVectors;
    if (filterDoc !== "all") {
        filteredVectors = allVectors.filter(v => v.doc_name === filterDoc);
    }

    if (filteredVectors.length === 0) {
        // No vectors matching the filter
        Plotly.purge("plotly-canvas");
        return;
    }

    // Grouping mapping
    let groups = {};
    if (colorBy === "doc_name") {
        filteredVectors.forEach(v => {
            if (!groups[v.doc_name]) groups[v.doc_name] = [];
            groups[v.doc_name].push(v);
        });
    } else if (colorBy === "category") {
        filteredVectors.forEach(v => {
            if (!groups[v.category]) groups[v.category] = [];
            groups[v.category].push(v);
        });
    } else {
        groups["Semua Dokumen"] = filteredVectors;
    }

    const traces = [];
    let colorIdx = 0;

    Object.keys(groups).forEach((groupName, idx) => {
        const groupPoints = groups[groupName];
        const groupColor = PALETTE[colorIdx % PALETTE.length];
        colorIdx++;

        const trace = {
            x: groupPoints.map(p => p.x),
            y: groupPoints.map(p => p.y),
            z: groupPoints.map(p => p.z),
            mode: 'markers',
            type: 'scatter3d',
            name: groupName,
            text: groupPoints.map(p => `${p.doc_name}<br>Hal: ${p.page_no}`),
            customdata: groupPoints, // Hold original metadata reference
            marker: {
                size: 5,
                color: groupColor,
                opacity: 0.85,
                line: {
                    color: 'rgba(255, 255, 255, 0.2)',
                    width: 0.5
                }
            },
            hoverinfo: 'text'
        };

        traces.push(trace);
    });

    // Save layout config
    currentLayout = {
        margin: { l: 0, r: 0, b: 0, t: 0 },
        paper_bgcolor: '#080A10',
        plot_bgcolor: '#080A10',
        scene: {
            xaxis: {
                title: 'PC1',
                backgroundcolor: '#080A10',
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                showbackground: true,
                zerolinecolor: 'rgba(255, 255, 255, 0.1)',
                tickfont: { color: '#6B7280' },
                titlefont: { color: '#9CA3AF' }
            },
            yaxis: {
                title: 'PC2',
                backgroundcolor: '#080A10',
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                showbackground: true,
                zerolinecolor: 'rgba(255, 255, 255, 0.1)',
                tickfont: { color: '#6B7280' },
                titlefont: { color: '#9CA3AF' }
            },
            zaxis: {
                title: 'PC3',
                backgroundcolor: '#080A10',
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                showbackground: true,
                zerolinecolor: 'rgba(255, 255, 255, 0.1)',
                tickfont: { color: '#6B7280' },
                titlefont: { color: '#9CA3AF' }
            },
            camera: {
                eye: { x: 1.5, y: 1.5, z: 1.25 }
            }
        },
        legend: {
            x: 0.02,
            y: 0.98,
            font: { color: '#F3F4F6', size: 10 },
            bgcolor: 'rgba(11, 15, 25, 0.7)',
            bordercolor: 'rgba(255, 255, 255, 0.08)',
            borderwidth: 1
        },
        showlegend: true
    };

    currentPlotData = traces;

    // Build Plotly Canvas
    Plotly.newPlot('plotly-canvas', currentPlotData, currentLayout, {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['sendDataToCloud', 'hoverCompareCartesian']
    });

    // Setup interactive events
    const myPlot = document.getElementById('plotly-canvas');
    myPlot.on('plotly_click', (data) => {
        if (data.points && data.points.length > 0) {
            const p = data.points[0];
            if (p.customdata) {
                showDetailCard(p.customdata, [p.x, p.y, p.z]);
            }
        }
    });

    myPlot.on('plotly_hover', (data) => {
        if (data.points && data.points.length > 0) {
            const p = data.points[0];
            if (p.customdata) {
                showDetailCard(p.customdata, [p.x, p.y, p.z]);
            }
        }
    });

    // Reapply search highlight if any text is typed
    applySearchHighlight();
}

// Show vector details on bottom card
function showDetailCard(meta, coords) {
    document.getElementById("selected-id-badge").textContent = `ID Vektor: #${meta.id}`;
    document.getElementById("info-doc").textContent = meta.doc_name;
    document.getElementById("info-cat").textContent = meta.category;
    document.getElementById("info-page").textContent = `Halaman ${meta.page_no}`;
    
    if (coords) {
        document.getElementById("info-coords").textContent = 
            `X: ${coords[0].toFixed(4)} | Y: ${coords[1].toFixed(4)} | Z: ${coords[2].toFixed(4)}`;
    }

    // Apply regex highlighting inside chunk text if search is active
    const searchVal = document.getElementById("search-highlight").value.trim().toLowerCase();
    let text = meta.chunk_text;

    if (searchVal) {
        const regex = new RegExp(`(${searchVal})`, 'gi');
        text = text.replace(regex, `<span class="text-highlight-match">$1</span>`);
    }

    document.getElementById("info-text").innerHTML = text;
}

// Real-time keyword filter/highlight
function applySearchHighlight() {
    const searchVal = document.getElementById("search-highlight").value.trim().toLowerCase();
    const badge = document.getElementById("search-results-count");

    if (!searchVal) {
        // Reset traces to default styling
        badge.style.display = "none";
        
        currentPlotData.forEach((trace, traceIdx) => {
            const defaultColor = PALETTE[traceIdx % PALETTE.length];
            Plotly.restyle('plotly-canvas', {
                'marker.size': 5,
                'marker.color': defaultColor,
                'marker.opacity': 0.85
            }, [traceIdx]);
        });
        return;
    }

    let matchCount = 0;

    currentPlotData.forEach((trace, traceIdx) => {
        const sizes = [];
        const colors = [];
        const opacities = [];
        const defaultColor = PALETTE[traceIdx % PALETTE.length];

        trace.customdata.forEach(p => {
            if (p.chunk_text.toLowerCase().includes(searchVal)) {
                sizes.push(10);
                colors.push("#F59E0B"); // Glowing Orange for matches
                opacities.push(1.0);
                matchCount++;
            } else {
                sizes.push(3);
                colors.push("rgba(100, 116, 139, 0.2)"); // Transparent grey
                opacities.push(0.15);
            }
        });

        // Restyle this specific trace
        Plotly.restyle('plotly-canvas', {
            'marker.size': [sizes],
            'marker.color': [colors],
            'marker.opacity': [opacities]
        }, [traceIdx]);
    });

    // Update search result badge
    badge.style.display = "inline-block";
    badge.textContent = `${matchCount} ditemukan`;
}

// Project custom query vector via backend
async function projectCustomQuery() {
    const query = document.getElementById("query-text").value.trim();
    if (!query) {
        alert("Masukkan teks kueri terlebih dahulu!");
        return;
    }

    showLoading(true, "Menghubungi Ollama & memproyeksikan kueri baru...");
    
    try {
        const response = await fetch("/api/project_query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ query: query, top_k: 5 })
        });

        const data = await response.json();
        if (response.status !== 200) {
            alert(`Error: ${data.detail || "Gagal memproyeksikan kueri"}`);
            showLoading(false);
            return;
        }

        const q3d = data.query_3d;
        const neighbors = data.neighbors || [];

        // Remove previous query traces if any
        clearQueryTracesFromPlot();

        // 1. Create trace for Query Point
        queryVectorTrace = {
            x: [q3d.x],
            y: [q3d.y],
            z: [q3d.z],
            mode: 'markers',
            type: 'scatter3d',
            name: 'Kueri Kustom',
            text: [`KUERI: "${query.substring(0, 40)}..."`],
            marker: {
                size: 14,
                color: '#EF4444', // Red star
                symbol: 'diamond',
                line: {
                    color: '#FFFFFF',
                    width: 2
                }
            },
            hoverinfo: 'text'
        };

        // 2. Draw lines to top-K nearest neighbors
        queryLinesTraces = [];
        const lineColors = ['#F59E0B', '#10B981', '#3B82F6', '#EC4899', '#8B5CF6'];

        neighbors.forEach((nb, i) => {
            // Find coordinates of neighbor point from loaded vectors
            let neighborPoint = null;
            for (let v of allVectors) {
                if (v.id === nb.id) {
                    neighborPoint = v;
                    break;
                }
            }

            if (neighborPoint) {
                const lineTrace = {
                    x: [q3d.x, neighborPoint.x],
                    y: [q3d.y, neighborPoint.y],
                    z: [q3d.z, neighborPoint.z],
                    mode: 'lines',
                    type: 'scatter3d',
                    name: `Tetangga #${i+1} (Sim: ${(nb.similarity * 100).toFixed(1)}%)`,
                    line: {
                        color: lineColors[i % lineColors.length],
                        width: 4
                    },
                    hoverinfo: 'name'
                };
                queryLinesTraces.push(lineTrace);
            }
        });

        // 3. Temporarily update trace of nearest neighbors to make them bigger
        currentPlotData.forEach((trace, traceIdx) => {
            const sizes = [];
            const opacities = [];
            
            trace.customdata.forEach(p => {
                const isNeighbor = neighbors.some(nb => nb.id === p.id);
                if (isNeighbor) {
                    sizes.push(11);
                    opacities.push(1.0);
                } else {
                    sizes.push(4);
                    opacities.push(0.3); // Fade out unrelated points
                }
            });

            Plotly.restyle('plotly-canvas', {
                'marker.size': [sizes],
                'marker.opacity': [opacities]
            }, [traceIdx]);
        });

        // Add query traces to plot
        Plotly.addTraces('plotly-canvas', [queryVectorTrace, ...queryLinesTraces]);

        // Enable Clear Button
        document.getElementById("btn-clear-project").disabled = false;

        // Show query vector info in selected card
        document.getElementById("selected-id-badge").textContent = `Kueri Kustom`;
        document.getElementById("info-doc").textContent = "N/A";
        document.getElementById("info-cat").textContent = "N/A";
        document.getElementById("info-page").textContent = "N/A";
        document.getElementById("info-coords").textContent = `X: ${q3d.x.toFixed(4)} | Y: ${q3d.y.toFixed(4)} | Z: ${q3d.z.toFixed(4)}`;
        
        let neighborListHtml = `<strong>KUERI PROYEKSI:</strong> "${query}"<br><br><strong>Top 5 Tetangga Terdekat (Milvus):</strong><br><br>`;
        neighbors.forEach((nb, i) => {
            let p = allVectors.find(v => v.id === nb.id);
            if (p) {
                neighborListHtml += `${i+1}. [${p.doc_name} Hal. ${p.page_no}] (Sim: ${(nb.similarity * 100).toFixed(2)}%)<br>
                <span style="color: #9CA3AF;">"${p.chunk_text.substring(0, 150)}..."</span><br><br>`;
            }
        });
        document.getElementById("info-text").innerHTML = neighborListHtml;

    } catch (err) {
        console.error("Gagal memproyeksikan kueri:", err);
        alert("Gagal memproyeksikan kueri. Periksa koneksi backend/Ollama.");
    } finally {
        showLoading(false);
    }
}

// Clear custom query traces from plot
function clearCustomQuery() {
    clearQueryTracesFromPlot();
    rebuildPlot();
    
    document.getElementById("query-text").value = "";
    document.getElementById("btn-clear-project").disabled = true;
    document.getElementById("selected-id-badge").textContent = "Pilih salah satu titik di grafik";
    document.getElementById("info-doc").textContent = "-";
    document.getElementById("info-cat").textContent = "-";
    document.getElementById("info-page").textContent = "-";
    document.getElementById("info-coords").textContent = "-";
    document.getElementById("info-text").textContent = 
        "Silakan arahkan kursor Anda atau klik salah satu titik vektor di grafik 3D untuk melihat potongan teks hukum yang bersangkutan.";
}

// Helper to remove traces
function clearQueryTracesFromPlot() {
    const gd = document.getElementById('plotly-canvas');
    if (!gd || !gd.data) return;

    const indicesToRemove = [];
    gd.data.forEach((trace, idx) => {
        if (trace.name === 'Kueri Kustom' || trace.name.startsWith('Tetangga #')) {
            indicesToRemove.push(idx);
        }
    });

    if (indicesToRemove.length > 0) {
        Plotly.deleteTraces('plotly-canvas', indicesToRemove);
    }
    
    queryVectorTrace = null;
    queryLinesTraces = [];
}
