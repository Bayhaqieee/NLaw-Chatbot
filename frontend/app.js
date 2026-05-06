const API_BASE_URL = "http://localhost:8000/api";

const chatForm        = document.getElementById("chatForm");
const questionInput   = document.getElementById("questionInput");
const sendBtn         = document.getElementById("sendBtn");
const chatHistory     = document.getElementById("chatHistory");
const webSearchToggle = document.getElementById("webSearchToggle");
const uploadForm      = document.getElementById("uploadForm");
const pdfFile         = document.getElementById("pdfFile");
const uploadBtn       = document.getElementById("uploadBtn");
const uploadStatus    = document.getElementById("uploadStatus");
const fileNameEl      = document.getElementById("fileName");

function getSelectedModel() {
    const sel = document.querySelector('input[name="modelChoice"]:checked');
    return sel ? sel.value : "qwen3.5-9b-nlaw";
}

// File picker display
pdfFile.addEventListener("change", () => {
    if (pdfFile.files[0]) {
        fileNameEl.textContent = pdfFile.files[0].name;
    } else {
        fileNameEl.textContent = "Belum ada file dipilih";
    }
});

// ── Chat Submit ──────────────────────────────────
chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    const selectedModel = getSelectedModel();
    const modelLabel = selectedModel === "qwen3.5-9b-nlaw" ? "Fine-Tuned" : "Vanilla";

    appendMessage(question, "user-message", "U");
    questionInput.value = "";
    sendBtn.disabled = true;
    questionInput.disabled = true;

    const loadingId = appendTyping();

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                use_web_search: webSearchToggle.checked,
                model: selectedModel
            })
        });

        if (!response.ok) throw new Error("Terjadi kesalahan pada server.");

        const data = await response.json();
        document.getElementById(loadingId)?.remove();
        appendResponseWithSources(data.answer, data.sources, data.web_results, data.toon_tokens_saved, modelLabel);

    } catch (error) {
        document.getElementById(loadingId)?.remove();
        appendMessage("Maaf, sistem sedang mengalami gangguan. " + error.message, "system-message", "NL");
    } finally {
        sendBtn.disabled = false;
        questionInput.disabled = false;
        questionInput.focus();
        scrollToBottom();
    }
});

// ── Upload Submit ────────────────────────────────
uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = pdfFile.files[0];
    if (!file) return;

    uploadBtn.disabled = true;
    uploadStatus.textContent = "Mengunggah dan memproses...";
    uploadStatus.style.color = "var(--gold)";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            uploadStatus.textContent = `Sukses! ${data.chunks_embedded} chunks telah diindeks.`;
            uploadStatus.style.color = "var(--green)";
        } else {
            uploadStatus.textContent = data.status || "Gagal mengunggah dokumen.";
            uploadStatus.style.color = "var(--red)";
        }
    } catch (error) {
        uploadStatus.textContent = "Koneksi gagal. Pastikan backend aktif.";
        uploadStatus.style.color = "var(--red)";
    } finally {
        uploadBtn.disabled = false;
        uploadForm.reset();
        fileNameEl.textContent = "Belum ada file dipilih";
    }
});

// ── Helpers ──────────────────────────────────────
function appendMessage(text, className, avatarLabel) {
    const id = "msg-" + Date.now();
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${className}`;
    msgDiv.id = id;
    msgDiv.innerHTML = `
        <div class="avatar">${avatarLabel}</div>
        <div class="content">${escapeHtml(text).replace(/\n/g, "<br>")}</div>
    `;
    chatHistory.appendChild(msgDiv);
    scrollToBottom();
    return id;
}

function appendTyping() {
    const id = "msg-" + Date.now();
    const msgDiv = document.createElement("div");
    msgDiv.className = "message system-message";
    msgDiv.id = id;
    msgDiv.innerHTML = `
        <div class="avatar">NL</div>
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
    `;
    chatHistory.appendChild(msgDiv);
    scrollToBottom();
    return id;
}

function appendResponseWithSources(answer, sources, webResults, tokensSaved, modelLabel) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message system-message";

    const isFt = modelLabel === "Fine-Tuned";
    const modelBadge = modelLabel
        ? `<span style="display:inline-block;margin-bottom:8px;font-size:10px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;color:${isFt ? 'var(--gold)' : 'var(--t3)' };background:${isFt ? 'var(--gold-lo)' : 'var(--s3)'};border:1px solid ${isFt ? 'rgba(201,162,39,0.3)' : 'var(--b1)'};border-radius:4px;padding:2px 8px;">${modelLabel}</span>`
        : '';

    let sourcesHTML = "";
    if ((sources && sources.length > 0) || (webResults && webResults.length > 0)) {
        sourcesHTML += `<div class="source-box"><strong>Sumber Rujukan</strong><ul>`;
        (sources || []).forEach(s => {
            sourcesHTML += `<li>${s.sumber} (Halaman ${s.page_no})</li>`;
        });
        (webResults || []).forEach(w => {
            sourcesHTML += `<li><a href="${w.url}" target="_blank">Web: ${w.url}</a></li>`;
        });
        sourcesHTML += `</ul>`;
        if (tokensSaved) sourcesHTML += `<p class="toon-note">~${tokensSaved} token dihemat via TOON format.</p>`;
        sourcesHTML += `</div>`;
    }

    msgDiv.innerHTML = `
        <div class="avatar">NL</div>
        <div class="content">
            ${modelBadge}
            ${escapeHtml(answer).replace(/\n/g, "<br>")}
            ${sourcesHTML}
        </div>
    `;
    chatHistory.appendChild(msgDiv);
    scrollToBottom();
}

function escapeHtml(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function scrollToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight;
}
