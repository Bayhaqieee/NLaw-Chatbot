const API_BASE_URL = "http://localhost:8000/api";

const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const chatHistory = document.getElementById("chatHistory");
const webSearchToggle = document.getElementById("webSearchToggle");

const uploadForm = document.getElementById("uploadForm");
const pdfFile = document.getElementById("pdfFile");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    // Add user message to UI
    appendMessage(question, "user-message", "👤");
    questionInput.value = "";
    
    // Disable inputs
    sendBtn.disabled = true;
    questionInput.disabled = true;

    // Add loading message
    const loadingId = appendMessage("Menganalisis hukum dan mencari referensi...", "system-message", "🏛️");

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question,
                use_web_search: webSearchToggle.checked
            })
        });

        if (!response.ok) throw new Error("Terjadi kesalahan pada server.");

        const data = await response.json();
        
        // Remove loading
        document.getElementById(loadingId).remove();
        
        // Append response with sources
        appendResponseWithSources(data.answer, data.sources, data.web_results, data.toon_tokens_saved);

    } catch (error) {
        document.getElementById(loadingId).remove();
        appendMessage("Maaf, sistem sedang mengalami gangguan. " + error.message, "system-message", "❌");
    } finally {
        sendBtn.disabled = false;
        questionInput.disabled = false;
        questionInput.focus();
        scrollToBottom();
    }
});

uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = pdfFile.files[0];
    if (!file) return;

    uploadBtn.disabled = true;
    uploadStatus.textContent = "Mengunggah dan memproses...";
    uploadStatus.style.color = "blue";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        if (response.ok) {
            uploadStatus.textContent = `Sukses! ${data.chunks_embedded} chunks ditambahkan ke Milvus.`;
            uploadStatus.style.color = "green";
        } else {
            uploadStatus.textContent = data.status || "Gagal mengunggah dokumen.";
            uploadStatus.style.color = "red";
        }
    } catch (error) {
        uploadStatus.textContent = "Error koneksi.";
        uploadStatus.style.color = "red";
    } finally {
        uploadBtn.disabled = false;
        uploadForm.reset();
    }
});

function appendMessage(text, className, avatarEmoji) {
    const id = "msg-" + Date.now();
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${className}`;
    msgDiv.id = id;
    
    msgDiv.innerHTML = `
        <div class="avatar">${avatarEmoji}</div>
        <div class="content">${text.replace(/\\n/g, "<br>")}</div>
    `;
    
    chatHistory.appendChild(msgDiv);
    scrollToBottom();
    return id;
}

function appendResponseWithSources(answer, sources, webResults, tokensSaved) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message system-message";
    
    let sourcesHTML = "";
    if (sources.length > 0 || webResults.length > 0) {
        sourcesHTML += `<div class="source-box"><strong>Sumber Rujukan:</strong><ul>`;
        
        sources.forEach(s => {
            sourcesHTML += `<li>${s.sumber} (Halaman ${s.page_no})</li>`;
        });

        webResults.forEach(w => {
            sourcesHTML += `<li><a href="${w.url}" target="_blank">Web: ${w.url}</a></li>`;
        });

        sourcesHTML += `</ul><div style="margin-top: 5px; color: #198754;"><em>Menghemat ~${tokensSaved} token via TOON format.</em></div></div>`;
    }

    msgDiv.innerHTML = `
        <div class="avatar">🏛️</div>
        <div class="content">
            ${answer.replace(/\n/g, "<br>")}
            ${sourcesHTML}
        </div>
    `;
    
    chatHistory.appendChild(msgDiv);
    scrollToBottom();
}

function scrollToBottom() {
    chatHistory.scrollTop = chatHistory.scrollHeight;
}
