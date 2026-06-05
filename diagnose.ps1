###############################################################################
# NusantaraLaw System Diagnostic - tests every component in the RAG pipeline
###############################################################################
param(
    [switch]$Full   # Pass -Full to also run a real chat test at the end
)

$ErrorActionPreference = "Continue"
$pass = 0; $fail = 0

function Write-Step($name) {
    Write-Host ""
    Write-Host "------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "  [CHECK] $name" -ForegroundColor Cyan
    Write-Host "------------------------------------------------" -ForegroundColor DarkGray
}

function Write-OK($msg)   { $script:pass++; Write-Host "  [PASS] $msg" -ForegroundColor Green }
function Write-FAIL($msg) { $script:fail++; Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function Write-WARN($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-INFO($msg) { Write-Host "         $msg" -ForegroundColor Gray }

Write-Host ""
Write-Host " ====================================================" -ForegroundColor Yellow
Write-Host "   NusantaraLaw - Full System Diagnostic" -ForegroundColor Yellow
Write-Host " ====================================================" -ForegroundColor Yellow

# -- 1. Docker Containers ---------------------------------------------------
Write-Step "Docker Container Status"
$containers = @(
    "nusantaralaw-chatbot-backend-1",
    "nusantaralaw-chatbot-frontend-1",
    "nusantaralaw-chatbot-milvus-standalone-1",
    "nusantaralaw-chatbot-vector-visualizer-1",
    "nusantaralaw-chatbot-searxng-1"
)
foreach ($c in $containers) {
    $state = docker inspect --format '{{.State.Status}}' $c 2>$null
    if ($state -eq "running") {
        Write-OK "$c -> running"
    } else {
        Write-FAIL "$c -> $state (NOT running!)"
    }
}

# -- 2. Ollama Host Reachability --------------------------------------------
Write-Step "Ollama Host (http://localhost:11434)"
try {
    $ollamaResp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    $modelNames = @()
    if ($ollamaResp.models) {
        $modelNames = $ollamaResp.models | ForEach-Object { $_.name }
    }
    Write-OK "Ollama is reachable - $($modelNames.Count) model(s) available"
    foreach ($m in $modelNames) { Write-INFO "  - $m" }

    # Check required models
    $needed = @("qwen3-embedding:8b")
    foreach ($n in $needed) {
        $found = $false
        foreach ($mn in $modelNames) {
            if ($mn -eq $n) { $found = $true; break }
        }
        if ($found) { Write-OK "Required model '$n' is present" }
        else { Write-FAIL "Required model '$n' NOT FOUND - embedding will fail!" }
    }
    # Check at least one chat model
    $chatModels = @()
    foreach ($mn in $modelNames) {
        if ($mn -match "qwen3" -and $mn -notmatch "embedding") {
            $chatModels += $mn
        }
    }
    if ($chatModels.Count -gt 0) {
        Write-OK "Chat model(s) found: $($chatModels -join ', ')"
    } else {
        Write-FAIL "No chat model (qwen3.5:9b or qwen3.5-9b-nlaw) found!"
    }
} catch {
    Write-FAIL "Ollama is NOT reachable at localhost:11434"
    Write-WARN "Make sure Ollama is running on the host."
    Write-INFO "Error: $($_.Exception.Message)"
}

# -- 3. Ollama Embedding Test ------------------------------------------------
Write-Step "Ollama Embedding API (/api/embed)"
try {
    $embedBody = '{"model":"qwen3-embedding:8b","input":"tes hukum"}'
    $embedResp = Invoke-RestMethod -Uri "http://localhost:11434/api/embed" -Method Post -Body $embedBody -ContentType "application/json" -TimeoutSec 60
    if ($embedResp.embeddings -and $embedResp.embeddings.Count -gt 0) {
        $dim = $embedResp.embeddings[0].Count
        Write-OK "Embedding returned - dimension: $dim"
        if ($dim -ne 4096) { Write-WARN "Expected 4096 dimensions, got $dim" }
    } else {
        Write-FAIL "Embedding response had no embeddings array"
    }
} catch {
    Write-FAIL "Embedding call failed"
    Write-INFO "Error: $($_.Exception.Message)"
}

# -- 4. Milvus Connectivity -------------------------------------------------
Write-Step "Milvus Vector DB (localhost:9091)"
try {
    $milvusHealth = Invoke-WebRequest -Uri "http://localhost:9091/healthz" -TimeoutSec 5 -UseBasicParsing
    Write-OK "Milvus health endpoint returned HTTP $($milvusHealth.StatusCode)"
} catch {
    Write-FAIL "Milvus health check failed"
    Write-INFO "Error: $($_.Exception.Message)"
}

# -- 5. Backend Health -------------------------------------------------------
Write-Step "Backend API (http://localhost:8000)"
try {
    $healthResp = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -TimeoutSec 5
    Write-OK "Backend /api/health -> $($healthResp.status)"
} catch {
    Write-FAIL "Backend health check failed"
    Write-INFO "Error: $($_.Exception.Message)"
}

# -- 6. Backend -> Test Cases ------------------------------------------------
Write-Step "Backend Test Cases List (/api/test-cases)"
try {
    $testCases = Invoke-RestMethod -Uri "http://localhost:8000/api/test-cases" -TimeoutSec 10
    Write-OK "Backend /api/test-cases responded - $($testCases.Count) test cases loaded"
} catch {
    Write-FAIL "Backend /api/test-cases failed"
    Write-INFO "Error: $($_.Exception.Message)"
}

# -- 7. Live Backend Docker Logs (last 30 lines) ----------------------------
Write-Step "Backend Docker Logs (last 30 lines)"
$logs = docker logs nusantaralaw-chatbot-backend-1 --tail 30 2>&1
foreach ($line in $logs) {
    $lineStr = "$line"
    if ($lineStr -match "error|Error|ERROR|Exception|Traceback|FAIL") {
        Write-Host "     $lineStr" -ForegroundColor Red
    } elseif ($lineStr -match "WARNING|WARN") {
        Write-Host "     $lineStr" -ForegroundColor Yellow
    } else {
        Write-Host "     $lineStr" -ForegroundColor DarkGray
    }
}

# -- 8. Frontend Reachability ------------------------------------------------
Write-Step "Frontend (http://localhost:3000)"
try {
    $feResp = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 5 -UseBasicParsing
    Write-OK "Frontend is reachable - HTTP $($feResp.StatusCode)"
} catch {
    Write-FAIL "Frontend not reachable"
    Write-INFO "Error: $($_.Exception.Message)"
}

# -- 9. Optional: Full Chat Pipeline Test -----------------------------------
if ($Full) {
    Write-Step "Full Chat Pipeline Test (/api/chat POST)"
    Write-INFO "Sending: 'Apa itu hukum pidana?' - this may take 30-120s..."
    try {
        $chatBody = '{"question":"Apa itu hukum pidana?","use_web_search":false,"model":"qwen3.5-9b-nlaw","use_hf":false}'
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $chatResp = Invoke-RestMethod -Uri "http://localhost:8000/api/chat" -Method Post -Body $chatBody -ContentType "application/json" -TimeoutSec 300
        $sw.Stop()
        $elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        if ($chatResp.answer -and $chatResp.answer.Length -gt 10) {
            Write-OK "Chat responded in ${elapsed}s - answer length: $($chatResp.answer.Length) chars"
            $preview = $chatResp.answer.Substring(0, [Math]::Min(200, $chatResp.answer.Length))
            Write-INFO "Preview: ${preview}..."
        } else {
            Write-FAIL "Chat returned empty or very short answer after ${elapsed}s"
        }
    } catch {
        Write-FAIL "Chat POST failed"
        Write-INFO "Error: $($_.Exception.Message)"
    }
}

# -- Summary -----------------------------------------------------------------
Write-Host ""
Write-Host "====================================================" -ForegroundColor Yellow
if ($fail -eq 0) {
    Write-Host "  RESULT: ALL $pass CHECKS PASSED" -ForegroundColor Green
} else {
    Write-Host "  RESULT: $pass passed, $fail FAILED" -ForegroundColor Red
}
Write-Host "====================================================" -ForegroundColor Yellow
Write-Host ""

if (-not $Full) {
    Write-Host "  Tip: Run .\diagnose.ps1 -Full to also test a live chat request." -ForegroundColor DarkGray
    Write-Host ""
}
