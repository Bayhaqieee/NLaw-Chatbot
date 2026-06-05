param(
    [switch]$TailOnly
)

if ($TailOnly) {
    Write-Host "Fetching NusantaraLaw Backend Docker Logs (last 100 lines)..." -ForegroundColor Cyan
    docker logs nusantaralaw-chatbot-backend-1 --tail 100
} else {
    Write-Host "Streaming live NusantaraLaw Backend logs (Press Ctrl+C to stop)..." -ForegroundColor Cyan
    docker logs -f nusantaralaw-chatbot-backend-1
}

Write-Host "`nFor the internal website logs API, you can visit http://localhost:8000/api/logs" -ForegroundColor Green
