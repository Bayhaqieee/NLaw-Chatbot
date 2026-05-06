Write-Host "Fetching NusantaraLaw Backend Docker Logs..." -ForegroundColor Cyan
docker logs nusantaralaw-chatbot-backend-1 --tail 100
Write-Host "`nFor the internal website logs API, you can visit http://localhost:8000/api/logs" -ForegroundColor Green
