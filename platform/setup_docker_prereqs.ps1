#!/usr/bin/env pwsh
# setup_docker_prereqs.ps1
# REQUIERE: ejecutar como Administrador
# Habilita WSL 2 + VirtualMachinePlatform para Docker Desktop

Write-Host "=== Habilitando prerequisitos para Docker Desktop ===" -ForegroundColor Cyan

# 1. Subsistema de Windows para Linux
Write-Host "`n[1/3] Habilitando WSL..." -ForegroundColor Yellow
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# 2. Virtual Machine Platform (necesario para WSL 2)
Write-Host "`n[2/3] Habilitando VirtualMachinePlatform..." -ForegroundColor Yellow
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 3. Descargar el kernel de WSL 2
Write-Host "`n[3/3] Descargando actualizacion del kernel de WSL 2..." -ForegroundColor Yellow
$wslKernel = "$env:TEMP\wsl_update_x64.msi"
Invoke-WebRequest -Uri "https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi" `
    -OutFile $wslKernel -UseBasicParsing
Write-Host "Instalando kernel WSL 2..."
Start-Process msiexec.exe -ArgumentList "/i `"$wslKernel`" /quiet" -Wait

# 4. Establecer WSL 2 como versión predeterminada
Write-Host "`nEstableciendo WSL 2 como version predeterminada..." -ForegroundColor Yellow
wsl --set-default-version 2

Write-Host "`n=== LISTO ===" -ForegroundColor Green
Write-Host "Ahora:" -ForegroundColor White
Write-Host "  1. REINICIA el equipo" -ForegroundColor White
Write-Host "  2. Instala Docker Desktop desde:" -ForegroundColor White
Write-Host "     https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe" -ForegroundColor Cyan
Write-Host "  3. Vuelve aqui y ejecuta: docker compose up -d" -ForegroundColor White
