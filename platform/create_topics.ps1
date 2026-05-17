#!/usr/bin/env pwsh
# create_topics.ps1 — Crea todos los Kafka topics de Los Ojos
# Uso: .\create_topics.ps1

$topics = @(
    "los_ojos.market.data",
    "los_ojos.market.orderbook",
    "los_ojos.market.funding",
    "los_ojos.market.normalized",
    "los_ojos.macro.indicators",
    "los_ojos.macro.regime",
    "los_ojos.macro.recession_alert",
    "los_ojos.macro.signal",
    "los_ojos.macro.series",
    "los_ojos.onchain.whale_alert",
    "los_ojos.onchain.smart_money",
    "los_ojos.context.regime",
    "los_ojos.context.anomaly",
    "los_ojos.context.state",
    "los_ojos.ml.feature_vector",
    "los_ojos.features.vector",
    "los_ojos.signals.trading",
    "los_ojos.bot.kill_switch",
    "los_ojos.sec.signal",
    "los_ojos.derivatives.events",
    "los_ojos.dlq"
)

Write-Host "Creating Kafka topics..." -ForegroundColor Cyan
foreach ($topic in $topics) {
    docker-compose exec kafka kafka-topics `
        --create --if-not-exists `
        --bootstrap-server localhost:9092 `
        --partitions 4 `
        --replication-factor 1 `
        --topic $topic
    Write-Host "  OK: $topic" -ForegroundColor Green
}
Write-Host "`nAll topics created." -ForegroundColor Cyan
