#!/usr/bin/env bash
# Multi-seed battery: roda N seeds × M gens, agrega champion + median.
# Usage: ./scripts/battery.sh [seeds] [gens] [swarm] [speed]
#   seeds: list separated by comma, e.g. "7777,4242,1337,9001,5555"  (default: 5 seeds)
#   gens:  4
#   swarm: 60
#   speed: 16

set -euo pipefail
SEEDS="${1:-7777,4242,1337,9001,5555}"
GENS="${2:-4}"
SWARM="${3:-60}"
SPEED="${4:-16}"

cd "$(dirname "$0")/.."
[ -x ./target/release/universe ] || cargo build --release --quiet

REPORT="melhorias/battery-$(date +%Y%m%d-%H%M%S).md"
{
    echo "# Battery — $(date +%Y-%m-%d_%H:%M:%S)"
    echo ""
    echo "Config: seeds=$SEEDS, gens=$GENS, swarm=$SWARM, speed=$SPEED"
    echo ""
    echo "| Seed | Gen | NET max | Median | Deaths |"
    echo "|---|---|---|---|---|"
} > "$REPORT"

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
for seed in "${SEED_ARR[@]}"; do
    rm -f CHAMPION.dna
    out=$(timeout 600 ./target/release/universe --headless --gens "$GENS" --swarm "$SWARM" --speed "$SPEED" --seed "$seed" 2>&1 | grep "^📊 GEN")
    while IFS= read -r line; do
        # Format: "📊 GEN N | Laps: ... | NET max=Xm median=Ym | Deaths Z/W"
        gen=$(echo "$line" | sed -E 's/.*GEN ([0-9]+).*/\1/')
        netmax=$(echo "$line" | sed -E 's/.*NET max=([0-9]+)m.*/\1/')
        median=$(echo "$line" | sed -E 's/.*median=([0-9]+)m.*/\1/')
        deaths=$(echo "$line" | sed -E 's/.*Deaths ([0-9]+\/[0-9]+).*/\1/')
        echo "| $seed | $gen | $netmax | $median | $deaths |" >> "$REPORT"
    done <<< "$out"
done

# Aggregate per seed: max champion + max median across gens.
{
    echo ""
    echo "## Aggregate (per seed: max-of-gens)"
    echo ""
    echo "| Seed | Champion (best gen) | Best median |"
    echo "|---|---|---|"
} >> "$REPORT"

for seed in "${SEED_ARR[@]}"; do
    best_champ=$(grep "^| $seed |" "$REPORT" | awk -F'|' '{print $4}' | tr -d ' ' | sort -n | tail -1)
    best_med=$(grep "^| $seed |" "$REPORT" | awk -F'|' '{print $5}' | tr -d ' ' | sort -n | tail -1)
    echo "| $seed | ${best_champ}m | ${best_med}m |" >> "$REPORT"
done

echo "Report: $REPORT"
cat "$REPORT"
