#!/usr/bin/env bash
# Parallel pop training: spawn N seeds simultaneously, pick best DNA.
# Each process trains its own seed fresh, writes CHAMPION.dna_<seed>.
# Master selects highest fitness, promotes to CHAMPION.dna.
#
# Usage: ./scripts/parallel_train.sh [seeds] [gens] [swarm] [speed]
#   seeds: comma-list (default "9001,5555,7777,4242,1337")
#   gens:  10
#   swarm: 60
#   speed: 16

set -euo pipefail
SEEDS="${1:-9001,5555,7777,4242,1337}"
GENS="${2:-10}"
SWARM="${3:-60}"
SPEED="${4:-16}"

cd "$(dirname "$0")/.."
[ -x ./target/release/universe ] || cargo build --release --quiet

mkdir -p /tmp/universe_pop
rm -f /tmp/universe_pop/CHAMPION.dna.*
rm -f /tmp/universe_pop/log.*

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
echo "Parallel training ${#SEED_ARR[@]} seeds × $GENS gens × swarm $SWARM..."
PIDS=()

for seed in "${SEED_ARR[@]}"; do
    (
        WORKDIR="/tmp/universe_pop/seed_$seed"
        mkdir -p "$WORKDIR"
        cd "$WORKDIR"
        rm -f CHAMPION.dna universe_config.json race_reports.md
        timeout 1500 /home/leonardo/projects/LLM/Universe/target/release/universe \
            --headless --gens "$GENS" --swarm "$SWARM" --speed "$SPEED" --seed "$seed" \
            > "/tmp/universe_pop/log.$seed" 2>&1
        if [ -f CHAMPION.dna ]; then
            cp CHAMPION.dna "/tmp/universe_pop/CHAMPION.dna.$seed"
        fi
    ) &
    PIDS+=($!)
done

echo "PIDs: ${PIDS[*]}"
echo "Waiting all to finish..."
wait

# Pick best by fitness from dna_view header.
BEST_FIT=0
BEST_SEED=0
for seed in "${SEED_ARR[@]}"; do
    DNA="/tmp/universe_pop/CHAMPION.dna.$seed"
    if [ ! -f "$DNA" ]; then
        echo "  seed=$seed: NO DNA"
        continue
    fi
    cp "$DNA" /home/leonardo/projects/LLM/Universe/CHAMPION.dna
    FIT=$(cd /home/leonardo/projects/LLM/Universe && ./target/release/dna_view 2>/dev/null | grep "fitness" | awk '{print $3}' | cut -d. -f1)
    FIT=${FIT:-0}
    GEN_BEST=$(grep "^📊 GEN" "/tmp/universe_pop/log.$seed" | awk -F'NET max=' '{print $2}' | awk -F'm' '{print $1}' | sort -n | tail -1)
    echo "  seed=$seed fitness=$FIT gen_peak=${GEN_BEST:-?}"
    if [ "${FIT:-0}" -gt "$BEST_FIT" ] 2>/dev/null; then
        BEST_FIT=$FIT
        BEST_SEED=$seed
    fi
done

if [ "$BEST_SEED" != "0" ]; then
    cp "/tmp/universe_pop/CHAMPION.dna.$BEST_SEED" /home/leonardo/projects/LLM/Universe/CHAMPION.dna
    echo ""
    echo "BEST: seed=$BEST_SEED fitness=$BEST_FIT m"
    echo "CHAMPION.dna populated from seed $BEST_SEED"
else
    echo "No DNA crowned in any seed"
fi
