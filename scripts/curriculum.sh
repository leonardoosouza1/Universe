#!/usr/bin/env bash
# Curriculum train: wide → narrow gradual. DNA herda entre stages.
# Cars trained em pista larga TRANSFEREM skill pra narrow.
#
# Uso: ./scripts/curriculum.sh [seed] [gens_per_stage] [swarm] [widths...]
#   seed: 9001
#   gens_per_stage: 12
#   swarm: 60
#   widths: "36 32 28 26 24 22"

set -euo pipefail
SEED="${1:-9001}"
GENS="${2:-12}"
SWARM="${3:-60}"
shift 3 2>/dev/null || true
WIDTHS=("${@:-36 32 28 26 24 22}")
[ ${#WIDTHS[@]} -eq 0 ] && WIDTHS=(36 32 28 26 24 22)

cd "$(dirname "$0")/.."
[ -x ./target/release/universe ] || cargo build --release --quiet

rm -f CHAMPION.dna universe_config.json
echo "=== CURRICULUM seed=$SEED gens/stage=$GENS swarm=$SWARM ==="
for W in "${WIDTHS[@]}"; do
    echo ""
    echo "--- stage width=${W}m ---"
    timeout 1500 ./target/release/universe --headless --gens "$GENS" --swarm "$SWARM" --speed 16 --seed "$SEED" --width "$W" 2>&1 | grep "^📊 GEN" | tail -3
    FIT=$(./target/release/dna_view 2>/dev/null | grep "fitness" | awk '{print $3}' | cut -d. -f1)
    echo "  → DNA fitness: ${FIT}m"
done
echo ""
echo "Final DNA salvo. UI test:"
echo "  ./target/release/universe --seed $SEED --width ${WIDTHS[-1]}"
