#!/usr/bin/env bash
# Progress meter for docs/STYLE.md §2 (600 lines per .rs, 400 per .wgsl).
# Informational, never fails: prints the files over the cap, largest first, and
# the summary line the STYLE.md table is regenerated from.
#
#   bash scripts/file_size_meter.sh          # tracked src/ and crates/ files
#   bash scripts/file_size_meter.sh --table  # Markdown rows for docs/STYLE.md
set -euo pipefail
cd "$(dirname "$0")/.."
RS_CAP=600
WGSL_CAP=400
mode="${1:-}"
list() {
    git ls-files 'src/*.rs' 'src/**/*.rs' 'crates/*.rs' 'crates/**/*.rs' 'src/*.wgsl' 'src/**/*.wgsl' \
        | grep -v '^docs/' | while read -r f; do printf '%s %s\n' "$(wc -l < "$f")" "$f"; done
}
over() {
    list | awk -v rs="$RS_CAP" -v wgsl="$WGSL_CAP" \
        '($2 ~ /\.rs$/ && $1 > rs) || ($2 ~ /\.wgsl$/ && $1 > wgsl)' | sort -rn
}
if [ "$mode" = "--table" ]; then
    over | awk '{ printf "| `%s` | %d |\n", $2, $1 }'
    exit 0
fi
over | awk '{ printf "%6d  %s\n", $1, $2 }'
list | awk -v rs="$RS_CAP" -v wgsl="$WGSL_CAP" '
    $2 ~ /\.rs$/   { n_rs++;   t_rs += $1;   if ($1 > rs)   o_rs++ }
    $2 ~ /\.wgsl$/ { n_wgsl++; t_wgsl += $1; if ($1 > wgsl) o_wgsl++ }
    END {
        printf "\n.rs   files=%d lines=%d over %d: %d (%.0f%%)\n", n_rs, t_rs, rs, o_rs, 100*o_rs/n_rs
        printf ".wgsl files=%d lines=%d over %d: %d (%.0f%%)\n", n_wgsl, t_wgsl, wgsl, o_wgsl, 100*o_wgsl/n_wgsl
    }'
