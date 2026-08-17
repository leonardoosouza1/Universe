#!/usr/bin/env bash
# Report drift between the vendored IARA crates and their upstream copy.
#
# Why this exists: `vendor/` holds a copy of five crates from `../iara-runtime`,
# taken on 2026-08-17 so that a fresh clone of this repository builds without any
# untracked sibling directory. That upstream is also consumed by another project
# (byte-language-lab), so the two copies CAN drift. This script makes the drift
# visible instead of letting it rot silently.
#
# It never fails the build: upstream is not required to exist. On a contributor's
# machine it simply reports that there is nothing to compare against.
#
# Usage:  bash scripts/check_vendor.sh  [path-to-upstream]
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${1:-$REPO/../iara-runtime}"
CRATES=(iara-byte iara-cfc iara-gate iara-hdc iara-perception)

if [[ ! -d "$UPSTREAM" ]]; then
    echo "upstream not found at $UPSTREAM — nothing to compare."
    echo "This is the normal case for an outside contributor: vendor/ is the source"
    echo "of truth here, and the crates are self-contained."
    exit 0
fi

echo "comparing vendor/ against $UPSTREAM"
drift=0
for crate in "${CRATES[@]}"; do
    if [[ ! -d "$UPSTREAM/$crate" ]]; then
        echo "  $crate: not in upstream (vendored only)"
        continue
    fi
    # Compare sources only. Cargo.toml legitimately differs: the vendored manifests
    # inline `version`/`edition`, which upstream inherits from its workspace root.
    if diff -rq "$REPO/vendor/$crate/src" "$UPSTREAM/$crate/src" >/dev/null 2>&1; then
        echo "  $crate: identical"
    else
        echo "  $crate: DIFFERS"
        diff -rq "$REPO/vendor/$crate/src" "$UPSTREAM/$crate/src" 2>&1 | sed 's/^/      /'
        drift=1
    fi
done

if [[ $drift -eq 1 ]]; then
    cat <<'EOF'

Drift found. Decide deliberately which side is right — do not merge blindly:
  * upstream is newer  -> copy it in and re-run the test suite;
  * vendor is newer    -> push the change upstream so the other consumer gets it;
  * both changed       -> reconcile by hand, then record why in the commit message.

The permanent fix is to publish the upstream crates and depend on them by git URL
or version, which deletes vendor/ entirely. See docs/MIGRATION.md.
EOF
fi
exit 0
