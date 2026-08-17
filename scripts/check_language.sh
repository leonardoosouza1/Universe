#!/usr/bin/env bash
#
# check_language.sh — guard the "English everywhere" rule (see docs/STYLE.md).
#
# Universe was written in Portuguese and is being migrated to English one file at a
# time (see docs/MIGRATION.md). This script is what keeps the migration monotonic:
# already-migrated files may never regress, and files still waiting in the queue may
# not grow new Portuguese.
#
# MODES
#   (default)             Check only what this branch changed against the base ref.
#                         Exits non-zero on a violation. This is the CI gate.
#   --all                 Scan the whole repository and print a report sorted by
#                         number of hits. Always exits 0 — it is a progress meter,
#                         not a gate.
#   --generate-allowlist  Run --all and (re)write scripts/language_allowlist.txt.
#   --help                Show usage.
#
# ENVIRONMENT
#   BASE_REF   Ref to diff against in default mode. Falls back to origin/main,
#              then main, then the first commit reachable from HEAD.
#
# EXIT CODES
#   0  clean (or --all / --generate-allowlist, which never fail)
#   1  Portuguese found where English is required
#   2  bad usage / broken environment
#
# Interactive shells in this project may shadow `grep` with a wrapper function.
# Scripts must not inherit that, so drop any inherited function definitions first.
unset -f grep sed awk sort head 2>/dev/null || true

set -uo pipefail

# ---------------------------------------------------------------------------
# Repository root — every path printed by this script is repo-relative.
# ---------------------------------------------------------------------------
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "check_language.sh: not inside a git repository" >&2
    exit 2
}
cd "$REPO_ROOT" || exit 2

ALLOWLIST_FILE="scripts/language_allowlist.txt"

# ---------------------------------------------------------------------------
# What must be written in English.
#
# Source and build manifests are always in scope. Markdown is in scope only for
# the documents that are part of the public product: docs/ also holds experiment
# records written in Portuguese for the author's own use, and those are not a
# migration target.
# ---------------------------------------------------------------------------
CODE_EXTENSIONS='rs wgsl toml'

ENGLISH_DOCS=(
    'README.md'
    'CONTRIBUTING.md'
    'CODE_OF_CONDUCT.md'
    'SECURITY.md'
    'CHANGELOG.md'
    'NAMING.md'
    'docs/STYLE.md'
    'docs/MIGRATION.md'
)

# ---------------------------------------------------------------------------
# Detection patterns.
#
# Two independent signals, both deliberately conservative — a false positive
# costs a contributor an argument with CI, which is worse than a missed line
# that the next reader will catch anyway.
#
#   1. Portuguese diacritics. No English word carries them; every hit is real.
#   2. A short list of unambiguous Portuguese function words, in both the
#      accented spelling used in prose and the ASCII spelling used in
#      identifiers (`nao_sobe_com_apenas_2_boas`).
#
# Words that exist in BOTH languages are deliberately absent: "no", "a", "o",
# "e", "de", "um", "ate", "so", "e" would fire on ordinary English and on hex
# literals. The list only grows with words that cannot be English.
#
# Boundaries are [A-Za-z0-9] rather than \b, so `_` counts as a separator and
# snake_case identifiers are inspected word by word.
# ---------------------------------------------------------------------------
PATTERN_ACCENT='[áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ]'

PT_WORDS='nao|não|para|que|pelo|quando|entao|então|porque|tambem|também'
PT_WORDS="$PT_WORDS|apenas|cada|sobre|isso|este|esta|entre|sem"

PATTERN_WORDS="(?<![A-Za-z0-9])(?:${PT_WORDS})(?![A-Za-z0-9])"

# "com" needs its own rule: it is unambiguous as a word, but it is also the end
# of every .com host name (github.com/leonardoosouza1/Universe). The extra dot
# guard keeps URLs out of the report.
PATTERN_COM='(?<![A-Za-z0-9.])com(?![A-Za-z0-9])'

PATTERN="${PATTERN_ACCENT}|${PATTERN_WORDS}|${PATTERN_COM}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

# is_in_scope <path> -> 0 if the file must be written in English
is_in_scope() {
    local path="$1" ext doc
    [ -f "$path" ] || return 1

    ext="${path##*.}"
    for e in $CODE_EXTENSIONS; do
        [ "$ext" = "$e" ] && return 0
    done

    for doc in "${ENGLISH_DOCS[@]}"; do
        [ "$path" = "$doc" ] && return 0
    done

    # Issue/PR templates and workflow docs ship with the project.
    case "$path" in
        .github/*.md | .github/*/*.md) return 0 ;;
    esac

    return 1
}

# load_allowlist -> fills ALLOW[] with the patterns from the allowlist file
declare -a ALLOW=()
load_allowlist() {
    local line
    [ -f "$ALLOWLIST_FILE" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -n "$line" ] && ALLOW+=("$line")
    done < "$ALLOWLIST_FILE"
}

# is_allowlisted <path> -> 0 when the file is a known, not-yet-migrated file
is_allowlisted() {
    local path="$1" entry
    for entry in "${ALLOW[@]}"; do
        # shellcheck disable=SC2254 -- globs in the allowlist are intentional
        case "$path" in
            $entry) return 0 ;;
        esac
    done
    return 1
}

# scan_whole_file <path> -> emits "line:text" for every offending line
scan_whole_file() {
    grep -nPi -- "$PATTERN" "$1" 2>/dev/null
}

# scan_added_lines <path> -> emits "line:text" for offending lines this branch ADDED
#
# The awk stage rebuilds the new-file line number from the hunk header, so the
# report still points at a line a contributor can open. The line number is
# printed before the text, which is why grepping the result cannot produce a
# false positive from the path: digits never match the pattern.
scan_added_lines() {
    local path="$1"
    git diff -U0 "$BASE_SHA" -- "$path" 2>/dev/null | awk '
        /^\+\+\+/ { next }
        /^@@/ {
            # @@ -old,count +new,count @@
            n = $3
            sub(/^\+/, "", n)
            split(n, parts, ",")
            lineno = parts[1] + 0
            next
        }
        /^\+/ { print lineno ":" substr($0, 2); lineno++ }
    ' | grep -Pi -- "$PATTERN"
}

# check_one <path> <full|added> <label>
#   Prints "path:line:text" for every offending line and returns 1 if there was
#   at least one. The scan output is captured into a variable rather than piped:
#   a `... | while read` loop runs in a subshell, so any counter it increments is
#   lost and the check reports OK while printing violations. That exact bug was
#   in the first draft of this script.
MAX_LINES_PER_FILE=20
check_one() {
    local path="$1" mode="$2" label="$3" out line n=0

    if [ "$mode" = added ]; then
        out=$(scan_added_lines "$path")
    else
        out=$(scan_whole_file "$path")
    fi
    [ -z "$out" ] && return 0

    while IFS= read -r line; do
        [ -z "$line" ] && continue
        n=$((n + 1))
        if [ "$n" -le "$MAX_LINES_PER_FILE" ]; then
            # Trim very long lines so the CI log stays readable.
            printf '  %s:%.200s\n' "$path" "$line"
        fi
    done <<< "$out"

    [ "$n" -eq 0 ] && return 0
    if [ "$n" -gt "$MAX_LINES_PER_FILE" ]; then
        printf '  %s: ... and %d more line(s)\n' "$path" "$((n - MAX_LINES_PER_FILE))"
    fi
    printf '    ^ %s\n' "$label"
    return 1
}

fix_instructions() {
    cat <<'EOF'

How to fix
----------
  Universe is English-only: identifiers, comments, doc comments, log and panic
  messages, test names, commit messages. See docs/STYLE.md for the full rule.

  Translate the offending lines. Keep every measurement, file:line reference and
  command intact -- those comments record bugs that cost real GPU hours. Translate
  the prose, never summarise away the number.

  Then re-check and rebuild:
      ./scripts/check_language.sh
      cargo build --release

  If the file is a large one you are only passing through, migrate it properly in
  its own commit and tick it off in docs/MIGRATION.md instead of half-translating.
EOF
}

# ---------------------------------------------------------------------------
# Mode: --all  (full report, never fails)
# ---------------------------------------------------------------------------
run_all() {
    local quiet="${1:-}"
    local path count total_files=0 total_hits=0
    local tmp
    tmp=$(mktemp) || exit 2
    # shellcheck disable=SC2064
    trap "rm -f '$tmp'" EXIT

    while IFS= read -r path; do
        is_in_scope "$path" || continue
        [ "$path" = "$ALLOWLIST_FILE" ] && continue
        count=$(grep -cPi -- "$PATTERN" "$path" 2>/dev/null || true)
        [ -z "$count" ] && count=0
        if [ "$count" -gt 0 ]; then
            printf '%s\t%s\n' "$count" "$path" >> "$tmp"
            total_files=$((total_files + 1))
            total_hits=$((total_hits + count))
        fi
    done < <(git ls-files)

    if [ "$quiet" != "quiet" ]; then
        echo "Portuguese still present, by file (English-only scope: .${CODE_EXTENSIONS// /, .} + product docs)"
        echo "-------------------------------------------------------------------------------"
        printf '%8s  %s\n' "LINES" "FILE"
        sort -t$'\t' -k1,1nr -k2,2 "$tmp" | while IFS=$'\t' read -r count path; do
            printf '%8s  %s\n' "$count" "$path"
        done
        echo "-------------------------------------------------------------------------------"
        printf '%8s  files with Portuguese\n' "$total_files"
        printf '%8s  offending lines total\n' "$total_hits"
        echo
        echo "This mode never fails. Track progress in docs/MIGRATION.md; the CI gate is"
        echo "the default mode, which only looks at what your branch changed."
    fi

    sort -t$'\t' -k1,1nr -k2,2 "$tmp" > "${tmp}.sorted"
    ALL_SORTED="${tmp}.sorted"
    ALL_FILES="$total_files"
    ALL_HITS="$total_hits"
    return 0
}

# ---------------------------------------------------------------------------
# Mode: --generate-allowlist
# ---------------------------------------------------------------------------
run_generate_allowlist() {
    run_all quiet
    {
        cat <<EOF
# scripts/language_allowlist.txt — files that still contain Portuguese.
#
# Generated by: ./scripts/check_language.sh --generate-allowlist
# Generated on: $(date +%Y-%m-%d)
#
# This list exists so CI can be green TODAY while the English migration runs file
# by file (docs/MIGRATION.md). Rules:
#
#   * A file ON this list is not scanned in full -- but the lines your branch ADDS
#     to it are. You cannot add new Portuguese to a legacy file.
#   * A file NOT on this list must be 100% English, every line.
#   * When you finish migrating a file, DELETE its line here in the same commit
#     and mark it done in docs/MIGRATION.md. The list only ever shrinks.
#
# One repo-relative path per line. Shell globs are allowed. '#' starts a comment.
EOF
        while IFS=$'\t' read -r _count path; do
            printf '%s\n' "$path"
        done < "$ALL_SORTED"
    } > "$ALLOWLIST_FILE"

    echo "Wrote $ALLOWLIST_FILE with $ALL_FILES entries ($ALL_HITS offending lines)."
}

# ---------------------------------------------------------------------------
# Mode: default (CI gate — only what this branch changed)
# ---------------------------------------------------------------------------
resolve_base() {
    local candidate
    if [ -n "${BASE_REF:-}" ]; then
        candidate="$BASE_REF"
    elif git rev-parse --verify --quiet origin/main >/dev/null; then
        candidate=origin/main
    elif git rev-parse --verify --quiet main >/dev/null; then
        candidate=main
    else
        candidate=""
    fi

    if [ -z "$candidate" ]; then
        BASE_SHA=$(git rev-list --max-parents=0 HEAD | tail -1)
        BASE_NAME="(root commit)"
        return 0
    fi

    BASE_SHA=$(git merge-base HEAD "$candidate" 2>/dev/null) || {
        echo "check_language.sh: cannot find a merge base with '$candidate'." >&2
        echo "  Fetch the base branch first, or set BASE_REF to something local." >&2
        exit 2
    }
    BASE_NAME="$candidate"
}

run_changed() {
    resolve_base
    load_allowlist

    local -a changed=()
    local path
    while IFS= read -r path; do
        [ -n "$path" ] && changed+=("$path")
    done < <(
        {
            git diff --name-only --diff-filter=ACMR "$BASE_SHA" HEAD
            git diff --name-only --diff-filter=ACMR HEAD
            git ls-files --others --exclude-standard
        } | sort -u
    )

    echo "check_language.sh: base=$BASE_NAME (${BASE_SHA:0:8}), ${#changed[@]} changed path(s)"

    local failed=0 checked=0
    for path in "${changed[@]}"; do
        is_in_scope "$path" || continue
        [ "$path" = "$ALLOWLIST_FILE" ] && continue
        checked=$((checked + 1))

        if is_allowlisted "$path"; then
            # Legacy file: only the lines this branch added have to be English.
            check_one "$path" added \
                "new Portuguese added to a file that is still on the allowlist" \
                || failed=1
        else
            check_one "$path" full \
                "this file is not on the allowlist, so it must be fully English" \
                || failed=1
        fi
    done

    echo "check_language.sh: $checked file(s) in the English-only scope"

    if [ "$failed" -ne 0 ]; then
        fix_instructions
        return 1
    fi

    echo "check_language.sh: OK"
    return 0
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
case "${1:-}" in
    --all)                run_all; exit 0 ;;
    --generate-allowlist) run_generate_allowlist; exit 0 ;;
    -h | --help)          usage; exit 0 ;;
    "")                   run_changed; exit $? ;;
    *)
        echo "check_language.sh: unknown option '$1'" >&2
        echo "Try: check_language.sh --help" >&2
        exit 2
        ;;
esac
