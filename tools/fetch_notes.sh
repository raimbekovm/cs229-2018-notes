#!/usr/bin/env bash
# Download the official CS229 Autumn 2018 class notes into cs229-notes/ and extract them.
#
#   tools/fetch_notes.sh          # every PDF listed in tools/notes_map.tsv
#   tools/fetch_notes.sh 7        # only the PDFs lecture 7 draws on
#
# The notes are the secondary source for a chapter — the lecture video is the
# primary one. Stanford no longer serves the 2018 per-topic PDFs, so they come from
# the maxim5/cs229-2018-autumn mirror; full URLs in the map are fetched as given.
# Everything lands in cs229-notes/, which is gitignored: it is Stanford's material
# and stays local.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIRROR="https://raw.githubusercontent.com/maxim5/cs229-2018-autumn/main"
MAP="$REPO/tools/notes_map.tsv"
OUT="$REPO/cs229-notes"

key=""
if [ -n "${1:-}" ]; then
  case "$1" in *[!0-9]*) key="$(printf '%s' "$1" | sed -E 's/^([0-9])([a-z])$/0\1\2/')" ;; *) key="$(printf '%02d' "$((10#$1))")" ;; esac
  grep -q "^$key	" "$MAP" || { echo "lecture $key is not in $MAP" >&2; exit 1; }
fi

mkdir -p "$OUT"
awk -F'\t' -v k="$key" 'k == "" || $1 == k {print $2}' "$MAP" | tr ' ' '\n' | grep -v '^-$' | sort -u |
  while read -r src; do
    case "$src" in http*) url="$src" ;; *) url="$MIRROR/$src" ;; esac
    pdf="$OUT/$(basename "$src")"
    if [ ! -s "$pdf" ]; then
      curl -sfL -o "$pdf" "$url" || { echo "failed: $url" >&2; rm -f "$pdf"; continue; }
    fi
    python3 "$REPO/tools/notes_text.py" "$pdf"
  done
