#!/usr/bin/env bash
# Grab one still of the board from a CS229 (Autumn 2018) lecture video.
#
#   tools/board_frame.sh 19 01:10:50             # lecture 19 at 1:10:50
#   tools/board_frame.sh 19 01:07:45 01:10:50    # several stills in one call
#   tools/board_frame.sh 2 00:20:00 --out /tmp/x # somewhere other than $TMPDIR/boards
#
# Captions carry what was said, not what was written. When the transcript says
# "this term here" or a sign on a board formula matters, a single frame settles it.
# Writes <out>/lecture-NN_HHMMSS.png and prints each path. The stills are for
# reading the lecture and never become figures; they stay out of the repository.
#
# Video formats sit behind YouTube's bot check, so this always reads cookies:
# YTDLP_COOKIES_BROWSER (default "chrome:Profile 3") or YTDLP_COOKIES_FILE. Run it
# outside the sandbox; see fetch_lecture.sh for the Full Disk Access caveat.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

N="${1:?usage: board_frame.sh <lecture-number|08b> <HH:MM:SS>... [--out DIR]}"
shift
OUTDIR="${TMPDIR:-/tmp}/boards"
STAMPS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUTDIR="${2:?--out needs a directory}"; shift 2 ;;
    *) STAMPS+=("$1"); shift ;;
  esac
done
[ ${#STAMPS[@]} -gt 0 ] || { echo "give at least one HH:MM:SS" >&2; exit 1; }

case "$N" in
  *[!0-9]*) NN="$(printf '%s' "$N" | sed -E 's/^([0-9])([a-z])$/0\1\2/')" ;;
  *) NN="$(printf '%02d' "$((10#$N))")" ;;
esac

for tool in yt-dlp ffmpeg; do
  command -v "$tool" >/dev/null || { echo "missing $tool" >&2; exit 1; }
done
row="$(awk -F'\t' -v n="$NN" '$1 == n {print; exit}' "$REPO/tools/playlist.tsv")"
[ -n "$row" ] || { echo "lecture $NN is not in tools/playlist.tsv" >&2; exit 1; }
# Quoted throughout: a bare watch?v= URL is a glob in zsh and fails with "no matches found".
URL="https://www.youtube.com/watch?v=$(printf '%s' "$row" | cut -f2)"

if [ -n "${YTDLP_COOKIES_FILE:-}" ]; then
  cookies=(--cookies "$YTDLP_COOKIES_FILE")
else
  cookies=(--cookies-from-browser "${YTDLP_COOKIES_BROWSER:-chrome:Profile 3}")
fi

mkdir -p "$OUTDIR"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
for t in "${STAMPS[@]}"; do
  [[ "$t" =~ ^[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]] || { echo "bad timestamp $t (want HH:MM:SS)" >&2; exit 1; }
  tag="${t//:/}"
  IFS=: read -r h m s <<<"$t"
  end=$((10#$h * 3600 + 10#$m * 60 + 10#$s + 2))
  # A zero-length section sometimes cuts between keyframes and yields no video stream;
  # two seconds always holds one, and the first frame is the requested time.
  yt-dlp -q --no-warnings "${cookies[@]}" -f 'bv*[height<=720]' \
    --download-sections "*$t-$end" -o "$work/$tag.%(ext)s" "$URL"
  clip="$(ls "$work/$tag".* | head -1)"
  if [ "$(wc -c <"$clip")" -lt 10000 ]; then
    # Seen on 08b: when YouTube's n-challenge fails, only progressive format 18 is offered
    # and every byte range of it comes back empty. Nothing on this side fixes that.
    echo "lecture $NN at $t: YouTube returned an empty clip (only a broken format offered); try later" >&2
    exit 2
  fi
  png="$OUTDIR/lecture-${NN}_$tag.png"
  ffmpeg -loglevel error -y -i "$clip" -frames:v 1 "$png"
  echo "$png"
done
