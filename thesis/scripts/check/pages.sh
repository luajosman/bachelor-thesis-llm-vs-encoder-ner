#!/usr/bin/env bash
# Seitenzahlen, Kapitelseiten und Satzwarnungen aus dem letzten Build.
# Aufruf: bash thesis/scripts/check/pages.sh   (von beliebigem Verzeichnis)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THESIS="$(cd "$HERE/../.." && pwd)"
LOG="$THESIS/build/main.log"
TOC="$THESIS/build/main.toc"
if [[ ! -f "$LOG" ]]; then echo "kein Build-Log: $LOG"; exit 1; fi
echo "Build: $(grep -o 'Output written on [^ ]* ([0-9]* pages' "$LOG" | tail -1)"
echo "Overfull: $(grep -c Overfull "$LOG")  Underfull: $(grep -c Underfull "$LOG")  undefined: $(grep -ci 'undefined' "$LOG")  multiply: $(grep -ci 'multiply defined' "$LOG")"
echo "Kapitel -> Startseite:"
grep -E 'contentsline \{chapter\}' "$TOC" | sed -E 's/.*\\numberline \{([^}]*)\}([^}]*)\}\{([0-9ivx]+)\}.*/  \1 \2 -> \3/; s/.*\{chapter\}\{([^}\\]*)\}\{([0-9ivx]+)\}.*/  \1 -> \2/'
FIRST=$(grep -E 'contentsline \{chapter\}\{\\numberline \{1\}' "$TOC" | sed -E 's/.*\}\{([0-9]+)\}\{.*/\1/' | head -1)
LIT=$(grep -E 'Literaturverzeichnis' "$TOC" | sed -E 's/.*\}\{([0-9]+)\}\{.*/\1/' | head -1)
if [[ -n "${FIRST:-}" && -n "${LIT:-}" ]]; then echo "Haupttext: Seiten ${FIRST} bis $((LIT-1)) = $((LIT-FIRST)) Seiten"; fi
