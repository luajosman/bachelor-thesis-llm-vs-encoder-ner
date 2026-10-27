#!/usr/bin/env bash
# Holt die Cluster-Commits und die Ergebnisdateien auf diesen Rechner.
#
# Aufruf im Repo-Ordner auf einem Mac mit Cluster-Zugang (ggf. VPN an):
#     bash sync_from_cluster.sh
#
# Das Skript sucht auf diesem Rechner nach bereits benutzten SSH-Zielen
# (Shell-Historie, ~/.ssh/config, bekannte Hosts, VS Code), lässt eines
# auswählen, prüft die Verbindung, holt die Commits aus dem Cluster-Repo per
# "git fetch", führt sie zusammen, fragt vor dem Push und kopiert danach die
# kleinen Ergebnisdateien (YAML/JSON/CSV, keine Modellgewichte) nach
# ba-ner/results/, das in Git ignoriert ist.
set -euo pipefail

cd "$(dirname "$0")"
if [ ! -d .git ]; then
    echo "Bitte im Repo-Ordner ausführen (dort, wo .git liegt)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Mögliche SSH-Ziele auf diesem Rechner einsammeln
# ---------------------------------------------------------------------------
CANDIDATES=()
ALIASES=()
add_candidate() {
    local c="$1" x
    [ -z "$c" ] && return 0
    case "$c" in -*|*HOST*|*DEIN-LOGIN*|*musterfrau*|*github*|*gitlab*) return 0 ;; esac
    for x in "${CANDIDATES[@]:-}"; do [ "$x" = "$c" ] && return 0; done
    CANDIDATES+=("$c")
}

if [ -f "$HOME/.ssh/config" ]; then
    while read -r h; do ALIASES+=("$h"); add_candidate "$h"; done < <(
        awk 'tolower($1)=="host" {for (i=2;i<=NF;i++) if ($i !~ /[*?]/) print $i}' "$HOME/.ssh/config")
fi
for f in "$HOME/.zsh_history" "$HOME/.bash_history"; do
    [ -f "$f" ] || continue
    while read -r h; do add_candidate "$h"; done < <(
        grep -a -oE '(^|;| )ssh +[A-Za-z0-9._@-]+' "$f" | sed -E 's/.*ssh +//' | sort -u)
done
if [ -f "$HOME/.ssh/known_hosts" ]; then
    while read -r h; do add_candidate "$h"; done < <(
        cut -d' ' -f1 "$HOME/.ssh/known_hosts" | grep -v '^|' | cut -d, -f1 | sort -u)
fi
VSDIR="$HOME/Library/Application Support/Code/User/workspaceStorage"
if [ -d "$VSDIR" ]; then
    while read -r h; do add_candidate "$h"; done < <(
        grep -a -h -oE 'ssh-remote%2B[A-Za-z0-9._%-]+' "$VSDIR"/*/workspace.json 2>/dev/null \
        | sed -E 's/^ssh-remote%2B//; s/%40/@/g' | sort -u)
fi

echo "Mit welchem Ziel loggst du dich auf dem Cluster ein?"
if [ "${#CANDIDATES[@]}" -gt 0 ]; then
    echo "Auf diesem Rechner wurden diese SSH-Ziele schon einmal benutzt:"
    i=1
    for c in "${CANDIDATES[@]}"; do echo "  $i) $c"; i=$((i+1)); done
    echo "Nummer eingeben, oder ein anderes Ziel tippen (z. B. losman@rechnername.dfki.de)."
else
    echo "Keine früheren SSH-Ziele gefunden. Bitte eintippen, z. B. losman@rechnername.dfki.de"
fi
read -r -p "Auswahl: " CHOICE
if [[ "$CHOICE" =~ ^[0-9]+$ ]] && [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le "${#CANDIDATES[@]}" ]; then
    TARGET="${CANDIDATES[$((CHOICE-1))]}"
else
    TARGET="$CHOICE"
fi
case "$TARGET" in
    ""|*HOST*|*DEIN-LOGIN*|*musterfrau*)
        echo "Das ist ein Platzhalter, kein echtes Ziel." >&2
        exit 1 ;;
esac
# Reiner Rechnername ohne Benutzer und ohne Alias: Cluster-Benutzer ergänzen
if [[ "$TARGET" != *@* ]]; then
    is_alias=0
    for a in "${ALIASES[@]:-}"; do [ "$a" = "$TARGET" ] && is_alias=1; done
    [ "$is_alias" -eq 0 ] && TARGET="losman@$TARGET"
fi
echo "Verwende: $TARGET"

# ---------------------------------------------------------------------------
# 2. Verbindung prüfen, Repo und Benutzer auf dem Cluster feststellen
# ---------------------------------------------------------------------------
echo "Teste die Verbindung (falls es lange hängt: VPN an?) ..."
ssh -o ConnectTimeout=20 "$TARGET" 'echo "Verbunden als $(whoami) auf $(hostname)"'
RUSER="$(ssh "$TARGET" 'whoami')"

REMOTE_REPO="$(ssh "$TARGET" 'for d in ~/bachelor-thesis-llm-vs-encoder-ner ~/bachelor-thesis-llm-vs-encoder-ner-1; do if [ -d "$d/.git" ]; then echo "$d"; break; fi; done')"
if [ -z "$REMOTE_REPO" ]; then
    echo "Kein Repo im Home-Verzeichnis auf dem Cluster gefunden. Vorhandene Ordner:" >&2
    ssh "$TARGET" 'ls -d ~/bachelor-thesis* 2>/dev/null; ls -d ~/* 2>/dev/null | head -20' || true
    exit 1
fi
echo "Repo auf dem Cluster: $REMOTE_REPO"

DIRTY="$(ssh "$TARGET" "git -C '$REMOTE_REPO' status --porcelain | wc -l | tr -d ' '")"
if [ "$DIRTY" != "0" ]; then
    echo "Hinweis: Im Cluster-Repo gibt es $DIRTY nicht committete Änderungen; die werden nicht geholt."
fi
REMOTE_BRANCH="$(ssh "$TARGET" "git -C '$REMOTE_REPO' rev-parse --abbrev-ref HEAD")"
echo "Branch auf dem Cluster: $REMOTE_BRANCH"

# ---------------------------------------------------------------------------
# 3. Commits holen, zusammenführen, auf Wunsch pushen
# ---------------------------------------------------------------------------
git remote remove cluster 2>/dev/null || true
git remote add cluster "$TARGET:$REMOTE_REPO"
git fetch cluster
echo "Neueste Commits auf dem Cluster:"
git log --oneline -5 "cluster/$REMOTE_BRANCH"

echo "Aktualisiere den lokalen Stand von GitHub und führe die Cluster-Commits zusammen ..."
git pull --no-rebase --no-edit origin main
git merge --no-edit "cluster/$REMOTE_BRANCH"
echo "Commits, die noch nicht auf GitHub sind:"
git log --oneline origin/main..HEAD || true

read -r -p "Jetzt zu GitHub pushen? [j/N] " ANSWER
if [ "$ANSWER" = "j" ] || [ "$ANSWER" = "J" ]; then
    git push origin main
else
    echo "Nicht gepusht. Später: git push origin main"
fi

# ---------------------------------------------------------------------------
# 4. Ergebnisdateien kopieren
# ---------------------------------------------------------------------------
echo "Hole die Ergebnisdateien von /netscratch/$RUSER/ba-ner/results/ ..."
mkdir -p ba-ner/results
rsync -avm --include='*/' --include='*.yaml' --include='*.json' --include='*.csv' --exclude='*' \
    "$TARGET:/netscratch/$RUSER/ba-ner/results/" ba-ner/results/
echo "Angekommen: $(find ba-ner/results -name '*.yaml' | wc -l | tr -d ' ') YAML-, $(find ba-ner/results -name '*.json' | wc -l | tr -d ' ') JSON-Dateien"
echo "Laufordner:"
find ba-ner/results -mindepth 2 -maxdepth 2 -type d | sort | head -60
echo "Fertig. Nächster Schritt: Claude Code in diesem Ordner starten und thesis/STATUS.md lesen lassen."
