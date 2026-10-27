#!/usr/bin/env bash
# Holt die Cluster-Commits und die Ergebnisdateien auf diesen Rechner.
#
# Aufruf im Repo-Ordner auf einem Mac mit Cluster-Zugang (ggf. VPN an):
#     bash sync_from_cluster.sh
#
# Das Skript fragt nach dem SSH-Ziel, prüft die Verbindung, holt die Commits
# aus dem Cluster-Repo per "git fetch", führt sie zusammen, fragt vor dem
# Push und kopiert danach die kleinen Ergebnisdateien (YAML/JSON/CSV, keine
# Modellgewichte) nach ba-ner/results/, das in Git ignoriert ist.
set -euo pipefail

cd "$(dirname "$0")"
if [ ! -d .git ]; then
    echo "Bitte im Repo-Ordner ausführen (dort, wo .git liegt)." >&2
    exit 1
fi

echo "Mit welchem Ziel loggst du dich auf dem Cluster ein?"
echo "Gemeint ist das, was du sonst hinter 'ssh' tippst, z. B. losman@rechnername.dfki.de"
if [ -f "$HOME/.ssh/config" ]; then
    echo "Aliase in ~/.ssh/config:"
    grep -i '^Host ' "$HOME/.ssh/config" | sed 's/^[Hh]ost /  /' || true
fi
read -r -p "SSH-Ziel: " TARGET
case "$TARGET" in
    ""|*HOST*|*DEIN-LOGIN*|*musterfrau*)
        echo "Das ist ein Platzhalter, kein echtes Ziel. Bitte den echten Login eingeben." >&2
        exit 1 ;;
esac

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

echo "Hole die Ergebnisdateien von /netscratch/$RUSER/ba-ner/results/ ..."
mkdir -p ba-ner/results
rsync -avm --include='*/' --include='*.yaml' --include='*.json' --include='*.csv' --exclude='*' \
    "$TARGET:/netscratch/$RUSER/ba-ner/results/" ba-ner/results/
echo "Angekommen: $(find ba-ner/results -name '*.yaml' | wc -l | tr -d ' ') YAML-, $(find ba-ner/results -name '*.json' | wc -l | tr -d ' ') JSON-Dateien"
echo "Laufordner:"
find ba-ner/results -mindepth 2 -maxdepth 2 -type d | sort | head -60
echo "Fertig. Nächster Schritt: Claude Code in diesem Ordner starten und thesis/STATUS.md lesen lassen."
