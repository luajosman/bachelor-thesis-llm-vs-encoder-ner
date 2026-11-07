#!/usr/bin/env bash
# Altvarianten der Terminologie finden. Aufruf: bash terms.sh DATEI [DATEI…]; Exit 1 bei Treffern.
set -u
PATTERN='Zero-Shot-(Setting|Einsatz)|Silber-Referenz|Silberannotation|geprompte|parameter-effizient|token-level|Micro-F|BIO-Tag|BIO-Sequenz|Few-Shot-Prompting.*[Kk]eraghel|[Kk]eraghel.*Few-Shot|DeBERTaV3|Strukturtreue|LoRA-Regime'
rc=0
for f in "$@"; do
  if grep -nE "$PATTERN" "$f" | grep -v '^[0-9]*:%' ; then rc=1; fi
done
[[ $rc -eq 0 ]] && echo "terms: keine Altvarianten"
exit $rc
