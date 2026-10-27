# Repo-Hinweise

- `ba-ner/`: Experimentcode (Python, SLURM-Skripte, Konfigurationen). `thesis/`: LaTeX-Quellen der Bachelorarbeit (deutsch).
- Thesis bauen: `cd thesis && latexmk -pdf main.tex`. Hilfsdateien landen in `thesis/build/`, das PDF in `thesis/main.pdf`; beides ist in Git ignoriert.
- Rote Platzhalter `\PH{...}` in `thesis/chapters/05_*.tex`, `06_*.tex` und `anhang.tex` sind mit Messwerten zu füllen. Die `% TODO`-Kommentare am Anfang jedes Abschnitts nennen die Quelldateien und die nötige Rechnung.
- Detaillierter Stand, Konventionen, offene Punkte und Zeitplan: `thesis/STATUS.md` (zuerst lesen).
- Ergebnisse liegen nicht im Repo (`ba-ner/results/` ist ignoriert), sondern auf dem Cluster unter `/netscratch/losman/ba-ner/results`.
- Keine Paper-PDFs committen (`thesis/literatur/` ist ignoriert). Modellgewichte, Adapter und Checkpoints bleiben ebenfalls draußen.
- Arbeitsregel bei zwei Rechnern: vor dem Arbeiten `git pull`, danach `git push`.
