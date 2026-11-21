# Stand der Bachelorarbeit (21.11.2026)

## Abbildungsprüfung (21.11., Claude Code)

- **Bericht:** `review/26_abbildungspruefung_2026-11-21.md`. Sechs Abbildungen (2.1 bis 2.3, 5.1 bis 5.3), keine im Anhang. Alle Werte der Diagramme aus den Rohvorhersagen nachgerechnet, keine Abweichung; Neulauf der Aggregation byteidentisch. Kein kritischer Befund.
- **Umgesetzt (Minimalvarianten):** Abb. 5.1 Marker versetzt, Achse bis 103, Unterschrift mit „Zero-Shot als Einzellauf; LLM-Werte bei 256 Ausgabetokens"; Abb. 5.2 Zero-Shot-Marker durchgezogen, Unterschrift mit „vergrößerter Ausschnitt" und 256 Ausgabetokens; Abb. 5.3 Achse „Mittlere Latenz", 4B-Quadrat unterscheidbar; Abb. 2.3 ΔW = (α/r)·BA wie Gleichung 2.2; Abb. 2.1 Quellenangabe „Abb. 1". Weiterhin 112 Seiten, alle Labels auf denselben Seiten.
- **Offen beim Verfasser:** GPU-Typ in Abb. 5.3, Hinweis auf seltene Typen und Fehlerbalken in Abb. 5.1, Werte bei 1 024 Tokens in Abb. 5.2 (verlangen Skriptänderung oder kosten eine Zeile); G-03 bis G-09.

## Betreuerfeedback zu Kapitel 4 und 5 (21.11., Claude Code)

- **Anlass:** handschriftliche Kommentare von Philippe Thomas vom 17.11. zur Fassung vom 12.11. (25 Anmerkungen: 22 in Kapitel 4, eine in 5.1.1, zwei in 6.1) und zwei Hinweise aus der Begleitmail (Trennung Grundlagen / verwandte Arbeiten / Methodik; Methodenkapitel stellenweise zu ausführlich).
- **Stand:** 112 physische Seiten, Haupttext S. 1–77 (Kapitel 1: 5, Kapitel 2: 12, Kapitel 3: 13, Kapitel 4: 22, Kapitel 5: 20, Kapitel 6: 5 Seiten), Literatur ab S. 78, Anhang ab S. 87. Build ohne Fehler und Warnungen. Datum auf Deckblatt und Erklärung: 26. November 2026.
- **Struktur:** „Datensätze und Korpora" steht jetzt als Abschnitt 3.1 in den verwandten Arbeiten (vorher 2.8) und enthält die MultiNERD-Referenzwerte (vorher 3.1.1). Der frühere Unterabschnitt 4.5.4 zur Analyse des Größeneinflusses ist in 4.2.3 aufgegangen; die Nachinferenz ist damit Abschnitt 4.5.4. Die Unterabschnitte von Anhang A.2 haben Labels (`app:vorlaeufe` bis `app:kontrollstichprobe`); Kapitel 4 und 5 verweisen nicht mehr pauschal auf A.2.
- **Kapitel 4:** Einleitung mit inhaltlichem Überblick; neues Listing 4.1 (Nutzernachricht, leerer Denkblock, Zielantwort); die drei Längengrenzen (256 neue Tokens, 1 024 Tokens Trainingssequenz, 1 024 Ausgabetokens nur in den Sensitivitätsanalysen) getrennt erklärt; Folgen von „nicht parsebar" und „verworfen" direkt bei den Verarbeitungsschritten; Wort-Tokenisierung der Evaluation benannt; Absatz zur hybriden Attention und Detailangaben in den Anhang verlegt; gestrichen sind die vom Betreuer markierten Stellen (rsLoRA/DoRA, SHA-256 im Haupttext, pauschaler Anhangssatz).
- **Kapitel 5 und 6:** kein Satz beginnt mehr mit „Tabelle", „Abbildung" oder „Listing" (gilt für alle Kapitel und den Anhang); eigener Absatz „Datenkontamination" in 5.7.1, Abschnitt 3.6 verweist nur noch darauf; die Asymmetrie der Vergleiche bei 256 und 1 024 Tokens steht ausdrücklich in 5.7.1 und 6.1; die Korrekturen des Branches `korrekturen-2026-11-25` (Rundungssatz, 126 gegenüber 151 Typfehler) sind übernommen.
- **Zusätzlich:** eindeutige Berichtigungen aus `review/24_…` (B-09, B-10, B-12 bis B-25, Q-02 bis Q-16) und die formalen Punkte aus `review/25_…` (Jahr und Revision für die vier Online-Quellen ohne Datum, Datum der 27B- und 397B-Karte, DOI bei Efron und Tibshirani entfernt, zwei unzitierte Einträge gelöscht).
- **Offen beim Verfasser:** B-01 bis B-08, B-11, B-26 aus Bericht 24 (darunter „vier oder sieben Hauptvergleiche" und die BF16-Messung); W6, W7, W9 bis W14, W16 und die G-Punkte aus Bericht 25; KI-Angabe in der Erklärung auf Vollständigkeit prüfen (zuletzt verwendetes Claude-Modell); englisches Abstract; Repository öffentlich schalten; Branch nach `main` zusammenführen und pushen.

## Kürzung und Kontrolllauf (12.11., Claude Code)

- **Stand:** 111 physische Seiten, Haupttext S. 1–77 (Kapitel 1: 5, Kapitel 2: 13, Kapitel 3: 12, Kapitel 4: 22, Kapitel 5: 20, Kapitel 6: 5 Seiten), Literatur S. 78–86, Anhang ab S. 87. Build ohne Fehler und Warnungen, 15 Skripttests, Aggregation byte-reproduzierbar. Die lange Fassung (118 Seiten, Haupttext 85) liegt als Branch `fassung-lang-2026-11-12` und lokal als `thesis/main_long12112026.pdf` vor.
- **Kürzung:** zwei Runden; was in welchem Abschnitt entfiel und warum, steht in `KUERZUNG_CHANGELOG.md`. Runde 1 nach festem Auftrag (2.3, 2.7, 3.2.1 bis 3.2.3, Tabelle 3.1, Wiederholungen gegenüber Tabellen in Kapitel 4 und 5, 5.7.5 verweist auf 6.2, 6.1, Abstract neu formuliert). Runde 2 mit der Vorgabe des Verfassers, dass aus Kapitel 4 und 5 keine wichtige Angabe verschwindet: dort nur Doppelungen gestrichen und zwei Details in den Anhang verschoben, dazu Straffungen in 2.5, 2.6 und Kapitel 3. Keine Messzahl geändert, keine Quelle entfallen außer `su2024roformer` (Runde 1).
- **Kontrolllauf:** Die neun Kontrollläufe (Jobs 3724608 bis 3724616, 9./10.11., Commit 1638272, H100) sind aggregiert und eingebaut. Je Lauf 1 000 zufällige Testsätze mit parsebarer Antwort, mit 1 024 Tokens neu erzeugt: 8 988 von 9 000 zeichengleich, keine Limit-Treffer. Im Text als Sensitivitätsanalyse (4.5.5, 5.2, 5.4, 5.7.3, 5.7.4, 6.1, 6.2, Anhang A.2.12, Tabelle A.8); Abstract und Hauptzahlen bei 256 Tokens unverändert. Rohdaten lokal unter `ba-ner/results/multinerd_control_1024/` (per rsync nur für diesen Ordner geholt), Auswertung in `scripts/aggregate_results.py` (`analyse_control_sample`, Schlüssel `control_sample_multinerd` in `summary.json`).
- **Umbruchregel:** Nummerierte Überschriften rücken weiter aus dem unteren Seitendrittel auf die nächste Seite; Absatzüberschriften (`\paragraph`) sind über die titlesec-Haken in `main.tex` ausgenommen.
- **Berichtigt:** Legende von Tabelle 4.4 (die Nachinferenz lief auch für das 4B-Modell auf einem anderen GPU-Typ als der Hauptlauf).
- **Offen beim Verfasser:** Folgenachricht an den Betreuer (`review/23_…`) mit der neuen PDF-Kopie versenden; Rückmeldung zum Kontrolllauf und zur Länge abwarten; die offenen Punkte des Abschnitts vom 8.11. unten (KI-Angabe, englisches Abstract, Repository) gelten weiter.
- **Lektorat (Branch `lektorat-2026-11-25`, nicht gepusht):** sprachliche und formale Durchsicht der gesamten Arbeit. 64 typografische Korrekturen direkt angewendet (geschützte Leerzeichen vor Klammerzitaten und Einheiten), kein Rechtschreibfehler gefunden, 205 Vorschläge zur Entscheidung (hoch 6, mittel 73, niedrig 126). Seitenstand unverändert (111, Haupttext 77). Bericht: `LEKTORAT_2026-11-25.md`.

Aufgeräumt am 12.11.: Auf Wunsch des Verfassers sind `KUERZUNG_CHANGELOG.md`, `LEKTORAT_2026-11-25.md` und die Notizen 16, 22 und 23 aus `review/` entfernt, ebenso die lokale PDF-Kopie `main_betreuerentwurf_f233071.pdf`. Verweise darauf in diesem Dokument beziehen sich auf den Stand davor; die Dateien sind über die Git-Historie abrufbar, z. B. `git show 9ebc572:thesis/LEKTORAT_2026-11-25.md`.

Aufgeräumt am 09.11.: Die abgearbeiteten Prüfberichte 01–08a, 10–15 und 18–21 sind aus `review/` entfernt; die nie committeten Codex-Berichte 09 und 17 ebenfalls. Die Verweise unten beziehen sich auf den Stand davor; die Berichte sind über die Git-Historie abrufbar, z. B. `git show 1aa3a1b:thesis/review/15_umsetzungsprotokoll_2026-11-07.md`.

## Umsetzung der 1,0-Überarbeitung (7./8.11., Claude Code)

- **Stand:** 116 physische Seiten, Haupttext S. 1–82 (Ausgangsstand 7.11.: 112/90; nach den Korrekturen der Phase A 119/95). Überschriften im unteren Seitenviertel rücken per titlesec-Regel auf die nächste Seite (`main.tex`). Build ohne Fehler, Overfull/Underfull 0, keine offenen Verweise oder Zitate; Aggregationsskript mit 10 Tests, zwei Läufe byte-identisch. Tag `betreuerentwurf-2026-11-08` markiert den Stand nach Phase A (Korrekturen), der aktuelle `main` enthält zusätzlich die Verdichtung.
- **Umgesetzt (Details, Quellenfundstellen und Prüfergebnisse je Block):** `review/15_umsetzungsprotokoll_2026-11-07.md`. Kurz: Literatur ergänzt und vereinheitlicht (Ramponi, Mayhew, GNER, Bootstrap, Modellkarten, TRL-Doku; Preprints gekennzeichnet; Codex' Bereinigung übernommen; unzitierte Einträge entfernt); Kapitel 3 Quellenwiedergabe, Forschungslücke als zwei Lücken, Erwartungen E1–E3; Kapitel 1 abgestimmt; Kapitel 4 Datennutzungstabelle, Fairness, Belege, Checkpoint-Diagnose; Kapitel 5 Überdeutungen zurückgenommen, Synthesetabelle, alternative Erklärungen; Kapitel 6 Antworten je FF mit Erwartungsabgleich, Limitationen je FF, priorisierter Ausblick; Kapitel 2 technische Präzisierungen; Anhang mit Repositoryzugang, Laufdaten, Vorläufen, neuem Abschnitt A.2 „Ergänzungen zur Durchführung"; KI-Angabe; Deckblattdatum; Verdichtung in zwei Runden (Haupttext −13 % Wörter).
- **Audit 8.11. (Bericht `review/18_verdichtungsaudit_2026-11-08.md`):** freigegebene Auswahl umgesetzt (alle Korrekturen und A-Vorschläge, fünf B-Vorschläge), −232 Fließtextwörter bei unveränderten 116 Seiten; korrigiert u. a. die Speicherspanne in 5.7.4 (ab 0,7 GiB) und den Verweis zu Zhans LoRA-Zahlen in 4.4.5. Anhang unverändert.
- **Prüfungen:** Zahlen-Audit aller Kapitel (alle Werte aus `generated/summary.json` oder Quelltext belegt), Quellenprüfung je Block, 8-Wort-Overlap gegen 77 Quelltexte 0 Treffer, Argumentationsgang (Lücke → FF → Erwartungen → Antworten → Abstract). Zwei unabhängige Bewertungs-Agenten am 8.11.: Gesamt 7,5 bzw. 8 von 10, geschätzte Note jeweils 1,7 (Spielraum 1,3–2,0).
- **KI-Angabe (8.11.):** Modelle aus den lokalen Protokollen eingetragen (Claude Fable 5.1, Opus 5.5, Sonnet 5.5; Zeitraum für Claude und OpenAI Juli bis November 2026 entsprechend der Bearbeitungszeit; OpenAI: gpt-5.6-luna, gpt-6-luna, gpt-6-astra, dazu gpt-5.3-codex, gpt-5.4 und gpt-5.5 als Vorarbeit vor der Bearbeitungszeit); Claude-Modelle vor dem 27.10. sind wegen der 30-Tage-Aufbewahrung nicht belegt und als „jeweils aktuelle Modelle" formuliert, ChatGPT im Browser ist nicht protokolliert.
- **Offen beim Verfasser:** KI-Angabe auf Vollständigkeit prüfen (ältere Claude-Modelle, ChatGPT-Modelle); Abgabedatum auf dem Deckblatt prüfen; KI-Angabe und Erklärungswortlaut mit Betreuer bzw. Prüfungsamt abstimmen; englisches Abstract klären; eigene Sprachdurchsicht mindestens von 1.2, 3.5, 5.7 und 6.1; Begleitnotiz (`review/16_begleitnotiz_entwurf_2026-11-08.md`) anpassen und versenden; Repository vor Abgabe öffentlich schalten. Inhaltlich nur durch neue Läufe zu verbessern: vollständiger Neulauf mit 1 024 Tokens auf einheitlicher Hardware, Zero-Shot-Wiederholungen mit validierungsbasierter Prompt-Wahl.
- **Werkzeuge:** Projekt-Agenten `thesis-schreiber`/`thesis-pruefer` (Sonnet) unter `.claude/agents/`; Prüfskripte unter `scripts/check/`.


## Aktueller Arbeitsplan vom 07.11.

- Abgabe am **26.11.2026** vom Verfasser bestätigt; internes Fertigstellungsziel **25.11.2026**. Die unten historisch erwähnte mögliche Verlängerung wird nicht eingeplant.
- Vollständiger Zwischenstand für den Betreuer am Abend des **08.11.2026**: dafür zehn Stunden am 07.11. und acht Stunden am 08.11. eingeplant.
- Aktuelles PDF nach Überarbeitung von Abschnitt 1.2 und Formatierungsprüfung: **112 physische Seiten, 90 Seiten Haupttext**. Alle 31 Tabellen und sieben Abbildungen sowie Listings und Deckblatt wurden visuell geprüft; Build ohne Fehler, offene Verweise/Zitate oder Overfull-/Underfull-Meldungen.
- Höchste fachliche Priorität haben die weiterhin offenen Quellenkorrekturen aus `review/08_literatur_zitationsaudit_2026-11-06.md`, ihre Folgen für Forschungslücke und Diskussion sowie letzte methodische Präzisierungen. Die Formatierungsänderungen haben diese inhaltlichen Punkte nicht erledigt.
- Aktuelle Gesamtbewertung, Kapitelwerte und konkreter 18-Stunden-Plan: `review/09_gesamtbewertung_und_18h_plan_2026-11-07.md`. Planungsbudget etwa 60 konzentrierte Stunden einschließlich der 18 Stunden bis zum Betreuerentwurf; anschließend etwa 42 Stunden. Endziel ungefähr 70–75 Seiten Haupttext, Arbeitsziel 73 Seiten.
- Die aktuelle Prüfung hat Bewertungs- und Planungsdokumente aktualisiert; die vorgeschlagenen inhaltlichen Manuskriptkorrekturen sind noch umzusetzen.

## Methodische Überarbeitung am 06.11.

- Alle geplanten Trainings-, Testinferenz- und 1.024-Token-Läufe sind abgeschlossen. Die unten dokumentierten laufenden Jobs vom 28.10. sind historische Zwischenstände.
- Die Regel „Differenz größer als Summe der Standardabweichungen“ wurde aus Manuskript und Aggregationscode entfernt. Seed-SD bleibt deskriptiv; ausgewählte Differenzen werden mit gepaarten Satz-Bootstrap-Intervallen eingeordnet. Kein Gleichwertigkeitsnachweis durch ein Intervall mit null.
- CrossNER erhält eine reproduzierbare, nach Domäne geschichtete Bootstrap-Auswertung aus den gespeicherten Vorhersagen für beide Ausgabelimits, mit geprüfter Satz-/Referenzpaarung und einer neuen Tabelle `generated/tab-crossner-ci.tex`.
- CrossNER bei 1.024 Tokens: Qwen-27B minus DeBERTa-large gepoolt +2,13 Punkte, 95-%-Intervall [1,47; 2,80]. Domänenweise schließen drei Intervalle null aus (Musik, Politik, Naturwissenschaften); KI und Literatur nicht. Keine Korrektur für Mehrfachvergleiche.
- Kapitel 1, 4–6 und Abstract grenzen Aussagen auf die ausgeführten Konfigurationen ein: kein isolierter Anpassungs-/Architektureffekt, keine behauptete Folgenlosigkeit der 200er-Validierung, keine unbekannte Sampling-Streuung als vernachlässigbar ausgegeben.
- Ausgabelimit-Analysen in Schlussfolgerungen und Empfehlungen integriert; MultiNERD-Nachinferenz als selektive Sensitivitätsanalyse eingeordnet. Ressourcenwerte unterschiedlicher Limits werden nicht zu einer ungemessenen Konfiguration kombiniert.
- Testgestützte Zero-Shot-Entwicklung bleibt ein historischer methodischer Vorbehalt. Epochenänderung 27B (2 → 3) wird genannt; eine damals nicht belegte Entscheidungsbegründung wird nicht nachträglich erfunden.
- Speicherwerte der eigenen Auswertung korrekt als GiB bezeichnet; keine garantierten GPU-Mindestgrößen oder allgemeinen Trainingszeiten aus den Messungen abgeleitet. Tatsächliche Trainingskonfigurationen und Rohartefakte bleiben erhalten.
- Methodische Quellen zu Selektionsverzerrung und Äquivalenzaussagen ergänzt. Neue Auswertungstests prüfen Paarung, grobe Typen, gepoolte Zähler, Seed-Mittel und Reproduzierbarkeit.
- Prüfung abgeschlossen: vollständige Aggregation, fünf Tests, PDF-Build mit 110 Seiten; 0 Overfull-Boxen, 0 offene Verweise/Zitate, 0 LaTeX-Fehler. Details und verbleibende Grenzen in `review/06_methodische_korrekturen_2026-11-06.md`.

Die folgenden Einträge dokumentieren den bisherigen Verlauf; ihre damaligen Bewertungen und offenen Experimentpunkte sind durch die Angaben oben und den aktuellen Überarbeitungsbericht zu ergänzen.

## Historischer Stand vom 28.10. bis 30.10.

Abgabe: 26.11.2026 (geplant 25.11.); mit zweiwöchigem Attest (Entscheidung bis 6.11.) verschiebt sich die Abgabe auf ca. 10.12.2026 (geplant 8.12.). Abstract nur deutsch (Entwurf steht). Angemeldeter Titel laut Anmeldung vermutlich „Fine-Tuning für NER: Ein Vergleich von BERT-basierten Modellen und LLMs mit LoRA“; das Deckblatt schreibt „Named Entity Recognition“ aus – **exakten Wortlaut prüfen**. Die Abweichung zu DeBERTa/QLoRA/Zero-Shot ist jetzt in 1.3 erklärt.

Der ausführliche Plan mit Bewertung je Kapitel, Audit-Listen und beiden Zeitplänen liegt unter `~/.claude/plans/guck-dir-bitte-kapitel-quiet-fairy.md` (Kopie der Kernpunkte unten).

## Ablage und Bauen

- LaTeX: `thesis/` in diesem Repo. Auf diesem Mac ist kein TeX installiert; Build über Docker Desktop:
  `cd thesis && docker run --rm -v "$PWD":/work -w /work texlive/texlive:latest latexmk -pdf main.tex`
  (Image `texlive/texlive:latest`, 9 GB, einmal gezogen). Hilfsdateien in `thesis/build/`, PDF `thesis/main.pdf`. Stand: 106 Seiten, 0 Overfull-Boxen, 0 offene Verweise. Seiten rendern: `gs -sDEVICE=png16m -r80 -dFirstPage=N -dLastPage=N -o build/pages/p.png main.pdf` im selben Container. Achtung: die Shell-Funktion `grep` schweigt bei der ISO-8859-Logdatei, `/usr/bin/grep -a` benutzen.
- Ergebnisse: `ba-ner/results/` (in Git ignoriert), seit 28.10. vollständig vom Cluster synchronisiert (alle YAML/JSON/CSV/MD/TXT/OUT, keine Gewichte; `bash sync_from_cluster.sh`, das in der aktuellen Fassung nur YAML/JSON/CSV holt). Zusätzlich dort: `slurm_accounting_2026-08.csv` (sacct-Export aller Jobs Aug.–Okt.), `crossner_overlap_check_2026-10-05.json`, `METHODIK_FACTS.md` (Beweis-Audit vom 8.10.).
- Aggregation: `/Users/luajosman/miniforge3/bin/python3 thesis/scripts/aggregate_results.py` (braucht PyYAML und numpy; das Repo-`.venv` hat beides nicht). Schreibt `thesis/generated/` (Tabellenkörper, Abbildungsdaten, `summary.json`), prüft strikte Werte gegen die gespeicherten, bricht bei Abweichung ab; Bootstrap mit festem Seed, Ausgabe byte-reproduzierbar. Erzeugt `tab-limit.tex` und `tab-limit-multinerd.tex` erst, wenn die 1024-Läufe vollständig vorliegen.
- Cluster: `losman@login1.pegasus.kl.dfki.de`, Repo `/home/losman/bachelor-thesis-llm-vs-encoder-ner`, Ergebnisse `/netscratch/losman/ba-ner/results`. Commit 137bffd (Zero-Shot) liegt jetzt als Tag `zeroshot-runs-137bffd` auf GitHub. `METHODIK_FACTS.md` wurde aus dem Cluster-Repo nach `/home/losman/` verschoben (Orchestrator verlangt sauberen Baum).

## Heute erledigt (28.10.)

- Rohdaten vollständig lokal; Zahlen-Audit von Kapitel 5/6 gegen `summary.json` und Rohdaten; Kapitel-4-Audit gegen Code/Configs.
- Faktenfehler korrigiert: Trainingsdauer 27B (vier Slurm-Zuteilungen, 265,0/264,2/269,5 h statt „≥ 50,5 h, rund 70 h“), „Hälfte des Recall-Rückstands“ (tatsächlich fast der ganze), „in keinem der neun Läufe“ (Einzelfälle), Speicher-„Größenordnungen“, CrossNER-Latenz/GPU, Validierungssplit-Anteil, SD-Regel bei CEL/VEHI/EVE, Latenzfaktoren, Rundungen, Adapter-Präzision (27B FP32), vierte Fehlerklasse, Kontamination, LogitMatch, FF2.
- Terminologie vereinheitlicht (Mikro-/Makro-F$_1$, F$_1$ mit Index, QLoRA-Regime, IOB2, Formtreue, Entity-Level), alle Tabellen booktabs, Kurztitel, Abbildungs- und Tabellenverzeichnis, Leerseite beseitigt, Abb. 2.1/2.2 als TikZ neu.
- Abstract-Entwurf, Erklärung mit KI-Angabe und Unterschriftszeile, Titelabsatz in 1.3, A.1 ergänzt (Tag, Vorlauf-Commits, Artefakte, Adaptergrößen).
- Aggregationsskript: sacct-Trainingszeiten, Testinferenzdauer, gepaartes Satz-Bootstrap (Tab. 5.9), Span-Zähler, CrossNER-Anhangstabellen, Hooks für die 1024-Auswertungen.
- Experimente laufen (Stand 28.10., 15:30): CrossNER mit 1024 Tokens komplett (Jobs 3673475–3673477, Aggregation 3673478; Erstabgabe 3672273–3672275 scheiterte an der Frozen-Commit-Prüfung, weil der Cluster-Checkout nach der Einreichung weiterbewegt wurde). MultiNERD-Nachinferenz der abgeschnittenen Antworten (`ba-ner/src/evaluate/multinerd_output_limit.py`, 75 Tests, Preflight ok): Smoke-Job 3673162 abgeschlossen (2/2 Präfix-Treffer, 5/5 Kontrollsätze byte-gleich, Pakete identisch, GPU H100 HBM3 statt NVL), neun Jobs 3673515–3673523 eingereicht. Alle Jobs auf Partition `H100-SLT,H100` (Gruppenpartition, startet binnen Minuten statt Tagen); Cluster-HEAD ff91342 darf bis zum Start aller Jobs nicht bewegt werden (kein Bundle/Pull).

- Abends (28.10., Commits dc9d045 bis 5aad2a0): vier Zweitgutachten eingeholt (`thesis/review/`), alle bestätigten Sachfehler in Kap. 1–6 und Abstract korrigiert; Begründungen in Kap. 4 ergänzt (Ausgabelimit 256 mit Längenverteilung der Referenzantworten aus `ba-ner/scripts/answer_lengths.py`, Checkpoint-Wahl auf 200 Sätzen, Modellwahl, Zero-Shot-Konfundierung); Bootstrap-Intervalle in Kap. 6; römischer Vorspann, Verzeichnisse im Inhaltsverzeichnis, einzeilige Verzeichnisse; ICLR-Fassungen für LoRA/DeBERTa, RoFormer- und InstructUIE-Zitate, CoNLL-DOI; Abb. 2.3/2.4 als TikZ/pgfplots; Gleitobjekt-Parameter; konkrete Anhangsverweise; Tabellenformalia (Tab. 5.3/5.5/5.7); Listing A.1 als Float; Tab. A.8-Caption. Build 107 Seiten, 0 Overfull, 0 offene Verweise.

- Nachts (28./29.10., Commit 892035d): Nachinferenz abgeschlossen und eingebaut. Ergebnis: Mikro-F₁ mit 1 024 Tokens 0.8B 94,27, 4B 95,11, 27B 95,64 (je +1,1 bis +1,2), keine Antwort bleibt abgeschnitten, 594/604 Präfixe reproduziert; 27B liegt 0,18 hinter DeBERTa-large (KI 0,02–0,33) und 0,14 vor DeBERTa-base (KI −0,02–0,30, nicht unterscheidbar). Neue Tab. 5.4 (`generated/tab-limit-multinerd`), 1 024-Zeilen in Tab. 5.9; Text in 5.2, 5.5, 5.7.1, 5.7.2, 5.7.5, 6.1–6.3, Abstract. Ergebnisse lokal unter `ba-ner/results/multinerd_output_limit_1024/`.

- 30.10. früh (Commits 532853d, bb786f2): CrossNER-1024 abgeschlossen (3 Seeds × 5 Domänen, je 9 h auf H100-SLT) und eingebaut: gepoolter F₁ des 27B 71,7 → 87,1 (+15,4; Recall 60,4 → 85,5, Precision unverändert), keine Antwort erreicht 1 024 Tokens; das LLM liegt damit vor DeBERTa-large (85,0) in allen fünf Domänen (unterscheidbar in vier); Transferverlust gegen die MultiNERD-Referenz (PER/ORG/LOC) 12,4 (LLM) vs. 14,5 (Encoder). Tab. 5.9 (CrossNER-Sensitivität) aus `generated/tab-limit`; Text 5.6, 5.7.4, 6.1, 6.2, Abstract. Strukturpunkte der Gutachten umgesetzt: FF2-Ressourcen in 5.7.3, 6.1 verdichtet, 6.2 nach Wirkung geordnet (je Punkt betroffene FF und Richtung/Betrag), Empfehlungssatz aus 5.4 und Zhan-Vergleich aus 5.5 in 5.7 verschoben. Build 109 Seiten, 0 Overfull. Ergebnisse lokal unter `ba-ner/results/crossner_output_limit_1024/`.

## Offene Punkte (Pflicht vor Abgabe)

1. ~~Experimente einarbeiten~~ erledigt (30.10.). Noch zu prüfen: `aggregate_results.py` laufen lassen (erzeugt `tab-limit-multinerd.tex`; `tab-limit.tex` sobald `comparison.yaml` vollständig; Zweig mit synthetischen Daten geprüft). Dann CrossNER-1024 (Tab. 5.8 durch `generated/tab-limit` ersetzen; 4.6, 5.6, 5.7.1, 6.1–6.3) und MultiNERD-Nachinferenz (5.2, 5.5, 5.7.1 Hochrechnung → Messwert, 5.7.5, 6.1–6.3, Abstract-Satz, Tab. 4.4 Hardware, Tab. A.1 Zeilen mit Commit ff91342).
2. **Kapitel 5 und 6 in eigene Sprache bringen** (Nutzeraufgabe; Stilmuster „nicht X, sondern Y“ und Redundanz 5.7.4/6.1 reduzieren).
3. Datensatzstatistik Tab. 4.3 und Kürzungszahlen (48/11) auf dem Cluster nachrechnen; Lizenzen/Modellkarten und Literaturzahlen prüfen (Rechercheagent läuft).
4. README/Manifest-Kommentare, die Greedy-Zero-Shot behaupten, berichtigen.
5. Frontmatter: Titelwortlaut, Überschrift „Abstract“ vs. „Zusammenfassung“, KI-Angabe bestätigen, Datum/Unterschrift bei Abgabe.
6. Betreuer-Feedback (Entwurf bis 8.11. bzw. 11.11. versenden), externes Korrekturlesen, Umfang (106 Seiten) mit Fachgebiet klären.

Weitere offene Punkte aus den Gutachten: Paragraph-Rahmen in Kap. 6 auflösen und Redundanz 5.3↔5.7.3 (Teil des Umschreibens in eigener Sprache); Kap. 3 Suchstrategie (Nutzer: Datenbanken, Suchbegriffe, Stichtag), GNER einordnen oder Ausschluss begründen, Unterabschnitte in 3.1/3.2, Redundanz zu Kap. 4; Kap. 2 Abb. 2.2-Caption, Tupelnotation Abb. 2.1, 2.7 kürzen; Repository-Fundort (Nutzer: URL/DOI oder Zugangsregel) in A.1 und 4.4.1; Tab. A.3/A.5/A.7; Bib: Preprint-Kennzeichnung, ungenutzte Einträge, Keraghel-Fassung. **Cluster-Regel:** Cluster-Checkout erst bewegen (Bundle/Pull), wenn alle Jobs 3673475–3673478 und 3673515–3673523 gestartet sind (Frozen-Commit-Prüfung).

## Bewertung (28.10. abends, nach vier Zweitgutachten in `thesis/review/`; 1–10, 10 = abgabefertig)

| Teil | Note | Hauptgründe |
|---|---|---|
| Frontmatter | 7 | Abstract fertig bis auf den Satz zum Ausgabelimit; Deckblatt ohne Datum/Institut; Erklärung nicht 1:1 Fak.-IV-Vorlage; Vorspann arabisch paginiert |
| 1 Einleitung | 8 | sauber; Doppelungen 1.2/1.3, DFKI-Absatz, „vier Fragen“ vs. zwei Lücken in 3.5 |
| 2 Grundlagen | 8 | drei Sachfehler korrigiert (28.10.); offen: RoPE-Primärquelle, Typkürzel-Glosse, Abb. 2.3/2.4, 2.7 kürzen |
| 3 Verwandte Arbeiten | 8 | zwei Überzeichnungen korrigiert; offen: Suchstrategie, InstructUIE/GNER, Redundanz zu Kap. 4, Unterabschnitte in 3.2 |
| 4 Methodik | 8 | Widersprüche zu Kap. 3 korrigiert; offen: Begründung Ausgabelimit 256, Checkpoint-Wahl auf 200 Sätzen, Modellwahl, Zero-Shot-Konfundierung, Länge |
| 5 Ergebnisse und Diskussion | 6 | sechs Tabellenwidersprüche korrigiert; offen: Messwerte statt Hochrechnung, Trennung Darstellung/Interpretation, FF2 in 5.7.3, Redundanz zu Kap. 6, Tabellen-/Abbildungsformalia, eigene Sprache |
| 6 Fazit und Ausblick | 7 | Bootstrap und vier Präzisionsfehler korrigiert; offen: Messwerte, Redundanz zu 5.7, Limitationen nach Wirkung ordnen, Paragraph-Rahmen auflösen, eigene Sprache |
| Anhang | 8 | offen: Repository-Fundort (URL/DOI), Listing A.1 ungebrochen, Labels aus dem Haupttext, Tab. A.8/A.3/A.5/A.7 |
| Literatur | 8 | offen: ICLR-Fassungen für LoRA/DeBERTa, Preprint-Kennzeichnung, CoNLL-DOI, ungenutzte Einträge |
| **Gesamt** | **7** | Zur 1,0: Messwerte einbauen, Struktur (Redundanz, Limitationen, Darstellung/Interpretation), Begründungslücken Kap. 3/4, Formalia, eigene Sprache in 5/6, Betreuer-Schleife |

Detaillierte Mängellisten mit Zeilennummern (Stand 99d9a56) und Stilstellen für das Umschreiben: `thesis/review/01_…` bis `04_…`.

## Zeitplan (Kurzform; Details im Plan)

- **A (Abgabe 25./26.11.):** bis 1.11. Experimente laufen, Text-Korrekturen (erledigt); 2.–8.11. Experimente einarbeiten, Abstract final, **8.11. Gesamtentwurf an Betreuer**, 6.11. Attest-Entscheidung; 9.–15.11. Kap. 1–4 gegenlesen, Formalien klären; 16.–22.11. Feedback einarbeiten, Korrekturlesen; 23.–25.11. finaler Build, Abgabe.
- **B (Attest, Abgabe 8./10.12.):** gleiche ersten zwölf Tage, **11.11. Entwurf an Betreuer**; 9.–22.11. eigene Sprache in Ruhe, optionale Zero-Shot-Greedy-Kontrollläufe; 23.–29.11. Feedback, zweite Runde bis 27.11.; 30.11.–6.12. zweites Feedback, finaler Build; 7./8.12. Abgabe.

## Zentrale Befunde (Kurzfassung, Stand 256-Token-Hauptauswertung)

- Mikro-F$_1$ (Mittel ± SD über Seeds 42/123/456): DeBERTa-v3-base 95,50 ± 0,04; DeBERTa-v3-large 95,82 ± 0,03; Qwen3.5-0.8B/4B/27B QLoRA 93,14 ± 0,10 / 93,97 ± 0,05 / 94,48 ± 0,08; Zero-Shot 6,17 / 34,93 / 69,41 (Einzelläufe).
- Encoder vorn (1,34 Punkte, Bootstrap-KI 0,99–1,68); Vorsprung fast ausschließlich im Recall; der größte Teil des Recall-Rückstands liegt in den 65 am 256-Token-Limit abgeschnittenen Antworten (2,6 % der Referenzentitäten). Makro-F$_1$ gleichauf (86,7 vs. 86,9; KI schließt 0 ein).
- Effizienz: Latenz 8,9/16,5 ms (Encoder) gegenüber 1 876 bis 5 505 ms (QLoRA); Speicher 0,7/1,7 GB gegenüber 0,9 bis 18,7 GB; Training 0,4/1,2 h gegenüber 12,1/62,9/266 h; Testinferenz 3–5 min gegenüber 8,6/21,4/25,2 h.
- CrossNER (PER/ORG/LOC): DeBERTa-large 85,0, Qwen-27B 71,7; Vorstudie mit 1024 Tokens hebt den Recall auf ~90 (vollständiger Lauf läuft).
