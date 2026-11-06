# Methodische Korrekturen – 6. November 2026

Die Änderungen setzen die methodischen Punkte aus `05_gesamtanalyse_2026-11-06.md` und der anschließenden Besprechung um. Sie betreffen die Auswertung gespeicherter Vorhersagen und die Argumentation der Thesis. Historische Trainings- und Inferenzartefakte sowie ihre Konfigurationen wurden nicht verändert. Es wurden keine neuen Trainings- oder Modellinferenzläufe gestartet.

## Umgesetzt

| Punkt | Korrektur |
|---|---|
| SD-Heuristik | Die Regel „Differenz größer als Summe der Standardabweichungen“ einschließlich `sd_sum` und `distinguishable` wurde aus Aggregation und aktuellem Manuskript entfernt. Seed-SD bleibt eine deskriptive Angabe. |
| Unsicherheitsinterpretation | Gepaarte Satz-Bootstrap-Intervalle sind die Grundlage ausgewählter Differenzvergleiche. Bedingtheit auf vorhandene Modelle, feste Seeds, fehlende Dokumentcluster und fehlende Korrektur für Mehrfachvergleiche werden explizit genannt. Ein Intervall mit null belegt keine Äquivalenz. |
| CrossNER | Neue, nach Domäne geschichtete Bootstrap-Auswertung für die Differenz Qwen-27B minus DeBERTa-large bei 256 und 1.024 Tokens, integriert in `aggregate_results.py`, `summary.json` und die neue Tabelle `tab-crossner-ci.tex`. |
| Testnutzung | Testgestützte Zero-Shot-Entwicklung wird als explorativ eingeordnet. Mögliche Selektionsverzerrung wird nicht als quantitativ bekannt dargestellt; Durchsatzmessung und leistungsbezogene Auswahl werden unterschieden. Neue Seeds stellen Testunabhängigkeit nicht wieder her. |
| Versuchsablauf | Die Erhöhung der 27B-Epochenzahl von zwei auf drei wird im Ablauf genannt. Eine nicht belegte historische Entscheidungsbegründung wurde nicht ergänzt. |
| Zero-Shot/QLoRA | Unterschiede werden als Vergleich vollständiger Konfigurationen beschrieben; kein isolierter Fine-Tuning-Effekt bei gleichzeitig veränderten Prompts, Decoding-Verfahren und Präzisionsregimen. Keine Behauptung einer „oberen Abschätzung“ oder vernachlässigbarer unbekannter Sampling-Streuung. |
| Checkpoint-Auswahl | Behauptete Folgenlosigkeit der Auswahl anhand der ersten 200 Validierungssätze entfernt. Eine mögliche Änderung der Checkpoint-Wahl unter anderer Validierung bleibt offen. |
| Ausgabelimit | Sensitivitätsanalysen bestimmen die zentrale Schlussfolgerung in Diskussion, Fazit und Abstract mit. Der verbleibende MultiNERD-Abstand wird nicht der Architektur kausal zugeschrieben. Selektive Nachinferenz wird von einer vollständigen Wiederholung unterschieden. |
| Ressourcen | Aussagen an Hardware, Batchgröße, Limit und Messumfang gebunden. Keine garantierten GPU-Mindestgrößen, allgemeinen 27B-Trainingszeiten oder ungemessenen Batching-Vorteile. Keine Kombination der 1.024er-Qualität mit unpassenden 256er-Ressourcen zu einer neuen Pareto-Aussage. Eigene Speicherwerte korrekt als GiB bezeichnet. |
| Weitere Interpretationen | Typbezogene Rangfolgen ohne Intervalle als deskriptiv eingeordnet. Niedrige Zero-Shot-Werte entkräften mögliche Vortrainingskontamination nicht. Erfolgreiche Änderung des Typinventars per Prompt bleibt ungetestet. |
| Konsistenz | Kapitel 1, 4–6, Abstract, README und Status angepasst; Cawley/Talbot (Selektionsverzerrung) und Lakens (Äquivalenz) als methodische Quellen ergänzt. 4B-Zero-Shot: 51 % direkt leere Arrays von 61 % leeren Vorhersagen nach Validierung unterschieden. |

## Neue CrossNER-Auswertung

2.000 Bootstrap-Replikate, Zufallssaat 42. Innerhalb jeder Domäne werden so viele Sätze mit Zurücklegen gezogen wie im jeweiligen Testsplit. Alle Modelle, Seeds und Ausgabelimits verwenden dieselbe Stichprobe. Für den gepoolten Mikro-F₁ werden zuerst die Entitätszähler über Domänen summiert, anschließend die F₁-Werte je Seed berechnet und gemittelt. Satz-IDs, Reihenfolge, Tokens, Referenzlabels und Sequenzlängen werden vor der Paarung geprüft. Punktschätzungen werden gegen die bestehenden Aggregate abgeglichen.

Differenz **Qwen-27B mit 1.024 Tokens minus DeBERTa-large**, F₁-Punkte:

| Domäne | Differenz | 95-%-Intervall |
|---|---:|---:|
| Künstliche Intelligenz | +0,72 | [−1,27; +2,71] |
| Literatur | +0,90 | [−0,37; +2,26] |
| Musik | +2,99 | [+1,40; +4,62] |
| Politik | +1,40 | [+0,41; +2,42] |
| Naturwissenschaften | +4,26 | [+2,28; +6,32] |
| Gepoolt | +2,13 | [+1,47; +2,80] |

Damit schließen drei domänenweise Intervalle null aus. Die vorherige SD-Regel hatte vier Unterschiede ausgewiesen. Die Vergleiche bleiben explorativ und ohne Korrektur für Mehrfachvergleiche; die gepoolte Aussage gilt für die feste Zusammensetzung dieser fünf Domänen.

## Prüfung

- Vollständige Aggregation aus lokalen Vorhersageartefakten erfolgreich. Alle bisherigen Gruppenwerte, MultiNERD-Bootstrap-Intervalle, Ressourcenwerte und CrossNER-Aggregate stimmen mit dem vor der Überarbeitung reproduzierten Stand überein. Änderungen bestehender generierter Dateien betreffen `summary.json` (neue Intervalle, entfernte Heuristik) und `tab-effizienz.tex` (GiB statt GB); eine neue Tabelle kommt hinzu.
- Fünf Tests in `thesis/scripts/test_aggregate_results.py` bestanden: korrektes Pooling über Domänen, Mittel der Seed-F₁ statt Pooling über Seeds, blockgrößenunabhängige Reproduzierbarkeit, Null-Differenz identischer gepaarter Vorhersagen, Ablehnung unpassender Referenzen/Reihenfolgen/IDs/Längen und strikte Auswertung der drei gemeinsamen Typen.
- Aufruf: `/Users/luajosman/miniforge3/bin/python3 -m unittest discover -s thesis/scripts -p 'test_*.py'`.
- Vollständiger PDF-Build mit dem vorhandenen TeX-Docker-Image erfolgreich: **110 Seiten, keine LaTeX-Fehler, keine undefinierten Zitate/Verweise, keine Overfull-Boxen**. Abstract und Seite mit der neuen CrossNER-Tabelle visuell geprüft. Die Beschriftung der Ressourcenabbildung benennt nun GiB, die logarithmischen x-Achsen und das zugehörige 256-Token-Limit.
- `git diff --check` ohne Befund.

## Historisch verbleibende Grenzen

Die frühere Testnutzung, die nicht zufällige 200er-Validierung, fehlende Wiederholungen der Zero-Shot-Inferenz, unterschiedliche Hardware und die tatsächlich ausgeführte QLoRA-Variante lassen sich mit dieser Überarbeitung nicht rückwirkend ändern. Sie sind jetzt einschließlich ihrer Konsequenzen dokumentiert. Eine unabhängige Bestätigung würde eine neue, bis zur Festlegung des Protokolls unberührte Testgrundlage benötigen. Eine allgemeine Überlegenheit einer Modellklasse oder ein isolierter Anpassungseffekt wird nicht beansprucht.

Der Bericht dokumentiert die methodische Überarbeitung. Er ersetzt weder die fachliche Enddurchsicht des Verfassers noch die übrigen offenen Punkte der Gesamtanalyse und enthält keine Notenzusage.
