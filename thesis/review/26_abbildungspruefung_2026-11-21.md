# Prüfbericht Abbildungen (21.11.2026)

Auftrag: alle Abbildungen in Haupttext und Anhang auf inhaltliche Richtigkeit, Genauigkeit, Nachvollziehbarkeit und Darstellung prüfen. Es wurde nichts geändert: keine Abbildung, kein Skript, keine Datendatei, keine LaTeX-Datei, kein PDF. Dieser Bericht ist die einzige neue Datei. Umgesetzt wird erst nach ausdrücklicher Freigabe einzelner Punkte.

## 1. Geprüfte Fassung und verfügbare Dateien

**Fassung.** Branch `lektorat-2026-11-25`, Commit `ea6eede` (21.11., 16:44), Arbeitsverzeichnis ohne offene Änderungen. PDF: `thesis/main.pdf` vom 21.11., 16:44, 112 Seiten.

Während der Prüfung hat eine andere Sitzung drei Commits erstellt (`a6163cb`, `fdbe7f4`, `ea6eede`) und das PDF neu gebaut. Begonnen habe ich mit einem Schnappschuss des Arbeitsverzeichnisses von 16:38 und einem eigenen Build im Scratchpad. Der Abgleich mit dem Endstand ergab: Die sechs `figure`-Umgebungen, alle Verweissätze und die Datendateien sind identisch, und die sechs Abbildungen sind im Schnappschuss-Build und in `thesis/main.pdf` bei 300 dpi pixelgleich, auf denselben Seiten. Der Bericht gilt damit für `ea6eede`.

**Gefundene Abbildungen.** Sechs nummerierte Abbildungen, alle im Haupttext. Der Anhang enthält keine Abbildung (nur Tabellen und Listings). Alle sechs sind Vektorgrafiken, die LaTeX beim Build aus TikZ/pgfplots-Code erzeugt; eingebundene Bilddateien gibt es nur für die beiden Logos auf dem Deckblatt (keine nummerierten Abbildungen, hier nicht inhaltlich geprüft).

| Abb. | Seite (PDF-Seite) | Quelle im Repo | Daten |
|---|---|---|---|
| 2.1 | 7 (16) | `figures/ner_grundprinzip.tex` | keine (Schema) |
| 2.2 | 10 (19) | `figures/vaswani_transformer.tex` | keine (Schema) |
| 2.3 | 14 (23) | `figures/lora_adapter.tex` | keine (Schema) |
| 5.1 | 56 (65) | `chapters/05_…tex`, Z. 195–227 | `generated/fig-typen.dat` |
| 5.2 | 59 (68) | `chapters/05_…tex`, Z. 366–418 | `generated/fig-groesse.dat` |
| 5.3 | 62 (71) | `chapters/05_…tex`, Z. 543–616 | `generated/fig-tradeoff.dat` |

**Skripte und Daten.** `thesis/scripts/aggregate_results.py` (Funktion `fig_data`, Z. 1877–1897) schreibt die drei `.dat`-Dateien aus den Laufartefakten unter `ba-ner/results/multinerd/` (je Lauf `test_predictions.json` und `inference_metrics.yaml`). Diese Artefakte liegen lokal vor.

**Mehrere Versionen.** In `thesis/figures/` liegen zusätzlich vier PDF-Dateien (Stand 9.11.) und `kaplan_scaling.tex`/`.pdf`. Der Build bindet die `.tex`-Dateien ein, nicht die PDFs; die PDFs sind eigenständige Kopien ohne Einfluss auf die Arbeit. `kaplan_scaling` (frühere Abbildung 2.4) wird seit der Kürzung vom 12.11. nirgends mehr eingebunden.

**Was ich prüfen konnte.**

| Prüfung | Stand |
|---|---|
| Visuelle Prüfung im PDF | ja, alle sechs: Seiten bei 130 dpi, Abbildungen zusätzlich als Ausschnitt bei 300 dpi |
| Werte der Diagramme 5.1 bis 5.3 | vollständig nachgerechnet (Abschnitt 2) |
| Schemata 2.1 bis 2.3 gegen die Quellen | Sichtvergleich mit Abb. 1 bei Li et al. und Hu et al. (arXiv-Fassungen, heute geladen) und mit Abb. 1 bei Vaswani et al. (lokale PDF) |
| Messverfahren Latenz und Speicher | Messcode gelesen (`ba-ner/src/encoder/inference.py`, `src/decoder/inference.py`, `src/evaluate/efficiency.py`) |

**Nur eingeschränkt beurteilbar.**

- Latenz und Speicher (Abb. 5.3): Geprüft ist die Übernahme der gespeicherten Kennzahlen in die Abbildung und das Messverfahren im Code. Die Latenzen je Satz sind nicht gespeichert, der Mittelwert lässt sich also nicht aus Einzelmessungen nachrechnen, und nachmessen kann ich nicht.
- Die Zero-Shot-Läufe stammen aus dem Codestand `137bffd`; gelesen habe ich den aktuellen Messcode. Dass beide Stände gleich messen, belegen nur die Felder der Metrikdateien (`latency_measurement: amortized_per_sample`, `inference_batch_size: 1`).
- Dass die lokalen Ergebnisdateien den Cluster-Dateien entsprechen, habe ich heute nicht erneut geprüft.
- Lesbarkeit im Schwarz-Weiß-Druck und bei Farbfehlsichtigkeit habe ich aus Farbwerten und Symbolformen abgeleitet, nicht an einem Ausdruck oder einer Simulation geprüft.

## 2. Zahlenprüfung der Diagramme

Zwei unabhängige Wege, beide ohne Schreibzugriff auf das Repo:

1. **Eigene Nachrechnung aus den Rohvorhersagen.** Ein neu geschriebenes Skript (ohne Code aus `aggregate_results.py`) liest die 18 Dateien `test_predictions.json` (je 16 454 Sätze), bildet strikte IOB2-Spans, zählt TP/FP/FN je Typ und berechnet Mikro-F₁ je Lauf, Seed-Mittel, Stichprobenstandardabweichung und typbezogene F₁-Werte; Latenz und Speicher kommen aus `inference_metrics.yaml`. Ergebnis:
   - Die F₁-Werte aller 18 Läufe stimmen mit den in den Läufen gespeicherten Werten überein.
   - `fig-typen.dat`: 15 Typen × 3 Reihen und alle Support-Werte identisch (0 Abweichungen; Support-Summe 25 645).
   - `fig-groesse.dat` und `fig-tradeoff.dat`: Mikro-F₁, Standardabweichung, mittlere Latenz und Speicher aller acht Konfigurationen identisch auf zwei Nachkommastellen.
   - GPU-Typ je Lauf laut Laufmetadaten: stimmt mit Tabelle 4.4 und Tabelle 5.4 überein.
2. **Neulauf der Aggregation** mit umgeleitetem Ausgabeordner: Alle 23 erzeugten Dateien (3 `.dat`, 19 Tabellen, `summary.json`) sind byteidentisch mit `thesis/generated/`. Die Abbildungsdaten gehören also zum aktuellen Ergebnisstand.

Im Text genannte Werte zu den Abbildungen, mit ungerundeten Werten nachgerechnet, alle zutreffend:

| Stelle | Aussage | Nachrechnung |
|---|---|---|
| 5.1.2 | PER, ORG, LOC stellen 80 % der Testentitäten | 80,3 % |
| 5.1.2 | Abstand der Modellklassen dort höchstens 1,8, bester Encoder zu bestem LLM höchstens 1,4 (LOC) | 1,79 und 1,43 |
| 5.1.2 | ANIM, BIO, FOOD, PLANT: 58,1 bis 79,4 (trainiert), 3,2 bis 45,0 (Zero-Shot) | 58,14 bis 79,42; 3,22 bis 45,04 |
| 5.1.2 | Richtung je Typ (Encoder vorn bei ANIM, FOOD, PLANT, DIS, EVE, PER, LOC; LLM bei INST, MYTH, TIME; MEDIA und ORG gleich) | stimmt für alle 15 Typen |
| 5.3 | Zugewinne 0,3; 0,82; 0,51; Zero-Shot 28,8 und 34,5 | 0,32; 0,82; 0,51; 28,76; 34,48 |
| 5.3 | Parameterfaktoren 5,5-fach, 6-fach, 4-fach, 146-fach | 5,59; 6,39; 4,09; 146,2 |
| 5.4 | Latenzfaktoren 210, 335, 617 | 210,2; 334,6; 616,9 |
| 5.4 | Speicher Zero-Shot zu QLoRA: Faktor 1,7 bis 2,7 | 1,68; 2,35; 2,71 |
| 5.4 | Pareto-Front in beiden Teilbildern nur die Encoder; jedes LLM von DeBERTa-v3-base dominiert; 4B (QLoRA) innerhalb der LLMs nicht dominiert | rechnerisch bestätigt |

Die im rechten Teilbild von Abbildung 5.2 sichtbaren Unterschiede sind durch die gepaarten Bootstrap-Intervalle in Tabelle 5.9 gedeckt (large − base, 4B − 0.8B, 27B − 4B, base − 27B, base − 0.8B, 4B − base: kein Intervall enthält null). Die Abbildung legt bei 256 Tokens keinen Unterschied nahe, den die Auswertung nicht stützt.

## 3. Übersicht je Abbildung

| Abb. | Kap., Seite | Inhalt und Funktion | Geprüfte Grundlage | Zahlenprüfung | Achsen, Einheiten, Legende, Beschriftung | Abgleich mit Text, Tabellen, Quellen | Fehler oder Risiken | Empfehlung | Sicherheit |
|---|---|---|---|---|---|---|---|---|---|
| **2.1** Grundprinzip der Named Entity Recognition | 2.2, S. 7 | Schema der Aufgabe: Tokenfolge, NER, Tripel; führt den einschließenden Endindex ein | TikZ-Quelle, Text 2.2, Li et al. Abb. 1 | entfällt | Beschriftung klar; Beispiel wie Tabelle 2.1 | Tripel und Indexkonvention passen zum Text; Aufbau entspricht Abb. 1 bei Li et al., die Unterschrift nennt nur deren Formalisierung | Quellenangabe ungenau (A-07) | behalten; Quellenangabe präzisieren | hoch |
| **2.2** Transformer-Architektur mit Encoder und Decoder | 2.3, S. 10 | Aufbau des Originalmodells; Grundlage für die drei Modellfamilien | TikZ-Quelle, Text 2.3, Vaswani et al. Abb. 1 | entfällt | „N×" nicht erklärt; zwei Begriffe weichen vom Text ab | Komponenten, Reihenfolge und Datenfluss stimmen mit dem Original überein | nur Kleinigkeiten (G-03) | behalten; kleine Angleichungen | hoch |
| **2.3** Aufbau eines LoRA-Adapters | 2.6, S. 14 | Schema zu Gleichung 2.2; nötig für das QLoRA-Regime | TikZ-Quelle, Gl. 2.2, Hu et al. Gl. 3 und Abb. 1 | entfällt | Dimensionen von A, B, W₀ korrekt | ΔW = BA in Abbildung und Unterschrift, in Gl. 2.2 dagegen ΔW x = (α/r) BAx | formaler Widerspruch (A-06) | behalten; ΔW vereinheitlichen | hoch |
| **5.1** Typbezogene F₁-Werte der drei Kernkonfigurationen | 5.1.2, S. 56 | Typprofil von bestem Encoder, bestem LLM (QLoRA) und Zero-Shot; zeigt das Muster schneller als Tabelle A.9 | `.dat`, Rohvorhersagen, Tabellen A.9, A.10 | alle 45 Werte bestätigt | Achse 0 bis 100, Einheit Punkte, Legende eindeutig; Marker verdeckt und oben abgeschnitten; Bedingung 256 Tokens und Stützung der seltenen Typen fehlen | Text korrekt, aber mehrere Textaussagen sind in der Grafik nicht ablesbar | A-03, A-04, G-06, G-07 | behalten; Marker versetzen, Unterschrift ergänzen | hoch |
| **5.2** Erkennungsleistung in Abhängigkeit von der Parameterzahl | 5.3, S. 59 | Zentrale Evidenz zu FF2: F₁ über Parameterzahl, je Familie und Regime | `.dat`, Rohvorhersagen, Tabellen A.11, 5.3, 5.9, 4.1 | alle 8 Werte und Standardabweichungen bestätigt | Log-Achse korrekt; rechts Ausschnitt 92,5 bis 96,5 nicht als solcher benannt; Fehlerbalken erklärt; 256 Tokens fehlen | Text korrekt; Parameterzahlen wie Tabelle 4.1 | A-01, A-02, G-01, G-09 | behalten; Unterschrift ergänzen | hoch |
| **5.3** Erkennungsleistung gegenüber Inferenzlatenz und GPU-Spitzenspeicher | 5.4, S. 62 | Zentrale Evidenz zu FF2 und FF3: Qualität gegen Latenz und Speicher | `.dat`, `inference_metrics.yaml`, Tabellen 5.4, 4.4, Messcode | alle 16 Werte bestätigt (Übernahme; Messung selbst nicht wiederholbar) | Log-Achsen benannt, ms und GiB korrekt; „Latenz" ohne Angabe Mittelwert; GPU-Typ nicht erkennbar; Legende uneinheitlich gesetzt | Pareto-Aussagen rechnerisch bestätigt; 256 Tokens und Hardwarevorbehalt in der Unterschrift | A-05, G-02 | behalten; GPU-Typ und „mittlere" ergänzen | hoch für Werte, mittel für die Vergleichbarkeit über GPU-Typen |

Zu den in Auftragsabschnitt 8 genannten Themen gibt es **keine Abbildung**: Ausgabelimits 256 und 1 024, selektive Nachinferenz, CrossNER, falsch positive Spans nicht abgebildeter Typen, Bootstrap-Intervalle, Haupt- und explorative Vergleiche. Diese Inhalte stehen nur in Tabellen (5.3, 5.6 bis 5.9, A.8, A.12 bis A.14) und wurden hier nur so weit herangezogen, wie es der Abgleich mit den Abbildungen verlangt.

## 4. Befunde nach Priorität

### Kritisch

Keine. Kein Wert, keine Achse und keine Einheit ist falsch, und keine Abbildung widerspricht einem zentralen Ergebnis.

### Wichtig

**A-01: Abbildung 5.2 nennt die Bedingung „256 Ausgabetokens" nicht.**
1. Fundstelle: Bildunterschrift, `05_…tex` Z. 410–416; S. 59.
2. Beobachtet: Das rechte Teilbild zeigt die QLoRA-Modelle 1,0 bis 2,7 Punkte unter den Encodern. Nach Tabelle 5.3 liegen die QLoRA-Werte mit 1 024 Tokens bei 94,27, 95,11 und 95,64; der Abstand von DeBERTa-v3-large zum 27B-Modell beträgt dann 0,18 Punkte [0,02; 0,33], und gegenüber DeBERTa-v3-base liegt das 27B-Modell 0,14 Punkte vorn (Intervall enthält null). Wer nur die Abbildung liest, sieht einen Abstand, den die Arbeit selbst an anderer Stelle einschränkt.
3. Vergleich: `tab-limit-multinerd.tex`, `tab-ci.tex`; Unterschrift von Abbildung 5.3, die die Bedingung nennt.
4. Status: bestätigt.
5. Vorschlag: In der Unterschrift ergänzen: „LLM-Werte bei 256 Ausgabetokens; mit 1 024 Tokens siehe Tabelle 5.3." Wahlweise zusätzlich die drei Werte der Nachinferenz als hohle Quadrate im rechten Teilbild, in der Legende als Sensitivitätsanalyse gekennzeichnet. Das verlangt eine Erweiterung von `fig_data` und einen Neulauf der Aggregation.
6. Unverändert: alle Datenwerte; die 256-Token-Werte bleiben die Hauptwerte; Werte der Nachinferenz dürfen nicht mit den Hauptwerten verschmelzen (selektiv neu erzeugt, anderer GPU-Typ bei 4B und 27B).
7. Offen: Entscheidung des Verfassers zwischen reiner Ergänzung der Unterschrift und zusätzlichen Markern.

**A-02: Abbildung 5.2, rechter Ausschnitt und Aussagekraft der Fehlerbalken.**
1. Fundstelle: rechtes Teilbild und Unterschrift; S. 59.
2. Beobachtet: Die y-Achse rechts reicht von 92,5 bis 96,5 und trägt keine eigene Beschriftung; die Unterschrift sagt „rechts nur die trainierten", aber nicht, dass der Wertebereich vergrößert ist. Für ein Liniendiagramm ist der Ausschnitt fachlich angemessen, und das linke Teilbild zeigt die volle Skala; es fehlt nur der Hinweis. Die Fehlerbalken zeigen die Standardabweichung über drei Seeds (0,03 bis 0,10 Punkte). Die Bootstrap-Intervalle der Einzelwerte in Tabelle 5.9 sind mit ±0,25 bis ±0,42 Punkten vier- bis achtmal so breit. Die Balken sind korrekt benannt, bilden aber nur die Seed-Streuung ab.
3. Vergleich: `tab-ci.tex` (unterer Block), `fig-groesse.dat`.
4. Status: bestätigt. Irreführend ist die Abbildung nicht, weil die gepaarten Intervalle alle sichtbaren Unterschiede stützen (Abschnitt 2).
5. Vorschlag: Unterschrift: „rechts vergrößerter Ausschnitt (92,5 bis 96,5 Punkte) nur der trainierten Konfigurationen" und „Fehlerbalken: Standardabweichung über drei Seeds; Stichprobenunsicherheit in Tabelle 5.9".
6. Unverändert: Achsenbereiche, Werte, zweiteilige Darstellung.
7. Offen: nichts.

**A-03: Abbildung 5.1, verdeckte und abgeschnittene Marker.**
1. Fundstelle: Diagramm, `05_…tex` Z. 198–221; S. 56.
2. Beobachtet (300 dpi): Der Punkt für DeBERTa-v3-large liegt bei ORG, MEDIA, EVE und VEHI fast vollständig hinter dem Quadrat des QLoRA-Modells, bei DIS, FOOD, PLANT und TIME teilweise. Bei PER und LOC (99,7) schneidet der obere Rahmen (ymax = 100) den Punkt ab. Die im Text beschriebene Richtung der Unterschiede (Encoder vorn bei PER, LOC, EVE; gleich bei MEDIA und ORG) ist in der Grafik deshalb nicht ablesbar.
3. Vergleich: `fig-typen.dat`; Text 5.1.2 (Z. 149–172).
4. Status: bestätigt.
5. Vorschlag: Die drei Reihen je Typ seitlich um wenige Punkt versetzen und die Achse bis etwa 103 führen (Skalenstriche weiter bis 100). Mindestlösung ohne Eingriff in das Diagramm: Hinweis in der Unterschrift, dass sich die Werte der trainierten Konfigurationen bei mehreren Typen überdecken.
6. Unverändert: Werte, Typreihenfolge, Farben und Symbole (gleiches Schema wie in 5.2 und 5.3), Achsenbeginn bei null.
7. Offen: nichts.

**A-04: Abbildung 5.1, keine Kennzeichnung der schwach gestützten Typen.**
1. Fundstelle: Diagramm und Unterschrift (Z. 223–225).
2. Beobachtet: Die größten sichtbaren Abstände zwischen den beiden trainierten Modellen liegen bei Typen mit sehr wenigen Referenzentitäten: INST (12 Entitäten; 82,7 gegenüber 97,2), BIO (8; 79,0 gegenüber 67,5), CEL (41), MYTH (32). Die Standardabweichung über die Seeds beträgt dort nach meiner Nachrechnung bis 7,7 Punkte (BIO, 27B), 4,9 und 4,8 (INST), 4,6 (CEL, Encoder). Der Text nennt diese Unsicherheit (Z. 173–178), Abbildung und Unterschrift nicht. Außerdem fehlen in der Unterschrift „256 Ausgabetokens" und der Hinweis, dass Zero-Shot ein Einzellauf ist.
3. Vergleich: eigene Nachrechnung je Typ und Seed; Tabellen A.9 und A.10; `summary.json` enthält die Standardabweichungen je Typ.
4. Status: bestätigt.
5. Vorschlag: Unterschrift ergänzen, etwa: „BIO, INST, MYTH, VEHI und CEL haben 8 bis 41 Referenzentitäten; LLM-Werte bei 256 Ausgabetokens, Zero-Shot als Einzellauf." Wahlweise Fehlerbalken (Seed-Standardabweichung) für die beiden trainierten Reihen; dafür müsste `fig-typen.dat` zwei Spalten mehr erhalten.
6. Unverändert: Werte; keine Aussage über Signifikanz je Typ, die die Arbeit nicht berechnet hat.
7. Offen: Entscheidung, ob Fehlerbalken gewünscht sind.

**A-05: Abbildung 5.3, GPU-Typ nicht erkennbar und Latenzkennzahl nicht benannt.**
1. Fundstelle: beide Teilbilder und Unterschrift (Z. 610–614); S. 62.
2. Beobachtet: Die Latenzwerte stammen von drei GPU-Typen (H100: beide Encoder, 0.8B QLoRA; A100: 4B QLoRA, 0.8B und 4B Zero-Shot; H100 NVL: 27B in beiden Regimen). Die Unterschrift warnt allgemein, die Abbildung zeigt aber nicht, welcher Punkt zu welcher GPU gehört. Hardwaregleich sind nur Encoder gegenüber 0.8B (QLoRA) sowie Zero-Shot gegenüber QLoRA bei 4B und bei 27B. Die x-Achse heißt „Latenz je Satz", die Unterschrift „Inferenzlatenz"; gemeint ist der Mittelwert, während das 95. Perzentil der LLMs beim 3,4-Fachen des Medians liegt.
3. Vergleich: Spalte `gpu` in `fig-tradeoff.dat` (vorhanden, im Diagramm ungenutzt), Tabellen 5.4 und 4.4, Laufmetadaten, Messcode. Latenz und Speicher werden für Encoder und LLMs gleich gemessen: Batchgröße 1, nur der Modellaufruf, GPU-Synchronisation davor und danach; Speicher als `max_memory_allocated` in MiB, geteilt durch 1 024.
4. Status: bestätigt. Die Aussagen des Textes zur Pareto-Front sind korrekt; der Abstand zwischen Encodern und LLMs beträgt zwei Größenordnungen und hängt nicht am GPU-Typ.
5. Vorschlag: „mittlere Latenz" in Unterschrift oder Achsenbeschriftung. GPU-Typ entweder in der Unterschrift aufzählen oder die Punkte direkt beschriften (Kurzname und GPU, etwa „Q4B, A100"); die Spalten `short` und `gpu` stehen dafür schon in der Datendatei. Direkte Beschriftung würde auch die Legende mit acht Einträgen ersetzen.
6. Unverändert: Werte bei 256 Tokens; keine Vermischung mit den auf der H100 geschätzten Latenzen bei 1 024 Tokens aus Tabelle A.8.
7. Offen: nichts. Eine hardwaregleiche Messung aller Konfigurationen existiert nicht und lässt sich aus den Daten nicht herstellen.

**A-06: Abbildung 2.3 und Gleichung 2.2 definieren ΔW verschieden.**
1. Fundstelle: `figures/lora_adapter.tex` Z. 23, Unterschrift `02_…tex` Z. 446–447, Gleichung Z. 414; S. 14.
2. Beobachtet: Abbildung und Unterschrift schreiben ΔW = BA. Gleichung 2.2 setzt W₀x + ΔW x = W₀x + (α/r) BAx, also ΔW = (α/r) BA.
3. Vergleich: Hu et al., Gl. 3 (h = W₀x + ΔWx = W₀x + BAx) und der Folgesatz, nach dem ΔWx anschließend mit α/r skaliert wird.
4. Status: bestätigt.
5. Vorschlag: Eine Schreibweise wählen. Entweder in Abbildung und Unterschrift „ΔW = (α/r) BA" oder in der Abbildung nur „trainierbarer Low-Rank-Pfad BA, skaliert mit α/r" und in der Unterschrift ΔW streichen.
6. Unverändert: Dimensionen von A, B und W₀, Formel für h, Quellenangabe.
7. Offen: nichts.

**A-07: Abbildung 2.1, Quellenangabe nennt nur die Formalisierung.**
1. Fundstelle: Unterschrift, `02_…tex` Z. 78–79; S. 7.
2. Beobachtet: Der Aufbau (Tokenfolge mit Indizes, Kasten „Named Entity Recognition", Ausgabe als Tripel mit Entitätstext) entspricht Abbildung 1 bei Li et al., mit eigenem Beispielsatz und anderer Leserichtung. Die Unterschrift lautet „in Anlehnung an die Formalisierung bei Li et al.".
3. Vergleich: Li et al., arXiv 1812.09449, S. 2, Fig. 1 (Sichtvergleich).
4. Status: bestätigt für die arXiv-Fassung; die Nummer der Abbildung in der Zeitschriftenfassung habe ich nicht eingesehen.
5. Vorschlag: „Eigene Darstellung in Anlehnung an Li et al. (2022, Abb. 1)".
6. Unverändert: Beispiel, Hinweis auf den einschließenden Endindex.
7. Offen: Abbildungsnummer in der gedruckten Fassung, falls danach zitiert wird.

### Geringfügig

- **G-01, Abbildung 5.2:** Die Dreiecke der Zero-Shot-Reihe sind selbst gestrichelt gezeichnet und wirken im Diagramm und in der Legende gebrochen. Ursache: `dashed` wirkt auch auf den Marker. Abhilfe: `mark options={solid}`.
- **G-02, Abbildung 5.3:** Die Legende ist uneinheitlich gesetzt (verschiedene Abstände zwischen Symbol und Text, dritte Spalte versetzt). Im Latenzbild verdeckt das Quadrat des 27B-Modells das des 4B-Modells weitgehend (4 685 und 5 505 ms). Abhilfe: einheitliche Spaltenabstände; 4B nach 27B zeichnen oder direkt beschriften (siehe A-05).
- **G-03, Abbildung 2.2:** „N×" ist weder in der Unterschrift noch im Text erklärt. Die Abbildung schreibt „Feed-Forward-Netzwerk" und „Encoder-Decoder-Attention", Unterschrift und Text „Feed-Forward-Netz" (Z. 172, 187) und „Encoder-Attention" (Z. 198). Die Beschriftung „Encoder-Ausgabe (K, V)" liegt auf der Ecke des Pfeils; der Abzweig der untersten Residualverbindung fällt mit dem Rahmen des Stapels zusammen.
- **G-04, Abbildung 2.1:** Das Satzzeichen-Token steht mittig statt auf der Grundlinie. Die Tripel verwenden t₁, t₂ statt der im Text definierten Indizes; das folgt Li et al. und ist unkritisch.
- **G-05, Schrift:** Abbildungen 2.1 und 2.2 sind serifenlos gesetzt, 2.3 und alle Diagramme mit Serifen.
- **G-06, Text zu Abbildung 5.1:** „Kernkonfigurationen" kommt nur in 5.1.2 und in der Unterschrift vor und ist nicht definiert. „Über 97,5 Punkten" stimmt (Minimum 97,525), Tabelle A.9 zeigt aber gerundet 97,5; „mindestens 97,5" wäre unangreifbar.
- **G-07, Abbildung 5.1:** Die Typen stehen in den drei Gruppen aus Tabelle A.9 (zuverlässig annotiert; häufig verwechselt; übrige). Die Grafik zeigt die Gruppen nicht; zwei senkrechte Trennlinien würden die Reihenfolge erklären.
- **G-08, Dateien:** `figures/kaplan_scaling.tex` und `.pdf` sowie die vier PDF-Kopien in `figures/` sind versioniert, aber ungenutzt. Kein Fehler; Aufräumen nur auf Wunsch.
- **G-09, Abbildung 5.2:** Die Unterschrift von 5.3 nennt die logarithmische x-Achse, die von 5.2 nicht.

### Keine Änderung nötig

- Alle Datenwerte der Abbildungen 5.1 bis 5.3, die Zuordnung der Reihen, Aggregation (Seed-Mittel, Stichprobenstandardabweichung), Rundung und Einheiten (Punkte, ms, GiB).
- Achsenbeginn bei null in 5.1, 5.2 links und 5.3; logarithmische Achsen für Parameterzahl, Latenz und Speicher; alle Punkte liegen innerhalb der Achsenbereiche.
- Farb- und Symbolschema: DeBERTa blauer Kreis, QLoRA oranges Quadrat, Zero-Shot offenes oranges Dreieck, durchgängig in allen drei Diagrammen. QLoRA und Zero-Shot unterscheiden sich durch Form und Füllung, nicht durch Farbe; Blau und Orange unterscheiden sich deutlich in der Helligkeit.
- Nummern, Kurztitel und Seiten im Abbildungsverzeichnis; alle neun Verweise (sieben im Haupttext, zwei in Anhang A.3) zeigen auf die richtige Abbildung; jede Abbildung steht auf der Seite des ersten Verweises oder der folgenden. Kein Verweis ist offen, keine Abbildung unreferenziert.
- Komponenten, Reihenfolge und Datenfluss in Abbildung 2.2 (gegen Vaswani et al. Abb. 1) und der Aufbau von Abbildung 2.3 (gegen Hu et al. Abb. 1).

## 5. Text und Abbildung: Doppelungen und Funktion

| Abb. | Funktion | Doppelung mit dem Text | Empfehlung |
|---|---|---|---|
| 2.1 | Verständnis der Aufgabe und der Indexkonvention | gering; derselbe Beispielsatz wie Tabelle 2.1, aber anderer Zweck (Spans statt Tags) | behalten |
| 2.2 | Hintergrund; die Arbeit nutzt nur Encoder- und Decoder-only-Modelle | keine; der Text setzt die Abbildung für die drei Familien voraus | behalten |
| 2.3 | erforderlich für das QLoRA-Regime | keine | behalten |
| 5.1 | macht das Typprofil und den Zero-Shot-Abstand sichtbar | gering; der Text nennt Zahlen, die aus Tabelle A.9 stammen und in der Grafik nicht ablesbar sind | behalten; nach A-03 trägt die Grafik die Textaussagen besser |
| 5.2 | zentrale Evidenz zu FF2 | mäßig; die Zugewinne stehen im Text, in Tabelle A.11 und in der Grafik. Der Text ordnet aber ein (abnehmende Zugewinne, Familienvergleich) | behalten, Text unverändert |
| 5.3 | zentrale Evidenz zu FF2 und FF3 | keine; der Text interpretiert (Pareto-Front) | behalten |

Keine Abbildung sollte entfernt oder in den Anhang verschoben werden. Eine zusätzliche Abbildung (etwa der Intervalle aus Tabelle 5.9) empfehle ich nicht: Kapitel 5 endet am Seitenende, jede weitere Abbildung kostet eine Seite.

**Hinweis zum Seitenumbruch.** Jede Ergänzung einer Bildunterschrift in Kapitel 5 kann eine Zeile kosten; Kapitel 5 ist auf die Seite genau gesetzt. Nach einer Umsetzung ist der Umbruch zu prüfen und nötigenfalls an anderer Stelle eine Zeile einzusparen.

## 6. Gesamteinschätzung

- **Sachlich und rechnerisch korrekt?** Ja für die drei Diagramme: Alle dargestellten Werte sind aus den Rohvorhersagen nachgerechnet und stimmen. Die drei Schemata geben ihre Quellen korrekt wieder, mit einer formalen Unstimmigkeit bei ΔW (A-06).
- **Abweichungen zwischen Abbildungen, Text, Tabellen und Daten?** Keine bei den Zahlen. Zwei formale: ΔW in Abbildung 2.3 gegenüber Gleichung 2.2 und die Quellenangabe von Abbildung 2.1.
- **Achsen, Legenden, Einheiten, Unsicherheitsmaße?** Achsen und Einheiten sind korrekt. Es fehlen: die Bedingung „256 Ausgabetokens" in 5.1 und 5.2, der Hinweis auf den vergrößerten Ausschnitt in 5.2, die Angabe „mittlere" Latenz und der GPU-Typ in 5.3. Das einzige dargestellte Unsicherheitsmaß (Seed-Standardabweichung in 5.2) ist korrekt benannt.
- **Interpretation im Text?** Zutreffend und zurückhaltend; keine Textaussage geht über die Daten hinaus. Mehrere Aussagen zu Abbildung 5.1 sind wegen verdeckter Marker in der Grafik nicht zu sehen.
- **Unnötige Doppelung?** Nein.
- **Vor der Abgabe am wichtigsten:** A-01 (256 Tokens in 5.2), A-03 (verdeckte Marker in 5.1), A-05 (GPU-Typ und Mittelwert in 5.3), A-06 (ΔW). Danach A-02, A-04, A-07 und G-01.
- **Nicht zuverlässig prüfbar:** die Latenz- und Speichermessungen selbst (nur Übernahme und Verfahren geprüft) und die Vergleichbarkeit der Latenzen über GPU-Typen; hardwaregleiche Messungen aller Konfigurationen liegen nicht vor.

**Trennung der Sicherheit.** Vollständig verifiziert: Werte, Aggregation, Zuordnung und GPU-Typen der Abbildungen 5.1 bis 5.3; Nummern, Verweise, Seiten. Visuelle Einschätzung: Verdeckungen, Beschnitt, Legendensatz, Unterscheidbarkeit der Symbole. Offene Prüfpunkte: Messwerte für Latenz und Speicher an sich; Abbildungsnummer bei Li et al. in der Zeitschriftenfassung; Druckbild in Schwarz-Weiß.

## 7. Nachvollziehbarkeit

Die Prüfskripte und Zwischenergebnisse liegen im Scratchpad dieser Sitzung und nicht im Repo: Nachrechnung `recompute.py` mit Ergebnis `recomputed.json`, Neulauf der Aggregation mit Protokoll, gerenderte Seiten und Ausschnitte. Der Ordner ist sitzungsgebunden; auf Wunsch lege ich die Nachrechnung dauerhaft ab.

## 8. Nachtrag: Umsetzung vom 21.11.2026 (nach Freigabe)

Geändert wurden nur `chapters/02_grundlagen.tex`, `chapters/05_ergebnisse_und_diskussion.tex` und `figures/lora_adapter.tex`; keine Datendatei, kein Skript. Prüfbuild in einer Kopie: 112 Seiten, keine Warnungen, alle Labels (Kapitel, Abschnitte, Abbildungen, Tabellen, Listings) mit denselben Nummern und Seiten wie bei `ea6eede`; Text geändert nur auf den fünf Abbildungsseiten (PDF-Seiten 16, 23, 65, 68, 71). Anschließend `thesis/main.pdf` neu gebaut (textgleich mit dem Prüfbuild) und zusammen mit diesem Bericht committet.

| Punkt | Umgesetzt | Anmerkung |
|---|---|---|
| A-01 | ja | Abb. 5.2: „LLM-Werte bei 256 Ausgabetokens." in der Unterschrift; weiterhin fünf Zeilen |
| A-02 | teilweise | Abb. 5.2: „rechts als vergrößerter Ausschnitt"; der Verweis auf die Bootstrap-Intervalle ist nicht ergänzt |
| A-03 | ja | Abb. 5.1: Encoder und QLoRA je Typ um ±2,6 pt versetzt, Achse bis 103 (Skalenstriche bis 100); kein Marker mehr verdeckt oder abgeschnitten |
| A-04 | teilweise | Abb. 5.1: „Zero-Shot als Einzellauf; LLM-Werte bei 256 Ausgabetokens"; weiterhin drei Zeilen. Der Hinweis auf die seltenen Typen hätte eine vierte Zeile gekostet und entfällt; keine Fehlerbalken |
| A-05 | teilweise | Abb. 5.3: Achse „Mittlere Latenz je Satz [ms]"; GPU-Typ nicht ergänzt |
| A-06 | ja | Abb. 2.3 und Unterschrift: ΔW = (α/r) BA, passend zu Gleichung 2.2 |
| A-07 | ja | Abb. 2.1: „in Anlehnung an J. Li et al. (2022, Abb. 1)" |
| G-01 | ja | Abb. 5.2: Zero-Shot-Dreiecke durchgezogen |
| G-02 | teilweise | Abb. 5.3, Latenzbild: 4B (QLoRA) zusätzlich obenauf mit weißem Rand gezeichnet. Die reine Zeichenreihenfolge hätte nichts geändert, weil beide Quadrate dieselbe Farbe ohne Rand haben. Legendensatz unverändert |
| G-03 bis G-09 | nein | wie besprochen |

`figures/lora_adapter.pdf` (ungenutzte Kopie) ist aus der geänderten `.tex`-Quelle neu erzeugt (standalone, 12 pt, Latin Modern, Rand 4 pt).
