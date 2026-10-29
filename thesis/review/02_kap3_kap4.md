# Zweitgutachten Kapitel 3 und 4 (28.10.2026, Stand 99d9a56)

Geprüft: `chapters/03_verwandte_arbeiten.tex` (741 Zeilen), `chapters/04_methodik.tex` (1318 Zeilen), `references.bib`, Code/Configs in `ba-ner/`. Alle Methodik-Stichproben gegen den Code stimmen (Prompt-Listing wortgleich, Parsing-Kette, seqeval-Modus, Zero-Shot-Decoding am Tag, LoRA-Hyperparameter, Modulzahlen, Double-Quant-Fehlbenennung, Checkpoint-Auswahl, Encoder-Einstellungen, CrossNER-Typabbildung, Nachinferenz-Auswahlregel, Tab. 4.3-Summen). Alle 46 Bib-Schlüssel existieren; keine `\PH{}`-Reste; keine Overfull-Boxen.

## 1. Kapitel 3 – Note 8/10
**Stark:** explizite Einschlusskriterien, stringente Gliederung, kritische Synthesen je Abschnitt, Einordnung der eigenen Ausgabestrategie, Kontamination offen diskutiert, Tab. 3.1 korrekt gelesen, aktuell. **Schwächer:** keine Suchstrategie; zwei Überzeichnungen (korrigiert); InstructUIE nicht zitiert; 3.2 ohne Zwischenüberschriften; Redundanz zu Kap. 4.

## 2. Kapitel 4 – Note 8/10
**Stark:** Reproduzierbarkeit auf sehr hohem Niveau; Parser-Regeln vollständig; Metrik/Bootstrap/Fehlerklassen definiert; Effizienzmessung präzise abgegrenzt; ehrlich (Vorläufe, Testnutzung, Early Stopping, Double-Quant-Bug, DeltaNet-Schichten, 27B-Zuteilungen, Tab. 4.9). **Schwächer:** Ausgabelimit 256, Checkpoint-Auswahl auf 200 Sätzen und Wahl der Qwen-Größen unbegründet; zwei Widersprüche zu Kap. 3 (korrigiert); Zero-Shot-Decoding mit Prompt-Änderung konfundiert; 22 Seiten.

## 3. Mängel nach Schwere

### A. Inhaltliche Fehler / Widersprüche
- **A1** `03:472–476` „Die vier Arbeiten konvergieren“ – Lu et al. finden bleibenden Rückstand. (korrigiert)
- **A2** `03:487–489` „einzige Größenklasse“ vs. „3 bis 9 Mrd.“. (korrigiert)
- **A3** `04:62–64` „Encoder besitzen kein Zero-Shot-Pendant“ vs. GLiNER in 3.1. (korrigiert)
- **A4** `04:287–289` „Anschlussfähigkeit an die Literatur“ bei fehlenden Referenzwerten. (korrigiert)

### B. Argumentationslücken (offen)
- **B1** `04:602–603, 1135–1137` Ausgabelimit 256 nirgends begründet; wichtigste unbegründete Entscheidung. Vorschlag: Verteilung der Zielantwortlängen im Trainingssplit (Anteil > 256 Tokens) plus Kostenerwägung; Zeile in Tab. 4.9.
- **B2** `04:641–648` Checkpoint-Auswahl auf den ersten 200 Validierungssätzen: Begründung (Generierungsdauer) und Verzerrungshinweis fehlen; Zeile in Tab. 4.9.
- **B3** `04:609–617` Sampling (T = 1,0) mit Greedy-Vorlauf des 0.8B begründet, aber Prompt zeitgleich geändert → konfundiert; Übertragung auf 4B/27B unbegründet; Quelle der Parameter fehlt.
- **B4** `04:195–200` Modellwahl Qwen3.5 und Größen 0.8B/4B/27B nicht begründet (Kriterien: dichte Größenleiter, gemeinsamer Tokenizer/Trainingsrezept, Budget, Abdeckung unter/über Literaturbereich – steht in 3.4 Z. 542–545).
- **B5** `03:15–23` Einschlusskriterien ohne Suchstrategie (Quellen, Suchbegriffe, Stichtag, Schneeballsuche).
- **B6** `03:319–325, 344` InstructUIE genannt, nicht zitiert; GNER (Ding et al. 2024) fehlt oder Ausschluss begründen.
- **B7** `04:1138–1142` Auswahlregel „nicht parsebar“ als Proxy für „abgeschnitten“ explizit machen; Z. 1147 „anderen GPU-Knoten“ → „GPU-Typ“.
- **B8** `04:298–304` warum der manuell geprüfte Teildatensatz nicht als Gold-Kontrolle genutzt wird.
- **B9** `04:1247–1249` Ergebnisaussage im Methodikkapitel → „wird in 5.6 geprüft“.

### C. Formalia (offen)
- **C1** `04:319–321` doppeltes „geladen“.
- **C2** `04:669–670` manueller Autorname + `\cite` statt `\textcite`.
- **C3** `references.bib` hu2021lora, he2020deberta, he2021debertav3 als arXiv statt ICLR.
- **C4** `04:1274–1318` Tab. 4.9 allein auf Float-Seite (PDF S. 67).
- **C5** `04:103–104` doppeltes Label (`tab:experiment-matrix`, `tab:groessenklassen`).
- **C6** `04:37, 66, 124, 151` `\subsection*` nur in 4.1 → nummerieren oder `\paragraph`.
- **C7** `04:266–268, 1119` Kaplan/Hoffmann als Beleg für S/M/L-Einteilung dekorativ.
- **C8** `03:421` „5 biomedizinische“ → „fünf“.
- **C9** `04:636–638` 48/11 gekürzte Beispiele – auf dem Cluster bestätigt (28.10.).

### D. Stil / Sprache (offen)
- **D1** Sätze > 50 Wörter: `03:58–63, 510–515`, `04:849–859` (~80 Wörter, als Aufzählung), `04:924–930`.
- **D2** `04:1209–1211` „würden zum Abbruch führen“ kryptisch.
- **D3** `04:348` „entfielen“ → „würden entfallen“.
- **D4** „Entity-Level“ vs. „auf Entitätsebene“; „token-level NER“; „geprompte LLMs“ (Jargon).
- **D5** Redundanz Kap. 3 ↔ 4: `03:536–553` ↔ `04:257–268, 1116–1126`; `03:718–734` ↔ `04:136–149`/Tab. 4.9; „kein Suchlauf“ dreimal.
- **D6** 3.2 (Z. 176–377) und 3.1 in Unterabschnitte gliedern.
- **D7** Anhang-Kandidaten Kap. 4: Tab. 4.2 Vision/MTP-Parameter (Z. 222–255), Softwareversionen (Z. 464–469, 506–508), 27B-Zuteilungschronik (Z. 972–983), 25-Satz-Vorstudie (Z. 1250–1255); Vorläufe in 4.1 bündeln.

## 4. Was zur 10 fehlt
**Kapitel 3:** Suchstrategie; InstructUIE/GNER; Redundanzen zu Kap. 4 streichen; 3.1/3.2 strukturieren; Tab. 3.1 Spalte „Größen variiert“; ICLR-Fassungen.
**Kapitel 4:** Ausgabelimit 256 begründen (+Tab. 4.9); Checkpoint-Auswahl begründen (+Tab. 4.9); Zero-Shot-Konfundierung benennen; Modellwahl begründen; kürzen/entzerren (D7, C4), Kleinigkeiten (C1, C2, D1–D3).
