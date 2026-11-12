# Änderungsliste zur gekürzten Fassung (12.11.2026)

Ausgangsstand: Commit `4281263`, gesichert als Branch `fassung-lang-2026-11-12` und als PDF `thesis/main_long12112026.pdf` (118 Seiten). Die gekürzte Fassung ist `thesis/main.pdf` (114 Seiten nach der ersten Runde, 111 nach der zweiten; die zweite Runde steht in Abschnitt 7).

In der ersten Runde (Abschnitte 1 bis 6) wurden nur die im Auftrag genannten Passagen geändert. Unverändert sind `generated/`, `scripts/`, `references.bib`, `main.tex`, Deckblatt, Eigenständigkeitserklärung, Kapitel 1 und alle Tabellen und Abbildungen außer Tabelle 3.1 und der entfernten Abbildung 2.4. Im Anhang ist ein Absatz hinzugekommen (aus 4.4.3 verschobene Längenangaben), gestrichen wurde dort nichts.

## 1. Umfang alt und neu

| Teil | Seiten alt | Seiten neu | Wörter alt | Wörter neu |
|---|---:|---:|---:|---:|
| Abstract | 1 | 1 | 358 | 363 |
| 1 Einleitung | 5 | 5 | 1 550 | 1 550 |
| 2 Grundlagen | 15 | 14 | 4 140 | 3 770 |
| 3 Verwandte Arbeiten | 14 | 13 | 4 122 | 3 945 |
| 4 Methodik | 23 | 23 | 5 673 | 5 551 |
| 5 Ergebnisse und Diskussion | 22 | 20 | 5 703 | 5 271 |
| 6 Fazit und Ausblick | 6 | 6 | 1 728 | 1 602 |
| **Haupttext (Kapitel 1 bis 6)** | **85** | **81** | **22 916** | **21 689** |
| Literaturverzeichnis | 9 | 9 | | |
| Anhang | 15 | 15 | 1 786 | 1 845 |
| **PDF gesamt** | **118** | **114** | | |

Wörter nach `scripts/check/words.py` (Fließtext ohne Tabellen und Abbildungen).

## 2. Abstract

Länge und alle Zahlen sind gleich geblieben; der Abstract steht weiter auf einer Seite. Geändert sind der Einstieg (erst die Aufgabe, dann die beiden Wege), die Satzlänge (im Mittel 18 Wörter, kein Satz über 40) und der Übergang zu den Empfehlungen.

**Alt:**

> Named Entity Recognition (NER) lässt sich durch Token-Klassifikation mit Encodern oder durch strukturierte Textgenerierung mit Large Language Models (LLMs) lösen. Diese Arbeit vergleicht DeBERTa-v3-base und -large mit vollständigem Fine-Tuning sowie Qwen3.5-0.8B, -4B und -27B im Zero-Shot- und QLoRA-Regime auf dem englischen Teil von MultiNERD mit 15 Entitätstypen und automatisch erzeugter Referenzannotation. Die fünf trainierten Konfigurationen werden mit jeweils drei Seeds ausgewertet. Erfasst werden strikte Metriken auf Entitätsebene, die Formtreue der JSON-Ausgaben, Trainingsdauer, Inferenzlatenz und maximal allokierter GPU-Inferenzspeicher. Ein Transfertest auf CrossNER ergänzt den Vergleich.
>
> DeBERTa-v3-large erreicht einen Mikro-F1 von 95,8 Punkten, Qwen3.5-27B im QLoRA-Regime zunächst 94,5. Das Limit von 256 Ausgabetokens schneidet jedoch lange Antworten ab, die als leere Vorhersagen gewertet werden. Die selektive Nachinferenz mit 1 024 Tokens erhöht den Mikro-F1 des LLM auf 95,6 und verringert den Abstand von 1,34 auf 0,18 Punkte (gepaartes 95-%-Bootstrap-Intervall: 0,02 bis 0,33). Im Makro-F1 wird gegenüber dem großen Encoder kein eindeutiger Unterschied nachgewiesen. Auf CrossNER mit den gemeinsamen Typen PER, ORG und LOC kehrt sich die Rangfolge bei höherem Limit um: Das LLM erreicht 87,1 Punkte gegenüber 85,0 für den Encoder; das Intervall der gepoolten Differenz beträgt 1,47 bis 2,80 Punkte.
>
> Die Inferenzlatenz der Encoder beträgt auf MultiNERD 9 bis 17 ms je Satz bei 0,7 bis 1,7 GiB Inferenzspeicher. Die QLoRA-Läufe mit 256 Ausgabetokens benötigen 1,9 bis 5,5 s und bis zu 18,7 GiB; ein Vergleich auf gleicher Hardware ergibt für Qwen3.5-0.8B gegenüber DeBERTa-v3-base einen Latenzfaktor von etwa 210. Die Zero-Shot-Konfigurationen erreichen 6,2 bis 69,4 F1-Punkte. Ihre Prompt- und Decoding-Wahl wurde durch Testbeobachtungen beeinflusst; da sich die Regime über das Training hinaus in Prompt, Decoding und numerischer Präzision unterscheiden, ist kein isolierter Fine-Tuning-Effekt bestimmbar.
>
> Auf MultiNERD ist damit unter jeder Latenz- und Speichergrenze ein Encoder die beste gemessene Wahl; das LLM ist nur im Transfer auf CrossNER und erst ab rund 13 s je Satz vorzuziehen. Der Qualitätsvergleich hängt jedoch wesentlich von Ausgabelimit, Datensatz und der Typabbildung auf PER, ORG und LOC ab. Die Arbeit zeigt so sowohl den Ressourcenunterschied der konkreten Pipelines als auch die Bedeutung ausreichender Ausgabelimits für die Bewertung generativer NER. Eine allgemeine Überlegenheit einer Modellklasse wird aus den untersuchten Konfigurationen nicht abgeleitet.

**Neu:**

> Named Entity Recognition (NER) findet benannte Entitäten wie Personen, Orte oder Organisationen in Texten. Encoder ordnen dazu jedem Token ein Label zu, Large Language Models (LLMs) erzeugen die Entitäten als strukturierten Text. Diese Arbeit vergleicht beide Wege: DeBERTa-v3-base und -large mit vollständigem Fine-Tuning sowie Qwen3.5-0.8B, -4B und -27B im Zero-Shot- und QLoRA-Regime auf dem englischen Teil von MultiNERD mit 15 Entitätstypen und automatisch erzeugter Referenzannotation. Die fünf trainierten Konfigurationen werden mit je drei Seeds ausgewertet. Erfasst werden strikte Metriken auf Entitätsebene, die Formtreue der JSON-Ausgaben, Trainingsdauer, Inferenzlatenz und maximal allokierter GPU-Inferenzspeicher; hinzu kommt ein Transfertest auf CrossNER.
>
> DeBERTa-v3-large erreicht einen Mikro-F1 von 95,8 Punkten, Qwen3.5-27B im QLoRA-Regime zunächst 94,5. Doch das Limit von 256 Ausgabetokens schneidet lange Antworten ab, die als leere Vorhersagen zählen. Die selektive Nachinferenz mit 1 024 Tokens hebt das LLM auf 95,6 und verringert den Abstand von 1,34 auf 0,18 Punkte (gepaartes 95-%-Bootstrap-Intervall: 0,02 bis 0,33). Im Makro-F1 wird gegenüber dem großen Encoder kein eindeutiger Unterschied nachgewiesen. Auf CrossNER mit den gemeinsamen Typen PER, ORG und LOC kehrt sich die Rangfolge bei höherem Limit um: Das LLM erreicht 87,1 Punkte, der Encoder 85,0 (Intervall der gepoolten Differenz: 1,47 bis 2,80 Punkte).
>
> Die Encoder benötigen auf MultiNERD 9 bis 17 ms je Satz und 0,7 bis 1,7 GiB Inferenzspeicher, die QLoRA-Läufe mit 256 Ausgabetokens 1,9 bis 5,5 s und bis zu 18,7 GiB. Auf gleicher Hardware ergibt sich für Qwen3.5-0.8B gegenüber DeBERTa-v3-base ein Latenzfaktor von etwa 210. Die Zero-Shot-Konfigurationen erreichen 6,2 bis 69,4 F1-Punkte. Ihre Prompt- und Decoding-Wahl wurde durch Testbeobachtungen beeinflusst; da sich die Regime über das Training hinaus in Prompt, Decoding und numerischer Präzision unterscheiden, ist kein isolierter Fine-Tuning-Effekt bestimmbar.
>
> Für die Praxis heißt das: Auf MultiNERD ist unter jeder Latenz- und Speichergrenze ein Encoder die beste gemessene Wahl. Das LLM ist nur im Transfer auf CrossNER und erst ab rund 13 s je Satz vorzuziehen. Der Qualitätsvergleich hängt jedoch wesentlich von Ausgabelimit, Datensatz und der Typabbildung auf PER, ORG und LOC ab. Die Arbeit zeigt damit zweierlei: den Ressourcenunterschied der konkreten Pipelines und die Bedeutung ausreichender Ausgabelimits für die Bewertung generativer NER. Eine allgemeine Überlegenheit einer Modellklasse wird aus den untersuchten Konfigurationen nicht abgeleitet.

## 3. Änderungen je Datei und Abschnitt

### `chapters/02_grundlagen.tex`

| Abschnitt | Entfallen oder geändert | Begründung |
|---|---|---|
| 2.3 Transformer-Architektur | Entfallen: Vergleich mit rekurrenten Netzen, Attention-Gleichung (2.1) mit Skalierung durch √d_k, Einzelheiten der Köpfe (Dimension, Rückprojektion, Beispiele), Sinus-Positionskodierung und Rotationskodierung (RoPE), Residualverbindung und Layer-Normalisierung, technische Umsetzung der Maske (−∞), Beispiel „Washington“, Hinweis auf Decoder mit Klassifikationskopf (steht in 3.3), Qualitätsangabe zu Grouped-Query-Attention, Delta-Regel und Gate von Gated DeltaNet. Behalten: Aufbau aus Encoder und Decoder samt Attention über die Encoder-Ausgabe, Self-Attention mit Query, Key und Value, Köpfe, Positionskodierung (ein Satz), Feed-Forward-Netz, kausale Maske, Encoder-only, Decoder-only, Encoder-Decoder, Zwischenspeichern von Keys und Values, Grouped-Query-Attention, lineare und hybride Attention. | Auftrag b. Behalten ist, was 2.4 bis 2.6 und die Kapitel 3 und 4 verwenden (Adapterplatzierung in 4.4.5, Architektur in 4.2.2). Abbildung 2.2 ist geblieben; der Text füllt jetzt knapp eine Seite, mit Abbildung rund anderthalb. |
| 2.7 Modellgröße und Parameteranzahl | Entfallen: Exponenten nach Kaplan et al., Abbildung 2.4 mit Verweis, Absatz zu Hoffmann et al. Behalten: die zwei Einleitungssätze, die den Begriff Skalierungsgesetz erklären, und der letzte Absatz mit den zwei Grenzen der Übertragbarkeit. | Auftrag c. Die zwei Einleitungssätze sind nötig, weil der letzte Absatz sonst mit „diese Befunde“ ins Leere verweist; 3.4 und 5.7.3 verweisen weiter auf 2.7. |

### `chapters/03_verwandte_arbeiten.tex`

| Abschnitt | Entfallen oder geändert | Begründung |
|---|---|---|
| 3.2.1, Einzelbeschreibungen | Semin et al.: Versuchsaufbau in einen Satz gezogen, Preprint-Hinweis entfallen (steht in der Kapiteleinleitung). Zhan et al.: Aufzählung der fünf Formate entfallen; Befund (Inline-Formate am besten, rund 92 gegenüber rund 54 Punkten) und Begründung der Autoren bleiben. Ravikumar et al.: Formatliste entfallen; Datensatz, Metrik und Befund bleiben. Dagdelen et al.: Modellnamen und Tokenbedarf von JSON entfallen; 99 % parsebar bleibt. Lu et al.: zwei Sätze zu einem; Faktor 1,76 bis 10,22 bleibt. Der Absatz zur Einordnung der eigenen Ausgabestrategie ist unverändert. | Auftrag d. Jede Arbeit steht weiter mit ihrem zentralen Befund und ihrer Fundstelle. |
| 3.2.2 | GPT-NER: Markersymbole und Satz zu verschachtelten Datensätzen entfallen. Xie et al.: syntaktische Hinweise und Mehrheitsvotum entfallen. PromptNER: Zugewinne zusammengezogen (rund vier auf CoNLL-2003 und FewNERD, rund neun auf GENIA); Art der Begründungen entfallen. | Auftrag d. |
| 3.2.3 | UniversalNER: zwei Sätze zum überwachten Setting zu einem. GoLLIE: Zusatz „auf nicht im Training enthaltenen Datensätzen“ und Vergleich mit der Baseline auf MultiNERD entfallen; 55,3, CrossNER-Spanne, 76,0, acht von 15 Typen und 32 908 Testbeispiele bleiben. GNER: Herkunft der Vergleichswerte entfallen (steht in Tabelle 3.1). | Auftrag d. |
| Tabelle 3.1 | Alle 19 Zeilen erhalten. Spaltenbreiten neu verteilt, Zeilenabstände zwischen den Einträgen entfernt, fünf Zellen gekürzt: Bogdanov („keine Latenz- oder Speichermessung“ entfallen), Wang („Tagger-Retrieval“), Sainz und Zhan („keine Inferenz“ entfallen), Ding („GPU-Stunden (Training)“), Zhan („teilweise“, in der Legende erklärt). | Auftrag d. Die Tabelle belegt statt rund 85 % noch rund 75 % der Seite. |

### `chapters/04_methodik.tex`

| Abschnitt | Entfallen oder geändert | Begründung |
|---|---|---|
| 4.1 | Satz zur Deaktivierung des Thinking-Modus entfallen. | Steht in Tabelle 4.3 (erste Zeile) und in 4.4.3; dort jetzt mit Verweis auf Tabelle 4.3. |
| 4.3.3 | Entfallen: Epochenzahlen in Klammern (zwei, drei) und Halbsatz zur Inferenzpräzision nach Durchsatzmessung. Der Verweis auf Tabelle 4.3 steht jetzt am Anfang des Absatzes. Kernsätze zur Testnutzung unverändert. | Beide Angaben stehen in Tabelle 4.3, die Präzisionswahl zusätzlich in 6.2. |
| 4.4.3 | Längengrenze: Median, 99. Perzentil, Maximum, Überschreitung von 512 Tokens sowie die Zahlen für Trainings- und Validierungssplit in den Anhang (A.2.3) verschoben. Geblieben: Limit ohne Längenanalyse gesetzt, 55 von 16 454 Referenzantworten über 256 Tokens, keine über 1 024, Begründbarkeit ohne Testsplit. | Auftrag e. |
| 4.5.3 | Vier Sätze zu den Fehlerklassen zu einer Aufzählung mit Quellen gekürzt. | Die Definitionen stehen in Tabelle 4.7 und in 4.4.6. |
| 4.6 | Beschreibung der Auswertung des Neulaufs auf einen Satz mit Verweis auf Tabelle 5.7 gekürzt. | Inhalt steht in Tabelle 5.7 und ihrer Legende. |
| 4.7 | Einleitungssatz verweist jetzt ausdrücklich auf Abschnitt 6.2. Tabelle 4.9 unverändert. | Auftrag g. |

### `chapters/05_ergebnisse_und_diskussion.tex`

| Abschnitt | Entfallen oder geändert | Begründung |
|---|---|---|
| 5.1.1 | Im Text entfallen: 95,5 und 95,8 (Tabelle 5.1), 34,9 (Tabelle 5.1), Abstände von DeBERTa-v3-base zu den drei QLoRA-Modellen (Tabelle 5.9). Alle Bootstrap-Intervalle bleiben. | Auftrag f. |
| 5.1.2 | Einzelne Referenzzahlen der fünf seltenen Typen durch „8 bis 41“ mit Verweis auf Tabelle 4.2 ersetzt. | Auftrag f. |
| 5.2 | Text-Abweichungen: „45, 12 und 4“ zu „höchstens 45“; „3 397 Elemente“ zu „Tausende Elemente“; Recall- und Precision-Änderung der Nachinferenz ohne Einzelwerte. | Werte stehen in den Tabellen 5.2 und 5.3. |
| 5.3 | „95,5 Punkte“ entfallen. | Tabelle 5.1. |
| 5.4 | Trainingsdauern als Spannen (0,4 bzw. 1,2 h; 12,1 bis 266 h); Zahlen begonnener Elemente entfallen. | Tabellen 5.4 und 5.2. |
| 5.5 | Unverändert. | Fehleranalyse war ausgenommen. |
| 5.6 | Im Text entfallen: 14,5 und 26,8 (Tabelle 5.6), 12,9 s und 14,7 s (Tabelle 5.7). Die Differenz von 12,2 Punkten und alle Intervallaussagen bleiben. | Auftrag f. |
| 5.7.1 | Entfallen: Wiederholung zu Recall-Rückstand und Auslassungen (jetzt ein Satz mit Verweis auf 5.2 und 5.5), Spanne „25 bis 87 Punkte“ (steht in 5.1.1 als Einzelwerte), Annotationswerte nach Tedeschi und Navigli (stehen in 6.2), Wiederholung der Beispiele aus 5.5. Absatz zur Datenkontamination gestrafft. Geblieben: Kernsatz mit 1,34 und 0,18 samt Intervallen, Tabelle 5.8, alle Literaturvergleiche, die Vergleiche bei 1 024 Tokens, die Span-Auszählung. | Auftrag f. |
| 5.7.2 | Entfallen: Wiederholung der Zero-Shot-Befunde aus 5.2 und 5.1.2, Satz zur Nachverarbeitung (steht in 4.4.6). Einstieg gestrafft. | Auftrag f. |
| 5.7.4 | Speicherspanne im ersten Absatz entfallen. | Steht in 5.4 und in den Schwellen zwei Absätze weiter. |
| 5.7.5 | Entfallen: Aussagen zu Seeds, bedingten Intervallen, Hauptvergleichen, Mehrfachvergleichen, Testnutzung, Silberreferenz, Hardware und Checkpoint-Auswahl; dafür ein Verweis auf 6.2. Die Intervalle bei 1 024 Tokens und auf CrossNER stehen in 5.7.1 und 5.6. Geblieben: Seed-Streuung, Vorläufe, Lesart von Tabelle 5.9 (sieben Vergleiche, Makro-Intervalle, Intervallbreiten), fehlende Relevanzschwelle. | Auftrag g. Jeder entfallene Punkt steht in 6.2. |

### `chapters/06_fazit_und_ausblick.tex`

| Abschnitt | Entfallen oder geändert | Begründung |
|---|---|---|
| 6.1, FF1 | Entfallen: 95,6 (Tabelle 5.8), Beispieltypen je Modellklasse, Elementzahlen 3 930 und 3 397 mit dem Satz zu den Zero-Shot-Formatfehlern. Makro-Aussage in einen Satz gezogen. E1 und E3 unverändert. | Auftrag h. |
| 6.1, FF2 | Ressourcen als Spannen in zwei Sätzen statt Einzelwerten; entfallen: 62,9 h des 4B-Modells und 50,7 GiB des Zero-Shot-Laufs. E2 steht jetzt hier, der Faktor 210 nur noch einmal. | Auftrag h. |
| 6.1, FF3 | Entfallen: Schwellenwerte der Encoder (stehen in FF2 und 5.7.4; die Zuordnung base/large bleibt ohne Zahlen), 25,7 ms des Encoders auf CrossNER, Hinweis auf die ungeprüfte Prompt-Erweiterung (steht in 5.7.4). | Auftrag h. |
| 6.2, 6.3 | Unverändert. | |

### `chapters/anhang.tex`

| Abschnitt | Geändert | Begründung |
|---|---|---|
| A.2.3 | Drei Sätze mit den Längenverteilungen der Referenzantworten ergänzt, wörtlich aus 4.4.3 übernommen. | Auftrag e (Verschiebung in den Anhang). |

## 4. Folgen im Dokument

- Abbildung 2.4 und Gleichung 2.1 gibt es nicht mehr; die übrigen Abbildungsnummern sind gleich geblieben, die Gleichungen in Kapitel 2 rücken um eine Nummer vor.
- Alle Tabellennummern sind gleich geblieben.
- Das Literaturverzeichnis enthält einen Eintrag weniger (Su et al., RoFormer), weil er nur in 2.3 zitiert war. `references.bib` ist unverändert.
- `figures/kaplan_scaling.tex` und `.pdf` liegen weiter im Repo, werden aber nicht mehr eingebunden.

## 5. Prüfungen

- **Build:** `latexmk` gemäß `.latexmkrc`, Exitcode 0, 114 Seiten, keine Overfull- oder Underfull-Meldungen, keine undefinierten Verweise oder Zitate, keine LaTeX-Warnungen.
- **Verweise:** Alle `\ref` zeigen auf vorhandene Labels. Entfallen sind nur die Labels `eq:attention` und `fig:skalierungsgesetz`; auf beide verweist nichts mehr.
- **Zahlen in den Quelltexten:** Jede Zahl in den sieben geänderten Dateien kam schon im alten Stand vor.
- **Zahlen im PDF (alt gegen neu, Textauszug beider PDFs):** Keine Zahl steht nur in der neuen Fassung. Ganz entfallen sind 0,050, 0,076, 0,095, 5,4, 8,8 und 400 (Skalierungsgesetze in 2.7), 568, 127063 und 10.1016 (Literatureintrag RoFormer) sowie drei Seitenzahlen des Anhangs.
- **Abstract gegen Tabellen 5.1, 5.8 und 5.9 (dazu 5.4 und 5.7):** 95,8 (95,82), 94,5 (94,48), 95,6 (95,64), 1,34, 0,18, 0,02 bis 0,33, 87,1 (87,12), 85,0 (84,99), 1,47 bis 2,80, 6,2 (6,17), 69,4 (69,41), 9 und 17 ms (8,9 und 16,5), 0,7 und 1,7 GiB, 1,9 und 5,5 s (1 876 und 5 505 ms), 18,7 GiB, Faktor 210 (1 876 / 8,9 = 210,8), 13 s (12,9). Alle stimmen überein.
- **Sprache:** `terms.sh` ohne Treffer; kein neuer Satz über 50 Wörter.
- **Skripttests:** 10 von 10 bestanden; `generated/` und `scripts/` ohne Änderung.
- **Gegenlesung:** Ein unabhängiger Prüfdurchlauf über den gesamten Diff fand keine verbotene Streichung; jede Limitation, auf die 5.7.5 jetzt verweist, steht in 6.2. Sieben Hinweise wurden eingearbeitet (Einführung von Encoder und Decoder in 2.3, Datensatz und Metrik bei Ravikumar et al., Baseline und Fundstellen bei GoLLIE, Trainingsdatenanteil bei PromptNER, eindeutiger Bezug in 5.7.1, Minutenangaben in 5.4, Zuordnung base/large in 6.1).

## 6. Seitenanfänge der Abschnitte (alt → neu)

| Abschnitt | alt | neu |
|---|---:|---:|
| 2.3 Transformer-Architektur | 9 | 9 |
| 2.4 Encoder-basierte Modelle | 11 | 10 |
| 2.7 Modellgröße | 17 | 16 |
| 2.8 Datensätze | 18 | 16 |
| 3 Verwandte Arbeiten | 21 | 20 |
| 3.2 LLM-basierte Ansätze | 24 | 23 |
| 3.3 PEFT-Vergleiche | 28 | 26 |
| 3.5 Forschungslücke | 31 | 29 |
| 4 Methodik | 35 | 33 |
| 4.4 Versuchsaufbau | 42 | 40 |
| 4.5 Evaluationskonzept | 49 | 47 |
| 4.7 Validität | 57 | 55 |
| 5 Ergebnisse und Diskussion | 58 | 56 |
| 5.7 Diskussion | 71 | 69 |
| 6 Fazit und Ausblick | 80 | 76 |
| 6.2 Limitationen | 82 | 78 |
| Literaturverzeichnis | 86 | 82 |
| Anhang | 95 | 91 |

## 7. Zweite Runde (12.11., nach Freigabe des Plans)

Vorgaben des Verfassers: rund 77 Seiten Haupttext; aus Kapitel 4 und 5 darf keine wichtige Angabe verschwinden; Absatzüberschriften von der Umbruchregel ausnehmen; Kontrolllauf als Sensitivitätsanalyse einbauen. Die Abschnitte 1 bis 6 oben beschreiben die erste Runde und sind unverändert.

### 7.1 Umfang nach der zweiten Runde

| Teil | lang | Runde 1 | Runde 2 | Wörter Runde 2 |
|---|---:|---:|---:|---:|
| 1 Einleitung | 5 | 5 | 5 | 1 550 |
| 2 Grundlagen | 15 | 14 | 13 | 3 545 |
| 3 Verwandte Arbeiten | 14 | 13 | 12 | 3 672 |
| 4 Methodik | 23 | 23 | 22 | 5 527 |
| 5 Ergebnisse und Diskussion | 22 | 20 | 20 | 5 403 |
| 6 Fazit und Ausblick | 6 | 6 | 5 | 1 641 |
| **Haupttext** | **85** | **81** | **77** | **21 338** |
| Literaturverzeichnis | 9 | 9 | 9 | |
| Anhang | 15 | 15 | 16 | 2 027 |
| **PDF gesamt** | **118** | **114** | **111** | |

### 7.2 Umbruchregel (`main.tex`)

Nummerierte Überschriften rücken weiter auf die nächste Seite, wenn sie im unteren Seitendrittel beginnen würden. Absatzüberschriften (`\paragraph`) laufen jetzt frei: Die titlesec-Haken `\sectionbreak` bis `\paragraphbreak` setzen `\bottomtitlespace` je Ebene, ein negativer Wert schaltet die Prüfung ab. Wirkung: Kapitel 6 eine Seite weniger.

### 7.3 Kontrollstichprobe (neu, Sensitivitätsanalyse)

| Stelle | Änderung |
|---|---|
| 4.5.5 | Die fünf Kontrollsätze des Vorlaufs sind durch die Kontrollstichprobe ersetzt (je Lauf 1 000 zufällig gezogene Testsätze mit parsebarer Antwort, 1 024 Tokens, H100). |
| Tabelle 4.3 | Zeile zu den Analysen mit 1 024 Tokens nennt jetzt auch die Kontrollstichprobe. |
| Tabelle 4.4 | Neue Zeile für die Kontrollstichprobe. Legende berichtigt: Die Nachinferenz lief auch für das 4B-Modell auf einem anderen GPU-Typ als der Hauptlauf (vorher war nur das 27B-Modell genannt). |
| 5.2 | Ergebnis: 8 988 von 9 000 Antworten zeichengleich, keine am Limit, F1 der Stichprobe je Lauf um höchstens 0,1 Punkte verändert; Abweichungen je Größe genannt. Weil auch das 0.8B-Modell ohne GPU-Wechsel Abweichungen zeigt, nennt der Text den GPU-Typ nicht mehr als alleinige mögliche Ursache (der alte Satz zu den zehn Präfix-Abweichungen tat das). |
| 5.4, 5.7.3, 5.7.4 | Geschätzte Latenzen bei 1 024 Tokens auf dem GPU-Typ der Encoder (1,7, 2,3 und 4,6 s; das 193-, 262- und 512-Fache von DeBERTa-v3-base; 2,7-fach vom 0.8B- zum 27B-Modell). Die Empfehlung zu FF3 ändert sich nicht. |
| 6.1, 6.2 | Je ein Satz: Für eine Pipeline mit 1 024 Tokens liegt nur diese Schätzung vor; die Stichprobe ersetzt keinen vollständigen Neulauf. |
| Anhang | A.1 (Tabellenzeile, Jobs, Datum), A.3 (Commit), neuer Abschnitt A.2.12 (Ziehung und Schätzverfahren), neue Tabelle A.8 aus `generated/tab-app-kontrolle.tex`. |
| `scripts/aggregate_results.py` | `analyse_control_sample` prüft die Zusammenfassung des Clusters gegen die Laufdateien und rechnet die F1-Werte nach; fünf neue Tests. |

Abstract und alle Hauptzahlen bei 256 Tokens sind unverändert.

### 7.4 Kapitel 4 und 5: gestrichene Doppelungen

| Abschnitt | Entfallen | Wo es weiter steht |
|---|---|---|
| Einleitung Kapitel 5 | Wiederholung von strikter Metrik, Seeds und Bootstrap-Beschreibung | 4.5.1; die Lesart der Intervalle und die Rundungsregel bleiben in der Einleitung |
| Einleitung Kapitel 4 | Aufzählung dessen, was der Anhang enthält | 4.4.1 nennt die Tabellen A.1, A.3 und A.4 |
| 4.5.2 | Hochrechnung der Schrittrate als Plausibilitätsprüfung | Fußnote a von Tabelle 5.4 |
| 4.6 | Satzzahlen je Domäne; ausgeschriebener GPU-Typ | Tabelle 5.6; Tabelle 4.4 |
| 4.5.5, 4.6 | Doppelte Verweise zusammengezogen | |
| 5.7.3 | Abnehmende Zugewinne, Vergleich über die Familien mit Parameterfaktoren | 5.3 |

In den Anhang verschoben (zwei Stellen, wie abgesprochen): die Sekundenabweichung zwischen Trainer-Protokoll und Slurm-Abrechnung (4.5.2 nach A.1; im Kapitel steht weiter, dass beide Messungen nur um Sekunden bis gut zwei Minuten auseinanderliegen) und die Einzelheiten, in denen die Zero-Shot-Sampling-Einstellung von der Modellkarte abweicht (standen schon in A.2.5; im Kapitel bleiben die Parameterwerte und die Tatsache der Abweichung).

### 7.5 Kapitel 2 und 3

| Abschnitt | Entfallen oder gekürzt | Begründung |
|---|---|---|
| 2.5 | Verstärkungslernen im Nachtraining, 397-Mrd.-Modell, Begriff Prefill, Einschränkung zum Constrained Decoding in Kurzform, Anforderungen der Open-Source-AI-Definition gestrafft, vier Vorwärtsverweise zusammengezogen | Begriffe kommen später nicht vor; das 397-Mrd.-Modell steht in 1.1, Constrained Decoding ausführlich in 4.4.4 |
| 2.6 | GPT-3-Speicherzahlen, Taxonomie der PEFT-Verfahren, Rechenbeispiel zum Rang, Nebensätze zu Lernrate und Rundungsfehler | Beispiele ohne spätere Verwendung; alle Quellen bleiben zitiert |
| 3 Einleitung | Kapitelübersicht in einen Satz gezogen | |
| 3.1.2, 3.1.3 | U-RaNER: Makro-F1 des englischen Tracks und Retrieval-Einzelheiten; Keraghel et al.: Einzelwerte auf WNUT2017; GLiNER: Bewertungsverfahren, Einordnung von GoLLIE-7B, Hinweis auf CPUs; NuNER: Vergleichsaufbau gestrafft | Einzelbeschreibungen; zentrale Befunde und Fundstellen bleiben |
| 3.1.1, 3.1 Schluss | Satz zu den MultiNERD-Werten anderer Arbeiten (jetzt Verweis auf 3.5); Satz zu fehlenden Referenzwerten und Silberreferenz | steht in 3.5 und 4.3.1 |
| 3.2.3, 3.2.4 | GoLLIE: Testumfang (jetzt Verweis auf 3.5; dort stehen weiter 32 908 Beispiele und die Angabe zu Standardsplits), Trainingsstunden je Größe; UniversalNER: „Ressourcen nicht gemessen“ | steht in 3.5 bzw. 3.2.4 |
| 3.3 | Vorschau im Einleitungssatz; Zhan et al.: Annahmevermerk (steht im Literatureintrag); Obeidat et al.: Social-Media-Korpus, Angleichung des LoRA-Rangs (steht mit Zahlen in 3.4), Entitätslänge, Ursache der Latenz (steht in 3.5); Lu et al.: Satz zur Modellgröße; Schlusssatz gekürzt | Doppelungen innerhalb des Kapitels |
| 3.5 | Vergleich mit Einschränkungen anderer Arbeiten; Schlusssatz zur Kontamination | Doppelungen |
| Tabelle 3.1 | Drei Zeilen niedriger (Typenzahl bei Tedeschi und Navigli steht in der letzten Zeile, Retrieval bei GPT-NER in die Modellspalte); steht jetzt mit Text auf einer Seite | |

### 7.6 Prüfungen der zweiten Runde

- Build ohne Fehler, Overfull-, Underfull- oder sonstige Warnungen; 111 Seiten; Abstract weiter auf Seite ii; Inhaltsverzeichnis weiter drei Seiten.
- `numbers.py --diff`: Alle neuen Zahlen der Kapitel sind in `summary.json` belegt; im Anhang bleiben nur Jobnummern und ein Commit als Kennungen.
- Keine weitere Quelle ist aus dem Literaturverzeichnis gefallen; alle Verweise zeigen auf vorhandene Labels.
- `terms.sh` ohne Treffer, kein Satz über 50 Wörter, 15 von 15 Skripttests, Aggregation zweimal byte-identisch.
- Gegenlesung: Ein unabhängiger Prüfdurchlauf über den Diff der zweiten Runde bestätigte alle Zahlen der Kontrollstichprobe gegen `summary.json` und fand keinen Informationsverlust in Kapitel 4 und 5. Neun kleinere Hinweise sind eingearbeitet (Ursachenangabe in 5.2, Bezeichnung „parsebare Antworten“, Zusatz „in den Hauptläufen“ in 5.7.4, zwei Bezüge, drei wiederhergestellte Angaben in 2.5, 3.4 und 3.5).

## 8. Offene Punkte

1. **Abschnitt 2.3:** Mit Abbildung 2.2 bleibt der Abschnitt bei rund anderthalb Seiten.
2. **Vollständiger Neulauf:** Die Kontrollstichprobe ersetzt ihn nicht. Ob er noch nötig ist, hängt von der Rückmeldung des Betreuers ab.
3. **Ungenutzte Dateien:** `figures/kaplan_scaling.tex` und `.pdf` liegen weiter im Repo, werden aber nicht mehr eingebunden.
