# Verdichtungs- und Präzisionsaudit Kapitel 1–6 (So 8.11.2026)

**Status:** reiner Prüfbericht. Keine Quelle, kein PDF und nichts im Anhang wurde geändert. Umgesetzt wird erst nach Auswahl.

**Leitlinie:** Ein Vorschlag steht nur hier, wenn Aussage, Begründung, Einschränkung, Quelle und Nachvollziehbarkeit erhalten bleiben und nur Redundanz entfällt. Wo das nicht gegeben ist, steht der Abschnitt unter „Nicht kürzen".

---

## 1 Versions- und Prüfbasis

| Punkt | Befund |
|---|---|
| Git-Stand | HEAD `4b9f602` (8.11.2026, 08:08); `main` = `origin/main` nach `git fetch` |
| Arbeitsbaum | Nur Codex-Dateien weichen ab: `review/README.md` geändert, Berichte 09 und 17 unversioniert. Keine Kapitel-, Präambel- oder Skriptdatei ist geändert. |
| Geprüftes PDF | `thesis/main.pdf`, erzeugt 8.11. 08:10 aus `4b9f602`; keine Quelldatei ist jünger |
| `main_betreuerentwurf_59a1363.pdf` | Stand `59a1363` (8.11. 00:50), also vor der Tabellenumstellung in `4b9f602`. **Nicht aktuell, nicht als Basis verwendet.** |
| Build-Log `build/main.log` | 0 Overfull, 0 Underfull, 0 undefined, 0 „Float too large" |
| Tests | `python -m unittest discover -s scripts -p 'test_*.py'`: 10 Tests, OK |
| Vollständig gelesen | Fließtext von `chapters/01` bis `06`; Floats nur, soweit der Text auf sie Bezug nimmt. Dazu `main.tex` Z. 93–100, `generated/tab-effizienz.tex`, `generated/tab-synthese.tex` und Tabelle 4.3 im Quelltext. |
| Eingeschränkt | `chapters/anhang.tex`: nur Gliederung, Labels, Verweise und Seitenzahlen, wie vorgegeben. Abbildungen nur über Caption und Quelltext. |
| Nicht geprüft | Inhalt des Literaturverzeichnisses, Inhalt des Anhangs |

---

## 2 Seitenumfang

### 2.1 Gesamtzählung

| Größe | Wert | Methode |
|---|---|---|
| Physische Seiten | 116 | `pypdf`: `len(reader.pages)` |
| Gedruckte Seitenlabels | i, i, ii–ix, 1–106 | `pypdf`: `reader.page_labels` |
| Vorspann | 10 physische Seiten (phys. 1–10) | Labels und Seitentext |
| Haupttext Kap. 1–6 | S. 1–82 = 82 Seiten (phys. 11–92) | `build/main.toc` und Labels |
| Literaturverzeichnis | S. 83–91 = 9 Seiten | `build/main.toc` |
| Anhang | S. 92–106 = 15 Seiten | `build/main.toc` |

**116 gegenüber 106:** 106 ist das letzte arabische Seitenlabel, 116 die Zahl der physischen Seiten. Die Differenz sind die 10 Vorspannseiten mit römischen Labels. Deckblatt und Erklärung tragen beide das Label „i". Deshalb reichen die römischen Labels nur bis ix, obwohl es 10 Seiten sind.

**Vorspann im Einzelnen:** Deckblatt (i), Eigenständigkeitserklärung mit KI-Angabe (i), Abstract (ii), Inhaltsverzeichnis (iii–v), Abbildungsverzeichnis (vi), Tabellenverzeichnis (vii), Abkürzungsverzeichnis (viii–ix).

### 2.2 Kapitel

| Kapitel | Seiten | Anzahl | Fließtextwörter | Füllung der letzten Seite |
|---|---|---|---|---|
| 1 Einleitung | 1–5 | 5 | 1 537 | 86 % |
| 2 Grundlagen | 6–20 | 15 | 4 180 | 95 % |
| 3 Verwandte Arbeiten | 21–33 | 13 | 4 109 | 98 % |
| 4 Methodik | 34–56 | 23 | 5 546 | 90 % (mit Tab. 4.9) |
| 5 Ergebnisse und Diskussion | 57–77 | 21 | 5 413 | 99 % |
| 6 Fazit und Ausblick | 78–82 | 5 | 1 558 | 99 % |
| **Haupttext** | 1–82 | 82 | 22 343 | |

**Methode:**
- Wörter mit `scripts/check/words.py`: Fließtext ohne Tabellen, Abbildungen, Listings und Formeln.
- Füllung ist der Anteil der Textfläche (y = 770 bis 71 pt) oberhalb der tiefsten Textzeile, gemessen mit `pypdf`.
- Eine reine Textseite enthält 348 bis 376 extrahierte Wörter (S. 3, 4, 12, 28, 71, 79). Die Schätzungen unten rechnen mit rund 370 Wörtern je Seite.

**Folgerung für alle Seitenschätzungen:** Jedes Kapitel beginnt auf einer neuen Seite. Eine Kürzung spart nur dann eine Seite, wenn die letzte Seite eines Kapitels leer wird. Bei 86 bis 99 % Füllung braucht das je Kapitel etwa 320 bis 370 Wörter. Innerhalb eines Kapitels können sich Floats und durch die Überschriftenregel verschobene Überschriften verlagern. Das ist ohne Build nicht vorhersagbar.

---

## 3 Prüfung der früheren Änderungen

| Änderung | Befund am aktuellen Stand |
|---|---|
| `tab:typen`, `tab:groesse`, `tab:crossner-ci` im Anhang A.3 | nur in `anhang.tex` definiert |
| GPU-Tabelle `tab:hardware` | unverändert in Kapitel 4 |
| Synthese-Tabelle | je zwei Zeilen MultiNERD und CrossNER (256 und 1 024 Tokens) |
| Intervall-Kopf „95-%-…" | in `tab-ci`, `tab-synthese`, `tab-crossner-ci`; die alte Form „95 %-KI" kommt nicht mehr vor |
| Fußnote der Effizienztabelle | verweist auf Anhang A.2 (`app:durchfuehrung`) |
| „Herstellerangabe" | in der Caption der Effizienztabelle 5.4 (Kap. 5, Z. 456) |
| Überschriftenregel | `nobottomtitles*` mit `\bottomtitlespace` = .30 (`main.tex` Z. 93 und 99) |
| Fluss im Anhang | `\FloatBarrier` statt `\clearpage` vor A.2 und A.3 |
| „Kein Informationsverlust" der dritten Kürzungsrunde | nur als Bericht des Prüfagenten belegt, nicht unabhängig nachgeprüft |

Kein Vorschlag unten nimmt eine dieser Änderungen zurück.

---

## 4 Maßstab

- **A – verlustfrei:** reine Doppelung; der Inhalt steht vollständig an der genannten anderen Stelle.
- **B – zu prüfen:** nach meiner Prüfung verlustfrei, berührt aber die Eigenständigkeit eines Abschnitts, eine früher verhandelte Formulierung oder den Lesefluss.
- **C – geringe Wirkung oder Risiko:** möglich, aber der Nutzen ist klein oder das Risiko größer als der Nutzen.
- **P – Präzisionskorrektur:** behebt eine Ungenauigkeit; spart keine oder kaum Wörter.
- **Nicht kürzen:** geprüft und bewusst unverändert.

Wortzahlen sind Schätzungen: gestrichene minus hinzugefügte Wörter im Quelltext. Seiteneffekte sind Schätzungen.

---

## 5 Vorschläge je Kapitel

### Kapitel 1 – Einleitung

#### K1-a · A · Ressourcenmaße doppelt (1.3)
1. **Ort:** Abschnitt 1.3, `01_einleitung.tex` Z. 233–237.
2. **Original:**
   ```tex
   Der Ressourcenbedarf umfasst die Trainingsdauer, die Inferenzlatenz bei
   einzeln verarbeiteten Sätzen und den GPU-Spitzenspeicher während der
   Inferenz (Abschnitt~\ref{sec:effizienzmetriken}).
   Nicht erfasst werden der Speicherbedarf während des Trainings, ein durch
   Batching optimierter Durchsatz sowie Energieverbrauch und Kosten.
   ```
3. **Problem:** Die drei Maße stehen schon in 1.2 (Z. 124–125): „Neben der Erkennungsleistung werden die Trainingsdauer, die Inferenzlatenz und der GPU-Spitzenspeicher während der Inferenz erfasst."
4. **Warum verdichtbar:** Neu sind in 1.3 nur die Bedingung „einzeln verarbeitete Sätze" und die Liste des Nicht-Erfassten.
5. **Vorschlag:**
   ```tex
   Die Inferenzlatenz wird für einzeln verarbeitete Sätze gemessen
   (Abschnitt~\ref{sec:effizienzmetriken}); nicht erfasst werden der
   Speicherbedarf während des Trainings, ein durch Batching optimierter
   Durchsatz sowie Energieverbrauch und Kosten.
   ```
6. **Erhalten bleibt:** Bedingung der Einzelsatzverarbeitung, Verweis auf 4.5.2, vollständige Liste des Nicht-Erfassten.
7. **Risiken:** keine; 1.2 steht eine Seite vorher.
8. **Wörter:** etwa 10.
9. **Seiteneffekt:** keiner; die letzte Seite von Kapitel 1 ist zu 86 % gefüllt.
10. **Dichte:** steigt leicht.

#### P-1 · P · Anhangsbeschreibung unvollständig (1.4)
1. **Ort:** Abschnitt 1.4, Z. 263–265.
2. **Original:**
   ```tex
   Der Anhang dokumentiert die Reproduzierbarkeit der Läufe, enthält die
   Systemnachricht des Zero-Shot-Prompts und ergänzt Tabellen je Seed,
   Entitätstyp und CrossNER-Domäne.
   ```
3. **Problem:** Der Anhang hat inzwischen drei Teile. A.2 „Ergänzungen zur Durchführung" mit dem QLoRA-Prompt (A.2.4) fehlt in der Beschreibung.
4. **Warum ändern:** Präzision, keine Kürzung. Der Fehler liegt in Kapitel 1, nicht im Anhang.
5. **Vorschlag:**
   ```tex
   Der Anhang dokumentiert die Reproduzierbarkeit der Läufe und die
   verwendeten Prompts, ergänzt Details der Durchführung und enthält
   weitere Ergebnistabellen, unter anderem je Seed, Entitätstyp und
   CrossNER-Domäne.
   ```
6. **Erhalten bleibt:** alle bisher genannten Inhalte.
7. **Risiken:** keine; der Anhang selbst bleibt unberührt.
8. **Wörter:** etwa 5 mehr.
9. **Seiteneffekt:** keiner.
10. **Dichte:** gleich; die Genauigkeit steigt.

**Nicht kürzen (Kapitel 1):** QLoRA-Begründung in 1.3 (Z. 195–200), Absatz zur Konfundierung (Z. 218–224), Absätze zu FF1 bis FF3, Beitragsabsatz. Sie tragen Forschungsfragen, Abgrenzung und den Vorbehalt gegen kausale Lesarten.

**Kurzbilanz Kapitel 1:** 5 Seiten. Optionen K1-a und P-1. Potenzial etwa 10 Wörter, P-1 kostet etwa 5. Erwartete Seitenersparnis 0. Keine Risiken. Das Kapitel ist bereits präzise.

---

### Kapitel 2 – Grundlagen

#### K2-a · A · Thinking-Modus zweimal (2.5)
1. **Ort:** Abschnitt 2.5, `02_grundlagen.tex` Z. 405–408 und Z. 430–431.
2. **Original:**
   ```tex
   Neuere Modelle der Qwen-Familie können vor der eigentlichen Antwort einen
   abgegrenzten Block mit Zwischenüberlegungen erzeugen; dieser
   \emph{Thinking-Modus} lässt sich ein- und
   ausschalten~\cite{qwen2025qwen3}.
   ...
   Zu dieser Gruppe gehört die Qwen-Familie von Alibaba; Qwen3 führte den
   Wechsel zwischen Thinking- und Non-Thinking-Modus ein~\cite{qwen2025qwen3}.
   ```
3. **Problem:** Die Umschaltbarkeit wird im selben Abschnitt zweimal mit derselben Quelle eingeführt.
4. **Warum verdichtbar:** Die zweite Stelle fügt nur hinzu, dass Qwen3 die Umschaltung eingeführt hat. Das lässt sich in die erste Stelle aufnehmen.
5. **Vorschlag:**
   ```tex
   Qwen-Modelle ab Qwen3 können vor der eigentlichen Antwort einen
   abgegrenzten Block mit Zwischenüberlegungen erzeugen; dieser
   \emph{Thinking-Modus} lässt sich ein- und
   ausschalten~\cite{qwen2025qwen3}.
   ...
   Zu dieser Gruppe gehört die Qwen-Familie von Alibaba.
   ```
6. **Erhalten bleibt:** Definition, Umschaltbarkeit, Einführung mit Qwen3, Quelle.
7. **Risiken:** keine.
8. **Wörter:** etwa 12.
9. **Seiteneffekt:** keiner; die letzte Seite von Kapitel 2 ist zu 95 % gefüllt.
10. **Dichte:** steigt; „Neuere Modelle" wird zugleich genauer.

#### K2-b · A · CoNLL-Satz wiederholt 2.2 und 2.9 (2.8)
1. **Ort:** Abschnitt 2.8, Z. 643–645.
2. **Original:**
   ```tex
   Seine Daten liegen im IOB1-Schema vor (Abschnitt~\ref{sec:ner}); die
   Auswertung zählt eine Entität nur bei exakter Übereinstimmung als
   korrekt (Abschnitt~\ref{sec:evaluation}).
   ```
3. **Problem:** IOB1 steht in 2.2 (Z. 114–115), die exakte Zählweise mit CoNLL-2003 in 2.9 (Z. 749–754). Beide Stellen werden hier nur verwiesen.
4. **Warum verdichtbar:** Der Satz enthält keine neue Aussage.
5. **Vorschlag:** Satz streichen.
6. **Erhalten bleibt:** beide Fakten an ihren Fachstellen.
7. **Risiken:** keine; CoNLL-2003 wird in dieser Arbeit nicht ausgewertet.
8. **Wörter:** etwa 22.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt.

#### K2-c · A · Lizenz doppelt zu 4.2.2 (2.5)
1. **Ort:** Abschnitt 2.5, Z. 437–442.
2. **Original:**
   ```tex
   Diese drei Modelle werden in der vorliegenden Arbeit verwendet und stehen
   laut ihren Modellkarten unter der
   Apache-2.0-Lizenz~\cite{qwen2026card27b,qwen2026card4b,qwen2026card08b};
   welche Komponenten geladen werden und dass der Thinking-Modus
   deaktiviert ist, beschreiben die Abschnitte~\ref{sec:llm-auswahl}
   und~\ref{sec:llm-pipeline}.
   ```
3. **Problem:** 4.2.2 (Z. 171–176) nennt dieselbe Lizenz mit denselben Quellen und nutzt sie als Auswahlgrund.
4. **Warum verdichtbar:** In 4.2.2 ist die Lizenz Teil der Begründung, in 2.5 nur Wiederholung.
5. **Vorschlag:**
   ```tex
   Diese drei Modelle werden in der vorliegenden Arbeit verwendet;
   welche Komponenten geladen werden und dass der Thinking-Modus
   deaktiviert ist, beschreiben die Abschnitte~\ref{sec:llm-auswahl}
   und~\ref{sec:llm-pipeline}.
   ```
6. **Erhalten bleibt:** Lizenz und Quellen in 4.2.2; die Einordnung als offene Gewichte in 2.5.
7. **Risiken:** keine; die drei Modellkarten bleiben in 4.2.2 zitiert.
8. **Wörter:** etwa 9.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt.

#### K2-d · C · Zwei Nutzungsarten vorweggenommen (2.1)
1. **Ort:** Abschnitt 2.1, Z. 48–51.
2. **Original:**
   ```tex
   Vortrainierte Modelle lassen sich dabei auf zwei Arten nutzen.
   Encoder liefern Repräsentationen, auf denen ein Klassifikator Labels
   vorhersagt (Abschnitt~\ref{sec:encoder_ner}); generative Modelle erzeugen
   die Lösung einer Aufgabe als Text (Abschnitt~\ref{sec:llms}).
   ```
3. **Problem:** 2.3 (Z. 231–239) beschreibt dieselbe Unterscheidung ausführlicher.
4. **Warum verdichtbar:** Es ist eine Vorschau auf 2.3 bis 2.5.
5. **Vorschlag:** Satzpaar streichen.
6. **Erhalten bleibt:** Inhalt in 2.3 bis 2.5.
7. **Risiken:** Die Brücke von der Geschichte der NER zu den beiden verglichenen Modellklassen fällt am Kapitelanfang weg.
8. **Wörter:** etwa 30.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt formal, der Lesefluss leidet.

#### K2-e · C · NSP ohne spätere Verwendung (2.4)
1. **Ort:** Abschnitt 2.4, Z. 298–302.
2. **Original:**
   ```tex
   \textcite{devlin2019bert} trainieren BERT mit zwei Aufgaben auf nicht
   annotiertem Text: Beim \emph{Masked Language Modeling} (MLM) wird ein
   Teil der Tokens maskiert und aus dem beidseitigen Kontext
   rekonstruiert, bei der \emph{Next Sentence Prediction} (NSP) sagt das
   Modell vorher, ob zwei Segmente aufeinander folgen.
   ```
3. **Problem:** NSP wird später nicht mehr gebraucht.
4. **Warum verdichtbar:** Für DeBERTa und die Pipeline spielt NSP keine Rolle.
5. **Vorschlag:**
   ```tex
   \textcite{devlin2019bert} trainieren BERT unter anderem mit
   \emph{Masked Language Modeling} (MLM): Ein Teil der Tokens wird maskiert
   und aus dem beidseitigen Kontext rekonstruiert.
   ```
6. **Erhalten bleibt:** MLM als zentrales Vortrainingsziel.
7. **Risiken:** Die Darstellung von BERT wird unvollständig.
8. **Wörter:** etwa 15.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt kaum; Grundlagenwissen geht verloren.

#### K2-f · C · BIOES/BILOU und seqeval-Schemaliste (2.2 und 2.9, nur gemeinsam)
1. **Ort:** 2.2 Z. 110–111 und 2.9 Z. 774–775.
2. **Original:**
   ```tex
   Erweiterungen wie BIOES oder BILOU markieren zusätzlich das letzte Token
   eines Spans (E bzw. L) und Spans aus einem einzigen Token (S bzw. U).
   ...
   Unterstützt werden IOB1, IOB2, IOE1 und IOE2 sowie im strikten Modus
   IOBES und BILOU~\cite{nakayama2018seqeval}.
   ```
3. **Problem:** Beide Schemata werden in der Arbeit nicht verwendet.
4. **Warum verdichtbar:** Die Pipeline nutzt nur IOB2.
5. **Vorschlag:** Beide Sätze gemeinsam streichen.
6. **Erhalten bleibt:** IOB1 und IOB2 sowie der strikte Modus mit festem Schema.
7. **Risiken:** Nur gemeinsam möglich, sonst stehen in 2.9 unerklärte Schemata. Der Überblick über Kodierungen wird schmaler.
8. **Wörter:** etwa 35.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt leicht; Kontextwissen geht verloren.

#### K2-g · C · Schichtzahl des Originalmodells (2.3)
1. **Ort:** 2.3, Z. 206–208.
2. **Original:** `Training tiefer Stapel; das Originalmodell verwendet sechs Schichten.`
3. **Problem:** Die Angabe wird später nicht genutzt.
4. **Warum verdichtbar:** Sie ist für den Vergleich ohne Bedeutung.
5. **Vorschlag:** `Training tiefer Stapel.`
6. **Erhalten bleibt:** Funktion von Residualverbindung und Normierung.
7. **Risiken:** keine; die Wirkung ist sehr klein.
8. **Wörter:** etwa 5.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt minimal.

#### K2-h · C · Verwendung von CrossNER vorweggenommen (2.8)
1. **Ort:** 2.8, Z. 688–690.
2. **Original:**
   ```tex
   Diese Arbeit verwendet ausschließlich die Testsplits als Transfertest
   und wertet nur die gemeinsamen Grobtypen aus
   (Abschnitt~\ref{sec:crossner}).
   ```
3. **Problem:** 1.2 und 4.6 sagen dasselbe.
4. **Warum verdichtbar:** Es ist eine Methodenaussage im Grundlagenkapitel.
5. **Vorschlag:** Satz streichen.
6. **Erhalten bleibt:** Inhalt in 4.6.
7. **Risiken:** 2.8 nennt für MultiNERD die Rolle in der Arbeit, für CrossNER dann nicht mehr.
8. **Wörter:** etwa 16.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt leicht; die Darstellung wird ungleich.

**Hinweis:** Die Doppelung „hybride Attention, relevant für die Adapter" zwischen 2.3 und 4.2.2 steht als K4-f bei Kapitel 4.

**Nicht kürzen (Kapitel 2):** QLoRA-Komponenten (nötig für die Abweichungen in Tabelle 4.9 und 6.2), Definitionen der Decoding-Verfahren, Modi von seqeval, „F₁ als Übereinstimmung mit der Referenz", Silberreferenz und manuell geprüfte Stichprobe, Abschnitte zu Kaplan und Chinchilla samt ihren zwei Grenzen (Z. 614–624).

**Kurzbilanz Kapitel 2:** 15 Seiten. A-Optionen K2-a bis K2-c mit etwa 43 Wörtern, C-Optionen mit etwa 100 Wörtern. Erwartete Seitenersparnis 0. Die C-Optionen entfernen Grundlagenwissen. Das Kapitel ist weitgehend präzise; es gibt nur lokale Doppelungen innerhalb von 2.x und zu 4.2.2.

---

### Kapitel 3 – Verwandte Arbeiten

#### K3-a · A · MultiNERD-Beschreibung wiederholt 2.8 (3.1.1)
1. **Ort:** Abschnitt 3.1.1, `03_verwandte_arbeiten.tex` Z. 52–58.
2. **Original:**
   ```tex
   \textcite{tedeschi2022multinerd} stellen mit MultiNERD einen Datensatz
   vor, der zehn Sprachen, die Genres Wikipedia und Wikinews sowie
   15 Entitätstypen abdeckt; der englische Teil umfasst rund 164\,000 Sätze.
   Die Annotation erfolgt automatisch über BabelNet-Synsets und ein
   iteratives Bereinigungsverfahren (Abschnitt~\ref{sec:datasets_corpora});
   das dafür auf den Silberdaten trainierte mBERT-Modell mit BiLSTM und CRF
   ist zugleich die einzige in der Arbeit trainierte Architektur.
   ```
3. **Problem:** Sprachen, Genres, 15 Typen, 164 000 Sätze, BabelNet und Bereinigung stehen in 2.8 (Z. 651–671).
4. **Warum verdichtbar:** Neu ist in 3.1.1 nur, dass das Bereinigungsmodell die einzige trainierte Architektur der Originalarbeit ist.
5. **Vorschlag:**
   ```tex
   Den Aufbau des automatisch annotierten Datensatzes MultiNERD
   \cite{tedeschi2022multinerd} beschreibt Abschnitt~\ref{sec:datasets_corpora};
   einzige in der Originalarbeit trainierte Architektur ist das für die
   iterative Bereinigung auf den Silberdaten trainierte mBERT-Modell mit
   BiLSTM und CRF.
   ```
6. **Erhalten bleibt:** Quelle, automatische Annotation, mBERT mit BiLSTM und CRF und seine Rolle, Verweis auf 2.8.
7. **Risiken:** 3.1.1 beginnt ohne Kurzbeschreibung; wer Kapitel 3 einzeln liest, muss zu 2.8 springen.
8. **Wörter:** etwa 27.
9. **Seiteneffekt:** keiner; die letzte Seite von Kapitel 3 ist zu 98 % gefüllt.
10. **Dichte:** steigt deutlich.

#### K3-b · A · Folgerung am Ende von 3.1 wiederholt Begründung, grammatisch schief (3.1.3)
1. **Ort:** 3.1.3, Z. 169–173.
2. **Original:**
   ```tex
   Da die MultiNERD-Originalarbeit keine Referenzwerte für die offiziellen
   Splits liefert und die Testreferenz eine Silberreferenz ist, müssen
   Vergleiche auf diesem Datensatz innerhalb einer gemeinsamen Pipeline
   erfolgen und absolute Werte als Übereinstimmung mit einer automatisch
   erzeugten Referenz lesen (Abschnitt~\ref{sec:datenauswahl}).
   ```
3. **Problem:** Die Begründung steht in 3.1.1 (Z. 65–67). Zudem ist „Vergleiche müssen … absolute Werte … lesen" grammatisch schief, weil Vergleiche nicht lesen.
4. **Warum verdichtbar:** Die Begründung kann als Verweis stehen; die Folgerung bleibt.
5. **Vorschlag:**
   ```tex
   Mangels Referenzwerten für die offiziellen Splits
   (Abschnitt~\ref{sec:k3-multinerd}) müssen Vergleiche auf MultiNERD
   innerhalb einer gemeinsamen Pipeline erfolgen; absolute Werte messen die
   Übereinstimmung mit einer Silberreferenz
   (Abschnitt~\ref{sec:datenauswahl}).
   ```
6. **Erhalten bleibt:** Begründung, methodische Folgerung, Lesart absoluter Werte, beide Verweise.
7. **Risiken:** keine.
8. **Wörter:** etwa 10.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt; die Grammatik wird korrigiert.

#### K3-c · A · Symbol-Experiment von Zhan zweimal (3.3 und 3.5)
1. **Ort:** 3.3, Z. 425–427, sowie 3.5, Z. 740–742.
2. **Original:**
   ```tex
   % 3.3
   Ersetzt man die Typbezeichnungen auf CoNLL-2003 und CleanCoNLL durch
   bedeutungslose Symbole, ändern sich die Ergebnisse kaum, was die Autoren
   gegen eine bloße Memorisierung von Entität-Typ-Paaren anführen.
   % 3.5
   Ihr Befund, dass die Ergebnisse nicht von den Typbezeichnungen abhängen,
   spricht gegen eine Memorisierung von Entität-Typ-Paaren, gilt aber für
   CoNLL-2003 und CleanCoNLL; für MultiNERD bleibt die Frage offen.
   ```
3. **Problem:** Derselbe Befund mit derselben Folgerung an zwei Stellen.
4. **Warum verdichtbar:** In 3.5 wird er für die Kontaminationsfrage gebraucht, in 3.3 nur referiert.
5. **Vorschlag:** In 3.3 streichen. In 3.5:
   ```tex
   Ihr Befund, dass sich die Ergebnisse beim Ersetzen der Typbezeichnungen
   durch bedeutungslose Symbole kaum ändern, spricht gegen eine
   Memorisierung von Entität-Typ-Paaren, gilt aber für CoNLL-2003 und
   CleanCoNLL; für MultiNERD bleibt die Frage offen.
   ```
6. **Erhalten bleibt:** Versuchsaufbau, Ergebnis „kaum Änderung", Folgerung, Begrenzung auf CoNLL.
7. **Risiken:** Die Darstellung von Zhan in 3.3 verliert einen Befund; Hauptvergleich und Fehleranalyse bleiben.
8. **Wörter:** etwa 20 netto.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt; „kaum ändern" ist genauer als „nicht abhängen".

#### K3-d · A · Verzicht auf Constrained Decoding vorweggenommen (3.2.1)
1. **Ort:** 3.2.1, Z. 259–260.
2. **Original:**
   ```tex
   Auf Constrained Decoding wird verzichtet, damit die Formtreue der
   Ausgaben selbst messbar bleibt (Abschnitt~\ref{sec:robustheit-ausgaben}).
   ```
3. **Problem:** 4.4.4 (Z. 689–697) begründet denselben Verzicht vollständig und mit Quellen.
4. **Warum verdichtbar:** 3.2.1 bespricht keine Literatur zu Constrained Decoding; der Satz ist eine vorweggenommene Methodenentscheidung.
5. **Vorschlag:** Satz streichen.
6. **Erhalten bleibt:** Begründung in 4.4.4.
7. **Risiken:** keine.
8. **Wörter:** etwa 15.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt.

#### K3-f · B · Parameterangleichung bei Obeidat zweimal (3.3 und 3.4)
1. **Ort:** 3.3, Z. 439–440; Doppelung in 3.4, Z. 532–535.
2. **Original:**
   ```tex
   Der LoRA-Rang ist so gewählt, dass die trainierbaren Parameter der LLMs
   etwa der Größe der Encoder entsprechen.
   ```
3. **Problem:** 3.4 nennt dieselbe Angleichung mit Zahlen (341 bzw. 353 Mio. gegenüber 336 bis 435 Mio.).
4. **Warum verdichtbar:** 3.4 ist vollständiger.
5. **Vorschlag:** Satz in 3.3 streichen.
6. **Erhalten bleibt:** Angleichung mit Zahlen in 3.4.
7. **Risiken:** Die Versuchsbeschreibung in 3.3 wird unvollständiger; Fairness des Vergleichs ist dort das Thema.
8. **Wörter:** etwa 16.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt; die Darstellung in 3.3 wird knapper als nötig.

#### K3-g · C · Ankündigung in der Einleitung von 3.3
1. **Ort:** 3.3, Z. 401–403.
2. **Original:**
   ```tex
   Diese Kombination ist in der gesichteten Literatur noch selten; die
   vier folgenden Arbeiten decken allgemeine und domänenspezifische
   Datensätze ab und erfassen unterschiedliche Ressourcen.
   ```
3. **Problem:** Der zweite Teil kündigt nur an.
4. **Warum verdichtbar:** Die vier Absätze folgen unmittelbar.
5. **Vorschlag:** `Diese Kombination ist in der gesichteten Literatur noch selten.`
6. **Erhalten bleibt:** die inhaltliche Aussage zur Seltenheit.
7. **Risiken:** Die Orientierung über den Aufbau von 3.3 fällt weg.
8. **Wörter:** etwa 14.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt leicht.

#### K3-h · C · Wahl von DeBERTa ein drittes Mal begründet (3.1.2)
1. **Ort:** 3.1.2, Z. 109–111.
2. **Original:**
   ```tex
   Die Studie stützt die Wahl von DeBERTa-v3 als Encoder-Referenz in dieser
   Arbeit (Abschnitt~\ref{sec:encoder-auswahl}); sie liegt bislang nur als
   Preprint vor.
   ```
3. **Problem:** 1.3 (Z. 182–186) und 4.2.1 (Z. 154–156) begründen die Wahl mit derselben Quelle.
4. **Warum verdichtbar:** Die Verbindung steht schon an zwei Stellen.
5. **Vorschlag:** `Die Studie liegt bislang nur als Preprint vor.`
6. **Erhalten bleibt:** Preprint-Hinweis; Begründung in 1.3 und 4.2.1.
7. **Risiken:** Die übliche Brücke von der Literatur zur eigenen Wahl fehlt in Kapitel 3.
8. **Wörter:** etwa 13.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt leicht.

**Nicht kürzen (Kapitel 3):** Tabelle 3.1; Absatz zu Auswahl und Recherche; Erwartungen E1 bis E3; Einordnung des eigenen Formats in 3.2.1 (Z. 236–253, nötig für die Zuordnung zur Indexing-Familie); CrossNER-Absatz in 3.2.4 (Z. 384–393); Einschränkungen zu Tan und Keraghel; Lücke 1 in 3.5 mit den veröffentlichten MultiNERD-Werten (Z. 678–684), weil dies die einzige vollständige Stelle mit Quellen ist und K4-a dorthin verweist.

**Kurzbilanz Kapitel 3:** 13 Seiten. A-Optionen K3-a bis K3-d mit etwa 72 Wörtern, B-Option K3-f mit etwa 16, C-Optionen mit etwa 27. Erwartete Seitenersparnis 0, weil die letzte Seite zu 98 % gefüllt ist und Tabelle 3.1 eine eigene Seite belegt. Risiko gering. Das Kapitel ist präzise bis auf Doppelungen zu Kapitel 2 und zwischen 3.3 und 3.5.

---

### Kapitel 4 – Methodik

#### K4-a · A · Veröffentlichte MultiNERD-Werte ein drittes Mal (4.3.1)
1. **Ort:** 4.3.1, `04_methodik.tex` Z. 232–238; dazu 3.5 Z. 683–684.
2. **Original:**
   ```tex
   Veröffentlichte MultiNERD-Werte beruhen dagegen auf eigener Aufbereitung
   (InstructUIE und UniversalNER: Testsplit mit 10\,000 Beispielen
   \cite{wang2023instructuie,zhou2024universalner}) oder
   einem abweichenden Testumfang (GoLLIE: 32\,908 Testbeispiele, etwa das
   Doppelte des englischen Testsplits \cite[Tab.~9]{sainz2024gollie});
   Werte auf den offiziellen Splits sind nicht nachweisbar
   (Abschnitt~\ref{sec:encoder_ner_related}).
   ```
3. **Problem:** 3.1.1 fasst denselben Befund zusammen, 3.5 belegt ihn vollständig mit Quellen. Der Verweis zielt zudem nur auf 3.1, nicht auf 3.5.
4. **Warum verdichtbar:** Für die Datensatzwahl genügt die Aussage mit Verweis. Einzig „etwa das Doppelte des englischen Testsplits" steht nur hier und wandert nach 3.5.
5. **Vorschlag:** In 4.3.1:
   ```tex
   Veröffentlichte MultiNERD-Werte beruhen dagegen auf eigener Aufbereitung
   oder abweichendem Testumfang; Werte auf den offiziellen Splits sind
   nicht nachweisbar (Abschnitte~\ref{sec:k3-multinerd}
   und~\ref{sec:research_gap}).
   ```
   In 3.5 statt „Bei \textcite{sainz2024gollie} bleibt das Verhältnis der 32\,908 Testbeispiele zu den offiziellen Splits offen.":
   ```tex
   Bei \textcite[Tab.~9]{sainz2024gollie} umfasst der Test 32\,908
   Beispiele, etwa das Doppelte des englischen Testsplits; ihr Verhältnis
   zu den offiziellen Splits bleibt offen.
   ```
6. **Erhalten bleibt:** alle Zahlen und Quellen in 3.5; `wang2023instructuie` bleibt in 3.2.3 (Z. 315) zitiert.
7. **Risiken:** Die Datensatzbegründung in 4.3.1 wird abstrakter.
8. **Wörter:** etwa 17 netto.
9. **Seiteneffekt:** keiner sicher vorhersagbar.
10. **Dichte:** steigt; der Verweis wird genauer.

#### K4-b · A · Zuverlässige Typen nach Tedeschi wiederholt 2.8 (4.3.1)
1. **Ort:** 4.3.1, Z. 255–257.
2. **Original:**
   ```tex
   Laut den Autoren gelten PER, ORG, LOC, CEL, DIS, EVE und MEDIA als
   besonders zuverlässig annotiert, andere Klassen teils als stark
   verrauscht \cite{tedeschi2022multinerd}.
   ```
3. **Problem:** 2.8 (Z. 661–670) nennt dieselben Gruppen ausführlicher, mit ausgeschriebenen Typen und INST.
4. **Warum verdichtbar:** Für die Methodik genügt, dass die Qualität je Typ schwankt.
5. **Vorschlag:**
   ```tex
   Die Annotationsqualität unterscheidet sich dabei stark zwischen den
   Typen (Abschnitt~\ref{sec:datasets_corpora}).
   ```
6. **Erhalten bleibt:** Typgruppen, Quelle und Zahlen in 2.8.
7. **Risiken:** keine.
8. **Wörter:** etwa 10.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt.

#### K4-f · A · „Hybride Attention ist relevant für die Adapter" zweimal (4.2.2 oder 2.3)
1. **Ort:** 4.2.2, Z. 193–194; Doppelung in 2.3, Z. 270–272.
2. **Original:**
   ```tex
   Die hybride Attention ist für die Platzierung der LoRA-Adapter relevant
   (Abschnitt~\ref{sec:lora-konfiguration}).
   ```
3. **Problem:** 2.3 sagt dasselbe mit demselben Verweis.
4. **Warum verdichtbar:** Eine der beiden Stellen genügt; 4.4.5 erklärt die Platzierung selbst.
5. **Vorschlag:** Satz in 4.2.2 streichen. Alternative: in 2.3 den Nebensatz „was für die Platzierung der Adapter relevant ist (Abschnitt~\ref{sec:lora-konfiguration})" streichen, etwa 10 Wörter.
6. **Erhalten bleibt:** Architekturbeschreibung in 4.2.2 und Erklärung in 4.4.5.
7. **Risiken:** keine.
8. **Wörter:** etwa 12.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt.

#### K4-c · B · Kapiteleinleitung besteht nur aus Anhangverweisen
1. **Ort:** Kapitelanfang, Z. 12–15.
2. **Original:**
   ```tex
   Zu den hier beschriebenen Läufen ordnet
   Anhang~\ref{app:reproduzierbarkeit} jedem Lauf Konfiguration, Codestand,
   Revisionen und Ausgabeverzeichnis zu;
   Anhang~\ref{app:durchfuehrung} ergänzt Durchführungsdetails.
   ```
3. **Problem:** 4.4.1 (Z. 509–512) verweist auf dieselben Anhangtabellen genauer; A.2 wird in Kapitel 4 dreizehnmal an der jeweils passenden Stelle verwiesen.
4. **Warum verdichtbar:** Der Satz ist keine inhaltliche Einleitung.
5. **Vorschlag:** Satz streichen.
6. **Erhalten bleibt:** alle Anhangverweise an den Fachstellen.
7. **Risiken:** Kapitel- und Abschnittsüberschrift stehen dann direkt untereinander, wie schon in Kapitel 1.
8. **Wörter:** etwa 22.
9. **Seiteneffekt:** wenige Zeilen auf S. 34.
10. **Dichte:** steigt.

#### K4-e · B · Unsicherheit seltener Typen und Makro-Satz (4.5.1)
1. **Ort:** 4.5.1, Z. 908–915.
2. **Original:**
   ```tex
   Wegen der unausgewogenen Typverteilung (Abschnitt~\ref{sec:datenaufteilung},
   Tabelle~\ref{tab:datensplits}) sind typbezogene Werte seltener Klassen mit
   erheblicher Unsicherheit behaftet.

   Ergänzend wird der makro-gemittelte F$_1$-Wert als ungewichtetes Mittel der
   typbezogenen F$_1$-Werte über alle 15 Typen berichtet
   (Abschnitt~\ref{sec:evaluation}); er reagiert entsprechend stark auf die
   seltenen Typen.
   ```
3. **Problem:** Z. 888–889 legt schon fest, dass typbezogene Vergleiche deskriptiv bleiben. 5.1.2 (Z. 182–186) nennt die Unsicherheit mit konkreten Zahlen.
4. **Warum verdichtbar:** Der Grund lässt sich in den Makro-Satz aufnehmen.
5. **Vorschlag:**
   ```tex
   Ergänzend wird der makro-gemittelte F$_1$-Wert als ungewichtetes Mittel der
   typbezogenen F$_1$-Werte über alle 15 Typen berichtet
   (Abschnitt~\ref{sec:evaluation}); er reagiert stark auf die seltenen,
   mit wenigen Referenzentitäten besetzten Typen (Tabelle~\ref{tab:datensplits}).
   ```
6. **Erhalten bleibt:** Grund (wenige Referenzentitäten), Tabellenverweis, Makro-Definition, deskriptive Regel in Z. 888.
7. **Risiken:** Die Methodik nennt die Unsicherheit seltener Typen nicht mehr als eigenen Satz.
8. **Wörter:** etwa 12.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt leicht.

#### K4-g · B · Thinking-Entscheidung im Ansatz wiederholt Tabelle 4.3 (4.1)
1. **Ort:** 4.1, Z. 69–72.
2. **Original:**
   ```tex
   Der Thinking-Modus wurde nach der generativen Validierung eines
   abgebrochenen 4B-Vorlaufs auf Validierungsdaten für alle Läufe deaktiviert
   (Tabelle~\ref{tab:datennutzung}, Abschnitt~\ref{sec:llm-pipeline},
   Anhang~\ref{app:durchfuehrung}).
   ```
3. **Problem:** Tabelle 4.3 enthält dieselbe Zeile („generative Validierung eines abgebrochenen 4B-Vorlaufs", Testbezug „keiner"). 4.4.3 und A.2.1 beschreiben die Umsetzung.
4. **Warum verdichtbar:** Der Satz beschreibt eine Tabellenzeile im Fließtext.
5. **Vorschlag:** Satz streichen.
6. **Erhalten bleibt:** Entscheidung, Datengrundlage und Testbezug in Tabelle 4.3; Umsetzung in 4.4.3; Daten in A.2.1.
7. **Risiken:** Das Wort „Validierungsdaten" steht dann nur noch implizit („generative Validierung", Testbezug „keiner"). Die Testnutzung ist ein sensibles Thema.
8. **Wörter:** etwa 25.
9. **Seiteneffekt:** wenige Zeilen auf S. 34 oder 35.
10. **Dichte:** steigt.

#### K4-h · B · Testnutzung: zwei Sätze wiederholen Tabelle 4.3 (4.3.3)
1. **Ort:** 4.3.3, Z. 385–398.
2. **Original:**
   ```tex
   Bei den Zero-Shot-Konfigurationen beruhen dagegen Decoding und
   Prompt-Erweiterung auf Beobachtungen eines Vorlaufs auf dem Testsplit
   (Abschnitt~\ref{sec:llm-pipeline}), die Inferenzpräzision auf
   Durchsatzmessungen an einer Teilmenge des Testsplits.
   Die Zero-Shot-Ergebnisse werden deshalb als explorative Ergebnisse der
   nach Testbeobachtungen gewählten Konfigurationen berichtet. Solche
   Auswahlentscheidungen können zu einer optimistischen Verzerrung führen
   \cite{cawley2010selectionbias}; deren Umfang lässt sich mit den vorhandenen Daten
   nicht beziffern. Die Durchsatzmessung zur Präzisionswahl ist dabei von der
   an beobachteter Erkennungsleistung orientierten Prompt- und Decoding-Wahl
   zu unterscheiden. Auch die späteren Analysen des Ausgabelimits sind
   nachträgliche Sensitivitätsanalysen auf den bekannten Testsplits.
   Tabelle~\ref{tab:datennutzung} ordnet jeder Entscheidung Datengrundlage
   und Testbezug zu.
   ```
3. **Problem:** Die beiden vorletzten Sätze wiederholen die Spalte „Testbezug" von Tabelle 4.3 („direkt, nicht leistungsbezogen" und „post hoc").
4. **Warum verdichtbar:** Die Unterscheidung passt als Halbsatz in den ersten Satz; die Sensitivitätsanalysen lassen sich im Tabellenverweis nennen.
5. **Vorschlag:**
   ```tex
   Bei den Zero-Shot-Konfigurationen beruhen dagegen Decoding und
   Prompt-Erweiterung auf Beobachtungen eines Vorlaufs auf dem Testsplit
   (Abschnitt~\ref{sec:llm-pipeline}), die Inferenzpräzision auf
   Durchsatzmessungen an einer Teilmenge des Testsplits, nicht auf der
   Erkennungsleistung.
   Die Zero-Shot-Ergebnisse werden deshalb als explorative Ergebnisse der
   nach Testbeobachtungen gewählten Konfigurationen berichtet. Solche
   Auswahlentscheidungen können zu einer optimistischen Verzerrung führen
   \cite{cawley2010selectionbias}; deren Umfang lässt sich mit den vorhandenen Daten
   nicht beziffern.
   Tabelle~\ref{tab:datennutzung} ordnet jeder Entscheidung Datengrundlage
   und Testbezug zu, einschließlich der nachträglichen
   Sensitivitätsanalysen zum Ausgabelimit.
   ```
6. **Erhalten bleibt:** Unterschied zwischen leistungsbezogener und nicht leistungsbezogener Wahl, Post-hoc-Charakter der Limit-Analysen, Cawley, Tabellenverweis.
7. **Risiken:** Der Absatz wurde in früheren Runden sorgfältig formuliert. „Solche Auswahlentscheidungen" bezieht sich danach auf alle drei Entscheidungen; die Einschränkung „nicht auf der Erkennungsleistung" steht direkt davor.
8. **Wörter:** etwa 23.
9. **Seiteneffekt:** wenige Zeilen in 4.3.3.
10. **Dichte:** steigt.

#### P-3 · P · Verweis zu den LoRA-Zahlen von Zhan zeigt auf den falschen Abschnitt (4.4.5)
1. **Ort:** 4.4.5, Z. 706–709.
2. **Original:**
   ```tex
   von \textcite[Anh.~D]{zhan2026generativeNER} zudem 0,14 bis 0,48
   F$_1$-Punkte vor vollständigem Fine-Tuning liegen
   (Abschnitt~\ref{sec:lora}).
   ```
3. **Problem:** `sec:lora` ist Abschnitt 2.6. Dort steht Zhan nicht; die Zahlen stehen in 3.3 (Z. 468–471).
4. **Warum ändern:** Der Leser findet die Quelle der Zahlen nicht.
5. **Vorschlag:** `(Abschnitte~\ref{sec:lora} und~\ref{sec:peft_ner_related})`. Option C: Zahlen hier streichen und nur „knapp vor" schreiben, etwa 4 Wörter.
6. **Erhalten bleibt:** alles.
7. **Risiken:** keine.
8. **Wörter:** etwa 2 mehr.
9. **Seiteneffekt:** keiner.
10. **Dichte:** gleich; die Nachvollziehbarkeit steigt.

**Nicht kürzen (Kapitel 4):**
- Definition der strikten Metrik in 4.5.1 (Z. 853–855): Sie ist die operative Festlegung der Methodik, auch wenn 2.9 den Hintergrund liefert.
- Vorbehalt „kein isolierter Effekt der Anpassung" in 4.4.3 (Z. 609–612): Das ist der methodische Ort dieses Vorbehalts.
- Grenzen der Größenanalyse in 4.5.4 (Z. 1088–1092): Nur hier steht, dass sich Größenunterschiede gegen die Seed-Streuung abgrenzen lassen.
- Nachinferenz 4.5.5, Kern der Testnutzung mit Cawley, Validierungsregeln, Trainingsdauer und Vergleichbarkeit samt Hardwarevorbehalt und GPU-Tabelle, Hinweise zur Formtreue, CrossNER-Methodik, Diagnose der Checkpoint-Wahl, Absatz zu Fairness und Silberreferenz, Bootstrap.

**Kurzbilanz Kapitel 4:** 23 Seiten. A-Optionen K4-a, K4-b, K4-f mit etwa 39 Wörtern, B-Optionen K4-c, K4-e, K4-g, K4-h mit etwa 82 Wörtern, dazu P-3. Erwartete Seitenersparnis 0. Die letzte Seite enthält Tabelle 4.9 und etwa 215 Wörter; eine Float-Verschiebung kann im günstigen Fall eine Seite sparen, das ist unsicher. Risiko: Die Absätze zur Testnutzung sind inhaltlich sensibel. Das Kapitel ist dicht; die verbleibenden Doppelungen beschreiben vor allem Tabelle 4.3 im Fließtext oder wiederholen Kapitel 2 und 3.

---

### Kapitel 5 – Ergebnisse und Diskussion

#### K5-f · A + P · Speicherspanne falsch und doppelt (5.7.4)
1. **Ort:** 5.7.4, `05_ergebnisse_und_diskussion.tex` Z. 1004–1008.
2. **Original:**
   ```tex
   Der hardwaregleiche Vergleich von
   DeBERTa-v3-base und Qwen3.5-0.8B ergibt einen Latenzfaktor von etwa 210;
   der maximal allokierte Inferenzspeicher reicht von 1,7~GiB für
   DeBERTa-v3-large bis 18,7~GiB für Qwen3.5-27B im QLoRA-Regime. Faktoren
   zwischen anderen Größen enthalten zusätzlich Hardwareeffekte.
   ```
3. **Problem:** Die Spanne beginnt laut Effizienztabelle und 5.4 (Z. 506–507) bei 0,7 GiB (DeBERTa-v3-base). Der Satz davor sagt selbst, dass DeBERTa-v3-base am wenigsten Speicher braucht. Die Formulierung entstand in der zweiten Kürzungsrunde (Commit `1353891`).
4. **Warum verdichtbar:** 5.4 nennt die korrekte Spanne.
5. **Vorschlag:**
   ```tex
   Der hardwaregleiche Vergleich von
   DeBERTa-v3-base und Qwen3.5-0.8B ergibt einen Latenzfaktor von etwa 210;
   Faktoren zwischen anderen Größen enthalten zusätzlich Hardwareeffekte.
   ```
   Alternative ohne Kürzung: „reicht von 0,7~GiB (DeBERTa-v3-base) bis 18,7~GiB (Qwen3.5-27B, QLoRA)".
6. **Erhalten bleibt:** Latenzfaktor, Hardwarevorbehalt; die Speicherwerte in 5.4, Effizienztabelle (Tabelle 5.4) und Abbildung 5.3.
7. **Risiken:** keine.
8. **Wörter:** etwa 17.
9. **Seiteneffekt:** keiner; die letzte Seite von Kapitel 5 ist zu 99 % gefüllt.
10. **Dichte:** steigt; ein Sachfehler entfällt.

#### K5-e · A · Illustrative Zusätze zweimal (5.7.1)
1. **Ort:** 5.7.1, Z. 866–870.
2. **Original:**
   ```tex
   Die Fehleranalyse stützt diesen Vorbehalt an Beispielen, nicht
   quantitativ: Unter den häufigsten Zusätzen finden sich in einer
   illustrativen Durchsicht Spans, die nach
   den Richtlinien des Datensatzes Entitäten wären, in der Referenz aber
   fehlen (Abschnitt~\ref{sec:erg-fehlertypen}).
   ```
3. **Problem:** 5.5 (Z. 651–657) beschreibt dieselbe Beobachtung mit Beispielen und dem Hinweis, dass sie illustrativ ist.
4. **Warum verdichtbar:** Die Diskussion braucht nur den Befund und seinen Status.
5. **Vorschlag:**
   ```tex
   Die Fehleranalyse stützt diesen Vorbehalt nur an illustrativen
   Beispielen: Unter den häufigsten Zusätzen finden sich Spans, die nach
   den Richtlinien des Datensatzes Entitäten wären
   (Abschnitt~\ref{sec:erg-fehlertypen}).
   ```
6. **Erhalten bleibt:** illustrativ statt quantitativ, Art der Belege, Verweis. „In der Referenz fehlen" ist durch „Zusätze" definiert.
7. **Risiken:** keine.
8. **Wörter:** etwa 10.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt.

#### K5-g · A · Empfehlung für Encoder zweimal hintereinander (5.7.4)
1. **Ort:** 5.7.4, Z. 1018–1025.
2. **Original:**
   ```tex
   DeBERTa-v3-large bleibt auf MultiNERD im Mikro-F$_1$ knapp
   vorn, was eine Empfehlung für Encoder bei vergleichbaren Daten und
   Anforderungen an geringe Latenz stützt, wie sie auch
   \textcite{obeidat2025llm} für kleine Qualitätsunterschiede diskutieren.

   Für Anwendungen mit einem festen Inventar von 15 Typen, einer Datenbasis
   wie MultiNERD und niedriger Latenztoleranz sind die untersuchten Encoder
   eine gut gestützte Wahl.
   ```
3. **Problem:** Zwei aufeinanderfolgende Sätze geben dieselbe Empfehlung.
4. **Warum verdichtbar:** Der zweite Satz ist genauer; der Quellenbezug kann dorthin wandern.
5. **Vorschlag:**
   ```tex
   DeBERTa-v3-large bleibt auf MultiNERD im Mikro-F$_1$ knapp vorn.

   Für Anwendungen mit einem festen Inventar von 15 Typen, einer Datenbasis
   wie MultiNERD und niedriger Latenztoleranz sind die untersuchten Encoder
   damit eine gut gestützte Wahl; eine solche Empfehlung bei kleinen
   Qualitätsunterschieden diskutieren auch \textcite{obeidat2025llm}.
   ```
6. **Erhalten bleibt:** Befund, Bedingungen der Empfehlung, Bezug auf Obeidat.
7. **Risiken:** keine.
8. **Wörter:** etwa 11.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt.

#### K5-h · A · Domänenweise CrossNER-Intervalle zweimal (5.7.5)
1. **Ort:** 5.7.5, Z. 1084–1088.
2. **Original:**
   ```tex
   Die nach Domäne geschichteten CrossNER-Intervalle
   (Tabelle~\ref{tab:crossner-ci}) stützen bei 1\,024 Tokens den gepoolten
   Vorsprung des LLM; domänenweise schließen sie in drei von fünf Domänen
   (Musik, Politik, Naturwissenschaften) null aus, in Künstlicher
   Intelligenz und Literatur bleibt die Richtung unsicher.
   ```
3. **Problem:** 5.6 (Z. 734–737) sagt fast wörtlich dasselbe.
4. **Warum verdichtbar:** 5.7.5 braucht die Bilanz, nicht die Domänenliste.
5. **Vorschlag:**
   ```tex
   Die nach Domäne geschichteten CrossNER-Intervalle stützen bei
   1\,024 Tokens den gepoolten Vorsprung des LLM, domänenweise nur in drei
   von fünf Domänen (Abschnitt~\ref{sec:erg-crossner},
   Tabelle~\ref{tab:crossner-ci}).
   ```
6. **Erhalten bleibt:** gepoolter Befund, Anteil der Domänen, Domänennamen in 5.6.
7. **Risiken:** keine.
8. **Wörter:** etwa 15.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt.

#### K5-b · B · Lesart der Intervalle wiederholt 4.5.1 (Kapitelanfang)
1. **Ort:** Kapitelanfang, Z. 58–65.
2. **Original:**
   ```tex
   Ausgewählte Differenzen werden mit den gepaarten
   Bootstrap-Konfidenzintervallen aus Abschnitt~\ref{sec:performanzmetriken}
   eingeordnet, die die Unsicherheit durch die Auswahl der Testsätze bei
   festen Modellen und Vorhersagen quantifizieren; enthält ein
   Differenzintervall null, ist kein eindeutiger Unterschied nachgewiesen,
   daraus folgt aber keine Gleichwertigkeit.
   Typbezogene Unterschiede ohne eigene Intervalle werden nur als
   beobachtete Punktwertdifferenzen beschrieben.
   ```
3. **Problem:** 4.5.1 (Z. 885–892) erklärt Bootstrap, Bedingtheit und Lesart; 5.7.5 (Z. 1070–1071 und 1106–1107) wiederholt die Bedingtheit.
4. **Warum verdichtbar:** Die Lesregel bleibt; die zweite Erklärung des Verfahrens entfällt.
5. **Vorschlag:**
   ```tex
   Ausgewählte Differenzen werden mit den gepaarten
   Bootstrap-Konfidenzintervallen aus Abschnitt~\ref{sec:performanzmetriken}
   eingeordnet; enthält ein Differenzintervall null, ist kein eindeutiger
   Unterschied nachgewiesen, daraus folgt aber keine Gleichwertigkeit.
   Typbezogene Unterschiede ohne eigene Intervalle bleiben deskriptiv.
   ```
6. **Erhalten bleibt:** Lesregel „null enthalten heißt nicht gleichwertig", deskriptiver Status typbezogener Werte, Verweis.
7. **Risiken:** Wer Kapitel 5 einzeln liest, findet die Bedingtheit auf feste Modelle erst in 4.5.1 oder 5.7.5.
8. **Wörter:** etwa 17.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt.

#### K5-c · B · Vorbehalt zu Zero-Shot-Werten länger als nötig (5.1.1)
1. **Ort:** 5.1.1, Z. 117–122.
2. **Original:**
   ```tex
   Die Werte beruhen auf je einem gesampelten Lauf und auf Prompt und
   Decoding-Einstellungen, die nach Beobachtungen auf dem Testsplit gewählt
   wurden (Abschnitt~\ref{sec:datenaufteilung}). Sie sind daher explorative
   Messwerte dieser Konfigurationen und weder eine unabhängige Bestätigung
   noch eine Obergrenze der Leistung unveränderter Checkpoints mit diesem
   Format.
   ```
3. **Problem:** 4.3.3 legt den explorativen Status fest; „keine unabhängige Bestätigung" ist darin enthalten.
4. **Warum verdichtbar:** Die Aussage lässt sich in einen Satz fassen, ohne einen Teil zu verlieren.
5. **Vorschlag:**
   ```tex
   Die Werte beruhen auf je einem gesampelten Lauf mit nach
   Testbeobachtungen gewählten Prompt- und Decoding-Einstellungen
   (Abschnitt~\ref{sec:datenaufteilung}); sie sind explorative Messwerte
   dieser Konfigurationen, keine Obergrenze der Leistung unveränderter
   Checkpoints mit diesem Format.
   ```
6. **Erhalten bleibt:** Einzellauf, Sampling, Testbezug, explorativer Status, keine Obergrenze.
7. **Risiken:** „Weder eine unabhängige Bestätigung" entfällt als ausdrückliche Formulierung.
8. **Wörter:** etwa 13.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt.

#### K5-d · B · Abstand QLoRA zu Zero-Shot, Vorbehalt gestrafft (5.1.1)
1. **Ort:** 5.1.1, Z. 124–127.
2. **Original:**
   ```tex
   Nach der Anpassung mit QLoRA beträgt der Abstand zum Zero-Shot-Regime
   87,0 (0.8B), 59,0 (4B) bzw. 25,1 Punkte (27B). Er beschreibt den Unterschied der vollständigen
   Konfigurationen, da zusätzlich zum Training Prompt, Decoding und
   Präzision variieren. Die Zero-Shot-Sampling-Streuung wurde nicht gemessen.
   ```
3. **Problem:** Der Vorbehalt steht ausführlich in 4.4.3, 5.7.1 und 5.7.5; der Einzellauf in Z. 117, die fehlende Streuungsmessung in 5.7.5 und 6.2.
4. **Warum verdichtbar:** An dieser Stelle genügt ein Halbsatz.
5. **Vorschlag:**
   ```tex
   Nach der Anpassung mit QLoRA beträgt der Abstand zum Zero-Shot-Regime
   87,0 (0.8B), 59,0 (4B) bzw. 25,1 Punkte (27B); er umfasst neben dem
   Training auch Unterschiede in Prompt, Decoding und Präzision.
   ```
6. **Erhalten bleibt:** alle drei Abstände und der Vorbehalt.
7. **Risiken:** Der Satz zur Sampling-Streuung entfällt an dieser Stelle. Variante: ihn behalten, dann etwa 5 Wörter.
8. **Wörter:** etwa 11.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt.

#### K5-i · B · Gründe gegen Skalierungsgesetze wiederholt 2.7 (5.7.3)
1. **Ort:** 5.7.3, Z. 990–994.
2. **Original:**
   ```tex
   Die Skalierungsgesetze aus Abschnitt~\ref{sec:model_size} sind auf
   diese Befunde nicht übertragbar: Sie beschreiben den Vortrainingsverlust
   eigens trainierter Modellreihen, hier wurden veröffentlichte
   Checkpoints mit größenabhängigen Hyperparametern und auf verschiedenen
   GPUs angepasst.
   ```
3. **Problem:** 2.7 (Z. 614–624) nennt dieselben zwei Gründe. Der Vorbehalt steht insgesamt an fünf Stellen (2.7, 4.5.4, 5.3, 5.7.3, 6.1).
4. **Warum verdichtbar:** Die Diskussion kann auf die Gründe verweisen.
5. **Vorschlag:**
   ```tex
   Die Skalierungsgesetze aus Abschnitt~\ref{sec:model_size} sind aus den
   dort genannten Gründen auf diese Befunde nicht übertragbar.
   ```
6. **Erhalten bleibt:** Aussage und Begründung in 2.7.
7. **Risiken:** Die Diskussion wird verweisabhängig; die Gründe stehen 50 Seiten vorher.
8. **Wörter:** etwa 18.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt.

#### K5-a · C · Kapiteleinleitung straffen
1. **Ort:** Kapitelanfang, Z. 29–42.
2. **Original:** „Dieses Kapitel berichtet die Ergebnisse der in Kapitel~\ref{chap:methodik} beschriebenen Experimente und diskutiert sie anschließend. … Abschnitt~\ref{sec:erg-diskussion} interpretiert die Befunde, ordnet sie in die verwandten Arbeiten ein, leitet daraus die Empfehlungen zu FF3 ab und benennt, welche Unterschiede angesichts der Streuung belastbar sind."
3. **Problem:** Wegweiser mit etwas Leerlauf („der in Kapitel 4 beschriebenen Experimente", „zusammen").
4. **Warum verdichtbar:** Die Zuordnung der Abschnitte zu FF1 bis FF3 bleibt; nur Füllwörter entfallen.
5. **Vorschlag:** Ersten Satz zu „Dieses Kapitel berichtet die Ergebnisse und diskutiert sie anschließend." kürzen und „zusammen" streichen.
6. **Erhalten bleibt:** Zuordnung zu den Forschungsfragen, Trennung von Bericht und Diskussion.
7. **Risiken:** keine.
8. **Wörter:** etwa 10.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt minimal.

#### K5-j · C · Versuchsaufbau am Anfang von 5.6 wiederholt 4.6
1. **Ort:** 5.6, Z. 671–675.
2. **Original:** „Tabelle~\ref{tab:crossner-erg} berichtet den Ergänzungsversuch aus Abschnitt~\ref{sec:crossner}: die jeweils größten Modelle beider Familien, auf MultiNERD trainiert, ausgewertet auf den Testsplits der fünf CrossNER-Domänen und beschränkt auf die Grobtypen PER, ORG und LOC."
3. **Problem:** 4.6 beschreibt den Aufbau.
4. **Warum verdichtbar:** Die Tabelle zeigt die Modelle.
5. **Vorschlag:** „Tabelle~\ref{tab:crossner-erg} berichtet den Transfertest aus Abschnitt~\ref{sec:crossner} auf den fünf CrossNER-Domänen, beschränkt auf PER, ORG und LOC."
6. **Erhalten bleibt:** Bezug, Domänen, Typbeschränkung, Hardwaresatz danach.
7. **Risiken:** Die Leseorientierung im Ergebnisteil wird schwächer.
8. **Wörter:** etwa 12.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt leicht.

#### K5-k · C · „bleibt diese Betrachtung deskriptiv" (5.3)
1. **Ort:** 5.3, Z. 431–432.
2. **Original:** „Mit zwei Encoder- und drei LLM-Größen bleibt diese Betrachtung deskriptiv (Abschnitt~\ref{sec:einfluss-parameteranzahl})."
3. **Problem:** wiederholt 4.5.4.
4. **Warum verdichtbar:** 5.7.3 diskutiert die Grenze.
5. **Vorschlag:** Satz streichen.
6. **Erhalten bleibt:** Aussage in 4.5.4 und 5.7.3.
7. **Risiken:** Der Ergebnisteil verliert die Erinnerung an die Grenze direkt am Befund.
8. **Wörter:** etwa 11.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt leicht.

#### K5-l · C · Begründung „kein isolierter Anpassungseffekt" (5.7.1)
1. **Ort:** 5.7.1, Z. 833–836.
2. **Original:** „…; wegen gleichzeitiger Unterschiede in Prompt, Decoding und Präzision ist das kein isolierter Anpassungseffekt (Abschnitt~\ref{sec:disk-validitaet})."
3. **Problem:** Die Begründung steht in 4.4.3, 5.1.1 und 5.7.5.
4. **Warum verdichtbar:** Der Verweis auf 5.7.5 trägt die Begründung.
5. **Vorschlag:** „…; das ist kein isolierter Anpassungseffekt (Abschnitt~\ref{sec:disk-validitaet})."
6. **Erhalten bleibt:** Vorbehalt und Verweis.
7. **Risiken:** keine.
8. **Wörter:** etwa 7.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt minimal.

#### K5-m · C · Zero-Shot-Formtreue in der Diskussion nacherzählt (5.7.2)
1. **Ort:** 5.7.2, Z. 916–918.
2. **Original:** „Dazu kommt beim 4B-Modell die Neigung, leere Arrays auszugeben, und beim 27B-Modell eine Typdefinition für TIME, die von der des Datensatzes abweicht."
3. **Problem:** 5.2 und 5.3 berichten beides.
4. **Warum verdichtbar:** Die Diskussion stützt sich auch ohne diesen Satz auf 5.2.
5. **Vorschlag:** Satz streichen.
6. **Erhalten bleibt:** Befunde in 5.2 und 5.3.
7. **Risiken:** Die Deutung „Typinventar begrenzt die Leistung" verliert ihr deutlichstes Beispiel (TIME).
8. **Wörter:** etwa 25.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt formal; die Begründung wird dünner.

#### K5-n · C · FF3 wird am Anfang von 5.7.4 neu formuliert
1. **Ort:** 5.7.4, Z. 999–1000.
2. **Original:** „FF3 fragt nach dem Verhältnis von Erkennungsleistung und Ressourcenbedarf unter Speicher- und Latenzbeschränkungen."
3. **Problem:** wiederholt die Forschungsfrage aus 1.2.
4. **Warum verdichtbar:** 5.7.3 beginnt knapper mit „Zu FF2 zeigen die Ergebnisse …".
5. **Vorschlag:** „Zu FF3: Auf MultiNERD erreichen die Encoder …" (Satz entfällt).
6. **Erhalten bleibt:** Bezug auf FF3.
7. **Risiken:** Lesende ohne Kapitel 1 vermissen die Frage.
8. **Wörter:** etwa 13.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt leicht.

**Nicht kürzen (Kapitel 5):**
- Konventionen zu Rundung, Seeds und Einzelläufen am Kapitelanfang.
- 5.1.2: Supportzahlen seltener Typen (Z. 182–186), einzige Stelle mit diesen Zahlen; Relativsatz zu Tedeschi (Z. 163–165) als Lesehilfe für die Werte.
- 5.2: Satz zu den 55 längeren Referenzantworten (Z. 265–267). Er ist ein Argumentationsschritt: Die Modellantworten sind länger als die Referenz.
- 5.4: Einleitung zum GPU-Typ als Lesehilfe der Tabelle und alle vier Absätze; jeder enthält eigene Werte.
- 5.5 vollständig; 5.6 einschließlich beider Werte zum Ausschluss der 21 Sätze (verschiedene Limits).
- 5.7.1: Eröffnung mit dem zentralen Befund zu FF1, Literaturvergleiche, Kontaminationsabsatz.
- 5.7.4: Vorbehalt zur Kombination von Nachinferenz und Pareto-Aussage; Geltungsbereich „einzelne Sätze bei Batchgröße 1".
- 5.7.5: Seed-Absatz, Absatz zu den sieben Vergleichen, `samepage`-Absatz mit „aus FF1 und FF2 abgeleitet".

**Kurzbilanz Kapitel 5:** 21 Seiten. A-Optionen K5-e bis K5-h mit etwa 53 Wörtern, B-Optionen K5-b, K5-c, K5-d, K5-i mit etwa 59 Wörtern, C-Optionen mit etwa 78 Wörtern. Erwartete Seitenersparnis 0, weil die letzte Seite zu 99 % gefüllt ist. Gewinn ist Reserve gegen Überlauf. Risiko: Die Diskussion wird stärker verweisabhängig. Der Ergebnisteil ist dicht; die Diskussion wiederholt Ergebnisse gezielt und dabei an wenigen Stellen wörtlich. K5-f behebt eine sachliche Unstimmigkeit.

---

### Kapitel 6 – Fazit und Ausblick

#### K6-a · B · Statistische Unsicherheit doppelt zu 5.7.5 (6.2)
1. **Ort:** 6.2, Absatz „Statistische Unsicherheit", Z. 192–204.
2. **Original:**
   ```tex
   Drei Seeds erfassen die Trainingsvariabilität nur deskriptiv, das
   gepaarte Bootstrap ist auf die vorhandenen Vorhersagen bedingt, und die
   Sampling-Streuung der Zero-Shot-Einzelläufe ist unbekannt.
   Abhängigkeiten zwischen Sätzen desselben Dokuments bleiben
   unberücksichtigt.
   Als Hauptvergleiche gelten nur die aus FF1 und FF2 abgeleiteten
   Vergleiche bei 256 Ausgabetokens (Abschnitt~\ref{sec:performanzmetriken}); die übrigen sind explorativ, keines der Intervalle ist für
   Mehrfachvergleiche korrigiert, und knappe Differenzen bei 1\,024 Tokens
   sowie Einzeldomänenvergleiche sind vorsichtig zu lesen
   (Abschnitt~\ref{sec:disk-validitaet}).
   Keine Unsicherheitsangabe korrigiert Verzerrungen durch Testnutzung,
   Referenzqualität oder Konfigurationswahl.
   Betroffen sind FF1 bis FF3.
   ```
3. **Problem:** 5.7.5 (Z. 1060–1064 und 1105–1125) enthält jeden dieser Punkte, teils wörtlich.
4. **Warum verdichtbar:** Die Limitationen können auf 5.7.5 verweisen und die Kernaussage behalten.
5. **Vorschlag:**
   ```tex
   Die Grenzen der Seeds und Intervalle sowie die Einteilung in die aus
   FF1 und FF2 abgeleiteten Hauptvergleiche bei 256 Ausgabetokens und
   explorative Vergleiche beschreibt Abschnitt~\ref{sec:disk-validitaet}.
   Keine Unsicherheitsangabe korrigiert Verzerrungen durch Testnutzung,
   Referenzqualität oder Konfigurationswahl.
   Betroffen sind FF1 bis FF3.
   ```
   Mildere Variante: alle Punkte behalten, nur den Halbsatz „und knappe Differenzen bei 1 024 Tokens sowie Einzeldomänenvergleiche sind vorsichtig zu lesen" streichen, etwa 12 Wörter.
6. **Erhalten bleibt:** alle Einzelpunkte in 5.7.5; in 6.2 die Haupt-/Explorativ-Einteilung mit der Formulierung „aus FF1 und FF2 abgeleitet", die Grenze der Unsicherheitsangaben und die FF-Zuordnung.
7. **Risiken:** Die Limitationen sind danach nicht mehr eigenständig lesbar. Viele Gutachter lesen 6.2 isoliert.
8. **Wörter:** etwa 50, mildere Variante etwa 12.
9. **Seiteneffekt:** keiner; die letzte Seite von Kapitel 6 ist zu 99 % gefüllt. Die Kürzung schafft Reserve.
10. **Dichte:** steigt im Gesamttext; 6.2 allein wird weniger vollständig.

#### P-4 · P · Absatz „Anpassung und Checkpoint-Auswahl" enthält nichts zur Checkpoint-Auswahl (6.2)
1. **Ort:** 6.2, Z. 128–131 im Absatz „Entwicklungs- und Testtrennung"; Ziel Z. 168–177.
2. **Original:**
   ```tex
   Die nach F$_1$ gewählte Epoche fällt nur in einem von neun LLM- und in
   keinem von sechs Encoder-Läufen mit dem Verlustminimum zusammen; 12 von
   15 Läufen wählten die letzte Epoche, ein Untertraining ist nicht
   auszuschließen.
   ```
3. **Problem:** Die Überschrift „Anpassung und Checkpoint-Auswahl" verspricht diesen Inhalt, der Satz steht aber unter „Entwicklungs- und Testtrennung".
4. **Warum ändern:** Überschrift und Inhalt passen danach zusammen. Der Satz betrifft FF1 und FF2, wie der Zielabsatz.
5. **Vorschlag:** Satz unverändert vor „Betroffen sind FF1 und FF2." im Absatz „Anpassung und Checkpoint-Auswahl" einfügen. Der Satz zur Validierungsstichprobe bleibt in der Testtrennung, weil er die Datennutzung betrifft.
6. **Erhalten bleibt:** alles.
7. **Risiken:** Kapitel 6 ist bis zur letzten Zeile gefüllt; neue Zeilenumbrüche können einen Überlauf auf S. 83 auslösen. Nach dem Build prüfen.
8. **Wörter:** 0.
9. **Seiteneffekt:** keiner erwartet, Überlaufrisiko klein.
10. **Dichte:** gleich; die Gliederung wird genauer.

#### K6-b · C · Intervall in 6.1 zweimal genannt
1. **Ort:** 6.1 FF1, Z. 35–36.
2. **Original:** „Das Intervall dieser Differenz, [0,02; 0,33], liegt knapp über null und gehört zu den explorativen Vergleichen."
3. **Problem:** Das Intervall steht schon in Z. 27.
4. **Warum verdichtbar:** reine Wiederholung der Zahl.
5. **Vorschlag:** „Das Intervall dieser Differenz liegt knapp über null und gehört zu den explorativen Vergleichen."
6. **Erhalten bleibt:** Zahl in Z. 27, Einordnung.
7. **Risiken:** keine.
8. **Wörter:** etwa 3.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt minimal.

#### K6-c · C · Pareto-Werte in FF3 wiederholen FF2
1. **Ort:** 6.1 FF3, Z. 86–88.
2. **Original:** „Bei 256 Ausgabetokens bilden sie allein die Pareto-Front: DeBERTa-v3-base genügt mit 8,9~ms und 0,7~GiB dem engsten Budget, DeBERTa-v3-large bringt 0,3 Punkte mehr bei 16,5~ms und 1,7~GiB."
3. **Problem:** Die vier Werte stehen in FF2 (Z. 69–71).
4. **Warum verdichtbar:** Die Empfehlung kann ohne die Werte auskommen.
5. **Vorschlag:** „…: DeBERTa-v3-base genügt dem engsten Budget, DeBERTa-v3-large bringt 0,3 Punkte mehr bei höherer Latenz und höherem Speicherbedarf."
6. **Erhalten bleibt:** Werte in FF2.
7. **Risiken:** Die Empfehlung verliert ihre konkrete Begründung an Ort und Stelle.
8. **Wörter:** etwa 6.
9. **Seiteneffekt:** keiner.
10. **Dichte:** nicht eindeutig; die Lesbarkeit der Empfehlung sinkt.

#### K6-d · C · Klammer zum Typinventar doppelt zu 5.7.4
1. **Ort:** 6.1 FF3, Z. 99–100.
2. **Original:** „…; eine Erweiterung des Typinventars über den Prompt blieb ungeprüft (unveränderter Prompt, tabellarische Abbildung der CrossNER-Typen)."
3. **Problem:** 5.7.4 (Z. 1035–1036) nennt dieselbe Klammer.
4. **Warum verdichtbar:** Die Aussage trägt ohne Klammer.
5. **Vorschlag:** Klammer streichen.
6. **Erhalten bleibt:** Aussage; Details in 5.7.4.
7. **Risiken:** keine.
8. **Wörter:** etwa 5.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt minimal.

#### K6-e · C · „zehn wichen ab" (6.2 Ausgabelimit)
1. **Ort:** 6.2, Z. 142–143.
2. **Original:** „594 von 604 Präfixen wurden reproduziert, zehn wichen ab."
3. **Problem:** Die Zahl zehn folgt aus 604 minus 594.
4. **Warum verdichtbar:** arithmetisch redundant.
5. **Vorschlag:** „594 von 604 Präfixen wurden reproduziert."
6. **Erhalten bleibt:** beide Zahlen.
7. **Risiken:** Die Abweichung wird weniger sichtbar.
8. **Wörter:** etwa 3.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt minimal.

#### K6-f · C · LogitMatch im Ausblick erneut erklärt
1. **Ort:** 6.3, Z. 238–242.
2. **Original:** „Beim Decoding könnte LogitMatch, das die erzeugten Spans auf gültige Abschnitte der Eingabe beschränkt \cite[Abschn.~4]{semin2026spanLabeling}, die freie Erzeugung numerischer Wortindizes ersetzen, …"
3. **Problem:** 5.7.2 erklärt LogitMatch.
4. **Warum verdichtbar:** Der Ausblick kann den Namen mit Quelle nennen.
5. **Vorschlag:** „Beim Decoding könnte LogitMatch \cite[Abschn.~4]{semin2026spanLabeling} die freie Erzeugung numerischer Wortindizes ersetzen, …"
6. **Erhalten bleibt:** Vorschlag, Quelle, Begründung mit Verweis auf 5.7.2.
7. **Risiken:** Der Ausblick wird ohne 5.7.2 schwerer verständlich.
8. **Wörter:** etwa 10.
9. **Seiteneffekt:** keiner.
10. **Dichte:** steigt leicht.

**Nicht kürzen (Kapitel 6):** die drei FF-Absätze mit ihren Zahlen, weil das Fazit eigenständig lesbar sein muss; der Satz „…, was auf das Limit von 256 Tokens hinweist" (frühere Korrektur); die Absätze Ausgabelimit, Datenbasis und Effizienzmessung; die drei Anschlussfragen im Ausblick.

**Kurzbilanz Kapitel 6:** 5 Seiten. B-Option K6-a mit etwa 50 oder 12 Wörtern, C-Optionen mit etwa 27 Wörtern, P-4 ohne Wortänderung. Erwartete Seitenersparnis 0. Kürzungen schaffen hier vor allem Reserve, weil die letzte Seite zu 99 % gefüllt ist; jede spätere Ergänzung im Fazit kann sonst eine Seite 83 erzeugen. Das Kapitel ist präzise; der Statistik-Absatz doppelt 5.7.5, und ein Absatz passt nicht zu seiner Überschrift.

---

## 6 Übersicht

| Kapitel | Seiten | A | B | C | P | Erwartete Ersparnis | Bereits präzise? |
|---|---|---|---|---|---|---|---|
| 1 | 5 | ~10 W | – | – | P-1 (+5 W) | 0 Seiten | ja |
| 2 | 15 | ~43 W | – | ~100 W | – | 0 | weitgehend |
| 3 | 13 | ~72 W | ~16 W | ~27 W | – | 0 | weitgehend |
| 4 | 23 | ~39 W | ~82 W | ~4 W | P-3 | 0, unsicher 1 | ja, dicht |
| 5 | 21 | ~53 W | ~59 W | ~78 W | in K5-f | 0 | Ergebnisteil ja, Diskussion mit Rest-Doppelungen |
| 6 | 5 | – | ~50 W | ~27 W | P-4 | 0 | ja |
| **Summe** | 82 | **~217 W** | **~207 W** | **~236 W** | 4 | **0, höchstens 1** | |

**Umrechnung:** A ≈ 0,6 Seiten Text, A + B ≈ 1,1 Seiten, alle Kategorien ≈ 1,8 Seiten. Diese Menge verteilt sich auf sechs Kapitel. Kein Kapitel erreicht die 320 bis 370 Wörter, die für eine eingesparte Seite nötig wären.

---

## 7 Anhang und Vorspann – nur Hinweise, keine Änderung

**Anhang (nicht verändert, nicht gekürzt, nicht umgestellt):**
- Alle Labels im Anhang lösen auf; das Build-Log meldet 0 undefinierte Verweise.
- `tab:app-limit-seeds` wird nur innerhalb des Anhangs verwiesen, nicht aus dem Haupttext. Das ist kein Fehler, nur ein Hinweis.
- Die Anhangsbeschreibung in 1.4 ist unvollständig. Die Korrektur steht als P-1 in Kapitel 1; der Anhang selbst bleibt unberührt.
- Inhaltlich wurde der Anhang nicht geprüft.

**Vorspann (nicht verändert):**
- Das Inhaltsverzeichnis nennt das Abkürzungsverzeichnis auf Seite ix; es beginnt auf viii.
- Deckblatt und Eigenständigkeitserklärung tragen beide das Seitenlabel „i".
- Das Abgabedatum auf dem Deckblatt („Berlin, 25. November 2026") ist als TODO markiert und noch zu bestätigen.

---

## 8 Layout – nur Hinweise, keine Änderung

- **Leerraum durch die Überschriftenregel:** Auf S. 8, 21, 57, 60, 66, 73 und 81 endet der Text bei 70 bis 80 % der Textfläche, weil die nächste Überschrift sonst im unteren Bereich begonnen hätte. Zusammen sind das rund 1,8 Seiten Leerraum innerhalb der Kapitel.
- **Leerraum durch Floats:** S. 38, 43, 54, 76 und 80 enden früh, weil eine Tabelle oder Abbildung nicht mehr passte.
- **Bekannte Ausnahme:** 4.4.1 steht direkt unter 4.4; titlesec prüft die Regel bei direkt aufeinanderfolgenden Überschriften nicht.
- **Hebel, bewusst nicht genutzt:** Ein Wert von .25 statt .30 würde Leerraum sparen, ließ aber schon einmal Überschriften bei 80 bis 88 % stehen. Schriftgröße, Zeilenabstand, Ränder und globale Abstände bleiben wie vorgegeben unverändert.

---

## 9 Gesamtbewertung

- **Dichte:** Nach drei Kürzungsrunden ist die Arbeit dicht. Verlustfrei streichbar sind noch etwa 220 Wörter, mit den B-Vorschlägen etwa 420.
- **Muster der Restredundanz:** Dieselben Vorbehalte erscheinen kapitelübergreifend. „Kein isolierter Effekt" steht an etwa zehn Stellen, die Grenze der Skalierungsgesetze an fünf, die veröffentlichten MultiNERD-Werte an drei. Die Vorschläge behalten jeden Vorbehalt an seinem methodischen Ort, in der Diskussion und im Fazit und streichen nur zusätzliche Wiederholungen.
- **Seitenzahl:** Erwartet bleibt sie bei 116 physischen Seiten und 82 Seiten Haupttext. Die Seitenzahl hängt an Kapitelanfängen, Floats und der Überschriftenregel, kaum an einzelnen Sätzen.
- **Nutzen trotzdem:** höhere Dichte, eine korrigierte Sachangabe (K5-f), ein korrigierter Verweis (P-3), eine stimmige Gliederung in 6.2 (P-4) und Reserve in Kapitel 5 und 6, deren letzte Seiten zu 99 % gefüllt sind.
- **Kapitel 4 und 5:** Beide sind dicht. In Kapitel 4 wiederholt der Fließtext an zwei Stellen Tabelle 4.3. In Kapitel 5 wiederholt die Diskussion den Ergebnisteil an drei Stellen fast wörtlich.
- **Keine Notenaussage:** Die Vorschläge verbessern Präzision und Lesbarkeit. Eine Aussage über die Note folgt daraus nicht.

**Empfohlene Auswahl:**
1. Alle P-Vorschläge und alle A-Vorschläge, zusammen etwa 220 Wörter mit geringem Risiko.
2. Von den B-Vorschlägen vor allem K4-h, K5-c und K5-i. Bei K5-b und K6-a ist zu entscheiden, ob 5.7.5 oder 6.2 die vollständige Fassung der statistischen Grenzen trägt.
3. C-Vorschläge nur auf ausdrücklichen Wunsch.

**Nach Freigabe:** Umsetzung nur der gewählten Punkte, Anhang unverändert. Danach Build, Tests, Prüfung von Warnungen, Seitenzahlen, Umbrüchen und Floats, `git diff` auf `anhang.tex` leer, Vorher-nachher-Vergleich mit getrennten Mess- und Schätzwerten.
