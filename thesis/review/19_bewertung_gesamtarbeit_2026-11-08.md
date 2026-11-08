# Vorläufige Bewertung der Bachelorarbeit (So 8.11.2026)

Eigene, vorläufige Orientierung. Keine offizielle Note und kein Ersatz für die Bewertung durch Betreuer und Prüfende.

---

## 1 Grundlage der Bewertung

| Punkt | Befund |
|---|---|
| Beurteilte Fassung | `thesis/main.pdf`, erzeugt 8.11.2026 08:58 aus Commit `66cf6f7`; keine Quelldatei ist jünger. 116 physische Seiten: Vorspann i–ix (10 Seiten), Haupttext S. 1–82, Literatur S. 83–91, Anhang S. 92–106. |
| Nicht beurteilt | Die älteren PDF-Kopien `main_betreuerentwurf_cb9d29a.pdf`, `_3ce9d0d.pdf` und `_59a1363.pdf`; sie entsprechen nicht dem aktuellen Stand. |
| Gelesen | Der vollständige Text aller 116 Seiten, aus dem PDF extrahiert, einschließlich Tabellen, Listings, Literaturverzeichnis und Anhang. Abbildungen nur über Achsenbeschriftung, Legende und Caption; die Grafik selbst habe ich nicht erneut visuell geprüft. |
| Formale Prüfungen | Build-Log ohne Overfull-, Underfull- oder Verweisfehler; alle Labels lösen auf. |
| Fachgebiet und Art | Bachelorarbeit, B.Sc. Wirtschaftsinformatik, TU Berlin, Fakultät IV, Fachgebiet Quality & Usability; entstanden am DFKI (Speech and Language Technology). Empirisch-vergleichende Studie im Bereich NLP und maschinelles Lernen. |
| Bewertungsraster | Im Repository liegt keine Prüfungsordnung, kein Bewertungsraster und keine Seitenvorgabe vor. Ich verwende deshalb allgemeine, vorläufige Kriterien (Abschnitt 4) und die vorgegebene Notenstufung. |

**Wichtige Einschränkung zur Unabhängigkeit:** Ich habe in dieser und früheren Sitzungen große Teile des Textes mit überarbeitet und verdichtet. Meine Bewertung ist deshalb nicht unabhängig. Sie kann Stil und Entscheidungen begünstigen, an denen ich beteiligt war. Frühere KI-Bewertungen in diesem Projekt kamen auf etwa 1,7. Ich habe diese Berichte für diese Bewertung nicht erneut gelesen, kenne aber ihr Ergebnis.

---

## 2 Kurzfazit

**Stärkste Aspekte**
- **Methodische Transparenz auf sehr hohem Niveau:**
  - Konfundierende Faktoren werden ausdrücklich benannt (S. 35–36).
  - Jede Konfigurationsentscheidung wird ihrem Testbezug zugeordnet (Tabelle 4.3, S. 41).
  - Alle Abweichungen vom idealen Design stehen gesammelt in Tabelle 4.9 (S. 56).
  - Der Anhang belegt die Reproduzierbarkeit lückenlos mit Commits, Revisionen und Job-Daten.
- **Statistisch saubere Einordnung:**
  - Gepaarte Satz-Bootstrap-Intervalle.
  - Klare Trennung von Haupt- und explorativen Vergleichen.
  - Ausdrücklich keine Gleichwertigkeitsaussage aus Intervallen, die null enthalten (S. 48–49, Tabelle 5.9).
- **Umfang und technische Durchführung:**
  - 18 Testevaluationen auf MultiNERD, darunter ein 27B-Modell mit QLoRA und drei Seeds.
  - Ressourcenmessung, Formtreue-Analyse, Fehlerprofile und ein CrossNER-Transfertest mit Sensitivitätsanalyse.
  - Das geht deutlich über den üblichen Rahmen einer Bachelorarbeit hinaus.
- **Kritische Literaturarbeit:** Quellen werden geprüft statt nur referiert, etwa die unklaren MultiNERD-Testumfänge bei GoLLIE und InstructUIE (S. 26–27, 31).
- **Vorsichtige, belegte Schlussfolgerungen:** Kaum eine Aussage geht über die Daten hinaus.

**Größte Risiken**
- **Ein Pipeline-Parameter prägt den Kernbefund zu FF1:**
  - Das Ausgabelimit von 256 Tokens wurde ohne Längenanalyse gesetzt (S. 43).
  - Die Korrektur erfolgt nur als selektive, nachträgliche Nachinferenz auf anderer Hardware (S. 53, 62–63).
  - Die Arbeit macht daraus einen Befund. Prüfende können darin aber auch einen vermeidbaren Designfehler sehen.
- **Viele selbst verursachte Einschränkungen:**
  - Testsplit in Vorläufen genutzt; Zero-Shot-Setup nach Testbeobachtungen gewählt (S. 39–41).
  - Double Quantization wegen eines Konfigurationsfehlers inaktiv; SFT-Verlust einschließlich Prompt; in drei von vier Schichten bleibt die Attention ohne Adapter (S. 44, 46).
  - Checkpoint-Wahl der LLMs auf 200 nicht zufällig gezogenen Sätzen (S. 44); drei GPU-Typen (S. 42).
  - Jede einzelne ist offen dokumentiert. Zusammen begrenzen sie die Beweiskraft.
- **Lesbarkeit:** Die Kapitel 4 und 5 sind sehr dicht und mit Vorbehalten gesättigt. Kernbotschaften gehen zwischen Zahlen und Einschränkungen leicht unter.
- **FF3 bleibt qualitativ:** Die Frage nennt „gegebene Speicher- und Latenzbeschränkungen" (S. 3). Konkrete Budgets werden aber nie festgelegt; im Fazit erscheint „unter 2 GiB" ohne vorherige Herleitung (S. 79).
- **Unklar bleibt, was nur Prüfende beurteilen können:** Umfang (82 Seiten Haupttext), Sprache des Abstracts und die Gewichtung der umfangreich erklärten KI-Unterstützung.

**Vorläufige Gesamteinschätzung:** **1,7 (17)**. Realistischer Bereich **1,3 (13) bis 2,0 (20)**, am wahrscheinlichsten 1,3 bis 1,7. Sicherheit: mittel bis gering.

---

## 3 Bewertung je Kapitel

Die Kapitelnoten sind diagnostisch. Sie zeigen, wo die Arbeit stark oder verbesserungsbedürftig ist, und sind keine Teilnoten.

| Kapitel | Note | Spanne | Sicherheit |
|---|---|---|---|
| Vorspann und Formalia | 1,3 (13) | 1,0–1,7 | mittel |
| 1 Einleitung | 1,3 (13) | 1,3–1,7 | mittel bis hoch |
| 2 Grundlagen | 1,3 (13) | 1,3–1,7 | mittel |
| 3 Verwandte Arbeiten | 1,3 (13) | 1,0–1,7 | mittel |
| 4 Methodik | 1,7 (17) | 1,3–2,0 | mittel |
| 5 Ergebnisse und Diskussion | 1,7 (17) | 1,3–1,7 | mittel |
| 6 Fazit und Ausblick | 1,3 (13) | 1,3–1,7 | mittel |
| Literaturverzeichnis | 1,0 (10) | 1,0–1,3 | mittel bis hoch |
| Anhang | 1,0 (10) | 1,0–1,3 | mittel bis hoch |

### Vorspann und Formalia (S. i–ix)

**Note: 1,3 (13).** Sicherheit mittel. Maßgeblich sind Vollständigkeit, Korrektheit und Vorgabenkonformität.

**Stärken**
- **Abstract:** dicht und vollständig, mit Ergebnis und Einschränkung (S. ii).
- **Verzeichnisse und KI-Angabe:** vollständig; die KI-Angabe nennt Hersteller, Modelle und Nutzungsart (S. i).

**Schwächen und Risiken (beobachtbar)**
- **Inhaltsverzeichnis:** Es führt das Abkürzungsverzeichnis auf S. ix, es beginnt aber auf S. viii.
- **Seitenlabel:** Deckblatt und Erklärung tragen beide „i".
- **Abstract:** Es steht nur auf Deutsch, unter der Überschrift „Abstract". Ob eine englische Fassung oder eine deutsche „Kurzfassung" verlangt ist, kann ich nicht prüfen.
- **KI-Angabe:** Für Juni bis Oktober nennt sie „jeweils aktuelle Modelle" statt Versionen.

**Wichtigste Änderungen**
1. **Seitenverweis und Label korrigieren:** kurzfristig, rein formal.
2. **Abstract-Anforderung klären:** Gegebenenfalls eine englische Fassung ergänzen; kurzfristig machbar.

**Nicht sprachlich behebbar:** keine.

### Kapitel 1 – Einleitung (S. 1–5)

**Note: 1,3 (13).** Spanne 1,3–1,7, Sicherheit mittel bis hoch. Maßgeblich sind Problemstellung, Forschungsfragen, Beitrag und Abgrenzung.

**Stärken**
- **Klare Motivation:** Leistung gegen Aufwand, mit belegten Zahlen aus der Literatur (S. 1–2).
- **Präzise Forschungsfragen:** FF1 bis FF3, klar getrennt; FF1 schließt die Formtreue ausdrücklich ein (S. 3).
- **Beitrag konkret benannt:** acht Konfigurationen, 18 Evaluationen, eigene Encoder-Referenz in derselben Pipeline (S. 3).
- **Ehrliche Abgrenzung:**
  - Die Abweichung vom angemeldeten Titel wird begründet.
  - Die Konfundierung der Regime ist ausdrücklich benannt: „isolierte kausale Effekte … nicht bestimmt" (S. 4–5).

**Schwächen und Risiken**
- **Übergeneralisierung (S. 2):** „Mit LoRA angepasste LLMs schließen diesen Abstand." Das gilt so nicht. Kapitel 3 berichtet bei Q. Lu et al. einen bleibenden Rückstand (S. 29), und die eigenen Ergebnisse zeigen bei 256 Tokens einen Rückstand des LLM.
- **Schwer lesbarer Absatz (S. 2–3):** Er zählt vier Aspekte auf und verweist zugleich auf zwei Lücken in 3.5 („Erstens … viertens").
- **Budget nicht operationalisiert:** FF3 spricht von „gegebenen Speicher- und Latenzbeschränkungen" (S. 3), ohne dass später Budgets definiert werden (siehe Kapitel 4 und 6).
- **Beispiel passt schlecht zum Datensatz:** Die Motivation nennt Arztbriefe und Gerichtsentscheidungen (S. 1). Die Daten sind Wikipedia-Texte. Das ist kein Fehler, schwächt aber die Brücke zur Anwendung.

**Wichtigste Änderungen**
1. **Satz auf S. 2 präzisieren:** etwa „verringern diesen Abstand deutlich; in einzelnen Vergleichen liegen sie gleichauf oder vorn". Kurzfristig.
2. **FF3 an Budgetszenarien knüpfen**, sobald diese in Kapitel 4 definiert sind (Punkt A1). Kurzfristig.
3. **Absatz auf S. 2–3 entflechten:** zuerst die zwei Lücken, dann die vier Aspekte als deren Inhalt. Kurzfristig, sprachlich.

**Nicht sprachlich behebbar:** keine.

### Kapitel 2 – Grundlagen (S. 6–20)

**Note: 1,3 (13).** Spanne 1,3–1,7, Sicherheit mittel. Maßgeblich sind fachliche Korrektheit, Definitionen, Relevanz für die Methode und Fokus.

**Stärken**
- **Fachlich korrekt und gut belegt:** eigene Formalisierung der NER als Tripel (S. 7), präzise Darstellung von IOB1 und IOB2 (S. 7–8).
- **Klare Erklärung der zentralen Verfahren:** DeBERTa-v3 (S. 12), LoRA und QLoRA mit allen drei Komponenten (S. 15–16). Darauf stützen sich später die Abweichungen in Tabelle 4.9.
- **Hripcsak und Rothschild:** F₁ als Übereinstimmung mit einer Referenz (S. 20). Das ist für eine Silberreferenz methodisch genau der richtige Gedanke.
- **Sauberer Methodenbezug:** seqeval-Modi und ihre Folgen für generative Ausgaben (S. 20).
- **Rückverweise:** Jeder Abschnitt verweist auf seine Verwendung in Kapitel 4.

**Schwächen und Risiken**
- **Inhalte ohne spätere Funktion:**
  - Die Exponenten und Konstanten der Skalierungsgesetze samt Abbildung 2.4 (S. 17) dienen nur der Aussage, dass sie hier nicht gelten.
  - Dazu kommen Next Sentence Prediction (S. 12) und die Schichtzahl des Original-Transformers (S. 10).
  - Das kostet Fokus, nicht Korrektheit.
- **Länge:** 15 Seiten Grundlagen sind für eine Bachelorarbeit viel. Ob das als Gründlichkeit oder als Ballast gilt, entscheiden die Prüfenden.

**Wichtigste Änderungen**
1. **Abschnitt 2.7 straffen:** Den Abschnitt zu Skalierungsgesetzen auf das beschränken, was die Begründung in 4.5.4 trägt. Kurzfristig, nur wenn dadurch der Fokus steigt.
2. **Sonst nichts Wesentliches:** Das Kapitel erfüllt seine Funktion.

**Nicht sprachlich behebbar:** keine.

### Kapitel 3 – Verwandte Arbeiten (S. 21–33)

**Note: 1,3 (13).** Spanne 1,0–1,7, Sicherheit mittel. Maßgeblich sind Auswahl, kritische Nutzung, Synthese und Herleitung der Lücke.

**Stärken**
- **Rechercheverfahren offengelegt:** Datenbanken, Stichwörter, Schneeballsuche und ausdrücklich keine Systematik (S. 21).
- **Quellenkritik statt Nacherzählung:**
  - Die 32 908 Testbeispiele bei GoLLIE gegenüber dem Testsplit (S. 27).
  - Übernommene BERT-Werte bei UniversalNER (S. 26).
  - Die Herkunft der Baselines bei Zhan (S. 28).
- **Zwischenfazit mit echter Synthese:** etwa zur Rolle des Ausgabeformats (S. 27–28).
- **Tabelle 3.1:** ordnet alle Arbeiten nach Vergleichsart und Ressourcenmessung (S. 32).
- **Saubere Lückenherleitung:** zwei Lücken, den Forschungsfragen zugeordnet, Erwartungen E1 bis E3 ausdrücklich als Lesehilfe ohne Hypothesentest (S. 31–33).

**Schwächen und Risiken**
- **Recherchezeitraum:** Er ist mit 2019 bis 2025 angegeben (S. 21). Die Arbeit erscheint im November 2026 und zitiert mehrere Arbeiten aus 2026. Prüfende könnten fragen, ob die Literatur von 2026 systematisch gesichtet wurde.
- **Beitrag eher inkrementell:** Die Lücke betrifft die Reichweite der bisherigen Vergleiche, nicht eine neue Frage (S. 31: „die Forschungslücke liegt in ihrer Reichweite"). Für eine Bachelorarbeit ist das angemessen, für eine Spitzenbewertung eher eine Grenze.
- **Dichte:** Einzelne Absätze reihen viele Studienwerte aneinander, etwa S. 23 und 26. Die analytische Linie trägt aber.

**Wichtigste Änderungen**
1. **Recherche für 2026 nachziehen:** gezielt für 2026, mit den gleichen Stichwörtern. Das Ergebnis mit Recherchedatum angeben und relevante neue Arbeiten einordnen oder ausdrücklich „keine weiteren einschlägigen" vermerken. Kurzfristig, mittlerer Aufwand.
2. **Den eigenen Beitrag in 3.5 in einem Satz zuspitzen**, etwa als erste gemeinsame Auswertung auf den offiziellen Splits mit Ressourcen und Größenstaffelung. Kurzfristig.

**Nicht sprachlich behebbar:** Der inkrementelle Charakter der Lücke.

### Kapitel 4 – Methodik (S. 34–56)

**Note: 1,7 (17).** Spanne 1,3–2,0, Sicherheit mittel. Maßgeblich sind Eignung des Designs, Nachvollziehbarkeit, Begründung der Entscheidungen und Umgang mit Verzerrungen. Die Dokumentation läge für sich allein bei 1,0. Die Note sinkt wegen der Designqualität.

**Stärken**
- **Explizites Versuchsdesign:** Variablen, Konstanten und Störgrößen sind getrennt benannt (S. 35–36).
- **Tabelle 4.3 (S. 41):** ordnet jede Konfigurationsentscheidung ihrem Testbezug zu: direkt, indirekt, post hoc oder keiner. Das ist ungewöhnlich ehrlich und methodisch reif.
- **Vollständig beschriebene Nachverarbeitung:**
  - Deterministische Validierungskette mit vier Regeln und Konfliktauflösung (S. 46–48).
  - Bestätigte Hin- und Rückkonvertierung der Referenz (S. 39).
- **Effizienzmessung präzise definiert:**
  - Aufwärmlauf, GPU-Synchronisation, ausgeschlossene Schritte (S. 50–51).
  - Allokierter statt reservierter Speicher, offen benannt.
- **Statistik:** Gepaartes Bootstrap mit Quellen (Efron und Tibshirani, Koehn), Hauptvergleiche vorab aus den Forschungsfragen abgeleitet (S. 48–49).
- **Nachinferenz:** Die Prüfung der Annahme ist mitgedacht, über identische Präfixe und Kontrollsätze (S. 53).
- **Tabelle 4.9 (S. 56):** Alle Abweichungen an einer Stelle.

**Schwächen und Risiken (im Text belegt)**
- **Ausgabelimit:** 256 Tokens „vor den Läufen ohne vorherige Längenanalyse festgelegt" (S. 43). Das Limit prägt später den Kernbefund. Eine Längenanalyse der Trainingsantworten hätte das Problem vorab gezeigt.
- **Testnutzung:**
  - Der Testsplit wurde in Vorläufen ausgewertet.
  - Zero-Shot-Prompt und Sampling wurden nach Testbeobachtungen gewählt.
  - Die Präzision wurde auf einer Testteilmenge gewählt (S. 39–41).
  - Ein unberührter Testsplit existiert damit für keine Konfiguration.
- **Schwaches Zero-Shot-Setup:**
  - Sampling mit Temperatur 1,0, ein einziger Lauf je Größe, die Einstellung aus einem 0.8B-Vorlauf übernommen (S. 43–44).
  - Das macht Erwartung E3 fast trivial erfüllbar und begrenzt die Aussage des Regimes.
- **QLoRA-Umsetzung mit Mängeln:**
  - Double Quantization „wegen eines fehlerhaft benannten Konfigurationsparameters nicht aktiviert".
  - Verlust über die gesamte Sequenz einschließlich Prompt.
  - Die Attention der Gated-DeltaNet-Schichten, drei von vier Schichten, bleibt ohne Adapter; adaptiert werden dort nur die Feed-Forward-Netze (S. 44, 46).
  - Ob das die LLMs benachteiligt, bleibt offen.
- **Checkpoint-Wahl:**
  - Bei den LLMs auf 200 nicht zufällig gezogenen Validierungssätzen ohne BIO und INST, bei den Encodern auf dem vollständigen Split (S. 44).
  - 12 von 15 Läufen wählten die letzte Epoche. Ein Untertraining ist nicht auszuschließen.
- **Hardware und Hyperparameter:**
  - Drei GPU-Typen; nur ein hardwaregleicher Encoder-LLM-Vergleich (S. 42, 51).
  - Keine Hyperparametersuche (S. 39).
- **Budgets für FF3 nicht operationalisiert:** Abschnitt 4.5.2 definiert keine Budgetszenarien.
- **Kleinigkeiten:**
  - Im Fließtext steht ein Repository-Pfad (`thesis/generated/validation_subset_stats.md`, S. 44).
  - Die hohe Detaildichte erschwert das Lesen. Das Kapitel umfasst 23 Seiten.

**Fachliche Einschätzung:** Jede Einzelentscheidung ist begründet oder als Einschränkung ausgewiesen. Die Zahl der Abweichungen ist für ein Vergleichsdesign aber hoch, und einige wären mit Vorplanung vermeidbar gewesen. Prüfende werden die Transparenz positiv sehen. Ob sie die Mängel als normale Studienrealität oder als Planungsschwäche werten, ist offen.

**Wichtigste Änderungen**
1. **Budgetszenarien in 4.5.2 definieren:** zwei oder drei, etwa „Latenz unter 50 ms und Speicher unter 2 GiB", „interaktiv unter 10 s mit einer 24-GiB-GPU" und „Offline-Verarbeitung ohne Latenzgrenze, Batching nicht gemessen". Auf sie stützen sich dann 5.7.4 und 6.1. Kurzfristig umsetzbar und für FF3 wesentlich.
2. **Gründe für die Abweichungen kurz einordnen:** Tabelle 4.9 um eine Spalte „vermutete Wirkrichtung" ergänzen, etwa „benachteiligt eher das LLM", „unklar" oder „betrifft nur Zero-Shot". So sehen Prüfende sofort, welche Abweichung welches Ergebnis wohin verschieben könnte. Kurzfristig, mittlerer Gewinn.
3. **Repository-Pfad auf S. 44** durch einen Verweis auf den Anhang oder das Repository ersetzen. Kurzfristig, formal.

**Nicht sprachlich behebbar:** Ausgabelimit, Testnutzung, Zero-Shot-Setup, QLoRA-Mängel und Hardwaremix. Dafür braucht es neue Läufe (Abschnitt 5, Kategorie D).

### Kapitel 5 – Ergebnisse und Diskussion (S. 57–77)

**Note: 1,7 (17).** Spanne 1,3–1,7, Sicherheit mittel. Maßgeblich sind Darstellung, Verhältnis von Beschreibung und Deutung, Verknüpfung mit Tabellen und Abbildungen, Angemessenheit der Schlüsse und Umgang mit Einschränkungen.

**Stärken**
- **Systematische Ergebnisdarstellung:**
  - Klare Trennung von Ergebnissen (5.1 bis 5.6) und Diskussion (5.7), angekündigt auf S. 57.
  - Konsistente Rundungsregeln (S. 57).
- **Text, Tabellen und Abbildungen greifen sauber ineinander:**
  - Tabelle 5.1 (S. 58), Abbildung 5.1 (S. 60), Tabelle 5.2 (S. 61), Tabelle 5.4 und Abbildung 5.3 (S. 65–66), Tabelle 5.8 als Synthese (S. 71), Tabelle 5.9 mit allen Intervallen (S. 77).
  - Zahlen im Text sind jeweils einer Tabelle zugeordnet.
- **Analyse statt Beschreibung:**
  - Die Recall-Lücke des LLM wird bis auf die abgeschnittenen Antworten zurückverfolgt, mit Recall auf genau diesen Sätzen für beide Modelle (S. 67).
  - Die TIME-Abweichung im Zero-Shot-Regime wird an Definition und Ziffernanteil festgemacht (S. 60).
- **CrossNER mit Sensitivitätsanalyse:** Sie zeigt eine Umkehr der Rangfolge (S. 69–70). Daraus folgt ein echter, für die Bewertung generativer NER relevanter Befund.
- **Literaturbezug in 5.7.1 differenziert:**
  - Abweichung von Obeidat nach Precision und Recall erklärt.
  - Das Fehlermuster von Zhan geprüft und nicht bestätigt (S. 70–72).
- **Belastbarkeit gesondert und nüchtern bewertet:** 5.7.5, inklusive Makro-Intervallen, die null enthalten (S. 76–77).

**Schwächen und Risiken**
- **Kernbefund FF1 bedingt:**
  - Bei 256 Tokens liegt der Encoder 1,34 Punkte vorn, nach selektiver Nachinferenz 0,18 Punkte (S. 70).
  - Der zweite Wert ist ausdrücklich explorativ und stammt von anderer Hardware.
  - Die Arbeit sagt das korrekt. Die Hauptaussage bleibt damit aber „hängt vom Ausgabelimit ab".
- **FF3 nur qualitativ beantwortet:**
  - Aussage: „gut gestützte Wahl … bei niedriger Latenztoleranz" (S. 75).
  - Es fehlt eine Entscheidungshilfe je Budget. Die Pareto-Front ist gezeigt, aber nicht in Szenarien übersetzt.
- **Erklärungen der CrossNER-Umkehr ungeprüft:**
  - Abbildung der Fachtypen, Vortrainingswissen, Annotationsunterschiede.
  - Die Fachtypen-Zusätze könnten aus den gespeicherten Vorhersagen ohne neue Läufe ausgezählt werden. S. 75 sagt: „domänenweise ausgezählt wurden sie nicht".
- **Silberreferenz nur illustrativ:** Der Vorbehalt wird an ausgewählten Beispielen gestützt, nicht quantitativ (S. 68, 72). Eine kleine manuelle Prüfung der Abweichungen würde ihn messbar machen.
- **Präzisionsmängel:**
  - **S. 58:** „Die Validierungs-F₁-Werte der ausgewählten Checkpoints (Tabelle A.6) liegen rund drei Punkte unter den Testwerten." Das gilt nur für die Encoder. Bei den LLMs liegen die Validierungswerte auf 200 Sätzen über den Testwerten (Tabelle A.6, S. 102). Der Satz sollte eingeschränkt werden.
  - **S. 61:** „Da nur 55 der 16 454 Referenzantworten länger als 256 Tokens sind …, fallen die Modellantworten länger aus als die Referenz." Die Schlusskette ist zu knapp; es fehlt der Vergleich mit rund 65 abgeschnittenen Modellantworten je Lauf und deren mittlerer Länge von 337 Tokens.
- **Dichte und Vorbehalte:**
  - Viele Absätze kombinieren vier bis sechs Zahlen mit zwei Einschränkungen, etwa S. 62, 67, 69.
  - Die Diskussion wiederholt Einschränkungen, die schon in Kapitel 4 stehen.
  - Kernaussagen sind vorhanden, aber nicht hervorgehoben.
- **Effizienz implementierungsabhängig:**
  - Der Latenzfaktor von etwa 210 gilt für HF-Generierung mit NF4-Dequantisierung bei Batchgröße 1.
  - Die Arbeit benennt das (S. 73, 76).
  - Praktisch relevanter wären Werte mit Batching oder einer Inferenz-Engine. Das braucht neue Messungen.

**Fachliche Einschätzung:** Die Darstellung ist auf hohem Niveau. Die Schlüsse sind durch die Daten gedeckt und eher zu vorsichtig als zu weit. Die Note wird nicht von der Schreibqualität begrenzt, sondern davon, dass die zentrale Vergleichsaussage auf einer nachträglich korrigierten Pipeline beruht und FF3 ohne operationalisierte Budgets bleibt.

**Wichtigste Änderungen**
1. **FF3 je Budgetszenario beantworten:** In 5.7.4 nach den Szenarien aus 4.5.2, mit den vorhandenen Werten. Ungemessenes, etwa Batching, als offen kennzeichnen. Kurzfristig, wesentlicher Gewinn.
2. **Fachtypen-Zusätze auf CrossNER auszählen:** aus den gespeicherten Vorhersagen, je Domäne. Damit wird eine der drei ungeprüften Erklärungen der Umkehr geprüft. Kurzfristig, ohne GPU, mittlerer Gewinn.
3. **Manuelle Prüfstichprobe der Abweichungen von der Silberreferenz** (Punkt A2). Kurzfristig machbar mit rund ein bis zwei Arbeitstagen; wesentlicher Gewinn für 5.5, 5.7.1 und 6.2.
4. **Zwei Präzisionsmängel beheben:** den Validierungssatz auf S. 58 auf die Encoder einschränken, die Schlusskette auf S. 61 ausformulieren. Kurzfristig.
5. **Je Unterabschnitt eine Kernaussage voranstellen:** in 5.1, 5.2, 5.4, 5.6 und 5.7.x ein bis zwei Sätze. Kurzfristig, Lesbarkeit.

**Nicht sprachlich behebbar:**
- Die Abhängigkeit des Kernbefunds vom Ausgabelimit.
- Die Hardwareheterogenität der Effizienzwerte.
- Das schwache Zero-Shot-Setup.

### Kapitel 6 – Fazit und Ausblick (S. 78–82)

**Note: 1,3 (13).** Spanne 1,3–1,7, Sicherheit mittel. Maßgeblich sind die Beantwortung der Forschungsfragen, Bündelung, Grenzen und Ausblick.

**Stärken**
- **Jede Forschungsfrage beantwortet:**
  - Eigener Absatz je Frage mit Abgleich der Erwartungen.
  - E1 wird ehrlich als „nur teilweise bestätigt" eingestuft (S. 78).
- **Limitationen nach Themen gruppiert:** Jede trägt den Vermerk, welche Forschungsfrage betroffen ist (S. 80–82). Das ist sehr nachvollziehbar.
- **Priorisierter, konkreter Ausblick:** Er folgt aus den eigenen Grenzen; die drei Anschlussfragen bauen logisch aufeinander auf (S. 82).
- **Keine neuen Ergebnisse im Schluss.**

**Schwächen und Risiken**
- **Kein ausdrücklicher Beitrag:**
  - Es fehlt ein Satz, was die Arbeit dem Forschungsstand hinzufügt.
  - Die methodische Lehre zum Ausgabelimit steht nur im Abstract („Bedeutung ausreichender Ausgabelängen", S. ii).
- **FF-Antworten zahlenlastig:** Die Bündelung ist präzise, aber eher ein dichtes Resümee als eine Verdichtung zu Kernaussagen (S. 78–79).
- **Budgetgrenze ohne Vorlauf:** „unter 2 GiB" erscheint erst hier (S. 79).
- **Limitationen ausführlicher als das Fazit:** zweieinhalb Seiten Limitationen, eineinhalb Seiten Antworten. Inhaltlich gerechtfertigt; es verstärkt aber den Eindruck einer stark eingeschränkten Studie.

**Wichtigste Änderungen**
1. **Abschließenden Absatz zum Beitrag ergänzen:** drei oder vier Sätze mit den Kernbefunden und der methodischen Lehre für generative NER, etwa „Ausgabelimits vor der Evaluation an der Antwortlänge prüfen". Kurzfristig, mittlerer bis wesentlicher Gewinn.
2. **FF3 im Fazit an die Budgetszenarien anbinden** (Punkt A1). Kurzfristig.

**Nicht sprachlich behebbar:** Die Breite der Limitationen. Sie folgt aus dem Design.

### Literaturverzeichnis (S. 83–91)

**Note: 1,0 (10).** Spanne 1,0–1,3, Sicherheit mittel bis hoch.

**Stärken**
- **Einheitlich:** APA-Stil durchgehend, DOIs, Preprints gekennzeichnet.
- **Nachprüfbar:** Abrufdaten bei Webquellen, versionsgenaue Software- und Modellkarten-Einträge.
- **Annahmestatus belegt:** Die Annahme von Zhan et al. ist mit dem arXiv-Eintrag belegt (S. 90–91).

**Schwächen**
- **Recherchestand:** siehe Kapitel 3.
- **Ungewöhnliche Quelle für das Format:** Die Wortindex-Variante wird mit einer Arbeit zur Fallacy-Erkennung belegt (Ramponi et al., Tab. 9). Das ist korrekt zitiert, aber erklärungsbedürftig. Es ist in 3.2.1 bereits erklärt.

**Wichtigste Änderung:** keine zwingende.

### Anhang (S. 92–106, nur bewertet, nicht verändert)

**Note: 1,0 (10).** Spanne 1,0–1,3, Sicherheit mittel bis hoch. Maßgeblich sind Vollständigkeit, Nachvollziehbarkeit, Verweise und Funktion.

**Stärken**
- **Lückenlose Reproduzierbarkeit:** Commits, Revisionen, Job-IDs, Konfigurationsdateien, Ausgabeverzeichnisse (Tabellen A.1 bis A.4).
- **Vollständige Prompts** (Listings A.1 und A.2).
- **Seed-Einzelwerte und Typwerte:** mit Precision und Recall.
- **Fehlerbeispiele** mit Satzindex (Tabelle A.14).
- **Funktion erfüllt:** Der Anhang entlastet Kapitel 4 und 5 und macht jede Zahl prüfbar.

**Schwächen**
- **Lauf-Artefakte nicht öffentlich:** Sie werden „auf Anfrage" bereitgestellt (S. 92).
- **Zero-Shot-Läufe ohne Revisions-Pins:** Das ist ein Mangel der Läufe, nicht des Anhangs.

**Änderungen:** keine; der Anhang bleibt wie vorgegeben unverändert.

---

## 4 Vorläufige Gesamtbewertung

**Gesamtnote: 1,7 (17).**
- **Realistischer Bereich:** 1,3 (13) bis 2,0 (20).
- **Am wahrscheinlichsten:** 1,3 bis 1,7.
- **Sicherheit:** mittel bis gering.

**Gewichtung als Orientierung, nicht als Formel:**
- **Methodik und Durchführung, Ergebnisse und Diskussion:** zusammen etwa die Hälfte. Hier liegt die eigentliche wissenschaftliche Leistung.
- **Fragestellung und Literatur:** etwa ein Viertel.
- **Darstellung, Sprache, Formalia und Schluss:** etwa ein Viertel.

Die Gesamtnote ist kein Mittel der Kapitelnoten. Sie folgt aus dem Kern der Arbeit.

**Begründung**
- **Passung:** Forschungsfrage, Methode, Ergebnisse und Schluss passen eng zusammen. Jede Forschungsfrage wird mit Daten beantwortet, und jede Schlussfolgerung bleibt im Rahmen der Daten.
- **Zentrale Leistung:** Ein großer, technisch anspruchsvoller Vergleich mit sauberer Statistik und vorbildlicher Dokumentation. Das trägt eine sehr gute Note.
- **Begrenzung durch das Design:** Mehrere Einschränkungen betreffen den Kern:
  - der Kernbefund zu FF1 hängt am Ausgabelimit;
  - kein unberührter Testsplit;
  - QLoRA-Mängel;
  - Hardwaremix bei den Effizienzwerten.
  
  Die Arbeit gleicht das durch Transparenz und Sensitivitätsanalysen teilweise aus, aber nicht vollständig.
- **Zentral oder klein:** Die Darstellungsschwächen in Dichte und Lesbarkeit sind klein und ausgleichbar. Die Designmängel sind zentral, weil sie die Beweiskraft der Hauptvergleiche begrenzen. Das trennt die Arbeit derzeit von einer 1,0.

**Gründe für eine bessere Bewertung**
- Umfang und technische Tiefe über dem Bachelorniveau.
- Wissenschaftliche Redlichkeit, sichtbar etwa in Tabelle 4.3, Tabelle 4.9, 5.7.5 und 6.2.
- Ein eigener, verallgemeinerbarer Befund zum Ausgabelimit mit messbarer Umkehr auf CrossNER.
- Kritische Literaturarbeit und eine sehr gute Reproduzierbarkeit.

**Risiken gegen eine bessere Bewertung**
- **Designmängel als Planungsschwäche:** Prüfende können sie so werten, statt als normale Studienrealität. Vor allem gilt das für das Ausgabelimit ohne Längenanalyse und den Konfigurationsfehler bei Double Quantization.
- **Lesbarkeit und Umfang:** 82 Seiten Haupttext mit hoher Dichte. Eine mögliche Seitenvorgabe ist mir nicht bekannt.
- **Gewichtung der KI-Unterstützung:** Die Erklärung nennt Entwurf, Überarbeitung und Verdichtung von Text „in allen Kapiteln" sowie Hilfe am Code. Wie das bei der Bewertung der Eigenständigkeit zählt, entscheiden allein die Prüfenden nach den Regeln der TU Berlin.
- **FF3 ohne operationalisierte Budgets.**

**Fehlende Informationen für eine genauere Einschätzung**
- Das Bewertungsraster des Fachgebiets oder der Prüfenden.
- Seiten- und Sprachvorgaben (Umfang, Abstract).
- Erwartungen von DFKI und Fachgebiet an Neuheit gegenüber Durchführung.
- Die Haltung zur KI-Nutzung.
- Die Bearbeitungsdauer und ob die Datenerhebung vor der Anmeldung als zulässige Vorarbeit gilt.

---

## 5 Priorisierter Verbesserungsplan

Legende:
- **Wirkung:** W = wissenschaftliche Qualität, L = Sprache und Lesbarkeit, K = kosmetisch.
- **Gewinn:** wesentlich, mittel oder gering.

### A. Hohe Priorität vor der endgültigen Abgabe

**A1 – Budgetszenarien für FF3 definieren und beantworten.** W, wesentlich.
- **Problem:** FF3 fragt nach gegebenen Speicher- und Latenzbeschränkungen (S. 3). Die Arbeit legt aber keine fest; „unter 2 GiB" erscheint erst im Fazit (S. 79).
- **Stelle:** 1.2, 4.5.2, 5.7.4, 6.1.
- **Warum:** FF3 ist die praktische Frage der Arbeit. Ohne Budgets bleibt die Antwort allgemein.
- **Handlung:**
  - In 4.5.2 zwei bis drei Szenarien mit Latenz- und Speichergrenzen definieren, etwa „eingebettet", „interaktiv" und „offline".
  - Dabei begründen, warum gerade diese Grenzen gewählt wurden.
  - In 5.7.4 und 6.1 je Szenario die beste gemessene Konfiguration nennen.
  - Für das Offline-Szenario ausdrücklich festhalten, dass Batching nicht gemessen wurde.
- **Risiko:** Die Szenarien sind nachträglich gewählt. Das muss offen gesagt werden, sonst wirken sie passend gemacht.

**A2 – Manuelle Prüfstichprobe zur Silberreferenz.** W, wesentlich.
- **Problem:** Ob Abweichungen Modell- oder Annotationsfehler sind, wird nur illustrativ gestützt (S. 68, 72).
- **Stelle:** 5.5, 5.7.1, 6.2.
- **Warum:** Die Silberreferenz ist die wichtigste inhaltliche Einschränkung aller MultiNERD-Werte. Eine Messung würde den Vorbehalt in einen Befund verwandeln und einen sichtbaren eigenen Analyseanteil schaffen.
- **Handlung:**
  - Aus den gespeicherten Vorhersagen eine geschichtete Zufallsstichprobe ziehen, etwa 150 bis 200 Abweichungen: Zusätze, Auslassungen und Typfehler, je Encoder und LLM, mit Schwerpunkt auf ANIM, FOOD und PLANT.
  - Jede Abweichung nach den MultiNERD-Richtlinien einordnen: Referenz korrekt, Vorhersage korrekt oder unklar.
  - Anteile mit Konfidenzintervall berichten.
- **Risiko:**
  - Nur ein Annotator, nicht blind gegenüber dem Modell. Das Protokoll muss das offenlegen.
  - Aufwand etwa ein bis zwei Arbeitstage.
  - Neue Zahlen müssen in 5.5, 5.7.1 und 6.2 konsistent eingearbeitet werden.

**A3 – Beitrag und Kernbefunde im Fazit ausdrücklich formulieren.** W und L, mittel bis wesentlich.
- **Problem:** Es fehlt ein Absatz, der den Beitrag und die methodische Lehre bündelt (S. 78–82).
- **Stelle:** Ende von 6.1 oder ein kurzer Schlussabsatz nach 6.3.
- **Handlung:** Drei oder vier Sätze. Kernbefunde zu FF1 bis FF3 ohne Intervalle, dazu die Lehre für generative NER: Ausgabelimit an der Antwortlänge prüfen, abgeschnittene Antworten getrennt ausweisen.
- **Risiko:** Keine neuen Aussagen einführen. Kapitel 6 ist auf der letzten Seite voll; der Absatz kann eine Seite hinzufügen.

**A4 – Recherche für 2026 nachziehen und datieren.** W, mittel.
- **Problem:** Der Recherchezeitraum endet 2025 (S. 21), die Abgabe ist im November 2026.
- **Stelle:** Kapitelanfang 3; gegebenenfalls 3.2, 3.3 und Tabelle 3.1.
- **Handlung:**
  - Gezielte Suche für 2026 mit den vorhandenen Stichwörtern.
  - Recherchedatum angeben.
  - Neue einschlägige Arbeiten kurz einordnen oder ausdrücklich festhalten, dass keine weiteren gefunden wurden.
- **Risiko:** Eine neue, stark einschlägige Arbeit kann die Lücke in 3.5 berühren. Dann muss die Einordnung angepasst werden.

**A5 – Präzisionsmängel beheben.** W, gering bis mittel; kurzfristig.
- **S. 2:** „schließen diesen Abstand" relativieren.
- **S. 58:** Die Aussage zu Validierung und Test auf die Encoder beschränken und den umgekehrten Befund bei den LLMs mit der 200-Satz-Teilmenge nennen.
- **S. 61:** Die Schlusskette zu den Antwortlängen ausformulieren.
- **S. 44:** Den Repository-Pfad im Fließtext ersetzen.

### B. Mittlere Priorität

**B1 – Fachtypen-Zusätze auf CrossNER auszählen.** W, mittel.
- **Problem:** S. 75 nennt die Abbildung der Fachtypen als ungeprüfte Erklärung der Umkehr und sagt: „domänenweise ausgezählt wurden sie nicht".
- **Handlung:** Aus den gespeicherten Vorhersagen je Domäne zählen, wie viele falsch positive Grobtyp-Spans auf nicht abgebildete Fachtypen fallen. Das Ergebnis kurz in 5.6 oder 5.7.4 berichten.
- **Risiko:** gering; kein neuer Lauf nötig.

**B2 – Kernaussage je Ergebnisabschnitt.** L, mittel.
- **Stelle:** 5.1 bis 5.6, 5.7.x.
- **Handlung:** Ein bis zwei Sätze am Anfang oder Ende jedes Abschnitts, ohne neue Zahlen.
- **Risiko:** Wiederholung vermeiden; nicht dieselben Zahlen erneut nennen.

**B3 – Tabelle 4.9 um die vermutete Wirkrichtung ergänzen.** W, mittel.
- **Handlung:** Eine Spalte mit Werten wie „eher zulasten LLM", „eher zugunsten Zero-Shot" oder „unklar", jeweils mit Kurzbegründung.
- **Risiko:** Nur einschätzen, was sich begründen lässt; „unklar" ist eine zulässige Antwort.

**B4 – Lange Sätze entflechten.** L, mittel.
- **Stelle:** vor allem 1.2 (S. 2–3), 4.3.3 (S. 39–40), 5.7.1 (S. 70–72) und 6.1 (S. 78–79).
- **Handlung:** Sätze mit mehr als zwei Einschränkungen teilen; Zahlen in Klammern reduzieren, wo Tabellen sie tragen.

**B5 – Kapitel 2 auf die Funktion fokussieren.** L, gering bis mittel.
- **Handlung:** Die Einzelheiten zu Skalierungsgesetzen (S. 17) auf das Nötige beschränken. Nur dort kürzen, wo der Fokus steigt und nichts Benötigtes verloren geht.

### C. Niedrige Priorität und Feinschliff

- **C1:** Den Verweis des Inhaltsverzeichnisses auf das Abkürzungsverzeichnis korrigieren (ix statt viii). K.
- **C2:** Das doppelte Seitenlabel „i" von Deckblatt und Erklärung beheben. K.
- **C3:** Abstract-Sprache und Deckblattdatum mit den Vorgaben abgleichen. K, wird zu A, falls eine englische Fassung Pflicht ist.
- **C4:** Einzelne Wiederholungen in den Vorbehalten kürzen, nur wenn verlustfrei. L.

### D. Nicht kurzfristig oder nicht ohne neue Arbeit lösbar

**D1 – Vollständiger Neulauf aller LLM-Testinferenzen mit ausreichendem Ausgabelimit auf einheitlicher Hardware.**
- **Wirkung:** Der Kernbefund zu FF1 und die Ressourcenwerte für FF3 würden sauber. Den Abstand zwischen Haupt- und Sensitivitätsanalyse gäbe es nicht mehr.
- **Aufwand:** Neun QLoRA-Läufe mit 8 bis 25 GPU-Stunden je Lauf, also grob 150 bis 200 GPU-Stunden. Danach wären Zahlen in Abstract und Kapitel 5 und 6 zu aktualisieren.
- **Einschätzung:** Mit freier Clusterkapazität in zwei bis drei Wochen technisch möglich, aber riskant. Nur sinnvoll, wenn der Betreuer das ausdrücklich für wesentlich hält.

**D2 – Zero-Shot-Regime neu aufsetzen.**
- **Inhalt:** Prompt und Decoding auf Validierungsdaten wählen; Greedy-Lauf plus mehrere Sampling-Seeds je Größe.
- **Aufwand:** Mittel bis hoch. Ein vollständiger Satz aller drei Größen braucht laut Tabelle 5.4 rund 41 GPU-Stunden; Greedy plus drei Sampling-Seeds sind also etwa 160 GPU-Stunden.
- **Wirkung:** Es würde E3 und die Zero-Shot-Werte belastbar machen. Für den Kern der Arbeit ist es weniger wichtig als D1.

**D3 – Auswertung auf der manuell geprüften MultiNERD-Stichprobe.**
- **Inhalt:** 1 000 englische Sätze, veröffentlicht von Tedeschi und Navigli.
- **Wirkung:** Sie würde den Hauptvergleich gegen eine Goldreferenz prüfen und die Silberreferenz-Einschränkung direkt adressieren.
- **Aufwand:** Nur Inferenz, also gering. Für die LLMs grob 1 000 Sätze × 9 Läufe × 2 bis 6 s, also rund 5 bis 15 GPU-Stunden; für die Encoder Minuten.
- **Voraussetzung:** Zuerst prüfen, ob die Stichprobe zugänglich ist und ob sie mit dem Trainingssplit überlappt. Bei Überlappung nur die nicht überlappenden Sätze auswerten.
- **Einschätzung:** Von allen neuen Analysen das beste Verhältnis von Aufwand zu Gewinn. Mit dem Betreuer abstimmen.

**D4 – Hyperparametersuche, vollständiges LLM-Fine-Tuning, korrigierte QLoRA-Konfiguration, Kontaminationsprüfung, Effizienz mit Batching oder Inferenz-Engine.** Alles nicht kurzfristig; als Ausblick bereits genannt.

**Abgrenzung der Wirkung:** Wissenschaftliche Qualität verbessern vor allem A1, A2, A4, B1, B3 und D1 bis D3. Lesbarkeit verbessern A3, B2, B4 und B5. Kosmetisch sind C1 bis C3.

---

## 6 Einschätzung zum Ziel 1,0

**Was eine 1,0 plausibel machen würde**
- Eine klar formulierte Frage mit sauberem Design, dessen Hauptvergleiche nicht von nachträglichen Korrekturen abhängen.
- Ergebnisse, die belastbar und eindeutig interpretierbar sind.
- Ein erkennbarer eigener Erkenntnisbeitrag über die reine Messung hinaus.
- Eine Darstellung, die Prüfende ohne Mühe durch die Argumentation führt.
- Fehlerfreie Formalia.

**Bereits sichtbar**
- Transparenz, Reproduzierbarkeit und statistische Sorgfalt auf einem Niveau, das für eine 1,0 spricht.
- Umfang und technische Durchführung.
- Kritische Literaturarbeit.
- Ein eigener Befund mit Allgemeinwert: Das Ausgabelimit verändert die Rangfolge auf CrossNER.

**Was derzeit gegen eine 1,0 spricht**
- **Zentrale Designmängel:**
  - Ausgabelimit ohne Längenanalyse.
  - Kein unberührter Testsplit.
  - Ein schwaches Zero-Shot-Setup.
  - Fehler in der QLoRA-Konfiguration.
  - Hardwaremix bei den Effizienzwerten.
  
  Sie sind offen benannt, begrenzen aber die Beweiskraft der Hauptvergleiche.
- **FF3 ohne operationalisierte Budgets.**
- **Hohe Dichte, schwer lesbar.**
- **Beitrag eher inkrementell:** Er liegt in der Reichweite der Vergleiche, nicht in einer neuen Frage.

**Kurzfristig verbesserbar:** FF3 mit Budgets (A1), messbarer Silberreferenz-Vorbehalt (A2), ausdrücklicher Beitrag (A3), Literatur 2026 (A4), Präzision und Lesbarkeit (A5, B2, B4).

**Nur mit neuen Läufen:** Ein sauberer Kernvergleich (D1), ein belastbares Zero-Shot-Regime (D2), ein Abgleich mit einer Goldreferenz (D3).

**Urteil: derzeit nicht plausibel als erwartbares Ergebnis, aber nicht ausgeschlossen.**
- **Warum nicht ausgeschlossen:** Prüfende, die Transparenz und Umfang stark gewichten, können eine 1,0 vergeben.
- **Warum nicht erwartbar:** Nach allgemeinen Maßstäben für empirische Abschlussarbeiten begrenzen die Designmängel die Note eher auf den Bereich 1,3 bis 1,7.
- **Realistisch bestmöglich:** Eine 1,3 (13) ist mit den A-Punkten realistisch erreichbar. Das hängt vor allem von der Gewichtung der Designmängel durch die Prüfenden ab, weniger vom Text.
- **Mögliche Stärkung:** D3 und eventuell D1 könnten den Kernbefund stärken. Ob sie die Bewertung verschieben, kann nur der Betreuer einschätzen.

---

## 7 Fragen an den Betreuer

**Konkrete Fragen zu den Bewertungsrisiken**
1. Reicht die Hauptauswertung bei 256 Tokens mit selektiver Nachinferenz aus, oder erwarten Sie vor der Abgabe einen vollständigen Neulauf mit ausreichendem Limit auf einheitlicher Hardware?
2. Wie bewerten Sie die Testnutzung in den Vorläufen und die testbasierte Wahl des Zero-Shot-Setups? Genügt die offene Dokumentation in Tabelle 4.3?
3. Lohnt sich eine Auswertung auf der manuell geprüften MultiNERD-Stichprobe, oder alternativ eine eigene Prüfstichprobe der Abweichungen?
4. Gibt es Vorgaben zu Umfang, Abstract (Deutsch, Englisch oder beides) und Gliederung? Sind 82 Seiten Haupttext angemessen?
5. Ist die KI-Angabe in Form und Umfang ausreichend? Wie fließt die KI-Unterstützung in die Bewertung der Eigenständigkeit ein?
6. Erwarten Sie für FF3 konkrete Budgetszenarien, oder genügt die qualitative Antwort über die Pareto-Front?
7. Ist die Abweichung des Inhalts vom angemeldeten Titel (BERT-basierte Modelle und LoRA) so akzeptiert?
8. Nach welchen Kriterien bewerten Sie und das Fachgebiet? Gibt es ein Raster?

**Punkte, für die Sie um Priorisierung bitten sollten**
- A1 bis A5 gegenüber neuen Läufen (D1 bis D3): Was bringt nach Ihrer Einschätzung mehr?
- Detailtiefe in Kapitel 4: Soll etwas in den Anhang? Vorher Ihre ausdrückliche Zustimmung, der Anhang bleibt sonst unverändert.
- Lesbarkeit gegenüber Vollständigkeit der Vorbehalte in Kapitel 5 und 6.
- Ob die Literatur von 2026 systematisch nachgezogen werden soll.

**Kritik sortieren nach Umsetzbarkeit**
- **Sofort umsetzbar:**
  - Formulierungen und Präzisionsmängel;
  - Budgetszenarien;
  - Kernaussagen;
  - Beitragsabsatz;
  - Formalia;
  - Abstract-Sprache;
  - Auszählung der Fachtypen.
- **Zusätzliche inhaltliche Arbeit:**
  - manuelle Prüfstichprobe;
  - Literatur 2026;
  - Auswertung auf der Goldstichprobe;
  - Neuläufe mit höherem Limit oder neues Zero-Shot-Setup;
  - Änderungen am QLoRA-Setup.

**Vorschlag zum Sortieren des Feedbacks**

| Klasse | Kriterium | Beispiel |
|---|---|---|
| Muss geändert werden | Formfehler, Vorgabenverstoß, sachlicher Fehler oder ausdrücklicher Änderungswunsch des Betreuers | fehlendes englisches Abstract, falsche Zahl |
| Sollte geändert werden | Betrifft eine Forschungsfrage oder die Beweiskraft und ist bis zur Abgabe machbar | Budgetszenarien, Prüfstichprobe |
| Kann geändert werden | Verbessert Lesbarkeit oder Stil, ohne Inhalt zu ändern | Kernaussagen, Satzlängen |
| Mit Begründung diskutieren | Erfordert neue Läufe oder widerspricht einer bewussten, dokumentierten Entscheidung | vollständiger Neulauf, andere Modellwahl |

Je Feedbackpunkt sollten Kapitel, Seite, Klasse, Aufwand in Stunden und Abhängigkeiten notiert werden. Wenn Sie das Feedback bereitstellen, ordne ich es den Kapiteln zu, vergleiche es mit dieser Einschätzung und passe die Noten mit derselben Skala nachvollziehbar an.

---

## 8 Unsicherheiten

- **Nicht unabhängig:** Ich habe am Text mitgearbeitet. Stil und Entscheidungen, die ich mitgeprägt habe, bewerte ich möglicherweise zu günstig.
- **Kein offizielles Raster:** Die Gewichtung ist meine Annahme. Ein Fachgebiet, das Neuheit stärker gewichtet als Durchführung, würde eher strenger urteilen. Eines, das Sorgfalt und Umfang stark gewichtet, eher milder.
- **KI-Unterstützung:** Wie Prüfende den erklärten Umfang bewerten, ist nicht abschätzbar. Das kann die Note unberührt lassen oder spürbar beeinflussen.
- **Umfang:** Ohne bekannte Seitenvorgabe kann ich nicht beurteilen, ob 82 Seiten positiv, neutral oder negativ wirken.
- **Abbildungen:** Ihre Gestaltung habe ich in dieser Bewertung nur über Beschriftungen und Captions geprüft.
- **Kapitel 4 und 5:** Die Einordnung hängt stark davon ab, ob Prüfende die Designmängel als Planungsschwäche oder als normale, gut dokumentierte Studienrealität sehen. Deshalb reicht die Spanne bei Kapitel 4 bis 2,0.
- **Fachlich nicht nachgeprüft:** Die Korrektheit der Messwerte habe ich nur über die Konsistenz zwischen Text, Tabellen und Anhang geprüft, nicht gegen die Rohdaten auf dem Cluster.
