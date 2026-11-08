# Prüfbericht: Informationsdichte, Präzision und wissenschaftliche Qualität (So 8.11.2026)

Nur Analyse. An Quellen, PDF und Anhang wurde nichts geändert. Umsetzung erst nach Freigabe einzelner Punkte.

---

## 1 Geprüfte Version und Umfang

**Quellstand**
- Geprüft wurde `main` auf Commit `35fef59`, synchron mit `origin/main` nach `git pull` (Stand 8.11., 11:00).
- Der Stand von Kapiteln, Vorspann, `main.tex`, `references.bib` und `generated/` ist identisch mit `66cf6f7`. Das war die letzte Änderung an diesen Dateien (8.11., 08:58). Danach kamen nur Review-Dateien hinzu: `722e0a3` (Bericht 19) und `35fef59` (Begleitnotiz).
- Damit ist dies derselbe Textstand, den Bericht 19 bewertet hat.

**PDF**
- Geprüft wurde `thesis/main.pdf` vom 8.11., 08:58:12, 1 183 253 Byte.
- **Zugehörigkeit verifiziert:** Ein frischer Build aus `git archive HEAD` im Scratchpad ergibt 116 Seiten mit derselben Bytezahl, und der extrahierte Text aller 116 Seiten ist identisch (pypdf).
- Build: Exit 0, keine Overfull- oder Underfull-Boxen, keine undefinierten Verweise.

**Arbeitskopie**
- Nicht committet sind nur Review-Dateien der parallelen Codex-Sitzung: `review/19_…` (geändert um 09:22), `review/README.md` sowie die unversionierten Berichte 09 und 17.
- Keine davon betrifft Kapitel, Vorspann oder Anhang. Ich habe sie nicht angefasst.
- **Achtung, Bericht 19:** Die Codex-Änderung an Bericht 19 enthält einen Sachfehler. Dort steht, Abschnitt 5.7.4 setze ein Speicherbudget „unter 2 GiB" an. Tatsächlich steht diese Grenze nur in 6.1 (Z. 84, S. 79) und beschreibend im Abstract; 5.7.4 nennt keine Budgetgrenze (`grep GiB` über Kapitel 5 und 6). Der Fehler sollte vor einem Commit dieser Änderung korrigiert werden.

**Vollständigkeit**
- Haupttext, Literaturverzeichnis (85 Einträge) und Anhang liegen vollständig vor.
- Es gibt keine `\PH`-Platzhalter. Offen ist nur das als TODO markierte Abgabedatum auf dem Deckblatt.

**Einschränkungen dieser Prüfung**
- Grundlage waren die LaTeX-Quellen aller Kapitel und des Anhangs, vollständig gelesen, sowie der PDF-Text.
- Abbildungen wurden in dieser Runde nicht neu gerendert und angesehen; das geschah in früheren Runden.
- Messwerte wurden nicht erneut gegen die Rohdaten auf dem Cluster abgeglichen. Geprüft wurden nur Stichproben gegen die generierten Tabellen (`tab-haupt`, `tab-app-seeds`, `tab-effizienz`).
- Quellen wurden nur dort gegen lokale Volltexte geprüft, wo ein Vorschlag davon abhängt: Tedeschi et al. 2022 und Dettmers et al. 2023.
- Die Verfügbarkeit der manuell geprüften MultiNERD-Stichprobe wurde online geprüft (Abschnitt 6.3).
- Eine Literaturrecherche für 2026 wurde **nicht** durchgeführt.
- **Unabhängigkeit:** Die Bewertung ist nicht unabhängig, weil ich große Teile des Textes mitüberarbeitet habe. Als Gegenprobe hat ein separater Prüfer (Sonnet) die Arbeit ohne Kenntnis der bisherigen Noten und Berichte bewertet (Abschnitt 8).

### Seitenumfang (exakt, aus PDF und `build/main.toc`)

| Teil | Seiten | Gedruckte Zählung |
|---|---:|---|
| PDF physisch | **116** | – |
| Vorspann (Deckblatt, Erklärung, Abstract, Verzeichnisse) | 10 | i, i, ii–ix |
| Kapitel 1 Einleitung | 5 | 1–5 |
| Kapitel 2 Grundlagen | 15 | 6–20 |
| Kapitel 3 Verwandte Arbeiten | 13 | 21–33 |
| Kapitel 4 Methodik | 23 | 34–56 |
| Kapitel 5 Ergebnisse und Diskussion | 21 | 57–77 |
| Kapitel 6 Fazit und Ausblick | 5 | 78–82 |
| **Haupttext** | **82** | 1–82 |
| Literaturverzeichnis | 9 | 83–91 |
| Anhang (A.1 S. 92, A.2 S. 96, A.3 S. 101) | 15 | 92–106 |

**Die Zahl von 116 Seiten ist bestätigt.**

**Wortzahlen** (Fließtext ohne Tabellen, Abbildungen und Gleichungen; `scripts/check/words.py`):

| Teil | Wörter |
|---|---:|
| Kapitel 1 | 1 534 |
| Kapitel 2 | 4 143 |
| Kapitel 3 | 4 042 |
| Kapitel 4 | 5 488 |
| Kapitel 5 | 5 350 |
| Kapitel 6 | 1 554 |
| **Haupttext** | **22 111** |
| Anhang (nur Fließtext) | 1 737 |
| Abstract | 316 |

**Füllung der letzten Kapitelseite**
- S. 5: 83 %
- S. 20: 90 %
- S. 33: 98 %
- S. 56: 90 %
- S. 77: 97 %
- S. 82: laut Bericht 18 rund 99 %

Eine Seite pro Kapitel spart man erst, wenn dort Text im Umfang fast der ganzen letzten Seite wegfällt, also etwa 300 bis 370 Wörter.

**Früh endende Seiten im Haupttext** (Textende bei 70 bis 79 % der Seite):
- 8, 21, 49, 50, 54, 57, 66, 73, 74, 76, 80 und 81;
- Ursache sind die Überschriftenregel oder Gleitobjekte, keine Textmenge.

---

## 2 Kurzfazit

**Stärken, im Text belegt**
1. **Offengelegte Datennutzung:** Tabelle 4.3 ordnet jeder Konfigurationsentscheidung Datengrundlage und Testbezug zu. Tabelle 4.9 sammelt 13 Abweichungen mit Abschnittsverweis.
2. **Statistik:** Die Arbeit verwendet gepaarte Satz-Bootstrap-Intervalle mit sauberer Lesart: Ein Intervall mit null belegt keine Gleichwertigkeit (4.5.1, Z. 878–881). Haupt- und explorative Vergleiche sind getrennt, und die Grenzen der Intervalle sind benannt (4.5.1, 5.7.5).
3. **Der eigene Befund trägt:** Bei 256 Tokens liegt der Encoder 1,34 [0,99; 1,68] Punkte vorn, nach Nachinferenz nur 0,18 [0,02; 0,33]. Auf CrossNER kehrt sich die Rangfolge um (Tab. 5.8 und 5.9). Die Formtreue-Analyse belegt den Mechanismus: Antworten enden ohne schließende Klammer, und die Auslassungen konzentrieren sich auf die abgeschnittenen Sätze.
4. **Kritische Literaturarbeit:** Ein Beispiel ist die Unstimmigkeit beim GoLLIE-Testumfang (32 908 Beispiele, genau das Doppelte des Testsplits; 3.2.3).

**Größte Risiken**
1. **Hauptvergleich auf fehlkonfigurierter Einstellung:** Der vorab definierte Hauptvergleich läuft bei 256 Tokens, also mit der fehlkonfigurierten Einstellung. Die korrigierte Aussage ist nur eine post-hoc-Sensitivitätsanalyse auf anderem GPU-Typ. Das begrenzt FF1 im Betrag und auf CrossNER in der Richtung.
2. **FF3 nicht operationalisiert:** FF3 fragt nach Konfigurationen „bei gegebenen Speicher- und Latenzbeschränkungen" (1.2, Z. 141–144). Kein Kapitel legt Budgets fest. Erst das Fazit nennt „unter 2 GiB" und das „engste Budget" (6.1, Z. 84–88), ohne Herleitung.
3. **Viele Abweichungen in der Summe:** Testnutzung in Vorläufen, Zero-Shot-Setup nach Testbeobachtungen, drei GPU-Typen, Checkpoint-Wahl auf 200 nicht zufällig gezogenen Sätzen. Jede Abweichung ist offengelegt; zusammen wirken sie wie eine Planungsschwäche.
4. **Neu gefundene Sachungenauigkeiten:** Sie lassen sich alle redaktionell beheben (A8, A10, A16, A17).
   - Die Validierungsaussage auf S. 58 gilt nur für die Encoder.
   - Die Schlusskette auf S. 61 ist zu pauschal.
   - Die Erklärung, warum Zero-Shot schneller ist, trifft auf das 27B-Modell nicht zu (S. 65).
   - Der CrossNER-Lauf mit 1 024 Tokens ist schneller als der mit 256 Tokens, der Text geht darüber hinweg (S. 69).

**Vorläufige Einschätzung**
- Note **1,7 (17)**, Bereich 1,3 (13) bis 2,0 (20). Begründung und Gegenprobe in Abschnitt 8.
- **Länger, nicht kürzer:** Verlustfrei lässt sich die Arbeit nicht mehr nennenswert kürzen. Die A-Vorschläge sparen netto rund 70 Wörter und keine Seite. Die wertvollsten Verbesserungen (FF3-Schwellen, Beitragsabsatz, Quantifizierung der Double Quantization) machen sie um 0,5 bis 1 Seite länger.

---

## 3 Vergleich mit der bisherigen Einschätzung

### 3.1 Stärken

| Bisherige Aussage | Befund am Text | Urteil |
|---|---|---|
| Transparenz über Tab. 4.3 und 4.9 | Beide vorhanden und vollständig verwiesen. Tab. 4.3 trennt „direkt", „indirekt" und „post hoc"; Tab. 4.9 hat 13 Zeilen mit Abschnittsverweisen. Keine Zeile für den nicht protokollierten Stoppgrund, der steht aber in 4.5.3 und 4.5.5. | **bestätigt** |
| Gepaarte Bootstrap-Intervalle, Haupt- getrennt von explorativen Vergleichen | 4.5.1, Z. 865–896; 5.7.5; 6.2. **Einschränkung:** Gerade der Hauptvergleich steht auf der Einstellung mit bekanntem Mangel, siehe Risiko 1. | **bestätigt, mit Einschränkung** |
| Umfang über Bachelorniveau, Quellen kritisch geprüft | Belege: Herkunft der Baselines bei Zhan („nennt keine Herkunft der Werte", 3.3), Testumfang bei GoLLIE, Lesart der Silberreferenz (2.9, Z. 788–793). Der Umfang von 82 Seiten Haupttext ist eher eine Frage an die Prüfenden als ein Qualitätsmerkmal. | **bestätigt** |
| Ausreichendes Limit kehrt die Rangfolge auf CrossNER um | 5.6, Z. 724–735: gepoolt null ausgeschlossen, domänenweise nur in drei von fünf Domänen. Post hoc, nur PER/ORG/LOC, Typabbildung ungeprüft (5.7.4, Z. 1031–1040). | **bestätigt als explorativer Befund** |

### 3.2 Risiken

Kategorien:
- (a) offengelegte Einschränkung;
- (b) begrenzt die Interpretation;
- (c) gefährdet eine zentrale Schlussfolgerung;
- (d) behebbar durch Text, Analyse oder Experiment.

| Risiko | Fundstellen | Einordnung | Folgen |
|---|---|---|---|
| Limit 256 ohne Längenanalyse | 4.4.3, Z. 569–575; Tab. 4.3; 4.5.5; 6.2 | (a) + (b); für die **Rangfolge DeBERTa-v3-large gegen 27B** knapp an (c) | Die Kernaussage „Abstand hängt vom Limit ab" ist so formuliert und gedeckt. Die exakte Rangfolge bei ausreichendem Limit beruht auf einer selektiven Nachinferenz auf anderer Hardware: 0,18 [0,02; 0,33], also knapp. Behebbar nur durch einen vollständigen Neulauf (D1). |
| Testsplit in Vorläufen genutzt | 4.3.3, Z. 371–388; Tab. 4.3; 6.2 | (a) + (b) | Bei den trainierten Konfigurationen blieben die Hyperparameter unverändert, mit Ausnahme der 27B-Epochenzahl. Die Verzerrung ist daher eher klein, aber nicht bezifferbar. |
| Zero-Shot-Setup nach Testbeobachtung gewählt | 4.4.3, Z. 589–601; Tab. 4.3 | (a) + (b) | Die Verzerrung wirkt **zugunsten** von Zero-Shot. Die Schlussfolgerung „Zero-Shot liegt weit zurück" (6,2 bis 69,4 gegenüber 93 bis 95) bleibt daher robust. Unsicher sind nur die absoluten Zero-Shot-Werte. Behebbar durch D2. |
| Double Quantization inaktiv | 4.4.5, Z. 737–747; Tab. 4.9 | (a), Wirkung klein | Laut Dettmers et al. (2023, Abschn. 3) spart Double Quantization rund 0,37 Bit je Parameter und kostet keine Leistung. Für das 27B-Modell sind das höchstens rund 1,2 GiB der gemessenen 18,7 GiB. Die Pareto-Aussage bleibt davon unberührt. Im Text bisher nur qualitativ („vor allem den Speicherbedarf"), siehe B5. |
| Drei GPU-Typen | Tab. 4.4; 4.5.2; 5.4; 6.2 | (a) + (b) | Der zentrale Effizienzbefund stützt sich auf den hardwaregleichen Vergleich auf der H100 (Faktor 210) und auf Unterschiede in Größenordnungen (ms gegenüber s). Laut A.2 macht der GPU-Typ etwa den Faktor 2 aus (8,9 gegenüber 17,6 ms). Nicht gefährdend. |
| FF3 ohne Budgets | 1.2, Z. 141–146; 4.5.2 ohne Budget; 5.7.4 qualitativ; 6.1, Z. 82–88 | (b), (d) durch Text behebbar | Größte **textlich** behebbare Lücke, siehe B1. |

### 3.3 Kapitelnoten

| Teil | Bisher | Jetzt | Separater Prüfer | Abweichung und Grund |
|---|---:|---:|---:|---|
| Vorspann und Formalia | 1,3 (13) | 1,3 (13) | 1,7 (17) | bestätigt; Formfehler im Inhaltsverzeichnis (A15) |
| 1 Einleitung | 1,3 (13) | 1,3 (13) | 1,3 (13) | bestätigt; Übergeneralisierung S. 2 (A1), FF3-Formulierung |
| 2 Grundlagen | 1,3 (13) | 1,3 (13) | 1,7 (17) | bestätigt; der Prüfer findet das Kapitel zu breit |
| 3 Verwandte Arbeiten | 1,3 (13) | 1,3 (13) | 1,3 (13) | bestätigt; Recherchezeitraum widersprüchlich (B4) |
| 4 Methodik | 1,7 (17) | 1,7 (17) | 2,0 (20) | bestätigt, mit Risiko Richtung 2,0; neu: A4, A5, B3, B5 |
| 5 Ergebnisse und Diskussion | 1,7 (17) | 1,7 (17) | 1,7 (17) | bestätigt; neu vier Sachungenauigkeiten (A8, A10, A16, A17) |
| 6 Fazit und Ausblick | 1,3 (13) | **1,7 (17)** | 1,7 (17) | **verändert:** Budget ad hoc (S. 79), kein Beitragsabsatz, E3 zu glatt |
| Literaturverzeichnis | 1,0 (10) | 1,0 (10) | 1,3 (13) | bestätigt; der Prüfer zieht wegen zentraler Preprints ab |
| Anhang (nur bewertet) | 1,0 (10) | 1,0 (10) | 1,3 (13) | bestätigt; der Prüfer nennt Artefakte „auf Anfrage" und nicht rekonstruierbare Vorlaufstände |
| **Gesamt** | **1,7 (17)** | **1,7 (17)** | **1,7 (17)** | Bereich bei allen 1,3 bis 2,0 |

Begründung der Gesamtnote in Abschnitt 8.

---

## 4 Kapitelweise Prüfung

### Vorspann
**Stärken**
- Abstract mit Zahlen, Grenzen und Kernbefund.
- KI-Angabe ausgefüllt.
- Abkürzungsverzeichnis mit dokumentierten Aufnahmeregeln.

**Schwächen**
- Das Inhaltsverzeichnis führt das Abkürzungsverzeichnis auf S. ix; es beginnt auf viii. Ursache: `\addcontentsline` steht in `main.tex`, Z. 220, hinter dem zweiseitigen Verzeichnis.
- Deckblatt und Erklärung tragen beide die Zählung „i".
- Das Abgabedatum ist ein TODO.
- Nur ein deutsches Abstract; ob ein englisches verlangt ist, ist offen.

**Einsparpotenzial:** keines.

### Kapitel 1 (5 Seiten)
**Stärken**
- Die drei Forschungsfragen sind klar.
- Die Abgrenzung in 1.3 benennt Konfundierung, Silberreferenz und nicht Gemessenes.

**Schwächen**
- **Übergeneralisierung, Z. 86:** „Mit LoRA angepasste LLMs schließen diesen Abstand." Das stimmt so nicht. Q. Lu et al. finden einen bleibenden Rückstand (3.3, Z. 457–460), und die eigenen Ergebnisse bei 256 Tokens zeigen ebenfalls einen.
- **Verschachtelter Absatz, Z. 98–110:** Vier Aspekte werden zu zwei Lücken gebündelt, die erst in 3.5 heißen, was sie sind. Der vierte Aspekt („während Größenstaffelungen nur innerhalb jeweils einer Modellfamilie untersucht wurden") ist ohne den Zusatz „ohne Encodervergleich" missverständlich, denn auch diese Arbeit staffelt nur innerhalb einer Familie.
- **FF3 setzt Budgets voraus,** die die Arbeit nicht festlegt (Z. 141–144).

**Verbesserungen:** A1, A2; B1 für FF3.

**Einsparpotenzial:** etwa 0. A1 kostet +11 Wörter, A2 spart etwa 15.

### Kapitel 2 (15 Seiten)
**Stärken**
- Fachlich genau, mit Seiten- und Tabellenangaben.
- Jeder Abschnitt endet mit dem Bezug zur eigenen Arbeit.
- Die Grenzen der Skalierungsgesetze sind sauber hergeleitet (2.7, Z. 611–621).

**Schwächen**
- Definitionen aus Kapitel 1 kehren fast wörtlich wieder: NLP (K1 Z. 16–18 und K2 Z. 18–20), LLM (K1 Z. 41–43 und K2 Z. 355–358). In einer Einleitung ist das üblich und kein Mangel.
- **Unpräzise, Z. 795–799:** „während Speicherbedarf, Durchsatz und Trainingszeit gesondert gemessen werden". Gemessen wird nur der Inferenzspeicher. Der Trainingsspeicher ist ausdrücklich nicht erfasst (1.3, 4.5.2), und Durchsatz nur ohne Batching.

**Verbesserung:** A3.

**Einsparpotenzial:** gering. Die früher als C eingestuften Kürzungen (Audit 18: NSP, Schichtzahl; dazu die OSI-Definition in 2.5) bleiben C.

### Kapitel 3 (13 Seiten)
**Stärken**
- Die Studien werden geprüft, nicht nur referiert.
- Tab. 3.1 ordnet die eigene Arbeit ein.
- Die Erwartungen E1 bis E3 sind ausdrücklich als Lesehilfe ohne Hypothesentest gekennzeichnet.

**Schwächen**
- **Widerspruch im selben Absatz (Z. 20–25):** Preprints aus 2026 werden ausdrücklich einbezogen („darunter zwei Arbeiten aus dem Jahr 2026"), der Recherchezeitraum ist aber mit „2019 bis 2025" angegeben.
- **Prioritätsaussage hängt am Recherchestand:** Die Aussage „soweit bekannt, erstmals" in 5.7.2 (Z. 900) hängt direkt an diesem Stand.
- Die Lücke ist inkrementell (Reichweite statt neue Frage). Für eine Bachelorarbeit ist das angemessen.

**Verbesserung:** B4.

**Einsparpotenzial:** gering. Die GLiNER- und NuNER-Details in 3.1.3 ließen sich straffen (C), tragen aber den Befund in 3.4 zu Größe gegenüber Architektur.

### Kapitel 4 (23 Seiten), besonders gründlich

**Stärken**
- Jede Abweichung ist an ihrem Ort begründet und in Tab. 4.9 gesammelt.
- Die Logik der Nachinferenz ist sorgfältig hergeleitet (Stellvertreterregel „nicht parsebar", Präfixprüfung; 4.5.5).
- Die Validierungskette ist vollständig spezifiziert (4.4.6).
- Die Lesart der Bootstrap-Intervalle ist korrekt.

**Schwächen in der Sache**
- **„Skalierungsanalyse" (Z. 157)** widerspricht 2.7 und 4.5.4, wo ausdrücklich keine Skalierungsgesetze bestimmt werden.
- **Z. 237–238:** „Gegenüber CoNLL-2003 und OntoNotes bietet MultiNERD damit einen feingranularen NER-Benchmark". Ungenau, denn OntoNotes hat 18 Typen (2.8) und MultiNERD 15. Kriterium 1 im selben Absatz sagt das Richtige bereits.
- **Z. 245–249:** „Die zusätzlich veröffentlichte, manuell geprüfte Stichprobe". In den öffentlichen Bezugsquellen ist sie nicht auffindbar (Abschnitt 6.3), und ihre Herkunft aus den Splits ist nicht angegeben. Die Begründung für den Verzicht nennt den wichtigeren Grund nicht.
- **Repository-Pfad im Fließtext** (Z. 635).
- **Budgets für FF3** fehlen in 4.5.2.
- **Zu vage:** „was vor allem den Speicherbedarf betrifft" (Z. 740), obwohl sich der Effekt aus der zitierten Quelle beziffern lässt.

**Redundanzen**

Die meisten sind gewollte Rückverweise an der Stelle, an der der Leser die Information braucht.

| Inhalt | Stellen | Bewertung |
|---|---|---|
| Hardware-Vorbehalt | 1.3; 4.1 (Z. 126–129); 4.4.1 und Tab. 4.4; 4.5.2 „Vergleichbarkeit"; 4.6 (Z. 1185–1188); Tab. 4.9 | 4.4.1, 4.5.2 und Tab. 4.9 sind nötig. 4.1 und 4.6 benennen den Vorbehalt dort, wo er gilt. Hier nichts kürzen. |
| Unsicherheit seltener Typen | 2.9 (Z. 783–785); 4.5.1 (Z. 881–882 und 901–903); 5.1.2 mit Supportzahlen | 4.5.1, Z. 901–903 ist entbehrlich. Im Audit 18 war das K4-e, damals auf Empfehlung nicht umgesetzt; nicht erneut vorgeschlagen. |
| Testnutzung | 4.1; 4.3.3; Tab. 4.3; 4.4.3; Tab. 4.9 | Bereits in Audit 18 (K4-h) verdichtet. Weitere Kürzung würde den sensibelsten Vorbehalt schwächen: **nicht ändern.** |
| Werte von Zhan zu LoRA gegenüber vollem Fine-Tuning (0,14 bis 0,48) | 3.3 (Z. 461–464); 4.4.5 (Z. 699–702) | Zahl doppelt, aber im Methodikkapitel Teil der Begründung. C. |
| Indexing-Einordnung und Ramponi-Herkunft | 1.2 (Z. 154–159); 3.2.1 (Z. 238–242); 4.3.2 (Z. 300–303); 4.4.4 (Z. 665–667) | Viermal. 1.2 und 3.2.1 regeln die Zuschreibung von Fremdem und Eigenem, das ist wichtig. 4.4.4, Z. 665–667 ist ein reiner Verweis. C. |
| Kapitelanfang aus Anhangverweisen | Z. 12–15 und 4.4.1, Z. 502–505 | Audit 18 K4-c, nicht umgesetzt. C. |

**Lange Sätze:** 11 Sätze mit mehr als 40 Wörtern, keiner über 50. Die meisten sind mit Doppelpunkt sauber gegliedert. Teilen lohnt sich am ehesten bei Z. 1101–1106 (48 W, Stellvertreterregel). Geringer Nutzen.

**Verbesserungen:** A4 bis A7; B1, B3, B5, B7.

**Einsparpotenzial:** etwa 15 bis 30 Wörter verlustfrei. Mehr geht nur, wenn Begründungen oder Vorbehalte entfallen. Das empfehle ich nicht.

### Kapitel 5 (21 Seiten), besonders gründlich

**Stärken**
- Die Ergebnisabschnitte interpretieren statt nachzuerzählen. Beispiele: Mikro und Makro ordnen die Klassen unterschiedlich, und die Arbeit erklärt warum (Z. 173–179); die TIME-Definition im Zero-Shot-Regime (Z. 186–197).
- **Mechanismus belegt:** Auslassungen, abgeschnittene Antworten und Nachinferenz greifen ineinander (5.2, 5.5).
- **Vorsichtige Pareto-Aussage:** 1 024-Token-Qualität und 256-Token-Ressourcen werden bewusst nicht vermischt (Z. 1009–1013).
- Die stichprobenhaft geprüften Zahlenwerte stimmen mit den Tabellen überein (Tab. 5.1, A.6, 5.4). Zwei Deutungen widersprechen aber Tabellenwerten (A16, A17).

**Schwächen in der Sache**
- **S. 58, Z. 100–104:** „Die Validierungs-F1-Werte der ausgewählten Checkpoints liegen rund drei Punkte unter den Testwerten." Das gilt nur für die Encoder:
  - DeBERTa: Validierung 92,6 bis 93,1, Test 95,5 bis 95,8;
  - LLMs: Validierung auf 200 Sätzen in acht von neun Läufen **über** dem Test, beim 27B-Modell z. B. 96,13 gegenüber 94,53 (Tab. A.6);
  - auch „innerhalb eines Zehntelpunkts" gilt nur für die Encoder (0.8B: 93,04 bis 93,24).
- **S. 61, Z. 263–265:** „Da nur 55 der 16 454 Referenzantworten länger als 256 Tokens sind, fallen die Modellantworten länger aus als die Referenz." Als Pauschalaussage nicht gedeckt. Gedeckt ist nur: Je Lauf brechen 62 bis 72 Antworten ab, also sind mindestens 7 bis 17 Modellantworten länger als ihre Referenz.
- **FF3 bleibt qualitativ** („bei niedriger Latenztoleranz", Z. 1015–1018; „hängt von der Zielanwendung ab", Z. 1024–1025).

**Vom separaten Prüfer gefunden, von mir an den generierten Tabellen bestätigt**
- **S. 65, Z. 493–499:** „Die Zero-Shot-Modelle erzeugen kürzere Ausgaben, beim 4B-Modell in 51 % der Fälle unmittelbar ein leeres Array (Abschnitt 5.2)".
  - Die 51 % stimmen: 8 389 von 16 454 laut `summary.json`. Sie stehen aber nicht in 5.2; dort stehen 61 % leere Vorhersagen (10 021).
  - Beim 27B-Modell trifft „kürzere Ausgaben" nicht zu. Im Zero-Shot-Regime beginnt es 33 621 Elemente, im QLoRA-Regime 25 859 (Tab. 5.2).
  - Als möglicher Faktor fehlt der ungemergte Adapter der QLoRA-Inferenz (4.4.3, Z. 649–650).
- **S. 69, Z. 743–744:** „Die Latenz je Satz lag auf dem Niveau der Hauptläufe".
  - Tab. 5.7 zeigt mit 1 024 Tokens 12,9 s, mit 256 Tokens 14,7 s; in Politik 16,5 gegenüber 21,2 s.
  - Bei Greedy-Decoding und identischem Präfix kann der Lauf mit höherem Limit nicht weniger rechnen. Der Unterschied muss also aus den Laufbedingungen stammen und ist unkommentiert.
  - „Hauptläufe" ist mehrdeutig (MultiNERD: 5,5 s).
- **S. 64 gegenüber S. 96:** Das 0.8B-Training dauert in der Hauptstudie auf der H100 12,1 h (sacct), im Vorlauf auf der A100 nur 8,25 h (Trainer).
  - Laut 4.5.2 unterscheiden sich beide Messgrößen nur um Sekunden. Die Abweichung ist im Text nicht erklärt.
  - Ob die drei 0.8B-Läufe einen Knoten teilten, habe ich nicht geprüft (sacct auf dem Cluster).
- **Unterschiedliche Bezugsgrößen:** „Rund die Hälfte" der Auslassungen (5.5) und „der größte Teil dieses Rückstands" (5.7.1) messen Verschiedenes. Beide Aussagen sind richtig (27B: Auslassungen 5,06 % → 2,52 %, Encoder 2,4 %), die Bezugsgrößen werden aber nicht genannt.

**Redundanzen**

| Inhalt | Stellen | Bewertung |
|---|---|---|
| Kennzahlen-Konventionen, Bootstrap-Lesart | Kapitelanfang Z. 44–65 wiederholt 4.5.1 und die Caption von Tab. 5.1 | Verdichtbar, B6 (erweitert K5-b aus Audit 18) |
| Epochenwahl 8 von 9 | 4.4.3 (Z. 630–632) und 5.1.1 (Z. 126–128) | A9 |
| 27B-Trainingsdauer, Knoten und Neustarts | 4.5.2 (Z. 937–950), Tab. 5.4 Fußnote a, 5.4 (Z. 465–470), A.2 | A11 |
| CrossNER-Hardware | 4.6 (Z. 1185–1188), 5.6 (Z. 673 und 718–722) | A13 |
| Kernbefund 1,34 und 0,18 | 5.1.1, 5.7.1 (Z. 770–776 und erneut 811–817), 6.1, Abstract | Zweite Nennung in 5.7.1 ist rhetorisch nötig („macht daraus einen Messwert"). Nicht ändern. |
| „kein isolierter Effekt" | 5 Stellen in Kapitel 5, 15 in der ganzen Arbeit | Jede gehört zu einer anderen Aussage (Architektur, Anpassung, Domäne). Nur in einer Fußnote gebündelt ließen sie sich streichen, das kostet Präzision. C. |

**Verbesserungen:** A8 bis A13, A16 bis A18; B1, B6, B8.

**Einsparpotenzial:** etwa 60 Wörter verlustfrei (A9, A11 bis A13), mit B6 etwa 110 Wörter. A8 und A10 kosten zusammen rund 15 Wörter mehr.

### Kapitel 6 (5 Seiten)
**Stärken**
- Jede Forschungsfrage wird mit Zahlen beantwortet.
- Die Erwartungen E1 bis E3 werden abgeglichen.
- Jede Limitation nennt die betroffene Forschungsfrage.
- Der Ausblick ist konkret.

**Schwächen**
- **Doppelt in 6.1:** Der Vergleich mit DeBERTa-v3-base und das Intervall [0,02; 0,33] stehen innerhalb von neun Zeilen zweimal (Z. 27–36).
- **Budget ad hoc:** „gemessener allokierter Inferenzspeicher unter 2 GiB" und „dem engsten Budget" (Z. 84 und 87) erscheinen ohne Herleitung.
- **Kein Beitragssatz:** Was die Arbeit dem Forschungsstand hinzufügt, steht nirgends ausdrücklich.
- **Methodische Lehre fehlt:** Die Lehre zum Ausgabelimit steht nur im Abstract.
- **E3 zu glatt:** „das bestätigt Erwartung E3" (Z. 50–51) blendet aus, dass das Zero-Shot-Setup nach Testbeobachtungen gewählt ist, mit Sampling bei Temperatur 1,0.

**Verbesserungen:** A14, A19; B1, B2.

**Einsparpotenzial:** etwa 20 Wörter. Die letzte Seite ist zu rund 99 % gefüllt, jede Ergänzung kostet also eine Seite (83 statt 82).

### Literaturverzeichnis (9 Seiten)
- 85 Einträge, einheitlicher APA-Stil, DOIs und Abrufdaten, Preprints gekennzeichnet.
- Ein Restpunkt aus der Codex-Notiz in Bericht 19: Für drei Quellen fehlt der Originalabruf (Manning und Schütze 1999, Manning et al. 2008, Qwen3.5-Blog). Von mir in dieser Runde **nicht** geprüft.

### Anhang (15 Seiten), nur bewertet, keine Änderung, keine Kürzungsvorschläge
**Funktion**
- A.1: Läufe, Commits, Revisionen, Einstellungen.
- A.2: Durchführung, Prompts, Sampling, 27B-Verlauf, Vorstudie.
- A.3: ergänzende Ergebnistabellen.

**Bewertung**
- Alle Verweise aus dem Haupttext lösen auf.
- Die Nachvollziehbarkeit ist hoch; z. B. sind Adapter-Dateigrößen gegen die Parameterzahl nachgerechnet.

**Hinweise, ohne Änderung**
- Der Quellkommentar in `anhang.tex`, Z. 3–5, nennt die Ergebnistabellen noch „A.2"; sie stehen in A.3. Nur ein Kommentar, im PDF unsichtbar.
- A.1 nennt die öffentliche Repository-URL. Das Repository ist laut STATUS noch nicht öffentlich, das muss vor der Abgabe erledigt sein.
- Die Auszählung der 200 Validierungssätze (Z. 633–635 in Kapitel 4) steht nicht im Anhang, nur im Repository. Deshalb verweist A6 auf A.1 statt auf eine Anhangtabelle.

---

## 5 Verdichtungs- und Präzisierungsvorschläge

Felder je Vorschlag:
1. Fundstelle;
2. Problem;
3. Warum ändern;
4. Original;
5. Vorschlag;
6. Was erhalten bleibt;
7. Risiken und Abhängigkeiten;
8. Wörter;
9. Seiten;
10. Nutzen;
11. Art.

Zeilen beziehen sich auf den Stand `35fef59`.

### A – verlustfreie Verdichtung oder Präzisierung (redaktionell, ohne neue Daten)

#### A1 · Kapitel 1 · Übergeneralisierung auf S. 2
1. 1.2, Z. 86 (S. 2).
2. Die Pauschalaussage widerspricht 3.3 (Q. Lu et al.: bleibender Rückstand) und den eigenen Werten bei 256 Tokens.
3. Steht auf der zweiten Textseite. Prüfende lesen sie als Überdehnung.
4. `Mit LoRA angepasste LLMs schließen diesen Abstand.`
5. `Mit LoRA angepasste LLMs verringern diesen Abstand deutlich und liegen in einzelnen Vergleichen gleichauf oder vorn.`
6. Die Zahlen von Zhan und Obeidat in den Folgesätzen bleiben unverändert und belegen den Satz.
7. keine.
8. +11.
9. 0.
10. mittel bis hoch (Präzision an exponierter Stelle).
11. redaktionell.

#### A2 · Kapitel 1 und 3 · Zwei Lücken statt vier Aspekte
1. 1.2, Z. 98–110; dazu 3.5, Z. 661–662 und 701–702.
2. Der Leser muss vier Aspekte auf zwei Lücken abbilden, die erst in 3.5 benannt werden. Der vierte Aspekt ist ohne „ohne Encodervergleich" missverständlich.
3. Klarere Argumentationslinie von der Lücke zur Forschungsfrage.
4. Z. 98–110: „Für die vorliegende Arbeit sind dabei vier Aspekte relevant, die Abschnitt 3.5 zu zwei Lücken bündelt: … Erstens … zweitens … Drittens … viertens betrachten sie LLMs nur einer Größenklasse, während Größenstaffelungen nur innerhalb jeweils einer Modellfamilie untersucht wurden."
5. Neu in 1.2:
   ```tex
   Abschnitt~\ref{sec:research_gap} bündelt die Grenzen dieser Vergleiche
   zu zwei Lücken.
   Die erste betrifft Datenbasis und Pipeline: Die direkten Vergleiche
   nutzen etablierte allgemeine Benchmarks oder Fachkorpora,
   veröffentlichte MultiNERD-Werte stützen sich auf übernommene Baselines
   oder eigene Aufbereitungen, und eine gemeinsame Auswertung auf den
   offiziellen englischen Splits mit eigener Encoder-Referenz fehlt.
   Die zweite betrifft Ressourcenerfassung und Modellgröße: Die direkten
   Vergleiche erfassen Erkennungsleistung und Ressourcenbedarf nur
   unvollständig gemeinsam (Inferenzzeit ohne Speicher oder
   Trainingskosten ohne Inferenz) und betrachten LLMs nur einer
   Größenklasse; Größenstaffelungen gibt es nur innerhalb einzelner
   Modellfamilien, ohne gemeinsamen Encodervergleich.
   ```
   In 3.5:
   - Z. 661–662 → `Die erste Lücke betrifft Datenbasis und Pipeline.`
   - Z. 701–702 → `Die zweite Lücke betrifft Ressourcenerfassung und Modellgröße.`
6. Alle vier Inhalte, beide Klammerbeispiele, der Verweis auf 3.5.
7. Die Verweise auf „vier Aspekte" in 3.5 müssen mitgeändert werden. Andere Stellen verweisen nicht darauf (`grep "Aspekt"`: nur diese drei).
8. etwa −15.
9. 0.
10. mittel (Lesbarkeit der Herleitung der Forschungsfragen).
11. redaktionell.

#### A3 · Kapitel 2 · Gemessene Ressourcen präzise benennen
1. 2.9, Z. 795–799 (S. 20).
2. „Speicherbedarf, Durchsatz und Trainingszeit" klingt nach Trainingsspeicher und Batch-Durchsatz. Beides wird laut 1.3 und 4.5.2 nicht gemessen.
3. Widerspruchsfreiheit mit 1.3.
4. `während Speicherbedarf, Durchsatz und Trainingszeit gesondert gemessen werden`
5. `während Inferenzspeicher, Latenz und Trainingsdauer gesondert gemessen werden`
6. Verweis auf 4.5.2.
7. keine.
8. 0.
9. 0.
10. gering bis mittel.
11. redaktionell.

#### A4 · Kapitel 4 · „Skalierungsanalyse"
1. 4.2.1, Z. 157 (S. 36).
2. Widerspricht 2.7 und 4.5.4: keine Skalierungsgesetze.
3. Begriffliche Konsistenz.
4. `Base- und Large-Variante erlauben eine Skalierungsanalyse innerhalb derselben Architektur;`
5. `Base- und Large-Variante erlauben einen Größenvergleich innerhalb derselben Architektur;`
6. alles.
7. keine.
8. 0.
9. 0.
10. gering bis mittel.
11. redaktionell.

#### A5 · Kapitel 4 · OntoNotes-Satz streichen
1. 4.3.1, Z. 237–238 (S. 37).
2. OntoNotes hat mit 18 mehr Typen als MultiNERD mit 15 (2.8, Tab. 2.2). „feingranularer Benchmark gegenüber OntoNotes" ist ungenau. Der Satz wiederholt zudem Kriterium 1 aus demselben Absatz.
3. Entfernt eine angreifbare Aussage und eine Wiederholung.
4. `Gegenüber CoNLL-2003 und OntoNotes bietet MultiNERD damit einen feingranularen NER-Benchmark \cite{sang2003conll,keraghel2024recent}.`
5. Satz streichen.
6. Kriterium 1, Z. 221–224: „geht damit über Person, Ort und Organisation hinaus". Beide Quellen sind mehrfach anderweitig zitiert.
7. keine.
8. −13.
9. 0.
10. mittel (beseitigt eine Ungenauigkeit).
11. redaktionell.

#### A6 · Kapitel 4 · Repository-Pfad im Fließtext
1. 4.4.3, Z. 633–635 (S. 44).
2. Im Fließtext steht ein Dateipfad.
3. Wissenschaftlicher Stil, Verweis über den Anhang.
4. `(\texttt{thesis/generated/validation\_subset\_stats.md})`
5. `(Auszählung im Repository, Anhang~\ref{app:reproduzierbarkeit})`
6. Die Nachprüfbarkeit, denn A.1 nennt das Repository und `thesis/generated/`.
7. Der Leser muss die Datei im Repository suchen. Der Anhang selbst bleibt unverändert.
8. −1.
9. 0.
10. gering (formal).
11. redaktionell.

#### A7 · Kapitel 4 · Verweis in 4.5.5
1. 4.5.5, Z. 1093–1095 (S. 53).
2. Das Limit wird in 4.4.3 definiert und eingeordnet (Längenstatistik), der Verweis zeigt aber auf 4.5.3.
3. Korrekter Rückverweis.
4. `… die bei langen Sätzen mit vielen Entitäten greift (Abschnitt~\ref{sec:robustheit-ausgaben}).`
5. `… (Abschnitt~\ref{sec:llm-pipeline}).`
6. alles.
7. keine.
8. 0.
9. 0.
10. gering.
11. redaktionell.

#### A8 · Kapitel 5 · Validierungs- und Testaussage auf Encoder einschränken (Hinweis S. 58)
1. 5.1.1, Z. 100–104 (S. 58).
2. Beide Sätze gelten nur für die Encoder. Bei den LLMs liegt die Validierung auf 200 Sätzen in acht von neun Läufen über dem Testwert (Tab. A.6), und die Seed-Spanne des 0.8B-Modells beträgt 0,20.
3. Sachliche Präzision. Wer Tab. A.6 ansieht, findet sonst einen Widerspruch.
4. `Die Streuung über die Seeds ist sehr gering; die drei Einzelwerte liegen jeweils innerhalb eines Zehntelpunkts. Die Validierungs-F$_1$-Werte der ausgewählten Checkpoints (Tabelle~\ref{tab:app-seeds}) liegen rund drei Punkte unter den Testwerten.`
5. `Bei beiden Encodern ist die Streuung über die Seeds sehr gering; die drei Einzelwerte liegen jeweils innerhalb eines Zehntelpunkts. Die Validierungs-F$_1$-Werte ihrer ausgewählten Checkpoints (Tabelle~\ref{tab:app-seeds}) liegen rund drei Punkte unter den Testwerten.`
   Optional zusätzlich: `Bei den LLMs, deren Validierung nur 200 Sätze umfasst, liegen sie in acht von neun Läufen darüber.` (+18 W)
6. Die Erklärung über die LOC-Verschiebung (Z. 105–109) bleibt.
7. Die Zahl „acht von neun" ist aus Tab. A.6 abgelesen: 0.8B Seed 42 liegt darunter, alle übrigen darüber.
8. +3, optional +21.
9. 0.
10. mittel (Sachpräzision).
11. redaktionell.

#### A9 · Kapitel 5 · Epochenwahl doppelt
1. 5.1.1, Z. 126–128 (S. 59).
2. Wiederholt 4.4.3, Z. 630–632.
3. Verlustfreie Wiederholung.
4. `Ausgewählt wurde in acht der neun Läufe der Adapter der dritten Epoche (Abschnitt~\ref{sec:llm-pipeline}); die Werte der generativen Validierung auf 200 Sätzen stehen in Tabelle~\ref{tab:app-seeds}.`
5. `Die Werte der generativen Validierung auf 200 Sätzen und die gewählten Epochen stehen in Tabelle~\ref{tab:app-seeds}.`
6. Die Epochenwahl steht in 4.4.3 und Tab. A.6. Der folgende Satz zum 4B-Modell nennt die gewählte zweite Epoche weiterhin.
7. keine.
8. −10.
9. 0.
10. gering.
11. redaktionell.

#### A10 · Kapitel 5 · Schlusskette zu den Antwortlängen (Hinweis S. 61)
1. 5.2, Z. 263–265 (S. 61).
2. Aus „55 Referenzantworten über 256" folgt nicht, dass „die Modellantworten" länger sind. Gedeckt ist: Je Lauf brechen 62 bis 72 Antworten ab, also mindestens 7 bis 17 mehr als Referenzantworten über 256 Tokens.
3. Logische Präzision.
4. `Da nur 55 der 16\,454 Referenzantworten länger als 256 Tokens sind (Abschnitt~\ref{sec:llm-pipeline}), fallen die Modellantworten länger aus als die Referenz.`
5. `Je Lauf brechen 62 bis 72 Antworten am Limit ab, obwohl nur 55 der 16\,454 Referenzantworten länger als 256 Tokens sind (Abschnitt~\ref{sec:llm-pipeline}); ein Teil der Modellantworten ist also länger als die zugehörige Referenz.`
6. Beide Zahlen, der Verweis.
7. 62 bis 72 stammt aus Z. 308. Die genaue Zahl der betroffenen Sätze ließe sich ohne GPU aus den gespeicherten Daten bestimmen (N3).
8. +12.
9. 0.
10. mittel.
11. redaktionell; N3 wäre eine Analyse.

#### A11 · Kapitel 5 · 27B-Trainingsdauer, Vorbehalt verdichten
1. 5.4, Z. 465–470 (S. 64).
2. Wiederholt 4.5.2 (Z. 937–950), die Fußnote a von Tab. 5.4 und A.2.
3. Die Details stehen dreimal anderswo.
4. `Die 266~h der 27B-Läufe (je Seed in Anhang~\ref{app:durchfuehrung}) sind die Summe von je vier Slurm-Zuteilungen am Drei-Tage-Limit der Partition (Abschnitt~\ref{sec:effizienzmetriken}). Die drei Läufe teilten sich zudem einen Knoten, sodass die Dauer durch die gegenseitige Auslastung erhöht ist (Vergleich mit dem allein laufenden Vorlauf in Anhang~\ref{app:durchfuehrung}).`
5. `Die 266~h der 27B-Läufe sind die Summe von je vier Slurm-Zuteilungen und durch die gemeinsame Belegung eines Knotens erhöht (Abschnitt~\ref{sec:effizienzmetriken}, je Seed in Anhang~\ref{app:durchfuehrung}).`
6. Beide Ursachen (Neustarts, Knoten) und beide Verweise. Drei-Tage-Limit und Vorlaufvergleich stehen in 4.5.2 und A.2.
7. keine.
8. −22.
9. 0.
10. gering bis mittel.
11. redaktionell.

#### A12 · Kapitel 5 · Klammer „seedgleich"
1. 5.5, Z. 627–629 (S. 67).
2. „(seedgleich 96,6/93,8/95,1)" ist ohne Erklärung kaum lesbar und enthält keine neue Spanne.
3. Lesbarkeit.
4. `einen Recall von 93,8 bis 96,6 (seedgleich 96,6/93,8/95,1), das LLM dort null …`
5. `einen Recall von 93,8 bis 96,6, das LLM dort null …`
6. Die Spanne.
7. **Verlust:** die Zuordnung zu gleichen Seeds. Sie trägt kein Argument. Der Begriff „seedgleich" wird auch in den Fußnoten von Tab. 5.8 und 5.9 verwendet und ist dort erklärt; deshalb ist der Punkt optional.
8. −4.
9. 0.
10. gering.
11. redaktionell. Wer die Zuordnung behalten will, schreibt stattdessen „(bei gleichem Seed 96,6, 93,8 und 95,1)".

#### A13 · Kapitel 5 · CrossNER-Hardware zweimal
1. 5.6, Z. 673 (S. 68).
2. „Beide Modelle liefen auf derselben H100 mit 80 GB." Z. 718–722 sagt es an der Stelle, wo es gebraucht wird: Latenzvergleich „auf demselben GPU-Typ". Dazu kommen 4.6 und Tab. 4.4.
3. Doppelung.
4. siehe 2.
5. Satz streichen.
6. Der Hardwarehinweis in Z. 719–722, Tab. 4.4 und 4.6.
7. keine.
8. −8.
9. 0.
10. gering.
11. redaktionell.

#### A14 · Kapitel 6 · FF1-Absatz ohne Doppelnennung
1. 6.1, Z. 28–36 (S. 78).
2. Der Vergleich mit DeBERTa-v3-base und das Intervall [0,02; 0,33] stehen innerhalb von neun Zeilen zweimal.
3. Klarheit im Fazit.
4. Z. 28–36, von „Zu DeBERTa-v3-base ergibt sich dann keine Rangfolge …" bis „… gehört zu den explorativen Vergleichen."
5.
   ```tex
   Auf CrossNER kehrt sich die Reihenfolge mit dem höheren Limit um
   (Tabelle~\ref{tab:synthese}).
   Erwartung E1 ist auf MultiNERD nur bei ausreichendem Ausgabelimit und
   nur teilweise bestätigt: Das mit QLoRA angepasste LLM liegt im
   Punktwert knapp vor DeBERTa-v3-base, ohne dass sich eine Rangfolge
   nachweisen lässt ([$-$0,02; 0,30]), bleibt aber knapp hinter
   DeBERTa-v3-large; dieses explorative Intervall liegt nur knapp über
   null.
   ```
6. Alle drei Zahlen, „explorativ", die Umkehr auf CrossNER, der E1-Abgleich.
7. Der Satz „Der verbleibende Abstand ist kein isolierter Architektureffekt" (Z. 37–38) bleibt.
8. etwa −20.
9. 0; schafft rund zwei Zeilen Puffer für B2.
10. mittel bis gering.
11. redaktionell.

#### A15 · Vorspann · Seitenzahl des Abkürzungsverzeichnisses im Inhaltsverzeichnis
1. `main.tex`, Z. 219–220.
2. Das Inhaltsverzeichnis nennt S. ix, das Verzeichnis beginnt auf S. viii. `\addcontentsline` steht hinter dem zweiseitigen Verzeichnis.
3. Formfehler.
4. `\input{frontmatter/abkuerzungsverzeichnis}` gefolgt von `\addcontentsline{toc}{chapter}{Abkürzungsverzeichnis}`
5. Reihenfolge tauschen: `\phantomsection\addcontentsline{…}` vor `\input`. Das vorausgehende `\clearpage` sorgt dafür, dass der Eintrag auf der Anfangsseite landet.
6. alles.
7. Keine Layoutparameter. Nach dem Build das Inhaltsverzeichnis prüfen.
8. 0.
9. 0.
10. gering (formal, aber billig).
11. redaktionell.

#### A16 · Kapitel 5 · Warum Zero-Shot schneller ist (S. 65)
1. 5.4, Z. 493–501.
2. „kürzere Ausgaben" gilt nicht für das 27B-Modell (Tab. 5.2: mehr begonnene Elemente). Der Verweis auf 5.2 belegt die 51 % nicht. Der ungemergte Adapter fehlt als Faktor.
3. Eine sachlich falsche Begründung. Prüfende mit Tab. 5.2 in der Hand sehen das.
4. `Zwei Faktoren wirken hier zusammen und lassen sich nicht trennen: Die Zero-Shot-Modelle erzeugen kürzere Ausgaben, beim 4B-Modell in 51\,\% der Fälle unmittelbar ein leeres Array (Abschnitt~\ref{sec:erg-formtreue}), und die BF16-Gewichte ersparen die Dequantisierung der NF4-Gewichte bei jedem Vorwärtsdurchlauf.`
5. `In Frage kommen kürzere Ausgaben sowie der Wegfall der Dequantisierung der NF4-Gewichte und des ungemergten Adapters (Abschnitt~\ref{sec:llm-pipeline}); getrennt wurden diese Faktoren nicht. Kürzere Ausgaben können nur beim 4B-Modell beitragen, das in 51\,\% der Fälle unmittelbar ein leeres Array ausgibt; das 27B-Modell beginnt im Zero-Shot-Regime sogar mehr Elemente als im QLoRA-Regime (Tabelle~\ref{tab:formtreue-erg}).`
6. Beide bisherigen Faktoren, die 51 %, der Hinweis, dass sie nicht trennbar sind.
7. Die 51 % stammen aus `summary.json`, Schlüssel `form.empty_array`, nicht aus einer Tabelle der Arbeit. Optional die absolute Zahl 8 389 nennen.
8. +20.
9. 0.
10. mittel (Sachkorrektur).
11. redaktionell, beruht auf vorhandenen Daten.

#### A17 · Kapitel 5 · Latenz des CrossNER-Neulaufs (S. 69)
1. 5.6, Z. 743–745.
2. „auf dem Niveau der Hauptläufe" ist mehrdeutig und verdeckt einen Widerspruch: 1 024 Tokens ergeben 12,9 s, 256 Tokens 14,7 s.
3. Messunsicherheit offen benennen statt glätten.
4. `Die Latenz je Satz lag auf dem Niveau der Hauptläufe (Tabelle~\ref{tab:limit}); der Ausschluss …`
5. `Die mittlere Latenz je Satz lag mit 12,9~s sogar unter den 14,7~s des Laufs mit 256 Tokens, obwohl die Antworten länger sind (Tabelle~\ref{tab:limit}); die Ursache wurde nicht geprüft, Latenzvergleiche zwischen einzelnen Läufen sind daher nur in der Größenordnung belastbar. Der Ausschluss …`
6. Der Ausschluss der 21 Sätze und der Tabellenverweis.
7. Die Aussagen in 5.7.4 und 6.1 (12,9 s gegenüber 25,7 ms) bleiben gültig, weil der Faktor um 500 robust gegenüber 10 bis 20 % Schwankung ist.
8. +25.
9. 0.
10. mittel.
11. redaktionell.

#### A18 · Kapitel 5 · Bezugsgröße des Recall-Rückstands (S. 71)
1. 5.7.1, Z. 811–814.
2. „der größte Teil dieses Rückstands" ist ohne Zahl nicht nachvollziehbar und klingt anders als „rund die Hälfte" in 5.5.
3. Nachvollziehbarkeit.
4. `… und der größte Teil dieses Rückstands entfällt auf die am Limit von 256 Ausgabetokens abgeschnittenen Antworten (Abschnitt~\ref{sec:erg-fehlertypen}).`
5. `… und der größte Teil dieses Rückstands entfällt auf die am Limit von 256 Ausgabetokens abgeschnittenen Antworten: Mit deren Nachinferenz sinkt der Anteil der Auslassungen des 27B-Modells von 5,1 auf 2,5\,\% gegenüber 2,4\,\% bei DeBERTa-v3-large (Tabellen~\ref{tab:limit-multinerd} und~\ref{tab:fehlertypen}).`
6. alles.
7. keine; Zahlen aus Tab. 5.3 und 5.5.
8. +25.
9. 0.
10. gering bis mittel.
11. redaktionell.

#### A19 · Kapitel 6 · Abgleich mit E3 (S. 78)
1. 6.1, Z. 50–51.
2. „das bestätigt Erwartung E3" berücksichtigt das nach Testbeobachtungen gewählte Setup nicht.
3. Angemessene Stärke der Aussage.
4. `Im Zero-Shot-Regime erreichen die drei Modelle nur 6,2 bis 69,4 F$_1$-Punkte; das bestätigt Erwartung E3.`
5. `Im Zero-Shot-Regime erreichen die drei Modelle unter den gewählten Prompt- und Decoding-Bedingungen nur 6,2 bis 69,4 F$_1$-Punkte; das stützt Erwartung E3.`
6. Die Zahlen und der E3-Abgleich.
7. +7 Wörter in Kapitel 6, das keine Reserve hat. Zusammen mit A14 umsetzen.
8. +7.
9. 0.
10. gering.
11. redaktionell.

#### A20 · Abstract · Latenzfaktor 210 zuordnen (S. ii)
1. Abstract, dritter Absatz.
2. „ein Vergleich auf gleicher Hardware bestätigt einen Latenzfaktor von etwa 210" sagt nicht, dass dies das kleinste LLM gegenüber DeBERTa-v3-base betrifft. Die Faktoren für das 27B-Modell (335 und 617) stammen von anderer Hardware.
3. Präzision an der meistgelesenen Stelle.
4. siehe 2.
5. `ein Vergleich auf gleicher Hardware ergibt für Qwen3.5-0.8B gegenüber DeBERTa-v3-base einen Latenzfaktor von etwa 210`
6. alles.
7. Das Abstract muss auf einer Seite bleiben; nach dem Build prüfen.
8. +5.
9. 0.
10. gering bis mittel.
11. redaktionell.

**Summe A1 bis A15:** netto etwa −70 Wörter (A1 +11, A8 +3, A10 +12; A2 −15, A5 −13, A9 −10, A11 −22, A12 −4, A13 −8, A14 −20, A6 −1).

**Summe A16 bis A20:** etwa +80 Wörter.

**Insgesamt:** etwa ±0 Wörter. **Seitenersparnis: 0.**

**Nutzen:** Er liegt in Präzision und Sachrichtigkeit. A1, A5, A8, A10, A16 und A17 beseitigen angreifbare Aussagen.

### B – fachlich sinnvoll, Entscheidung nötig

#### B1 · FF3 über Schwellen aus den Messwerten beantworten (siehe 6.1)
1. 4.5.2 (neuer Absatz), 5.7.4 (Absatz oder kleine Tabelle), 6.1 FF3 (Z. 82–88). Die Formulierung von FF3 in 1.2 bleibt.
2. FF3 setzt Budgets voraus, die nie festgelegt werden. Erst im Fazit erscheinen „unter 2 GiB" und „engstes Budget".
3. Das ist die größte rein textlich behebbare Lücke zwischen Forschungsfrage und Antwort.
4. 6.1: `sowie engem Latenz- und Speicherbudget (gemessener allokierter Inferenzspeicher unter 2~GiB) sind …`, `genügt mit 8,9~ms und 0,7~GiB dem engsten Budget`
5. Schwellen statt erfundener Budgets; Ausarbeitung in 6.1. Etwa 60 Wörter in 4.5.2, 100 bis 140 in 5.7.4, 6.1 umformuliert.
6. Pareto-Aussage, alle Vorbehalte (Batchgröße 1, GPU-Typen, 256 gegenüber 1 024 Tokens).
7. Die Schwellen werden nachträglich aus den Messwerten abgeleitet. Das muss so benannt werden, sonst wirkt es wie ein vorab festgelegtes Budget. Optional als Zeile in Tab. 4.3 („post hoc, nicht leistungsbezogen").
8. +150 bis +200.
9. +0,3 bis +0,5. Kapitel 5 hat 3 % Reserve; wahrscheinlich +1 Seite, sofern nicht mit B6 verrechnet.
10. **hoch** (FF3 vollständig beantwortet).
11. fachlich. **Benötigt deine Entscheidung** zwischen den Varianten in 6.1.

#### B2 · Beitragsabsatz im Fazit
1. 6.1, Ende, oder Anfang von 6.3.
2. Was die Arbeit dem Forschungsstand hinzufügt, steht nirgends ausdrücklich. Die methodische Lehre zum Ausgabelimit steht nur im Abstract.
3. Prüfende suchen diesen Satz. Er bündelt die Befunde zu einer Aussage.
4. –
5. Entwurf ohne neue Behauptungen; alle Zahlen stehen bereits in Kapitel 5:
   ```tex
   Der Beitrag der Arbeit liegt erstens in einem Vergleich beider
   Modellklassen auf den offiziellen englischen MultiNERD-Splits in einer
   gemeinsamen Pipeline, mit Größenstaffelung auf beiden Seiten und
   gemeinsamer Ressourcenmessung. Zweitens zeigt sie, dass das
   Wortindex-Format im QLoRA-Regime nahezu formtreu ist. Drittens folgt
   eine methodische Lehre für generative NER: Ausgabelimits sind vor der
   Evaluation an den Antwortlängen von Trainings- und Validierungsdaten
   zu prüfen, denn ein zu knappes Limit verschob hier den Abstand auf
   MultiNERD um gut einen Punkt und kehrte auf CrossNER die Rangfolge um.
   ```
6. –
7. **Nicht überhöhen:** kein „erstmals" ohne Recherche 2026 (B4). Kapitel 6 hat keine Reserve.
8. +70 bis +85.
9. +1 Seite (Haupttext 83), mit A14 eventuell nur wenige Zeilen über der Grenze.
10. mittel.
11. redaktionell, aber inhaltliche Gewichtung. **Bitte freigeben oder anpassen.**

#### B3 · Aussage zur manuell geprüften Stichprobe (4.3.1) und Bedingung im Ausblick (6.3)
1. 4.3.1, Z. 245–249 (S. 38); 6.3, Z. 216–221 (S. 82).
2. „Die zusätzlich veröffentlichte … Stichprobe": In den öffentlichen Bezugsquellen ist keine solche Datei zu finden (6.3). Tedeschi et al. sagen nur „randomly select a sample of 1K English sentences" (Abschn. 5.3, Fn. 13: mindestens 20 Instanzen je Klasse). Aus welchem Split sie stammt, steht dort nicht. Die angegebene Begründung (Pipeline auf offiziellen Splits, kleine Stichprobe) nennt die wichtigeren Gründe nicht: fehlende Verfügbarkeit und mögliche Überlappung mit dem Trainingssplit.
3. Es ist eine Aussage über die Daten. Wenn sie nicht stimmt, ist sie angreifbar.
4. `Die zusätzlich veröffentlichte, manuell geprüfte Stichprobe von 1\,000 zufällig gezogenen englischen Sätzen wird nicht verwendet, da die Pipeline auf den offiziellen Splits aufsetzt und typbezogene Aussagen auf der kleinen Stichprobe sehr unsicher wären \cite[Tab.~5]{tedeschi2022multinerd}.`
5. Vorschlag, je nach deinem Kenntnisstand:
   ```tex
   Die von den Autoren manuell geprüfte Stichprobe von 1\,000 zufällig
   gezogenen englischen Sätzen wird nicht verwendet: Sie ist nicht
   Bestandteil der veröffentlichten Splits, ihre Herkunft aus den Splits
   ist nicht angegeben, sodass Überschneidungen mit den Trainingsdaten
   nicht auszuschließen sind, und typbezogene Aussagen wären auf ihr sehr
   unsicher \cite[Tab.~5]{tedeschi2022multinerd}.
   ```
   In 6.3 nach „… Qualitätsmessung nutzen": `, sofern sie zugänglich ist und nicht mit dem Trainingssplit überlappt,`
6. Zitat, Unsicherheitsargument.
7. **Benötigt deine Angabe:** Hast du die Stichprobe je angefragt oder gefunden? War „veröffentlicht" belegt? Ich habe nur die drei öffentlichen Orte geprüft.
8. +15.
9. 0.
10. mittel.
11. fachlich (Datenaussage), deshalb nicht A.

#### B4 · Recherchezeitraum (Hinweis Literatur 2026)
1. 3, Einleitung, Z. 20–25 (S. 21).
2. Im selben Absatz stehen „darunter zwei Arbeiten aus dem Jahr 2026" und „(2019 bis 2025)". Die Prioritätsaussage „soweit bekannt, erstmals" (5.7.2) hängt am Recherchestand.
3. Innere Widerspruchsfreiheit; Nachvollziehbarkeit der Recherche.
4. `… in Google Scholar, arXiv und der ACL Anthology (2019 bis 2025) mit Stichwörtern …`
5. Zwei Varianten:
   - **Variante a (nur Text, sofort möglich):** `(Schwerpunkt 2019 bis 2025, ergänzt um die genannten Arbeiten aus 2026)`. Das behauptet keine systematische Suche für 2026.
   - **Variante b (Recherche):** eine dokumentierte Nachsuche für 2026 mit denselben Stichwörtern und Datum, dann z. B. `(2019 bis Oktober 2026; letzte Suche am …)`. Neue einschlägige Arbeiten wären einzuordnen, andernfalls wäre „keine weiteren einschlägigen" zu vermerken.
6. Stichwörter, Quellen, der Hinweis auf fehlende systematische Vollständigkeit.
7. Variante b kostet einen halben bis ganzen Tag und kann neue Quellen nach sich ziehen. Ich ergänze **keine** Quellen ohne Prüfung von Existenz, bibliografischen Angaben und Relevanz. Eine Recherche habe ich **nicht** durchgeführt.
8. Variante a: +5. Variante b: +10 bis +100.
9. 0 bis +0,3.
10. mittel.
11. Variante a redaktionell, Variante b fachlich. **Entscheidung nötig.**

#### B5 · Wirkung der fehlenden Double Quantization beziffern
1. 4.4.5, Z. 737–741 (S. 46).
2. Bisher nur „was vor allem den Speicherbedarf betrifft".
3. Mit einer Zahl wird aus einer Abweichung ein begrenzter, für die Schlüsse unerheblicher Effekt.
4. `… nicht aktiviert, was vor allem den Speicherbedarf betrifft;`
5. `… nicht aktiviert; nach \textcite[Abschn.~3]{dettmers2023qlora} spart sie im Mittel rund 0,37~Bit je Parameter ohne Leistungseinbuße, beim 27B-Modell also höchstens rund 1,2~GiB des gemessenen Inferenzspeichers;`
6. Die Abweichung bleibt in Tab. 4.9.
7. Gegen den lokalen Volltext geprüft: „saving an average of about 0.37 bits per parameter", „double quantization does not degrade performance". Rechnung: 26,9 · 10⁹ × 0,373 / 8 Byte ≈ 1,25 · 10⁹ Byte ≈ 1,17 GiB, eine Obergrenze, weil nicht alle Gewichte quantisiert sind. Das ist eine neue Zahl im Text, deshalb B.
8. +25.
9. 0.
10. mittel.
11. fachlich (neue Quantifizierung aus zitierter Quelle).

#### B6 · Kennzahlenkonventionen am Anfang von Kapitel 5 verdichten
1. Kapitel 5, Z. 44–65 (S. 57).
2. Etwa 140 Wörter wiederholen 4.5.1 und die Caption von Tab. 5.1. Neu sind nur die Rundungsregeln.
3. Der Leser beginnt Kapitel 5 mit der dritten Erklärung desselben Protokolls.
4. Absatz „Für alle Kennzahlen gelten … beobachtete Punktwertdifferenzen beschrieben."
5.
   ```tex
   Für alle Kennzahlen gelten die Festlegungen aus
   Abschnitt~\ref{sec:evaluationskonzept}: strikte Entity-Level-Werte
   nach IOB2 auf dem vollständigen englischen MultiNERD-Testsplit, für die
   trainierten Konfigurationen Mittelwert~$\pm$~Standardabweichung über
   die Seeds 42, 123 und 456, für Zero-Shot je ein Lauf mit Sampling.
   Werte stehen in Punkten (0 bis 100), in Tabellen mit Streuungsangabe
   auf zwei, im Text auf eine Nachkommastelle; Differenzen mit
   Intervallangabe erhalten auch im Text zwei Nachkommastellen.
   Enthält ein gepaartes Bootstrap-Intervall null, ist kein Unterschied
   nachgewiesen, aber auch keine Gleichwertigkeit; typbezogene
   Unterschiede ohne eigene Intervalle bleiben deskriptiv.
   ```
6. Alle Konventionen und die Lesregel.
7. **Verlust:** die Erläuterung, was das Bootstrap misst („Auswahl der Testsätze bei festen Modellen"). Sie steht in 4.5.1 und 5.7.5. In Audit 18 hatte ich nur die kleinere Variante K5-b vorgeschlagen, damals nicht umgesetzt.
8. etwa −45.
9. 0, schafft aber Reserve in Kapitel 5 für B1.
10. gering bis mittel.
11. redaktionell.

#### B7 · Trainer-Dauer ohne Klassenaufteilung
1. 4.5.2, Z. 933–936 (S. 50).
2. Die Aufteilung (Encoder 30 bis 49 s, LLM 32 bis 141 s, Maximum beim 4B-Modell mit Seed 42) trägt kein Argument. Sie belegt nur, dass der Mehraufwand gegenüber Stunden vernachlässigbar ist.
3. Dichte.
4. `… liegt bei den ununterbrochenen Läufen darunter: bei den Encodern um 30 bis 49~s, bei den 0.8B- und 4B-Läufen um 32 bis 141~s (Maximum beim 4B-Modell mit Seed~42).`
5. `… liegt bei den ununterbrochenen Läufen um 30 bis 141~s darunter.`
6. Spanne und Aussage.
7. **Kleiner Informationsverlust:** Aufteilung nach Klasse und Maximum. Nicht im Anhang.
8. −14.
9. 0.
10. gering.
11. redaktionell, aber nicht ganz verlustfrei, daher B.

#### B8 · Trainingsdauer des 0.8B-Modells klären
1. 5.4, Z. 461–464 (S. 64) gegenüber A.2 (S. 96).
2. Hauptlauf auf der H100: 12,1 h. Vorlauf auf der A100: 8,25 h. Unerklärt.
3. Wer Kapitel 5 und Anhang vergleicht, stößt darauf. Es berührt FF2 (Ressourcen nach Größe).
4. –
5. Auf dem Cluster mit `sacct` prüfen, ob die drei 0.8B-Läufe gleichzeitig auf einem Knoten liefen. Dann einen Satz in 5.4 ergänzen: entweder die Ursache, oder „die Ursache ist ungeklärt". Der Anhang bleibt unverändert.
6. –
7. Lesender Cluster-Zugriff. Ohne Prüfung keine Ursache behaupten.
8. +15 bis +25.
9. 0.
10. gering bis mittel.
11. fachlich (Datenprüfung).

### C – geringer Nutzen oder hohes Risiko (nicht vorrangig, nicht eigenständig umsetzen)

| Nr. | Fundstelle | Vorschlag | Was verloren ginge | Wörter | Nutzen |
|---|---|---|---|---:|---|
| C1 | 4.1, Z. 73–75 | „und sind Teil der Hauptauswertung" streichen | nichts (steht in Z. 60–63) | −5 | sehr gering |
| C2 | Kapitelanfang 4 (Audit 18 K4-c) | Anhang-Verweissatz streichen | Orientierung beim Kapitelanfang | −22 | gering |
| C3 | 4.4.4, Z. 665–667 | Indexing-Verweis streichen | Rückverweis auf 3.2.1 an der Formatdefinition | −20 | gering |
| C4 | 4.4.5, Z. 699–702 | Zhan-Zahlen durch Verweis auf 3.3 ersetzen | Zahl als Begründung vor Ort | −6 | gering |
| C5 | 4.5.4, Z. 1081–1085 | Skalierungsgrenzen nur per Verweis auf 2.7 | eigenständige Methodikaussage | −12 | gering |
| C6 | 5.4, Z. 437–440 | Hardware-Satz kürzen | Hardware-Vorbehalt direkt vor Tab. 5.4; du hast ausdrücklich gebeten, ihn zu erhalten | −8 | gering, Risiko |
| C7 | 5.7.1, Z. 811–819 | Rekapitulation „knapp hinter DeBERTa-v3-large" streichen | rhetorischer Schritt „macht daraus einen Messwert" | −20 | gering |
| C8 | 5.7.5 und 6.2 | „Hauptvergleiche" nur in 4.5.1 | gewollte Wiederholung an drei argumentativen Stellen (Revision vom 7.11.) | −40 | Risiko |
| C9 | alle Kapitel | „kein isolierter Effekt" bündeln | Präzision je Aussage | bis −80 | Risiko |
| C10 | 3.1.3 | GLiNER- und NuNER-Details straffen | Stütze für 3.4 | −40 | gering |
| C11 | 2.5 | OSI-Definition offener Gewichte straffen | begriffliche Präzision | −20 | gering |
| C12 | K4 und K2 | Sätze über 40 Wörter teilen | – | 0 | gering |
| C13 | 4.7, 5.7.5, 6.2 | Vorschlag des separaten Prüfers: alle Vorbehalte in einer Evidenz- und Validitätsmatrix bündeln, „etwa 8 bis 10 Seiten weniger" | Vorbehalte an der Stelle, an der sie gelten. Die Einsparung halte ich für weit überschätzt: Die echten Wiederholungen umfassen eher 1 bis 2 Seiten, und 6.2 ist als eigener Limitationsabschnitt erwartet. **Nicht empfohlen**, weil es wie Verstecken von Schwächen wirken kann. | – | Risiko |
| C14 | 1.2, 3.1.2, 5.7.1 | Keraghel-Werte 65,17 und 90,98 erscheinen dreimal | Funktion je Stelle: Motivation, Studie, Einordnung | −15 | gering |
| C15 | 2.3, 2.7 | Grundlagen straffen (Formeln, Skalierungsgesetze), wie der Prüfer anregt | 2.7 begründet die Grenze, auf die 4.5.4 und 5.7.3 verweisen | −100 bis −200 | gering, Risiko |

### Nicht ändern
- **Tab. 4.4 (GPU-Typen) und ihre fünf Verweise:** Sie tragen den Hardware-Vorbehalt bei Latenz, Nachinferenz und CrossNER.
- **Tab. 4.3, Testnutzungsabsatz 4.3.3, Zero-Shot-Absatz 4.4.3:** die zentralen Offenlegungen, bereits verdichtet (Audit 18).
- **Stellvertreterlogik in 4.5.5:** Sie begründet, warum die Nachinferenz zulässig ist.
- **Überschneidungsabsatz 4.6:** Er rechtfertigt den CrossNER-Vergleich.
- **5.7.5 Belastbarkeit und die Struktur von 6.2:** jede Limitation mit betroffener Forschungsfrage.
- **Definitionsdoppelungen zwischen Kapitel 1 und 2:** üblich und funktional.
- **Anhang:** vollständig, auch nicht sprachlich.

**Verschiebungen in den Anhang:** keine vorgeschlagen.

---

## 6 Wichtige fachliche Verbesserungen

### 6.1 FF3 und Budgets

**Befund**
- FF3 verlangt Konfigurationen „bei gegebenen Speicher- und Latenzbeschränkungen" (1.2). Die Frage setzt also Budgets voraus.
- 4.5.2 legt keine fest.
- 5.7.4 antwortet qualitativ („niedrige Latenztoleranz", „hängt von der Zielanwendung ab").
- 6.1 führt „unter 2 GiB" und das „engste Budget" ohne Herleitung ein.

**Vorschlag ohne erfundene Werte:** Schwellen aus den Messwerten (Tab. 5.4).

| Dimension | Schwelle | Ab dort zulässig |
|---|---|---|
| Latenz je Satz (Mittel, Batchgröße 1) | ab 8,9 ms | DeBERTa-v3-base |
| | ab 16,5 ms | zusätzlich DeBERTa-v3-large |
| | ab 1,9 s | erstes LLM (Qwen3.5-0.8B, QLoRA, H100) |
| | ab 5,5 s | alle QLoRA-Konfigurationen |
| Inferenzspeicher | ab 0,7 GiB | DeBERTa-v3-base |
| | ab 0,9 GiB | Qwen3.5-0.8B (QLoRA) |
| | ab 1,7 GiB | DeBERTa-v3-large |
| | ab 3,5 GiB | 4B (QLoRA) |
| | ab 18,7 GiB | 27B (QLoRA) |

**Folgerung, direkt aus Tab. 5.1 und 5.4**
- **256 Tokens auf MultiNERD:** Unter jeder Budgetkombination, die überhaupt eine Konfiguration zulässt, ist ein Encoder die beste gemessene Wahl. DeBERTa-v3-base (95,5) liegt vor jedem LLM (höchstens 94,5) und ist ab 8,9 ms und 0,7 GiB zulässig.
- **1 024 Tokens:** Auch mit der selektiven Nachinferenz ändert sich das nicht. 27B mit 95,6 liegt vor base (Intervall enthält null), aber hinter large, und large ist ab 1,7 GiB zulässig.
- **Einziger Fall für das LLM:** CrossNER mit 1 024 Tokens (87,1 gegenüber 85,0). Dafür braucht es eine Latenztoleranz von mindestens rund 12,9 s je Satz und rund 19 GiB Inferenzspeicher (18,6 GiB, gemessen bei 256 Tokens).
- **„Rechtfertigt sich der Mehraufwand?":** Die Messwerte liefern das Verhältnis, etwa +2,1 Punkte für das rund 500-Fache der Latenz. Die Abwägung selbst bleibt anwendungsabhängig, und das ist so auch zu sagen.

**Was das bringt**
- FF3 erhält eine scharfe, aus Daten abgeleitete Antwort.
- Für MultiNERD lautet sie: **für jedes Budget ein Encoder**.
- Die 2-GiB-Grenze entfällt als unbegründete Setzung.

**Varianten zur Entscheidung**
- **(i) Schwellen aus Messwerten wie oben.** Empfohlen. Als nachträglich abgeleitet gekennzeichnet; keine erfundenen Werte.
- **(ii) Konkrete Einsatzszenarien,** etwa „interaktiv unter X ms" oder „Offline-Verarbeitung". Die Werte müssten von dir oder dem Betreuer kommen. Ich setze keine.
- **(iii) FF3 umformulieren** zu einer reinen Pareto-Frage. Eine Forschungsfrage nachträglich zu ändern ist heikel; ich rate davon ab.

**Gültigkeitsgrenzen**, im Text zu nennen:
- Batchgröße 1;
- die jeweiligen GPU-Typen;
- 256 Tokens für die Ressourcen; für 1 024 Tokens gibt es keine MultiNERD-Ressourcenwerte.

### 6.2 Manuelle Prüfstichprobe der Abweichungen von der Silberreferenz (neue Analyse, keine GPU)

**Ziel:** Prüfen, ob die Vorteile einer Modellklasse echte Erkennung sind oder Anpassung an Fehler der Silberreferenz. Das ist die in 5.7.1 (Z. 804–806) ausdrücklich offene Vermutung. Sie berührt den knappen Hauptunterschied (0,18 bis 1,34 Punkte).

**Auswahl der Fälle**

Alle Fälle stammen aus dem Testsplit, je Modell Seed 42: DeBERTa-v3-large und Qwen3.5-27B (QLoRA) mit den zusammengeführten 1 024-Token-Vorhersagen, damit das Limit-Artefakt nicht dominiert.

Drei Schichten:
- S1: Encoder stimmt mit der Silberreferenz überein, das LLM nicht.
- S2: Das LLM stimmt überein, der Encoder nicht.
- S3: Beide Modelle sagen denselben Span vorher, der in der Referenz fehlt oder anders typisiert ist. Das sind Kandidaten für Fehler der Referenz.

Innerhalb jeder Schicht:
- Zufallsziehung mit festem Seed;
- 50 bis 70 Fälle, zusammen 150 bis 200;
- geschichtet nach verrauschten Typen (ANIM, FOOD, PLANT, DIS) und häufigen Typen (PER, ORG, LOC);
- Sätze, deren Antwort abgeschnitten war, ausgeschlossen oder gesondert ausgewiesen.

**Warum diese Größe:** Bei 50 Fällen je Schicht ist ein Anteil auf etwa ±14 Prozentpunkte genau (Wilson-Intervall, 95 %). Das reicht, um „überwiegend Referenzfehler" von „überwiegend Modellfehler" zu unterscheiden. **Nicht** genug für Aussagen je Typ.

**Durchführung**
- **Blind:** Weder Modellname noch Silberlabel sind sichtbar. Gezeigt werden Satz und Kandidatenspans mit Typ in zufälliger Reihenfolge.
- **Urteilskategorien:** Referenz richtig, Vorhersage richtig, beide falsch, unentscheidbar. Grundlage sind die Typdefinitionen von MultiNERD.
- **Selbstkonsistenz:** 30 Fälle nach einigen Tagen erneut beurteilen und Cohen's κ berichten.
- **Einschränkung:** nur ein Annotator; als Limitation zu nennen.

**Auswertung:** Anteile je Schicht mit Intervall. Entscheidend ist der Vergleich S1 gegen S2: Ist der Encoder-Vorteil in S1 häufig ein Referenzfehler?

**Aufwand**
- Skript über die gespeicherten Vorhersagen: etwa ein halber Tag.
- Annotation: 4 bis 6 Stunden.
- Einarbeitung in den Text: ein halber Tag; Text +150 bis +250 Wörter und eine kleine Tabelle, etwa +0,5 bis +1 Seite.

**Zur Unterscheidung**
- Eine **Textpräzisierung** kann den Vorbehalt nur schärfer benennen; das tut der Text schon (5.7.1, 6.2).
- **Neue Evidenz** liefert erst diese Analyse.
- Es wurden **keine** Fälle geprüft. Ergebnisse gibt es nicht.

### 6.3 Auswertung auf der manuell geprüften MultiNERD-Stichprobe (1 000 Sätze)

**Zugänglichkeit, heute geprüft**
- **GitHub-README `Babelscape/multinerd`:** verlinkt je Sprache einen Google-Drive-Ordner und den Hugging-Face-Datensatz. Eine Goldstichprobe wird nicht als Datei genannt.
- **Hugging Face `Babelscape/multinerd`:** enthält nur `train/`, `val/` und `test/` je Sprache.
- **Drive-Ordner EN:** zeigt laut Seitenabruf nur `train_en.tsv`, `dev_en.tsv` und `test_en.tsv`. Der Abruf wurde automatisch zusammengefasst, deshalb nur mittlere Sicherheit.
- **Ergebnis:** In diesen drei Quellen ist keine Goldstichprobe auffindbar. Ob die Autoren sie auf Anfrage herausgeben, ist offen.

**Passung und Überlappung**
- Laut Tedeschi et al. (Abschn. 5.3, Fn. 13) ist die Stichprobe zufällig aus den englischen Sätzen gezogen, mit mindestens 20 Instanzen je Klasse. Ein Split wird nicht genannt.
- Stammt sie aus dem gesamten englischen Teil, liegen erwartungsgemäß rund 80 % der Sätze im Trainingssplit. Rechnung: 131 280 von 164 144 Sätzen.
- Diese Sätze haben die trainierten Modelle mit Silberlabels gesehen. **Eine Überlappung ist also nicht auszuschließen, sondern wahrscheinlich.**
- Sauber auswertbar wären nur die etwa 100 Sätze aus dem Testsplit. Für Aussagen je Typ ist das zu wenig.

**Aufwand, wenn die Daten vorlägen:** Die grobe Schätzung von 5 bis 15 GPU-Stunden ist plausibel. Rechnung für die 27B-QLoRA-Inferenz: etwa 5,5 s × 1 000 × 3 Seeds ≈ 4,6 h, dazu 0.8B und 4B. Das ist eine Schätzung und nicht verifiziert.

**Empfehlung**
- Derzeit nicht verfolgen.
- Stattdessen die eigene Prüfstichprobe aus 6.2. Sie ist verfügbar, ohne Überlappungsproblem und gezielter.
- Den Text dazu über B3 korrigieren.
- Ich habe nichts gestartet und keine Ergebnisse erzeugt.

### 6.4 Präzisionspunkte aus der bisherigen Einschätzung

| Hinweis | Aktuelle Fundstelle | Befund | Vorschlag |
|---|---|---|---|
| „schließen diesen Abstand" (S. 2) | 1.2, Z. 86, S. 2 | **bestätigt** | A1 |
| Validierung und Test nur für Encoder (S. 58) | 5.1.1, Z. 100–104, S. 58 | **bestätigt**, auch für den Seed-Satz davor | A8 |
| Begründung zu Antwortlängen zu knapp (S. 61) | 5.2, Z. 263–265, S. 61 | **bestätigt**; die Schlussfolgerung ist zu pauschal | A10 |
| Repository-Pfad im Fließtext (S. 44) | 4.4.3, Z. 635, S. 44 | **bestätigt** | A6 |
| Recherchezeitraum endet 2025 | 3, Z. 24–25, S. 21 | **bestätigt**; Widerspruch zu „Arbeiten aus 2026" im selben Absatz | B4 |

### 6.5 Weitere Analysen ohne GPU

- **N2: Nicht abgebildete CrossNER-Fachtypen unter den falsch positiven Spans je Domäne auszählen.** 5.7.4 (Z. 1035–1036) sagt ausdrücklich „domänenweise ausgezählt wurden sie nicht". Prüft eine der drei ungeprüften Erklärungen der Umkehr. Aufwand etwa ein halber Tag, aus den gespeicherten Vorhersagen. Nutzen mittel.
- **N3: Wie viele abgeschnittene Sätze haben eine Referenzantwort ≤ 256 Tokens?** Macht A10 zum Messwert. Aufwand etwa eine Stunde, CPU (Tokenizer). Nutzen gering.

---

## 7 Prioritätenplan

| Priorität | Punkte | Art | Aufwand | Wirkung auf Umfang |
|---|---|---|---|---|
| **Hoch** | B1 (FF3-Schwellen), A1, A5, A8, A10, A16, A17 | Text, Datenableitung aus vorhandenen Tabellen | etwa 1 Tag | +150 bis +200 W |
| **Hoch, neue Evidenz** | 6.2 Prüfstichprobe (N1) | Analyse ohne GPU, deine Annotation | 1,5 bis 2 Tage | +0,5 bis +1 Seite |
| **Mittel** | B2 (Beitrag), B3 (Stichprobenaussage), B4a oder B4b (Recherche), B5 (Double Quantization), B8 (0.8B-Dauer), A2, A14, A18, A20, N2 | Text, Recherche, Analyse | 1 bis 2 Tage | +100 bis +150 W |
| **Gering** | A3, A4, A6, A7, A9, A11 bis A13, A15, A19, B6, B7 | Text | 2 bis 3 Stunden | −120 W |
| **Nur mit neuen Läufen** | D1 vollständiger Neulauf der neun QLoRA-Testinferenzen mit begründetem Limit auf H100 (etwa 165 GPU-h); D2 Zero-Shot mit auf Validierung gewähltem Setup (etwa 160 GPU-h); D3 Goldstichprobe (derzeit nicht verfügbar) | GPU | 4 bis 7 Tage Laufzeit | Tabellen und Text in 4, 5, 6; Anhangtabellen A.1 und A.3 bräuchten neue Zeilen, also deine Zustimmung |

**Zielkonflikt:** Die Punkte mit dem höchsten fachlichen Nutzen machen die Arbeit länger, nicht kürzer: B1, B2, B5, N1 und gegebenenfalls D1. Verlustfreie Kürzungen gleichen das nur zu etwa einem Drittel aus. Ich empfehle, die Länge in Kauf zu nehmen. Eine kürzere Arbeit mit offener FF3 wäre schwächer als eine um eine Seite längere mit beantworteter FF3.

**Empfohlene Reihenfolge bis zur Abgabe am 26.11.**
1. A-Punkte sowie B1, B3 und B5. Das ist reine Textarbeit, Entscheidung zu B1 nötig.
2. N1 und N2, weil sie neue Evidenz ohne GPU liefern.
3. B2 und B4. Für B4 entscheiden, ob Variante a oder b.
4. D1 nur nach Rückmeldung des Betreuers, Grenze 15.11.

---

## 8 Vorläufige Bewertung

Eigene, vorläufige Orientierung auf der Skala 1,0 (10) bis 5,0 (50). Ein offizielles Bewertungsraster lag nicht vor. Die Kapitelnoten sind Diagnosen, keine Teilnoten.

**Gewichtung, eigene Orientierung**
- Methodische Validität und Beantwortung der Forschungsfragen: hoch, etwa 40 %.
- Ergebnisdarstellung und Diskussion: etwa 25 %.
- Forschungsstand und Einordnung: etwa 15 %.
- Transparenz und Reproduzierbarkeit: etwa 10 %.
- Form und Sprache: etwa 10 %.

Die Gewichtung ist keine Formel. Sie zeigt nur, warum Kapitel 4 und 5 die Gesamtnote stärker prägen als Kapitel 2.

**Gesamtnote: 1,7 (17)**
- Realistischer Bereich 1,3 (13) bis 2,0 (20), am wahrscheinlichsten 1,7.
- Gegenüber dem Ausgangswert (1,7; Bereich 1,3 bis 2,0; „am wahrscheinlichsten 1,3 bis 1,7") verschiebt sich die Einschätzung leicht nach unten:
  - vier neu gefundene Sachungenauigkeiten in Kapitel 5;
  - ein separater Prüfer bewertet Kapitel 4 mit 2,0;
  - Kapitel 6 ist bei genauer Lesung eher 1,7 als 1,3.

**Was die Note trägt, belegt**
- **Ehrliche, vollständige Offenlegung:** Tab. 4.3 klassifiziert jede Entscheidung als „direkt", „indirekt" oder „post hoc".
- **Saubere Statistik:** gepaarte Intervalle, Haupt- getrennt von explorativen Vergleichen, „Gleichwertigkeit ist damit nicht nachgewiesen".
- **Eigenständig diagnostizierter Kernbefund:** Ausgabelimit, Präfixkontrolle 594 von 604, CrossNER +15,4.
- **Kritische Quellenarbeit.**

**Was eine bessere Note verhindert**
- **Der Hauptvergleich steht auf der fehlkonfigurierten Einstellung** (256 Tokens). Die korrigierte Aussage ist post hoc, selektiv, auf anderer Hardware und knapp (0,18 [0,02; 0,33]).
- **Das QLoRA-Rezept weicht an mehreren Stellen ab,** und es gibt keine Hyperparametersuche. Verglichen wird damit eine suboptimale LLM-Umsetzung gegen einen voll feinjustierten Encoder.
- **FF3 ist nicht operationalisiert.**
- **Einzelne Sachungenauigkeiten** in Ergebnissen und Diskussion.

**Unabhängigkeit**
- Meine Noten sind nicht unabhängig. Der separate Prüfer kannte keine Vornoten, ist aber ebenfalls ein Sprachmodell und hat ähnliche blinde Flecken.
- Dass beide auf 1,7 kommen, ist ein Indiz, kein Beleg.

---

## 9 Einschätzung zum Ziel 1,0

**Schon vorhanden**
- Transparenz und Reproduzierbarkeit auf hohem Niveau.
- Statistik und Diskussion auf hohem Niveau.
- Ein eigener, methodisch relevanter Befund.
- Gutes Literaturverzeichnis und guter Anhang.

**Was dem Ziel entgegensteht und sich textlich nicht beheben lässt**
- Der Hauptvergleich beruht auf einer bekannten Fehlkonfiguration.
- Der Testsplit wurde vor der Hauptstudie genutzt.
- Das Zero-Shot-Setup wurde nach Testbeobachtungen gewählt.
- QLoRA-Abweichungen und drei GPU-Typen.
- Diese Punkte sind Teil der Versuchsgeschichte. Offenlegung mildert sie, beseitigt sie aber nicht.

**Bis zur Abgabe realistisch verbesserbar**
- **Rein textlich, etwa 2 bis 3 Tage:** alle A-Punkte, B1 bis B5. Das schließt die FF3-Lücke und beseitigt alle angreifbaren Einzelaussagen.
- **Neue Evidenz ohne GPU, etwa 2 Tage:** Prüfstichprobe (6.2) und CrossNER-Auszählung (N2). Das schärft den wichtigsten offenen Vorbehalt, die Silberreferenz.
- **Mit GPU, nur nach Rückmeldung des Betreuers (Grenze 15.11.):**
  - D1: vollständiger Neulauf der Testinferenzen mit vorab aus Trainings- und Validierungslängen begründetem Limit auf einheitlicher Hardware. Damit würde der Hauptvergleich sauber.
  - D2: Zero-Shot-Setup auf Validierungsdaten wählen.

**Einschätzung**
- Mit der Textarbeit und der Prüfstichprobe halte ich **1,3 für erreichbar**, aber nicht sicher.
- **1,0 ist allein mit Textarbeit unwahrscheinlich.** Auch mit D1 und D2 bleiben Testnutzung und QLoRA-Abweichungen dokumentiert.
- Ob Prüfende das als normale Studienrealität oder als Planungsschwäche werten, lässt sich aus der Arbeit nicht vorhersagen.
- Eine bestimmte Note kann ich nicht zusagen.

---

## 10 Vorher-nachher-Bilanz

Noch keine Änderung umgesetzt. Ausgangswerte für den späteren Vergleich:
- 116 physische Seiten, Haupttext S. 1–82 (82 Seiten);
- Literaturverzeichnis 9 Seiten, Anhang 15 Seiten;
- 22 111 Wörter Haupttext, 1 737 Wörter Anhang-Fließtext;
- Build ohne Warnungen.

Nach Freigabe und Umsetzung:
- neuer Build;
- Tests;
- Prüfung von Warnungen, Querverweisen und Umbrüchen;
- `git diff` auf `anhang.tex` leer;
- Vergleich mit getrennten Mess- und Schätzwerten.

---

## 11 Unsicherheiten

**Nicht zuverlässig beurteilbar**
- **Bewertungsraster und Erwartungen des Fachgebiets:** Umfang (82 Seiten), Sprache des Abstracts, Gewichtung von Methodik gegenüber Ergebnissen, Umgang mit der umfangreich erklärten KI-Unterstützung.
- **Wertung der Abweichungen:** ob Prüfende die vielen dokumentierten Abweichungen als Ehrlichkeit honorieren oder als Planungsmängel abziehen.
- **Unabhängigkeit:** Meine Bewertung ist nicht unabhängig, der separate Prüfer nur teilweise.
- **Messwerte:** In dieser Runde nicht erneut gegen die Rohdaten auf dem Cluster geprüft; die generierten Tabellen habe ich als richtig vorausgesetzt.
- **Goldstichprobe:** Der Drive-Ordner wurde nur über eine automatische Seitenzusammenfassung gesichtet, die Verfügbarkeit hat daher nur mittlere Sicherheit. Ob die Autoren die Stichprobe auf Anfrage herausgeben, ist offen.
- **Literatur 2026:** Ob neue einschlägige Arbeiten existieren, ist unbekannt, weil keine Recherche stattfand.
- **Mündliche Verteidigung und Betreuung:** nicht beurteilbar und nicht bewertet.

**Besonders wichtige Rückmeldungen des Betreuers**
1. Neulauf der Testinferenzen (D1) ja oder nein. Bereits in der Begleitnotiz gefragt, Frist 10.11.
2. Erwartet er für FF3 konkrete Budgets, und welche? Oder genügen Schwellen aus den Messwerten (B1, Variante i)?
3. Sind 82 Seiten Haupttext angemessen, und muss es ein englisches Abstract geben?
4. Wiegt er eine eigene Prüfstichprobe zur Silberreferenz (6.2) höher als weitere Textarbeit?

**Entscheidungen, die ich von dir brauche**
1. Freigabe der A-Punkte, ganz oder einzeln.
2. B1: Variante i, ii oder iii.
3. B2 freigeben oder anpassen.
4. B3: Hast du die manuell geprüfte Stichprobe je gefunden oder angefragt?
5. B4: Variante a oder b.
6. B5 bis B8 einzeln.
7. Ob ich die Prüfstichprobe (6.2) und die Auszählung N2 vorbereiten soll. Beides ohne GPU, die Annotation machst du.
