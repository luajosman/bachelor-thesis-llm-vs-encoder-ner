# Prüfbericht Kapitel 4 und 5

Stand 13.11.2026. Reine Prüfung: An Quelldateien, Tabellen, Abbildungen, Daten und PDF wurde nichts geändert. Alle Vorschläge warten auf Freigabe.

Zeilenangaben beziehen sich auf `thesis/chapters/04_methodik.tex` (kurz 04) und `thesis/chapters/05_ergebnisse_und_diskussion.tex` (kurz 05) im Commit `b4898f1`. Seitenzahlen sind die gedruckten Seitenzahlen des PDF.

---

## 1 Geprüfte Version und Umfang der Prüfung

### 1.1 Maßgebliche Version

| Stand | Commit | Verhältnis zur geprüften Fassung |
|---|---|---|
| **`lektorat-2026-11-25` (geprüft)** | `b4898f1` | Arbeitsverzeichnis, sauber, identisch mit `origin`. Enthält gegenüber `main` die 64 typografischen Korrekturen, den Wegweiser am Anfang von Kapitel 4 und das neue Deckblatt. |
| `main` | `53415e6` | älter, nicht maßgeblich |
| `korrekturen-2026-11-25` | `a70c01b` | lokal, nicht zusammengeführt. Enthält zwei Textänderungen in Kapitel 5, die in der geprüften Fassung fehlen: die Typfehlerzahlen in 5.5 und den Satz zur Rundung in der Kapiteleinleitung. |

Das vorhandene `thesis/main.pdf` (12.11., 14:49) ist älter als der letzte Commit. Ich habe `b4898f1` deshalb in einer getrennten Kopie neu gebaut: 111 Seiten, keine Warnungen, und der Text aller 111 Seiten stimmt mit dem vorhandenen PDF überein. Kapitel 4 steht auf S. 31 bis 52, Kapitel 5 auf S. 53 bis 72.

### 1.2 Was geprüft wurde

| Gegenstand | Art der Prüfung |
|---|---|
| Kapitel 4 und 5 | vollständig gelesen, einschließlich aller 18 Tabellen, 3 Abbildungen (im gebauten PDF angesehen) und Beschriftungen |
| Angrenzende Teile | Abschnitt 1.2 (Forschungsfragen), 3.5 (Erwartungen), Kapitel 6 vollständig, die von Kapitel 4 genannten Anhangstellen zu Antwortlängen und Sampling |
| Zahlen in Kapitel 5 | Prüfskript gegen `summary.json` und die generierten Tabellen: 742 Zahlen, 716 automatisch belegt, die 24 gemeldeten einzeln nachgerechnet |
| Kernzahlen | **unabhängig nachgerechnet** aus den gespeicherten Vorhersagen aller 18 Hauptläufe und der 9 zusammengeführten 1 024-Token-Läufe, mit eigener strikter IOB2-Auswertung und eigenem gepaarten Bootstrap (ohne das Aggregationsskript der Arbeit) |
| Laufdaten | Latenz, Speicher, GPU-Typ, Ausgabelimit, Parser-Zähler aus den `inference_metrics.yaml` der Läufe; Nachinferenz und Kontrollstichprobe aus den Cluster-Aggregaten; CrossNER aus dem Cluster-Aggregat und dem Skript `crossner_unmapped_fp.py` (neu ausgeführt) |
| Konfiguration | Hyperparameter gegen `ba-ner/configs`, Quantisierungsaufruf gegen den Quellcode |
| Quellen | rund 50 zitierte Aussagen am Volltext der Originale geprüft (zwei delegierte Prüfläufe). Strittige Urteile habe ich selbst nachgelesen; ein Urteil des Prüflaufs war falsch und ist hier korrigiert (Abschnitt 4.1). |

### 1.3 Nicht geprüft

- **Trainings- und Validierungsspalte von Tabelle 4.2:** Die Rohdaten liegen lokal nicht vor. Geprüft ist nur, dass die Typzeilen sich zu den Summen addieren; die Testspalte stimmt mit den Vorhersagedateien überein.
- **Längenangaben unter dem Qwen-Tokenizer** (55 Referenzantworten über 256 Tokens, 48 und 11 gekürzte Sequenzen, 576 und 103 im Anhang): bräuchte Tokenizer und Datensatz.
- **Trainingsdauern, Schrittzeiten des 27B-Trainings, Statistik der 200 Validierungssätze:** nur gegen `summary.json`, nicht gegen die Slurm-Abrechnung.
- **Zählungen der Fehleranalyse in 5.5** (Verwechslungspaare, Grenzfehler je Typ): gegen `summary.json`, nicht unabhängig nachgezählt. Die Beispiele habe ich nicht in den Daten gesucht.
- **Bootstrap-Intervalle:** mit eigenem Zufallsstrom nachgerechnet; sie weichen deshalb in der zweiten Nachkommastelle ab und bestätigen die Werte der Arbeit nur näherungsweise.
- **Quellen ohne Volltext:** Efron und Tibshirani (Buch), Manning et al. 2008 (Server nicht erreichbar), der Qwen3.5-Blog (liefert ohne Browser keinen Inhalt). Han et al. und Wang et al. wurden nur gezielt durchsucht.
- **Satzbild nach Umsetzung:** Kein Vorschlag ist probeweise gesetzt. Kapitel 5 endet am Seitenende; Zeilenangaben zu Einsparungen sind Schätzungen.

---

## 2 Kurzfazit

### Was überzeugt

- **Die Zahlen stimmen.** Tabelle 5.1 ließ sich vollständig reproduzieren, ebenso die Abstände 1,34 und 0,18 Punkte, die Faktoren 210, 335 und 617 und die CrossNER-Werte.
- **Die Offenlegung ist ungewöhnlich gründlich.** Tabelle 4.3 (Testbezug jeder Entscheidung) und Tabelle 4.9 (Abweichungen) nennen die Schwächen, bevor ein Prüfer sie findet. Alle acht Punkte aus den früheren Einschätzungen sind in der Arbeit offengelegt (Abschnitt 3.1).
- **Bericht und Deutung sind getrennt.** 5.1 bis 5.6 berichten, 5.7 deutet; Vermutungen sind fast überall als solche gekennzeichnet.
- **Die Grenzen kommen in Kapitel 6 wieder an**, je Absatz mit der betroffenen Forschungsfrage.

### Wo die Risiken liegen

1. **Explorative Vergleiche tragen Richtungsaussagen (B-01, B-02).** Der Abstand von 0,18 Punkten stammt aus einer nachträglichen, nicht korrigierten Analyse; sein 95-%-Intervall liegt knapp über null, ein 98,75-%-Intervall enthält null. Kapitel 5 schreibt trotzdem „liegt vor ihm“. Zugleich ist unklar, ob vier oder sieben Vergleiche als Hauptvergleiche gelten.
2. **Das Ausgabelimit wird in 4.4.3 beschwichtigend eingeführt (B-03, B-04).** „Nahezu vollständig abgedeckt“ passt nicht zu einem Effekt von gut einem Punkt auf MultiNERD und einer Umkehr der Rangfolge auf CrossNER. Für die Zero-Shot-Läufe fehlt der Hinweis, dass sie demselben Limit unterliegen und nicht nachgerechnet wurden.
3. **Latenzaussagen (B-05, B-06).** Die Schätzung für das 0.8B-Modell bei 1 024 Tokens liegt unter dem Messwert bei 256 Tokens auf demselben GPU-Typ. Das zeigt eine Laufstreuung, die der Text nur bei CrossNER erwähnt. Ein Satz in 5.7.2 widerspricht dem eigenen Befund, obwohl die Daten eine klare Aussage tragen würden.
4. **Zwei kleine Zahlenfehler im Bezug (B-09, B-10):** „höchstens 45 je Lauf“ ist ein Seed-Mittel (Einzelläufe bis 50), „656 bis 670“ ebenfalls (Einzelläufe 638 bis 687).
5. **Quellen:** kein falscher Wert, aber rund ein Dutzend Stellen, an denen die Arbeit etwas stärker oder anders formuliert als die Quelle (Abschnitt 4.2).

Verlustfrei kürzbar ist wenig: in Kapitel 5 rund 15 Zeilen, in Kapitel 4 rund 4. Das reicht, um die vorgeschlagenen Ergänzungen (rund 8 Zeilen in Kapitel 5) ohne zusätzliche Seite aufzunehmen.

---

## 3 Inhaltliche und methodische Prüfung

### 3.1 Die acht Punkte aus früheren Einschätzungen

| Punkt | Ergebnis | Fundstelle in der Arbeit | Reicht die Beschreibung? |
|---|---|---|---|
| Limit von 256 Tokens zu knapp | **bestätigt.** Je Lauf 62 bis 72 abgeschnittene Antworten (0,4 %) mit 2,5 bis 2,7 % aller Referenzentitäten; mit 1 024 Tokens +1,12 bis +1,16 Punkte; auf CrossNER 10 % nicht parsebar | 04 Z. 567–580, Tab. 4.3, Tab. 4.9; 05 Z. 255–266 | offengelegt, aber in 4.4.3 zu beruhigend formuliert (B-03); Zero-Shot fehlt (B-04) |
| Nur nicht parsebare Antworten nachgerechnet | **bestätigt.** 604 Sätze über neun Läufe neu erzeugt, alle übrigen übernommen; dazu Kontrollstichprobe mit je 1 000 parsebaren Sätzen | 04 Z. 1100–1123; 05 Z. 306–334 | ja. Auswahlregel, ihre Schwäche und die zwei Kontrollen sind beschrieben |
| Vollständiger Neulauf nur als möglicher Schritt | **bestätigt.** Nicht durchgeführt; als Anschlussfrage genannt | 04 Z. 1132–1134; 05 Z. 332–334; Kap. 6.3 | ja |
| Testsplit schon in Vorläufen ausgewertet | **bestätigt** | 04 Z. 72–75, Z. 368–385, Tab. 4.3; Z. 895–896 | ja. Keine Stelle nennt einen späteren Lauf eine unabhängige Bestätigung; der Begriff steht nur in 6.3 für künftige Arbeit. Zwei Stellen in 5.6 sagen „bestätigt“ für die Deutung des Recall-Verlusts (Z. 718, Z. 746); „stützt“ wäre vorsichtiger |
| Double Quantization inaktiv | **bestätigt im Code.** Der Parameter heißt in `train.py:463`, `inference.py:261` und `crossner.py:273` `bnb_4bit_double_quant` statt `bnb_4bit_use_double_quant` | 04 Z. 736–741, Tab. 4.6, Tab. 4.9; Kap. 6.2 | ja. Einschränkung: Die Wirkung ist aus Quellcode und Bibliotheksstandard abgeleitet, ein Laufprotokoll der Quantisierung gibt es nicht. „Laut Quellcode“ wäre genauer |
| Drei GPU-Typen | **bestätigt** aus den Laufdateien aller 18 Läufe: H100 (Encoder, 0.8B QLoRA), A100 (4B QLoRA, 0.8B und 4B Zero-Shot), H100 NVL (27B) | Tab. 4.4, Tab. 5.4, 04 Z. 984–991 | ja; Ergänzung in B-05 |
| Tabellen 4.3 und 4.9 dokumentieren Entscheidungen und Abweichungen | **bestätigt.** 4.3: acht Entscheidungen mit Datengrundlage, Testbezug, Folge; 4.9: dreizehn Abweichungen mit Abschnittsverweis | S. 38, S. 52 | weitgehend; Lücke bei der BF16-Messung (B-08) |
| GPU-Tabelle nicht entfernen | **Tabelle 4.4 wird gebraucht.** Sie enthält vier Zeilen (CrossNER, Nachinferenz, Kontrollstichprobe), die in Tabelle 5.4 fehlen, und trägt den Hardware-Vorbehalt | S. 39 | im Kapitel belassen |

### 3.2 Die sechs genannten Ergebnisse

| Aussage | Prüfergebnis | Bezugsgröße und Bedingungen |
|---|---|---|
| 1,34 Punkte Vorsprung bei 256 Tokens | **verifiziert:** 95,82 gegen 94,48; eigenes Intervall [0,99; 1,70], Arbeit [0,99; 1,68] | Mikro-F₁ in Punkten (Skala 0 bis 100), strikt, Mittel über je drei Seeds, 16 454 Testsätze; Hauptvergleich |
| 0,18 Punkte nach selektiver Nachinferenz | **verifiziert:** 95,82 gegen 95,64; eigenes Intervall [0,02; 0,34], Arbeit [0,02; 0,33] | LLM-Vorhersagen zusammengeführt (62 bis 72 Antworten je Lauf neu, Rest unverändert); explorativ, nicht korrigiert. Alle drei Seed-Paare positiv (0,15 bis 0,21) |
| Umkehr auf CrossNER | **verifiziert:** 256 Tokens 84,99 gegen 71,69; 1 024 Tokens 84,99 gegen 87,12, Differenz 2,13 [1,47; 2,80] | nur PER, ORG, LOC; gepoolter Mikro-F₁ über 2 506 Sätze; vollständiger Neulauf nur des LLM; explorativ |
| Rund drei Viertel des CrossNER-Vorsprungs | **rechnerisch verifiziert, aber bedingt:** 2,13 schrumpft auf 0,54 (75 %), wenn man exakte **und** teilweise Treffer auf Entitäten nicht abgebildeter Typen bei beiden Modellen streicht. Nur exakte Treffer: 1,73 (19 %) | Die Bedingung steht in 5.7.4 und 6.1; der Wert 1,73 fehlt (B-11) |
| 0.8B rund 210-mal langsamer als DeBERTa-v3-base | **verifiziert:** 1 875,8 ms gegen 8,92 ms = 210,2 | mittlere Latenz des Modellaufrufs, Batchgröße 1, beide auf H100, LLM mit 256 Tokens. Die Kontrollstichprobe ergibt bei 1 024 Tokens Faktor 193 |
| FF3 über nachträgliche Schwellen | **bestätigt:** 04 Z. 993–1001 sagt ausdrücklich, dass keine Budgets vorab feststanden; die Schwellen in 05 Z. 1030–1035 stimmen mit Tabelle 5.4 überein | Schwellen gelten für Batchgröße 1 und den jeweiligen GPU-Typ; siehe B-07 |

„F₁-Punkte“ wird einheitlich für die Skala 0 bis 100 verwendet (Definition in der Einleitung von Kapitel 5), „Prozentpunkte“ nur für Typanteile. Das ist korrekt.

### 3.3 Befunde

#### Hohe Priorität

**B-01 Explorative Vergleiche werden wie gesicherte Rangfolgen formuliert**
- *Fundstellen:* 05 Z. 799–801 („Beide Intervalle schließen null aus“), Z. 1028 („bleibt … knapp vorn“), Z. 1040–1041 („das selbst mit 1 024 Tokens vor ihm liegt“), Z. 745–756 (CrossNER mit 1 024 Tokens).
- *Befund:* 04 Z. 893–896 stuft alle Vergleiche mit 1 024 Tokens als explorativ und nicht korrigiert ein. Kapitel 5 übernimmt diese Einstufung nicht; erst 6.1 sagt, das Intervall liege „nur knapp über null“.
- *Begründung:* In meiner Gegenrechnung liegen 1,1 % der Bootstrap-Stichproben bei einer Differenz von null oder darunter; schon ein 98,75-%-Intervall (Korrektur für vier Vergleiche) reicht von −0,02 bis +0,38. Die Rangfolge „Encoder vor LLM“ bei 1 024 Tokens ist also ein Punktwert, kein gesicherter Unterschied.
- *Vorschlag:* in 5.7.1 und 5.7.4 je eine Einschränkung (K-01, K-02); in 5.6 das Wort „explorativ“.

**B-02 Unklar, welche Vergleiche Hauptvergleiche sind**
- *Fundstellen:* 04 Z. 889–894; 05 Z. 1094–1096; Tabelle 5.9 (S. 72); Kap. 6.2.
- *Befund:* 4.5.1 zählt in der Klammer vier Vergleiche auf (large gegen 27B, drei Größenschritte). 5.7.5 spricht von „sieben dort ausgewiesenen Vergleichen“ bei 256 Tokens. 6.2 nennt „die übrigen“ explorativ. Für drei Vergleiche (27B gegen base, base gegen 0.8B, 4B gegen base) ist der Status damit offen, und Tabelle 5.9 kennzeichnet nichts.
- *Vorschlag:* Du entscheidest, ob vier oder sieben gelten. Danach ein Satz in 4.5.1, die Zahl in 5.7.5 und ein Halbsatz in der Beschriftung von Tabelle 5.9 (K-03). Der Tabellenkörper bleibt unverändert.

**B-03 Das Ausgabelimit wird in 4.4.3 verharmlosend eingeführt**
- *Fundstelle:* 04 Z. 568–577.
- *Befund:* „nahezu vollständig abdeckt“ zählt Sätze (55 von 16 454). Die betroffenen Sätze enthalten aber rund 2,6 % aller Referenzentitäten, und das Limit verschiebt das Hauptergebnis um gut einen Punkt. „Dasselbe Bild“ für Training und Validierung ist ungenau: Der Satz belegt dann nur, dass 1 024 Tokens reichen, nicht, wie oft 256 überschritten werden.
- *Begründung:* Kapitel 4 soll Einschränkungen nicht kleiner erscheinen lassen, als Kapitel 5 sie zeigt. Die eigene Lehre aus 6.1 („Ausgabelimits sind vor der Evaluation zu prüfen“) setzt voraus, dass das Limit erkennbar zu knapp war.
- *Vorschlag:* K-04.

#### Mittlere Priorität

**B-04 Zero-Shot-Werte unterliegen demselben Limit, ohne Sensitivitätsanalyse**
- *Fundstellen:* 05 Z. 111–120, Z. 301–304; 04 Z. 1100–1101; Tab. 4.9 Zeile „Längengrenzen“.
- *Befund:* 2 304, 256 und 139 nicht parsebare Zero-Shot-Antworten enden ohne schließende Klammer und enthalten 3,8 bis 16,2 % der Referenzentitäten. Nachgerechnet wurden nur die QLoRA-Läufe. Die Folge für den Abstand zwischen den Regimen (87,0, 59,0 und 25,1 Punkte) steht erst im Ausblick (6.3).
- *Vorschlag:* ein Satz in 5.1.1 (K-05).

**B-05 Laufstreuung der Latenz bleibt bei MultiNERD unerwähnt**
- *Fundstellen:* 05 Z. 495–505, Z. 976–979, Z. 1024–1028.
- *Befund:* Die Kontrollstichprobe schätzt für das 0.8B-Modell 1,72 s bei 1 024 Tokens. Der Hauptlauf maß 1,88 s bei 256 Tokens, auf demselben GPU-Typ. Die Schätzung liegt also 8 % niedriger, obwohl die Antworten nicht kürzer sind.
- *Begründung:* Unterschiede dieser Größe entstehen ohne Hardwarewechsel. Die Aussage, die Halbierung beim 4B-Modell spreche „für einen deutlichen Hardwareanteil“, bleibt plausibel, aber ihr Umfang ist offen. Bei CrossNER nennt die Arbeit dieselbe Erscheinung ausdrücklich (Z. 764–768).
- *Vorschlag:* K-06.

**B-06 Der Latenzsatz in 5.7.2 widerspricht dem eigenen Befund**
- *Fundstelle:* 05 Z. 955–959.
- *Befund:* Der Satz sagt, Decoding-Schritte und Dequantisierung fielen „bei jeder Entität“ an, „unabhängig davon, wie kompakt sie kodiert ist“. Die Zahl der Decoding-Schritte hängt aber gerade von der Kodierung ab. Der vorangehende Satz (Z. 950–952) und 5.2 (Z. 335–338) sagen das selbst.
- *Begründung aus den Daten:* In den Kontrollstichproben (Seed 42, je 1 000 Sätze) korreliert die Latenz je Satz mit der Zahl erzeugter Tokens zu r = 0,99 bis 1,00; das Verhältnis von 95. Perzentil zu Median beträgt bei den Tokens 3,6, bei der Latenz 3,3 bis 3,4. Das ist meine Gegenrechnung, keine Zahl der Arbeit.
- *Vorschlag:* Satz umformulieren (K-07). Wahlweise die Korrelation als Messwert aufnehmen; das bräuchte eine Erweiterung des Aggregationsskripts und eigene Freigabe. Dann wäre auch das „weil“ in Z. 506–508 belegt.

**B-07 FF3 fragt nach einem „Verhältnis“, beantwortet wird eine Auswahl unter Grenzen**
- *Fundstellen:* Abschnitt 1.2 (FF3: „das günstigste Verhältnis von Erkennungsleistung und Ressourcenbedarf“); 04 Z. 993–1001; 05 Z. 1030–1044.
- *Befund:* Die Methode bestimmt je Budget die zulässige Konfiguration mit dem höchsten Mikro-F₁. Das ist sauber, aber kein Verhältnis.
- *Vorschlag:* in 4.5.2 einen Satz, der die Operationalisierung benennt (K-08). Die Forschungsfrage selbst bleibt unverändert.

**B-08 BF16-Messung auf Testsätzen: Offenlegung unvollständig**
- *Fundstellen:* Tab. 4.3, 04 Z. 421–423; Tab. 4.9; Kap. 6.2.
- *Befund:* Die Tabelle nennt den Testbezug „direkt, nicht leistungsbezogen“ und die Datengrundlage „standardmäßig 96 Sätze“. Das Messskript (`ba-ner/src/decoder/benchmark_inference.py`, Z. 283 und 294) gibt neben dem Durchsatz auch den F₁ der Testteilmenge aus. Ausgaben dieser Messung liegen lokal nicht vor.
- *Begründung:* Ob der F₁ die Entscheidung beeinflusst hat, lässt sich aus den Dateien nicht ablesen. „Standardmäßig“ zeigt, dass der tatsächliche Umfang nicht protokolliert ist.
- *Vorschlag:* in der Tabellenzelle ergänzen, dass die Messung auch den F₁ der Teilmenge ausgab und der Durchsatz maßgeblich war, sofern das zutrifft; „standardmäßig“ durch „Voreinstellung 96 Sätze, tatsächlicher Umfang nicht protokolliert“ ersetzen. Das ist eine inhaltliche Angabe, die nur du bestätigen kannst.

**B-09 „höchstens 45 je Lauf“ ist falsch**
- *Fundstelle:* 05 Z. 272–274 (S. 56).
- *Befund:* Die Text-Abweichungen betragen bei den drei 0.8B-Läufen 50, 47 und 39. 45 ist das Seed-Mittel aus Tabelle 5.2.
- *Vorschlag:* „im Mittel über die Seeds höchstens 45“ oder „je Lauf höchstens 50“.

**B-10 „656 bis 670 Referenzentitäten“ sind Mittelwerte, der Satz spricht von Läufen**
- *Fundstelle:* 05 Z. 259–266.
- *Befund:* Der Satz beginnt mit „Je Lauf brechen 62 bis 72 Antworten … ab“ (das stimmt je Lauf). Die Entitätenzahl ist dagegen das Seed-Mittel je Modellgröße; je Lauf sind es nach meiner Zählung 638 bis 687.
- *Vorschlag:* „enthalten je Modellgröße im Seed-Mittel 656 bis 670 Referenzentitäten“.

**B-11 CrossNER-Analyse nennt nur die günstigere Variante**
- *Fundstelle:* 05 Z. 1066–1070; Kap. 6.1.
- *Befund:* „Drei Viertel“ gilt nur, wenn auch Teilspans gestrichen werden. Streicht man nur exakt deckungsgleiche Spans, bleiben 1,73 von 2,13 Punkten. Die Bedingung steht im Text, der zweite Wert nicht.
- *Vorschlag:* den Wert in Klammern ergänzen: „(nur bei exakt deckungsgleichen Spans: von 2,1 auf 1,7)“. Quelle ist die Ausgabe von `crossner_unmapped_fp.py`.

#### Niedrige Priorität

| Nr. | Fundstelle | Befund | Vorschlag |
|---|---|---|---|
| B-12 | 05 Z. 255–258 | „gehen auf das Limit … zurück“ steht als Tatsache vor dem Beleg. 04 Z. 1105–1114 nennt die Regel einen Stellvertreter, der Stoppgrund ist nicht protokolliert; 6.1 schreibt richtig „hinweist“ | „…, was auf das Limit von 256 Ausgabetokens hinweist; die Nachinferenz stützt das (siehe unten)“ |
| B-13 | 05 Z. 157–158 | „im Zero-Shot-Regime … zwischen 3,2 und 45,0“ gilt nur für das 27B-Modell | „im Zero-Shot-Regime (27B)“ |
| B-14 | 05 Z. 975–976 | „Das 4B-Modell fällt in der Trainingsdauer … aus der Reihe“ ist aus Tabelle 5.4 nicht erkennbar (12,1, 62,9, 266,2 h: Faktoren 5,2 und 4,2) | benennen, was gemeint ist (vermutlich die Latenz nahe am 27B-Modell), oder streichen |
| B-15 | 05 Z. 1099–1101 | „3,8 bis 5,2 Punkten Breite (trainierte Konfigurationen)“: 3,8 entsteht nur mit den 1 024-Token-Zeilen; bei 256 Tokens sind es 3,9 bis 5,2 | „einschließlich der Auswertungen mit 1 024 Tokens“ ergänzen oder 3,9 schreiben |
| B-16 | 05 Z. 479–483, Z. 291–293, Z. 527–531 | Ursachen als Tatsache: „der Aufwand entsteht vor allem durch“, „das ist die Hauptursache“, „geht vor allem auf … zurück“. Plausibel, aber nicht gemessen | je ein „dürfte“ oder „vermutlich“ |
| B-17 | 05 Z. 328–331 | „dort … in beiden Kontrollen“ ist schwer zuzuordnen | „Auch beim 0.8B-Modell, dessen Kontrollen auf dem GPU-Typ des Hauptlaufs liefen, treten in beiden Kontrollen Abweichungen auf; der GPU-Wechsel erklärt sie also nicht allein.“ |
| B-18 | 05 Z. 1053–1057 | Drei Sätze ohne Träger: Wessen „höhere Werte“? Warum steht hier die Änderung des Typinventars? | „Die höheren Punktwerte des LLM bei einzelnen seltenen MultiNERD-Typen bleiben …“; den letzten Satz mit Anlass versehen oder streichen |
| B-19 | 05 Z. 1071–1073 | „die Neuinferenz allein des LLM“ als Störgröße ist unklar: Die Encoder-Vorhersagen hängen nicht vom Ausgabelimit ab | erklären, welcher Effekt gemeint ist, oder streichen |
| B-20 | 05 Z. 858–859 | 65,17 gegen 90,98 sind Makro-F₁-Werte (so in 1.2), die eigenen Vergleichswerte Mikro-F₁ | „Makro-F₁“ ergänzen |
| B-21 | 04 Z. 1014–1015 | „halluzinierte Kategorien“ kommt nur hier vor; Kapitel 5 spricht von „unbekannten Typen“ | Satz streichen oder Begriff angleichen |
| B-22 | 04 Z. 1191–1193 gegen Z. 279–289 | Überlange Sätze: MultiNERD „gekürzt“, CrossNER „Abbruch“, obwohl „die Inferenz entspricht der MultiNERD-Testinferenz“. Ohne Wirkung, da kein Satz betroffen ist | Unterschied benennen oder Klammer streichen |
| B-23 | 04 Z. 1134–1136 | „Beide Auswertungen gehen in die Schlussfolgerungen ein (Abschnitt 5.2)“: Die Schlussfolgerungen stehen in 5.7.1 | Verweis auf 5.7.1 |
| B-24 | 04 Z. 870–875 | Der Startwert des Bootstrap (42 laut `summary.json`) steht nicht im Text | „(Startwert 42)“ ergänzen |
| B-25 | 05 Z. 31–36 | 5.5 und 5.6 sind keiner Forschungsfrage zugeordnet; 6.2 ordnet CrossNER FF1 und FF3 zu | Halbsatz „ergänzend zu FF1 und FF3“ |
| B-26 | Abschnitt 1.2, Tab. 4.3, 04 Z. 1230–1231 | Zählung der Sensitivitätsanalysen (zwei in 1.2, drei in Tab. 4.3; dazu der Ausschluss der 21 Sätze unter demselben Namen) | bekannt aus dem Lektorat, weiterhin offen |

In der geprüften Fassung fehlt außerdem der Rundungssatz. Wer aus den Tabellen nachrechnet, erhält 12,3 statt 12,2 (05 Z. 717) und 30 statt 28 (Z. 477). Das löst der Branch `korrekturen-2026-11-25`.

### 3.4 Roter Faden und Verbindung der Kapitel

- **Forschungsfragen:** Die Einleitung von Kapitel 5 ordnet 5.1 und 5.2 FF1 zu, 5.3 und 5.4 FF2; FF3 wird in 5.7.4 beantwortet. Das ist klar. Kapitel 4 nennt die Zuordnung nur verstreut (4.5.1 Hauptvergleiche, 4.5.2 Schwellen, 4.5.4 Leitfrage). Ein Satz am Anfang von 4.5 würde reichen; nötig ist er nicht.
- **Methode trägt die Analysen:** Jede Auswertung in Kapitel 5 hat ihre Beschreibung in Kapitel 4 (Bootstrap, Fehlerklassen, Formtreue-Zähler, Nachinferenz, Kontrollstichprobe, CrossNER-Abbildung, Schwellen). Ausnahme: Die Analyse der falsch positiven Spans auf nicht abgebildeten Typen (05 Z. 1063–1070) ist in 4.6 nicht angekündigt; dort steht nur die Zählregel (04 Z. 1161–1162).
- **Einschränkungen kehren wieder:** Testnutzung, Hardware, Limit, Silberreferenz und Seeds werden in 5.7 und 6.2 aufgegriffen. Lücken sind B-01 (Status der Vergleiche) und B-04 (Zero-Shot und Limit).
- **Erwartungen E1 bis E3** werden laut 3.5 erst in Kapitel 6 abgeglichen. Dass Kapitel 5 sie nicht nennt, ist also gewollt.
- **FF3 ist nur als Fließtext beantwortet.** 05 Z. 1030–1035 zählt neun Schwellen auf. Eine kleine Tabelle (Grenze, zulässige Konfigurationen, beste Wahl) wäre lesbarer und etwa gleich lang; das wäre eine neue Tabelle und braucht deine Zustimmung.

---

## 4 Quellenprüfung

### 4.1 Zuverlässig gestützt

Am Volltext bestätigt (Fundstelle in der Quelle in Klammern):

- **Obeidat et al.:** LLM auf vier von fünf Datensätzen 1,9 bis 7,9 Punkte vorn, Encoder verlieren vor allem Precision (Tab. III); Faktoren 40 und knapp 220 (Tab. IV); Empfehlung des Encoders bei tolerierbaren Unterschieden.
- **Zhan et al.:** LoRA 0,14 bis 0,48 Punkte vor vollständigem Fine-Tuning (Anh. D, Tab. 11); Zeichenoffsets im Mittel 53,91 (Tab. 3); Fehlermuster auf CoNLL-2003.
- **Keraghel et al.:** 65,17 gegen 90,98 (Tab. 5, Makro-F₁); Rangwechsel von base und large (Tab. 5). **Auch „mit Beispielentitäten im Prompt“ ist belegt:** Die Arbeit beschreibt die Prompts mit „multiple examples“ und zeigt je Typ Beispielentitäten. Der delegierte Prüflauf hatte das als unbelegt gemeldet; das war falsch.
- **Tan et al.:** 14,78 gegen 92,77 (Tab. 4). Kapitel 3 nennt Metrik, Split und System korrekt; Kapitel 5 verweist nur kurz darauf.
- **Sainz et al.:** 76,0 für GoLLIE-7B auf MultiNERD (Tab. 3); acht gesehene Typen (Tab. 6, aus der Textfassung abgelesen).
- **Zhou et al.:** 93,73 und 91,25 (Tab. 2).
- **Tedeschi und Navigli:** ANIM, BIO, FOOD, PLANT häufig verwechselt und oft nicht verlinkt (Abschn. 5.4, Tab. 5).
- **Dettmers et al.:** rund 0,37 Bit je Parameter (Abschn. 3); Paged Optimizer als Teil von QLoRA. Die Folgerechnung „höchstens rund 1,2 GiB“ stimmt.
- **TRL-Dokumentation 1.8.0:** `assistant_only_loss` standardmäßig `False`.
- **Qwen3.5-27B-Karte (festgeschriebene Revision, selbst gelesen):** Apache-2.0; „Pre-training & Post-training“; 16 Blöcke aus drei Gated-DeltaNet-Schichten und einer Attention-Schicht; keine Angabe zu Vortrainingsdaten.
- **Weitere:** Cawley und Talbot, Koehn, Lakens, Segura-Bedmar et al., seqeval, Geng et al., Liu et al. (CrossNER, MIT-Lizenz), Hu et al., Kaplan et al., Hoffmann et al., Wei et al. (ChatIE), Ramponi et al. (Tab. 9), Semin et al. (Abschn. 3.2, LogitMatch), Dagdelen et al., Lu et al., Modellkarten von DeBERTa-v3, Datensatzkarte von MultiNERD.

### 4.2 Stärker oder anders formuliert als die Quelle

| Nr. | Fundstelle | Aussage der Arbeit | Was die Quelle trägt | Vorschlag |
|---|---|---|---|---|
| Q-01 | 04 Z. 716–718 | Adapter in unterschiedlichen Linearschichten stellen „komplementäre Kapazität“ bereit (Dettmers, Han, Wang) | Dettmers: LoRA in allen Linearschichten ist nötig, um vollständiges Fine-Tuning zu erreichen. Han: Attention- und MLP-Schichten sind die wirksamsten Positionen. Bei Wang wurde keine Aussage zur Platzierung gefunden | „Befunden, nach denen Adapter in allen Linearschichten nötig sind, um die Leistung vollständigen Fine-Tunings zu erreichen“; Wang hier streichen oder Stelle nachtragen |
| Q-02 | 05 Z. 819–821 | LLM und spezialisierte Modelle auf flachen Datensätzen knapp auseinander, „dort mit dem LLM vorn“ | Tab. 3: auf CoNLL-2003 93,85 gegen 93,64 (LLM vorn), auf OntoNotes 5.0 91,28 gegen 92,07 (Spezialmodell vorn) | „(auf CoNLL-2003 mit dem besten LLM knapp vorn, auf OntoNotes 5.0 knapp dahinter)“ |
| Q-03 | 05 Z. 869–873 | „MultiNERD-Variante mit eigener Datenaufbereitung“ | Die Aufbereitung stammt aus InstructUIE; der BERT-base-Wert ist von dort übernommen. Kapitel 3 sagt das richtig | „auf der MultiNERD-Variante des InstructUIE-Benchmarks“ |
| Q-04 | 05 Z. 980–982 | Bei GoLLIE und UniversalNER übersteigt der Gewinn durch Anpassung den durch Größe (Abschn. 5.2) | Die Werte tragen das (GoLLIE: 42,3 ohne, 55,3 mit 7B, 57,2 mit 34B; UniNER 41,7 und 43,4 gegen ChatGPT 34,9). Die Quellen selbst betonen den Nutzen größerer Modelle und ziehen den Vergleich nicht | als eigene Lesart kennzeichnen: „nach den dort berichteten Werten“ |
| Q-05 | 05 Z. 991–994 | Fähigkeiten „wie das Befolgen komplexer Formatvorgaben“ treten erst ab einer gewissen Größe auf (Wei et al.) | Die Quelle behandelt das Befolgen von Aufgabeninstruktionen, nicht von Ausgabeformaten | „wie das Befolgen von Instruktionen“; der Satz ist schon als Vermutung gekennzeichnet |
| Q-06 | 05 Z. 995–999 | Architektur, Vortrainingsziel und Anpassungsverfahren überlagern die Größe (Zaratiana, Bogdanov) | Beide Arbeiten zeigen kleine Encoder auf dem Niveau großer LLMs und führen das vor allem auf Umfang und Vielfalt der Vortrainingsdaten zurück | „Architektur, Vortrainingsdaten und Anpassungsverfahren“ |
| Q-07 | 05 Z. 935–939 | „Formatvorgabe und Typinventar“ begrenzen ohne Training die Leistung, wie Lu et al. beschreiben | Lu et al. nennen die ungewohnten BIO-Symbole auf Tokenebene; das Typinventar nennen sie nicht | Bezug auf die Formatvorgabe beschränken |
| Q-08 | 05 Z. 943–944 | Tam et al. berichten Leistungseinbußen durch Formatzwang | Belegt für Reasoning-Aufgaben; bei Klassifikation verbessert der JSON-Modus die Genauigkeit | „bei Reasoning-Aufgaben“ ergänzen. 04 Z. 688 („können … beeinträchtigen“) ist in Ordnung |
| Q-09 | 04 Z. 846–847 | Metriken „nach IOB2-Schema“ mit Sang und De Meulder sowie Ramshaw und Marcus | Beide beschreiben IOB1. Kapitel 2 trennt das sauber | Belege hinter „auf Entitätsebene“ setzen und für IOB2 auf Abschnitt 2 verweisen |
| Q-10 | 04 Z. 123–124, Z. 224–226, Z. 313–316 | „offizielle Splits“ mit Tedeschi und Navigli | Der Aufsatz definiert keine Splits; sie kommen mit dem veröffentlichten Datensatz | zusätzlich die Datensatzkarte zitieren (steht schon im Literaturverzeichnis) |
| Q-11 | 04 Z. 240–244 | Stichprobe von „1 000 zufällig gezogenen englischen Sätzen“; „in den veröffentlichten Daten nicht enthalten“ | Zufällig mit mindestens 20 Instanzen je Klasse (Fußnote 13; so steht es in 6.3). Dass die Stichprobe in den Daten fehlt, sagt keine Quelle | „mit mindestens 20 Instanzen je Typ“ ergänzen; „soweit ersichtlich“ vor „nicht enthalten“ |
| Q-12 | 04 Z. 168–169 | Qwen3.5 baut auf der „Architektur- und Trainingstradition“ von Qwen2.5 und Qwen3 auf | Der Blog war nicht lesbar. Die Berichte zu den Vorgängern können das nicht belegen; die Karten beschreiben eine neue hybride Architektur | abschwächen („folgt in der Qwen-Familie auf Qwen2.5 und Qwen3“) oder Blog von Hand prüfen |
| Q-13 | 04 Z. 699 | Adapter senken Speicher- und Rechenbedarf (Hu, Han, Obeidat) | Obeidat et al. nutzen QLoRA und nennen den höheren Bedarf der LLMs, vergleichen aber nicht mit vollständigem Fine-Tuning | Obeidat an dieser Stelle streichen |
| Q-14 | 04 Z. 281–284 | Wert −100 für ignorierte Positionen (Devlin, Keraghel) | Devlin: erstes Subwort trägt das Label. −100 ist eine Konvention der Implementierung | Beleg hinter den ersten Halbsatz ziehen |
| Q-15 | 05 Z. 874–878 | Fehlermuster „der Encoder“ gegen „das LLM“ | Verglichen werden nur BERT-MRC und LLaMA3.1-8B | Modelle nennen |
| Q-16 | Anhang, Sampling (von 04 Z. 588–591 genannt) | Empfehlung für Reasoning: Temperatur 1,0, top-p 0,95, top-k 20, Presence-Penalty 1,5 | Die 27B-Karte nennt **zwei** Sätze: im Schnellstart diese Werte, unter „Best Practices“ top-p 1,0, top-k 40, Presence-Penalty 2,0. „Nur teilweise“ in 4.4.3 stimmt in beiden Fällen | im Anhang beide Varianten nennen (Ergänzung, keine Kürzung) |

### 4.3 Nicht verifiziert

- Efron und Tibshirani 1993, Manning et al. 2008: formal vorhanden, inhaltlich nicht geprüft.
- Qwen3.5-Blog: nicht lesbar (Q-12).
- Wang et al. 2024: nur durchsucht.
- Obeidat et al.: gelesen wurde die arXiv-Fassung, nicht die IEEE-Fassung.
- CrossNER-Lizenz: am aktuellen Stand des Repositorys bestätigt, nicht am festgeschriebenen Commit.
- Karten von Qwen3.5-0.8B und -4B sowie DeBERTa-v3: nur im delegierten Prüflauf gelesen.

### 4.4 Aussagen, die einen Beleg brauchen

- 04 Z. 242–244: Die Stichprobe sei in den veröffentlichten Daten nicht enthalten. Das ist eine eigene Beobachtung; sie braucht „soweit ersichtlich“ und am besten ein Prüfdatum.
- 05 Z. 479–483: Ursache des Trainingsaufwands. Entweder als Vermutung kennzeichnen (B-16) oder belegen.
- Weitere fehlende Belege habe ich nicht gefunden. Zitate im Text und Einträge im Literaturverzeichnis stimmen überein (44 Schlüssel in Kapitel 4, 16 in Kapitel 5, keiner fehlt).

---

## 5 Tabellen und Abbildungen

### 5.1 Darstellungen mit Empfehlung

**Tabelle 5.4 Ressourcenbedarf (S. 60), Text 05 Z. 469–486** · Empfehlung 2: Text kürzen
- *Doppelt:* 0,4 und 1,2 h; 12,1 bis 266 h; 3 und 5,5 Minuten; 8,6 bis 25,2 h.
- *Zusatz im Text:* Faktor 28, Erklärung des Aufwands, Hinweis auf Neustarts.
- *Muss bleiben:* Faktor 28 und der Hinweis, dass die 27B-Dauer überlagert ist.
- *Vorschlag:* „Das QLoRA-Training dauerte um ein Vielfaches länger als das vollständige Fine-Tuning der Encoder (Tabelle 5.4).“ und „Auch die Testinferenz über die 16 454 Sätze dauerte bei den QLoRA-Modellen Stunden statt Minuten.“
- *Wirkung:* rund 1,5 Zeilen. Die Spannen stehen weiterhin in Tabelle 5.4 und in 6.1.

**Tabelle 5.5 Fehlerprofile (S. 63), Text 05 Z. 621–624** · Empfehlung 2
- *Doppelt:* Die Definition der Zusätze steht in 4.5.1, im Einleitungssatz und in der Beschriftung.
- *Vorschlag:* „Tabelle 5.5 ordnet jede Referenzentität nach dem Schema aus Abschnitt 4.5.1 genau einer Klasse zu.“
- *Wirkung:* rund 1,5 Zeilen, kein Verlust.

**Tabelle 5.6 CrossNER (S. 64), Text 05 Z. 691–694** · Empfehlung 3: Text auf Verweis reduzieren
- *Doppelt:* Modelle, Training auf MultiNERD, fünf Domänen, Grobtypen stehen in 4.6 und in der Beschriftung.
- *Vorschlag:* „Tabelle 5.6 berichtet den Transfertest aus Abschnitt 4.6.“
- *Wirkung:* rund 3 Zeilen. Die folgende Deutung bleibt unverändert und ist nötig.

**Tabelle 5.8 Synthese (S. 66), Text 05 Z. 795–802** · Empfehlung 5: beides behalten, Funktion trennen
- *Doppelt:* Die beiden MultiNERD-Zeilen stehen auch in Tabelle 5.9, die CrossNER-Zeilen in der Anhangtabelle; der Text wiederholt beide Abstände samt Intervall.
- *Funktion:* Die Tabelle stellt die vier Bedingungen nebeneinander; das ist der Kern von FF1 und sollte bleiben.
- *Vorschlag:* im Text die Intervalle weglassen und den gewonnenen Platz für die Einstufung aus B-01 nutzen (K-01).

**Tabelle 5.9 Bootstrap-Intervalle (S. 72)** · Empfehlung 5
- *Befund:* Die Tabelle wird ab S. 53 mehrfach zitiert (5.1.1, 5.1.2, 5.3, 5.7.1), steht aber am Kapitelende. Wer die Intervalle prüfen will, blättert 18 Seiten vor.
- *Vorschlag ohne Verschieben:* Hauptvergleiche in der Beschriftung kennzeichnen (B-02).
- *Vorschlag mit Verschieben (nur mit deiner Zustimmung):* die Tabelle nach 5.1.1 setzen. Folge: neue Nummerierung aller Tabellen in Kapitel 5 (automatisch), neue Seitenumbrüche in einem knapp gesetzten Kapitel. Ich würde das erst nach den Textänderungen probeweise bauen.

**Abbildung 5.2 Modellgröße (S. 59), Text 05 Z. 420–439** · Empfehlung 6, eine wahlweise Kürzung
- Der Text nennt die Differenzen (0,3, 0,82, 0,51, 28,8, 34,5), die Abbildung zeigt den Verlauf, die Werte stehen im Anhang. Das ergänzt sich.
- *Wahlweise:* Die Klammer mit den fünf Parameterzahlen in der Beschriftung (Z. 415–416) wiederholt Tabelle 4.1, auf die derselbe Satz verweist. Rund 1 Zeile. Nachteil: Die Punkte auf der logarithmischen Achse sind dann schwerer zuzuordnen.
- Dem rechten Teilbild fehlt die Beschriftung der y-Achse; da beide Teilbilder verschiedene Wertebereiche haben, wäre sie nützlich (Änderung an der Abbildung, nur mit Zustimmung).

**Abbildung 5.3 Abwägung (S. 62), Text 05 Z. 533–541** · Empfehlung 5
- *Befund:* Auf einer y-Achse von 0 bis 100 liegen die fünf trainierten Konfigurationen (93,1 bis 95,8) übereinander. Die Aussage des Textes zur Pareto-Front lässt sich in der Leistungsrichtung nicht aus der Abbildung ablesen, nur aus den Tabellen 5.1 und 5.4.
- *Funktion heute:* Die Abbildung zeigt den Abstand in Latenz und Speicher und die Schwäche der Zero-Shot-Konfigurationen.
- *Vorschlag:* ein Satz in der Beschriftung: „Die Unterschiede zwischen den trainierten Konfigurationen sind in dieser Skalierung nicht ablesbar (Werte in Tabelle 5.1).“
- *Größere Lösung (nur mit Zustimmung):* ein drittes Teilbild oder ein Achsenausschnitt für die trainierten Konfigurationen, wie in Abbildung 5.2 rechts.

**Tabelle 4.6 LoRA-Hyperparameter, Text 04 Z. 703–710** · Empfehlung 2
- *Doppelt:* Rang und Skalierung je Größe.
- *Vorschlag:* „Rang und Skalierung ($\alpha = 2r$) nennt Tabelle 4.6; der höhere Rang des 27B-Modells folgt der Heuristik, größeren Modellen mehr Adapterkapazität zuzuweisen. Alternative Ränge wurden nicht untersucht.“
- *Wirkung:* rund 1 Zeile.

**Tabelle 4.4 GPU-Typen (S. 39), Text 04 Z. 1124–1128** · Empfehlung 1 für die Tabelle, wahlweise Kürzung im Text
- Die Beschriftung und 4.5.5 sagen beide, dass 4B und 27B in den Sensitivitätsanalysen auf einem anderen GPU-Typ liefen.
- *Wahlweise:* in 4.5.5 auf „liefen auf der H100, dem GPU-Typ der Encoder (Tabelle 4.4)“ kürzen, rund 1,5 Zeilen. Ich würde es eher lassen, weil 5.2 den GPU-Wechsel als mögliche Ursache der Abweichungen aufgreift.

### 5.2 Keine Änderung nötig

| Darstellung | Warum Text und Darstellung sich ergänzen |
|---|---|
| Tab. 4.1 Konfigurationen | Text erklärt Läufe, Vorläufe und Variablen; keine Nacherzählung |
| Tab. 4.2 Datensplits | Text nennt nur abgeleitete Anteile (74 %, 80 %, Verschiebung bei LOC und PER) |
| Tab. 4.3 Datengrundlage der Entscheidungen | Offenlegungstabelle; der Text in 4.3.3 nennt die Folgen. **Nicht kürzen** |
| Tab. 4.5 Encoder-Hyperparameter | Text begründet FP32 und beschreibt die Checkpoint-Auswahl |
| Tab. 4.7 Formtreue-Kennzahlen | Regeln stehen in 4.4.6, die Tabelle ordnet Zähler den Fehlerklassen zu; verschiedene Funktion |
| Tab. 4.8 CrossNER-Abbildung | Text nennt die Zählregeln |
| Tab. 4.9 Abweichungen | Übersicht mit Verweisen; bewusst zusammenfassend. **Nicht kürzen** |
| Tab. 5.1 Hauptergebnisse | Text berichtet Differenzen, Intervalle und das Profil aus Precision und Recall |
| Abb. 5.1 Typen | Zeigt den Abstand des Zero-Shot-Regimes und die schwachen Typen; die kleinen Unterschiede nennt der Text mit Verweis auf den Anhang |
| Tab. 5.2 Formtreue | Text liefert Zusatzangaben, die nicht in der Tabelle stehen (Satzlänge, Einzelfälle, Beispiele unbekannter Typen) |
| Tab. 5.3 Nachinferenz | Text nennt die Teilmengenwerte und die beiden Kontrollen |
| Tab. 5.7 Limit auf CrossNER | Text ordnet ein und nennt die unerklärte Latenz |

---

## 6 Kürzungs- und Präzisierungsvorschläge

Einsparungen sind geschätzte Zeilen im Satzspiegel.

### 6.1 Präzisierungsbedürftig

**K-01 (zu B-01) · 05 Z. 796–801**
- *Original:* „Bei 256 Tokens liegt Qwen3.5-27B im QLoRA-Regime 1,34 Punkte (Intervall 0,99 bis 1,68) hinter DeBERTa-v3-large, nach der Nachinferenz am Ausgabelimit mit 1 024 Tokens noch 0,18 Punkte (0,02 bis 0,33). Beide Intervalle schließen null aus; die beiden Abstände unterscheiden sich in ihrer praktischen Größenordnung jedoch deutlich.“
- *Vorschlag:* „Bei 256 Tokens liegt Qwen3.5-27B im QLoRA-Regime 1,34 Punkte hinter DeBERTa-v3-large, nach der Nachinferenz mit 1 024 Tokens noch 0,18 Punkte. Beide Intervalle schließen null aus. Der zweite Vergleich ist jedoch explorativ und nicht für Mehrfachvergleiche korrigiert, und sein Intervall liegt nur knapp über null.“
- *Erhalten:* beide Abstände; die Intervalle stehen in Tabelle 5.8 direkt darunter. *Risiko:* keines. *Länge:* etwa gleich.

**K-02 (zu B-01) · 05 Z. 1040–1041 und Z. 1028**
- *Original:* „… DeBERTa-v3-large, das selbst mit 1 024 Tokens vor ihm liegt.“
- *Vorschlag:* „… DeBERTa-v3-large, das auch mit 1 024 Tokens im Punktwert knapp vor ihm liegt.“ In Z. 1028 entsprechend „im Punktwert knapp vorn“.

**K-03 (zu B-02) · 04 Z. 889–894, 05 Z. 1094–1096, Beschriftung von Tabelle 5.9**
- *Vorschlag für 4.5.1, falls vier gelten:* „Als Hauptvergleiche gelten vier aus FF1 und FF2 abgeleitete Gegenüberstellungen bei 256 Ausgabetokens: DeBERTa-v3-large gegen Qwen3.5-27B im QLoRA-Regime und die drei Größenschritte innerhalb der Familien. Alle übrigen Vergleiche, auch die der nachträglichen Sensitivitätsanalysen, sind explorativ.“
- *Beschriftung:* „Hauptvergleiche: Zeilen 1 und 3 bis 5.“
- *Länge:* plus rund 1,5 Zeilen in Kapitel 4, plus eine halbe in Kapitel 5.

**K-04 (zu B-03) · 04 Z. 568–577**
- *Vorschlag:* „Dieser Wert wurde vor den Läufen ohne vorherige Längenanalyse festgelegt. Nachträglich zeigt sich, dass 55 der 16 454 Referenzantworten des Testsplits unter dem Qwen3.5-Tokenizer 256 Tokens überschreiten und keine 1 024; weil es die Sätze mit den meisten Entitäten trifft, wirkt sich das Limit dennoch spürbar aus (Abschnitt 5.2). An Training und Validierung hätte sich das vorab prüfen lassen: Auch dort überschreiten Referenzantworten 256 Tokens, während 1 024 Tokens alle Validierungsantworten und bis auf eine alle Trainingsantworten abdecken (Längenverteilungen in Anhang A.2).“
- *Erhalten:* alle Zahlen und die Begründung des Limits von 1 024. *Neu:* keine Zahl; die Werte 576 und 103 stehen im Anhang und können wahlweise genannt werden. *Länge:* plus rund 1 Zeile.

**K-05 (zu B-04) · nach 05 Z. 115**
- *Vorschlag:* „Auch diese Läufe unterliegen dem Limit von 256 Tokens; eine Nachinferenz mit 1 024 Tokens liegt für sie nicht vor (Abschnitt 5.2).“ Plus rund 1,5 Zeilen.

**K-06 (zu B-05) · nach 05 Z. 503**
- *Vorschlag:* „Beim 0.8B-Modell, das in beiden Messungen auf der H100 lief, liegt die Schätzung trotz des höheren Limits rund 8 % unter dem Wert des Hauptlaufs; Unterschiede dieser Größe entstehen also auch ohne Wechsel des GPU-Typs.“
- *Herkunft der Zahl:* 1 722 gegen 1 876 ms, beide in `summary.json`. Plus rund 2 Zeilen.

**K-07 (zu B-06) · 05 Z. 955–959**
- *Original:* „Bei Batchgröße 1 hängt die Latenz vermutlich von der Zahl sequenzieller Decoding-Schritte und von der Dequantisierung der NF4-Gewichte ab, die beide bei jeder Entität anfallen, unabhängig davon, wie kompakt sie kodiert ist; ihr jeweiliger Anteil wurde nicht getrennt gemessen.“
- *Vorschlag:* „Bei Batchgröße 1 wächst die Latenz mit der Zahl sequenzieller Decoding-Schritte, also mit der Länge der Antwort (Abschnitt 5.2). Ein kompakteres Format würde die Zahl der Schritte senken, nicht aber die Kosten je Schritt, zu denen vermutlich die Dequantisierung der NF4-Gewichte beiträgt; die Anteile wurden nicht getrennt gemessen.“
- *Länge:* etwa gleich.

**K-08 (zu B-07) · nach 04 Z. 999**
- *Vorschlag:* „Als günstigstes Verhältnis im Sinne von FF3 gilt damit die unter den jeweiligen Grenzen zulässige Konfiguration mit dem höchsten Mikro-F₁; eine Kennzahl, die Leistung und Ressourcen verrechnet, wird nicht gebildet.“ Plus rund 2 Zeilen.

Dazu die Einzelkorrekturen aus B-09 bis B-25 und Q-01 bis Q-16; sie ändern die Länge kaum.

### 6.2 Verlustfrei kürzbar

| Nr. | Fundstelle | Was entfällt | Warum ohne Verlust | Zeilen |
|---|---|---|---|---|
| K-09 | 05 Z. 691–694 | Beschreibung des Transfertests im Einleitungssatz von 5.6 | steht in 4.6 und in der Beschriftung von Tabelle 5.6 | 3 |
| K-10 | 05 Z. 623–624 | Definition der Zusätze | steht in 4.5.1 und in der Beschriftung von Tabelle 5.5 | 1,5 |
| K-11 | 05 Z. 1042–1043 | „Ein LLM wird erst ab mehr als dem Hundertfachen der Encoder-Latenz zulässig;“ | derselbe Satz steht 16 Zeilen vorher (Z. 1026–1027) | 1 |
| K-12 | 05 Z. 470–472, Z. 484–486 | Spannen aus Tabelle 5.4 | siehe Abschnitt 5.1 | 1,5 |
| K-13 | 05 Z. 299–300 | „Die Zähler sind absolute Häufigkeiten …“ | steht in 4.5.3 und in der Beschriftung von Tabelle 4.7 | 1,5 |
| K-14 | 05 Z. 855–857 | zweite Nennung von „kein isolierter Anpassungseffekt“ | steht in 5.1.1 (Z. 118–119); hier reicht „(Abschnitt 5.1.1)“ | 1 |
| K-15 | 05 Z. 55–59 | zwei Sätze zur Lesart der Intervalle | fast wörtlich in 4.5.1 (04 Z. 878–882) | 4,5 |
| K-16 | 05 Z. 415–416 | Parameterzahlen in der Beschriftung von Abbildung 5.2 | Tabelle 4.1, im selben Satz zitiert | 1 |
| K-17 | 04 Z. 703–705 | Rang und Skalierung im Fließtext | Tabelle 4.6 | 1 |
| K-18 | 04 Z. 1014–1015 | Satz zu „halluzinierten Kategorien“ | Begriff wird nicht wieder verwendet (B-21) | 1,5 |

K-15 und K-16 sind Geschmackssache: Beide helfen Lesern, die Kapitel 4 überspringen. Ich würde K-15 nur streichen, wenn der Platz gebraucht wird.

### 6.3 Notwendig ausführlich

- **4.3.3, Absatz zur Testnutzung (04 Z. 368–385):** trägt die Einstufung der Zero-Shot-Ergebnisse.
- **4.4.3, Checkpoint-Auswahl (04 Z. 628–651):** lang, aber jede Angabe wird in 5.1.1 oder 6.2 gebraucht.
- **4.5.1, Bootstrap (04 Z. 865–896):** vollständig und nachrechenbar; meine Gegenrechnung kam mit dieser Beschreibung aus.
- **4.5.5, Nachinferenz (04 Z. 1097–1136):** Die Begründung der Auswahlregel und ihre Grenze sind der Kern des Vorbehalts.
- **5.2, Absatz zur Nachinferenz (05 Z. 306–339):** dicht, aber ohne Wiederholung.
- **5.5, Verwechslungspaare (05 Z. 656–677):** viele Zahlen im Fließtext, die in keiner Tabelle stehen. Sie belegen, dass beide Modellklassen dieselben Fehler machen.
- **5.7.4, Schwellen (05 Z. 1030–1044):** die eigentliche Antwort auf FF3.

### 6.4 Nicht kürzen

- **Tabellen 4.3, 4.4 und 4.9.**
- **Hardware-Vorbehalte an den Stellen, an denen Latenzen gedeutet werden** (05 Z. 446–449, Z. 521–522, Z. 538–541, Beschriftung von Abbildung 5.3). Sie wiederholen sich, stehen aber jeweils neben der Zahl, die sie einschränken.
- **Kennzeichnungen als Vermutung** in 5.7 („denkbar, aber ungeprüft“, „bleibt eine Vermutung“).
- **Der Anhang.**

Mehr verlustfreie Kürzungen sehe ich nicht. Die Kapitel sind nach den beiden Kürzungsrunden bereits dicht; weitere Streichungen würden Befunde oder Einschränkungen kosten.

---

## 7 Prioritäten

### Zuerst zu klären (fachlich und methodisch)

1. **B-02:** Vier oder sieben Hauptvergleiche? Davon hängt die Kennzeichnung in Tabelle 5.9 ab.
2. **B-01:** Einstufung der 1 024-Token-Vergleiche in 5.7.1 und 5.7.4 nachziehen.
3. **B-03 und B-04:** Ausgabelimit in 4.4.3 nüchtern formulieren; Zero-Shot-Hinweis in 5.1.1.
4. **B-08:** Was zeigte die BF16-Messung, und wie viele Sätze umfasste sie? Das kannst nur du beantworten.
5. **B-09 und B-10:** die beiden Zahlenbezüge berichtigen.
6. **Branch `korrekturen-2026-11-25` zusammenführen**, damit der Rundungssatz in der Fassung steht.

### Wichtige sprachliche und argumentative Verbesserungen

7. **B-06:** Latenzsatz in 5.7.2.
8. **B-05:** Laufstreuung beim 0.8B-Modell.
9. **B-07:** Operationalisierung von FF3.
10. **B-11:** zweiter Wert der CrossNER-Analyse.
11. **Q-01, Q-02, Q-09, Q-12:** die vier Quellenstellen mit dem größten Abstand zwischen Formulierung und Quelle.

### Geringfügig und wahlweise

12. B-12 bis B-25, Q-03 bis Q-08, Q-10, Q-11, Q-13 bis Q-16.
13. Kürzungen K-09 bis K-18, soweit Platz gebraucht wird.
14. Beschriftung von Abbildung 5.3; Achsenbeschriftung in Abbildung 5.2.
15. Tabelle 5.9 nach vorn setzen oder FF3-Schwellen als Tabelle: beides nur mit deiner Zustimmung und nach einem Probesatz.

### Besser unverändert lassen

- Tabellen 4.3, 4.4, 4.9 und die Abschnitte 4.3.3, 4.5.1, 4.5.5.
- Die Trennung von Bericht (5.1 bis 5.6) und Deutung (5.7).
- Die Formulierungen zur Silberreferenz (04 Z. 249–256, 05 Z. 879–897) und zur Datenkontamination (05 Z. 898–904).
- Die Zahlen: Kein Wert in Tabellen oder Abbildungen muss geändert werden.

---

## Anhang zum Bericht: Gegenrechnungen

Skripte und Ausgaben liegen im Sitzungsordner `/private/tmp/claude-501/-Users-luajosman-bachelor-thesis-llm-vs-encoder-ner/bcca6b11-2224-4431-8727-15e3ad30a229/scratchpad/k45/` (`verify_mn.py`, `verify_ci.py`, `num04.txt`, `num05.txt`, Quellentexte unter `quellen_a/` und `quellen_b/`, gebautes PDF `main_b4898f1.pdf`).

| Größe | Arbeit | Gegenrechnung |
|---|---|---|
| Mikro-F₁ aller elf Konfigurationen (Tab. 5.1, 5.3) | wie berichtet | identisch auf zwei Nachkommastellen; Abweichung zu den Laufdateien 0,000000 |
| large minus 27B, 256 Tokens | +1,34 [0,99; 1,68] | +1,34 [0,99; 1,70] |
| large minus 27B, 1 024 Tokens | +0,18 [0,02; 0,33] | +0,18 [0,02; 0,34]; 98,75 %: [−0,02; 0,38] |
| 27B (1 024) minus base | +0,14 [−0,02; 0,30] | +0,14 [−0,02; 0,30] |
| abgeschnittene Antworten je Lauf | 62 bis 72 | 62 bis 72 |
| Referenzentitäten darin | 656 bis 670 | je Lauf 638 bis 687; Seed-Mittel 656, 664, 670 |
| Recall 27B auf den übrigen Sätzen | 96,3 | 96,24 bis 96,27 |
| Recall large auf den abgeschnittenen Sätzen | 96,6, 93,8, 95,1 | 96,6, 93,8, 95,1 |
| Faktoren 210, 335, 617 | wie berichtet | 210,2, 334,6, 616,9 |
| P95 zu Median, QLoRA | 3,4 | 3,37 bis 3,42 |
| Kontrollstichprobe | 8 988 von 9 000; 1, 7, 4 Abweichungen | identisch |
| Präfixabgleich | 594 von 604 | identisch (2, 4, 4 Abweichungen je Größe) |
| CrossNER, 1 024 Tokens | 2,13; ohne Treffer auf nicht abgebildeten Typen 0,54 | 2,13; nur exakte gestrichen 1,73; exakte und teilweise 0,54 |
| Latenz gegen erzeugte Tokens | nicht berichtet | r = 0,988, 0,995, 0,999 (Kontrollstichproben Seed 42) |
