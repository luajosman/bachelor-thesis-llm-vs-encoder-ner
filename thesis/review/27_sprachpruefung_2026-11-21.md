# Prüfbericht Sprache: Natürlichkeit, Verständlichkeit, Präzision (21.11.2026)

Auftrag: die gesamte Arbeit sprachlich darauf prüfen, ob der Text natürlich, verständlich und präzise ist. Es wurde nichts geändert; dieser Bericht ist die einzige neue Datei. Umgesetzt wird erst nach ausdrücklicher Freigabe einzelner Punkte.

Der Bericht beurteilt Formulierungen nach Verständlichkeit, Angemessenheit und Präzision. Er trifft keine Aussage darüber, wer oder was einen Satz geschrieben hat; das lässt sich aus einem Text nicht sicher feststellen.

## 1. Geprüfte Fassung und Grenzen der Prüfung

**Fassung.** Branch `lektorat-2026-11-25`, Commit `b9de051` (Textstand wie `36df1ee` vom 21.11., 17:14), `thesis/main.pdf` vom 21.11., 17:14, 112 Seiten. Die Arbeit war vollständig verfügbar.

**Vollständig gelesen** (LaTeX-Quellen, Satz für Satz): Abstract, Eigenständigkeitserklärung, Abkürzungsverzeichnis, Kapitel 1 bis 6, Anhang A.1 bis A.3, alle Tabellen- und Abbildungsbeschriftungen, die Fußnoten und Legenden der erzeugten Tabellen, das Deckblatt.

**Nicht sprachlich geprüft:**
- die Einträge des Literaturverzeichnisses (kein Fließtext; formale Prüfung in Bericht 25);
- die englischen Prompts in den Listings (Versuchsmaterial, kein Text der Arbeit);
- Zahlen und Quellenaussagen auf Richtigkeit (Berichte 24 bis 26); wo mir beim Lesen ein inhaltlicher Widerspruch auffiel, ist er als solcher gekennzeichnet;
- das Schriftbild im PDF. Das PDF diente nur der Seitenzuordnung (über SyncTeX aus den Quellzeilen); Trennungen und Umbrüche habe ich nicht Seite für Seite kontrolliert.

**Ergänzende Zählungen** (Skript `scripts/check/sentences.py` und Suchmuster über alle Kapitel): rund 1 180 Sätze, im Mittel 20 Wörter; 38 Sätze über 40 Wörter, nur einer über 50 (Anhang A.2, siehe AH-02). Die im Auftrag genannten Floskeln kommen praktisch nicht vor: „Zusammenfassend", „Im Rahmen" und „Es ist wichtig" null Mal, „Darüber hinaus" ein Mal, „zeigt sich" drei Mal, jeweils unauffällig.

**Seitenangaben** sind die gedruckten Seitenzahlen; Zeilenangaben beziehen sich auf die jeweilige `.tex`-Datei im genannten Commit. In den zitierten Originalen stehen für Literaturverweise die Autorennamen und für Querverweise die aufgelösten Nummern; die `\cite`- und `\ref`-Befehle bleiben in jedem Vorschlag unverändert.

**Wichtig für jede Umsetzung:** Das Abstract füllt seine Seite genau, Kapitel 3 und Kapitel 5 enden am Seitenende. Vorschläge, die einen Satz teilen, sind meist längenneutral; trotzdem muss nach jeder Änderung in diesen Teilen der Umbruch geprüft werden.

## 2. Übersicht

Gezählt sind Einzelbefunde und die Muster M-01 bis M-06; mehrere Niedrig-Befunde bündeln je Kapitel einige kleine Stellen.

| Priorität | Anzahl | Befunde |
|---|---|---|
| Hoch | 3 | K4-01 (Begründung der Nachinferenz schwer verständlich), K6-01 (Fazit reicht weiter als die Messung), AH-01 (Satz im Anhang widerspricht Tabelle A.1) |
| Mittel | 52 | unklare Bezüge, verdrehte oder überladene Sätze, zwei Grammatikfehler, Lehnübersetzungen an tragenden Stellen |
| Niedrig | 18 | optionale stilistische Verbesserungen, meist gebündelt je Kapitel |

„Sicher" heißt: Der Vorschlag ändert nur die Form. „Fachlich zu prüfen" heißt: Die gemeinte Bedeutung ist nicht eindeutig oder der Vorschlag spricht etwas aus, das bisher nur mitgemeint war; hier entscheidet der Verfasser.

## 3. Wiederkehrende Muster

**M-01 · Mittel · Objekt vor Verweis („… nennt Tabelle X")**
- Beobachtung: Rund 50 Sätze stellen das Objekt an den Anfang und Tabelle, Abbildung oder Abschnitt ans Ende („Die Erkennungsleistung aller acht Konfigurationen fasst Tabelle 5.1 zusammen"). Das vermeidet Satzanfänge mit „Tabelle" oder „Abbildung" und ist in kurzen Sätzen unauffällig („Den Aufbau zeigt Abbildung 2.2."). In längeren Sätzen muss man Subjekt und Objekt erst sortieren, und die Häufung wirkt schematisch.
- Am deutlichsten:
  - 5.2, Z. 238–239: „Die Kennzahlen zur Formtreue aus Tabelle 4.7 berichtet Tabelle 5.2 für die sechs LLM-Konfigurationen." Vorschlag: „Für die sechs LLM-Konfigurationen stehen die Kennzahlen zur Formtreue (Definition in Tabelle 4.7) in Tabelle 5.2."
  - A.3, Z. 624–626: „… um Precision und Recall ergänzt die Typwerte Tabelle A.10." Vorschlag: „Precision und Recall zu den Typwerten stehen in Tabelle A.10."
  - 4.4.1, Z. 504–506: „Konfigurationsdatei, Commit und Ausgabeverzeichnis jedes Laufs nennt Tabelle A.1, die Revisionen von Modellen und Daten Tabelle A.3." Vorschlag: „Für jeden Lauf stehen Konfigurationsdatei, Commit und Ausgabeverzeichnis in Tabelle A.1, die Revisionen von Modellen und Daten in Tabelle A.3."
  - 5.3, Z. 361–364: „Die Erkennungsleistung über die Größenklassen aus Abschnitt 4.2.3 stellt Abbildung 5.2 getrennt nach Familie und Regime dar; die Werte mit Standardabweichungen nennt Tabelle A.11 …" Vorschlag: „In Abbildung 5.2 ist die Erkennungsleistung über die Größenklassen aus Abschnitt 4.2.3 getrennt nach Familie und Regime dargestellt; die Werte mit Standardabweichungen stehen in Tabelle A.11 …"
  - 4.7, Z. 1264–1266: zwei solche Sätze an einem Semikolon.
- Empfehlung: Nicht alle Stellen ändern. Die kurzen bleiben; bei den genannten und bei Sätzen mit zwei Verweisen auf „In Tabelle X stehen …", „… (Tabelle X)" oder ein Passiv ausweichen. Die Regel, keinen Satz mit „Tabelle" zu beginnen, bleibt dabei eingehalten.
- Sicherheit: sicher.

**M-02 · Mittel · Ketten aus Hauptsätzen an Semikolons**
- Beobachtung: 426 Semikolons im Fließtext, also in etwa jedem dritten Satz; in Kapitel 5 sind es 138 auf rund 250 Sätze. 25 Sätze enthalten zwei oder mehr, davon elf in Kapitel 5. Oft trägt das Semikolon sinnvoll eine unmittelbare Einschränkung („…; getrennt gemessen wurde er nicht."). Wo dagegen eine neue Aussage beginnt, entsteht ein Satz mit zwei oder drei Gedanken, dessen Kernaussage in der Mitte liegt.
- Einzelstellen: V-01, K1-04, K4-06, K5-11, dazu 5.1.2 Z. 166–172, 5.6 Z. 771–774, 5.7.3 Z. 1018–1021.
- Empfehlung: Semikolon behalten, wenn der zweite Teil den ersten direkt einschränkt oder begründet; Punkt setzen, wenn ein neuer Befund folgt. Das ist die Stelle, an der der Text am leichtesten an Lesbarkeit gewinnt, ohne ein Wort zu verlieren.
- Sicherheit: sicher.

**M-03 · Niedrig · Wiederholte Vorbehalte**
- Beobachtung: Wendungen wie „wurde nicht geprüft/untersucht/gemessen/bestimmt" stehen rund 33 Mal im Text, „isoliert" 15 Mal. Die meisten sind notwendige Einschränkungen und sollen bleiben. Ein Vorbehalt kehrt aber fast wörtlich wieder: Dass kein isolierter Effekt von Anpassung oder Architektur bestimmt wird, steht im Abstract, in 1.3, 4.4.3, 5.1.1, zweimal in 5.7.1, zweimal in 6.1 und in 6.2. Ähnlich „dienen … nur der Einordnung" (dreimal in 5.7.1 und 5.7.2).
- Empfehlung: Den Vorbehalt dort lassen, wo er zum Lesen einer Zahl nötig ist, und in 6.2. Kandidaten für eine Zusammenlegung sind die beiden Sätze in 6.1 (Z. 40–41 und 52–53), weil 6.2 unmittelbar folgt. Das ist eine Abwägung zwischen Vorsicht und Wiederholung, keine Fehlerkorrektur.
- Sicherheit: Entscheidung des Verfassers; keine Einschränkung darf ersatzlos entfallen.

**M-04 · Mittel · Lehnübersetzungen und Jargon an tragenden Stellen**
- „ist auf … bedingt" (4.5.1, 6.2), „unter dem Tokenizer" und „unter Greedy-Decoding" (4.4.3, 4.5.4, A.2.3), „adressiert" (3.6), „für die Modelle ungesehen" (3.4), „nicht limitierend" (A.2.2), „mit übermäßiger Sicherheit" (3.3.2), „verrauschte Typen" (5.1.1, 5.7.1, 6.3).
- Etablierte englische Fachbegriffe (Fine-Tuning, Prompt, Greedy-Decoding, Constrained Decoding, Recall) sind davon nicht betroffen und sollen bleiben. Gemeint sind nur wörtlich übertragene Wendungen, für die es eine gebräuchliche deutsche Form gibt. Einzelvorschläge stehen bei K3-06, K3-13, K4-01, K4-03, K4-13, K5-13, K6-07, AH-03.

**M-05 · Niedrig · Wechselnde Bezeichnungen für dieselbe Sache**
- „GPU-Spitzenspeicher während der Inferenz", „GPU-Spitzenspeicher der Inferenz", „Inferenzspeicher", „maximal allokierter GPU-Speicher", „GPU-Inferenzspeicher".
- „in derselben Pipeline" und „gemeinsame Pipeline" (1.2, 3.2, 3.6, 6, 6.1) neben „die beiden Pipelines" (Kapitel 4). Hier ist die Bedeutung nicht nur eine Stilfrage, siehe K1-02.
- „nicht parsebare Sätze" und „abgeschnittene Sätze" für Sätze, deren Antwort nicht parsebar oder abgeschnitten ist (5.2, 5.5).
- „F₁-Score" (Kapitel 1 und 2), „F₁-Wert" (Kapitel 4), „F₁" (Kapitel 5 und 6). Unkritisch; eine Form je Kapitel genügt.
- Empfehlung: je Größe eine Hauptbezeichnung festlegen und beim ersten Vorkommen in Kapitel 4 definieren.

**M-06 · Mittel · Zeitformen im Methodenkapitel:** siehe K4-08.

## Vorspann

**V-01 · Mittel · Abstract, S. ii · Z. 37–40**
- Original: „Ihre Prompt- und Decoding-Wahl wurde durch Testbeobachtungen beeinflusst; da sich die Regime über das Training hinaus in Prompt, Decoding und numerischer Präzision unterscheiden, ist kein isolierter Fine-Tuning-Effekt bestimmbar."
- Befund: Zwei verschiedene Vorbehalte hängen an einem Semikolon. „Ihre" bezieht sich auf die Zero-Shot-Konfigurationen, die zweite Hälfte betrifft beide Regime. „Prompt- und Decoding-Wahl wurde … beeinflusst" ist eine sperrige Nominalfügung.
- Vorschlag: „Prompt und Decoding dieser Läufe wurden nach Testbeobachtungen gewählt. Da sich die Regime zudem über das Training hinaus in Prompt, Decoding und numerischer Präzision unterscheiden, ist kein isolierter Fine-Tuning-Effekt bestimmbar."
- Erhalten: beide Vorbehalte (Testbezug der Zero-Shot-Einstellungen; kein isolierter Effekt).
- Sicherheit: sicher. Das Abstract füllt die Seite genau; der Vorschlag ist etwa gleich lang, der Umbruch muss trotzdem geprüft werden.

**V-02 · Niedrig · Abstract, S. ii · Z. 14–16**
- Original: „… Trainingsdauer, Inferenzlatenz und maximal allokierter GPU-Inferenzspeicher"
- Befund: Für dieselbe Größe stehen in der Arbeit mehrere Bezeichnungen („GPU-Spitzenspeicher während der Inferenz" in 1.2, „Inferenzspeicher" im Abstract zwei Absätze später, „maximal allokierter GPU-Speicher" in 4.5.2). Kein Fehler, aber die längste Variante steht hier in einer Aufzählung ohne Artikel und liest sich hart.
- Vorschlag: „… Trainingsdauer, Inferenzlatenz und GPU-Spitzenspeicher der Inferenz"
- Erhalten: gemeint ist der von PyTorch maximal allokierte Speicher während der Testinferenz (Definition in 4.5.2 bleibt).
- Sicherheit: sicher; optional.

**V-03 · Hinweis, kein Sprachbefund · Eigenständigkeitserklärung, S. i · Z. 24–26**
- Original: „zuletzt Claude Fable 5.1, Claude Opus 5.5 und Claude Sonnet 5.5"
- Befund: Die Sitzungen vom 21.11. liefen mit Claude Fable 5.2. Ob die Aufzählung vollständig sein soll, entscheidet der Verfasser (in `STATUS.md` bereits als offen geführt).
- Sicherheit: fachlich zu prüfen.

Keine Änderung nötig: übriger Abstract (klar, konkrete Zahlen, Einschränkungen genannt), Abkürzungsverzeichnis, formelhafte Teile der Erklärung.

## Kapitel 1: Einleitung

**K1-01 · Mittel · 1.2, S. 2 · Z. 105–110**
- Original: „Die direkten Vergleiche erfassen neben der Erkennungsleistung nur Teile des Ressourcenbedarfs (Inferenzzeit ohne Speicher oder Trainingskosten ohne Inferenz) und betrachten LLMs nur einer Größenklasse; Größenstaffelungen gibt es nur innerhalb einzelner Modellfamilien, ohne gemeinsamen Encoder-Vergleich."
- Befund: Die Klammer ist so verknappt, dass man sie erst beim zweiten Lesen versteht („Inferenzzeit ohne Speicher" liest sich zunächst wie eine Messgröße). Dazu dreimal „nur" in einem Satz.
- Vorschlag: „Die direkten Vergleiche erfassen neben der Erkennungsleistung lediglich Teile des Ressourcenbedarfs, entweder die Inferenzzeit ohne den Speicher oder die Trainingskosten ohne die Inferenz, und betrachten LLMs einer einzigen Größenklasse. Größenstaffelungen gibt es nur innerhalb einzelner Modellfamilien und ohne gemeinsamen Encoder-Vergleich."
- Erhalten: beide Teilbefunde (unvollständige Ressourcenerfassung; eine Größenklasse) und die Einschränkung zu den Größenstaffelungen.
- Sicherheit: sicher.

**K1-02 · Mittel · 1.2, S. 3 · Z. 147–149**
- Original: „Der Beitrag der Arbeit liegt in der gemeinsamen Untersuchung dieser Fragen anhand von acht Konfigurationen mit rund 184 Mio. bis 27 Mrd. Parametern und einer in derselben Pipeline trainierten Encoder-Referenz."
- Befund: Unklarer Bezug an zwei Stellen. „Acht Konfigurationen … und einer … Encoder-Referenz" klingt, als käme die Referenz zu den acht hinzu; die beiden Encoder gehören aber zu den acht. „In derselben Pipeline" ist mehrdeutig, weil Kapitel 4 von zwei Pipelines (Encoder und LLM) spricht.
- Vorschlag (eine mögliche Lesart): „Der Beitrag der Arbeit liegt darin, diese Fragen gemeinsam an acht Konfigurationen mit rund 184 Mio. bis 27 Mrd. Parametern zu untersuchen; die Encoder-Referenz ist dabei selbst trainiert und auf denselben Splits mit derselben Metrik ausgewertet."
- Erhalten: acht Konfigurationen, Parameterspanne, eigene Encoder-Referenz.
- Sicherheit: fachlich zu prüfen. Was „in derselben Pipeline" genau heißen soll, kann nur der Verfasser festlegen.

**K1-03 · Mittel · 1.2, S. 3 · Z. 135–138**
- Original: „Untersucht wird, wie sich Erkennungsleistung, Trainingsdauer, Inferenzlatenz und GPU-Spitzenspeicher während der Inferenz innerhalb der betrachteten Konfigurationen beider Modellfamilien mit der Modellgröße verändern."
- Befund: Vier Größen, dann drei Präpositionalgruppen hintereinander („während … innerhalb … beider … mit"); der eigentliche Bezug (je Familie über die Größen) geht darin unter.
- Vorschlag: „Sie fragt, wie sich Erkennungsleistung, Trainingsdauer, Inferenzlatenz und GPU-Spitzenspeicher der Inferenz mit der Modellgröße verändern, jeweils innerhalb der betrachteten Konfigurationen einer Modellfamilie."
- Erhalten: alle vier Größen; Beschränkung auf die betrachteten Konfigurationen und auf den Vergleich innerhalb der Familien.
- Sicherheit: sicher.

**K1-04 · Niedrig · 1.1, S. 1 · Z. 37–40**
- Original: „Encoder-Modelle wie BERT oder DeBERTa berechnen für jedes Token eine kontextabhängige Repräsentation; ein darauf trainierter Klassifikator weist jedem Token ein Label zu; aus der Labelfolge ergeben sich die Entitäten (Sequenzlabeling)."
- Befund: Drei Hauptsätze an zwei Semikolons. Verständlich, aber als Kette gesetzt. Das Muster „Hauptsatz; Hauptsatz" kehrt in der ganzen Arbeit sehr häufig wieder (siehe Gesamteinschätzung).
- Vorschlag: „… eine kontextabhängige Repräsentation. Ein darauf trainierter Klassifikator weist jedem Token ein Label zu, und aus der Labelfolge ergeben sich die Entitäten (Sequenzlabeling)."
- Erhalten: die drei Schritte und der Begriff Sequenzlabeling.
- Sicherheit: sicher; optional.

**K1-05 · Niedrig · 1.1, S. 2 · Z. 68–70**
- Original: „… wann der zusätzliche Aufwand eines LLM durch eine bessere Erkennungsleistung gerechtfertigt ist und wann ein kompakter Encoder effizienter ist."
- Befund: Die zweite Hälfte ist unscharf: Im Ressourcenbedarf ist der Encoder nach dem vorangehenden Absatz immer sparsamer. Gemeint ist vermutlich, wann er genügt.
- Vorschlag: „… gerechtfertigt ist und wann ein kompakter Encoder die bessere Wahl bleibt."
- Erhalten: Abwägung von Aufwand und Erkennungsleistung als Leitfrage.
- Sicherheit: fachlich zu prüfen (Lesart von „effizienter").

**K1-06 · Niedrig · 1.2, S. 3 · Z. 128–131**
- Original: „Die erste Forschungsfrage (FF1) untersucht, wie sich …"
- Befund: Eine Frage untersucht nichts; FF3 ist mit „fragt" passender formuliert.
- Vorschlag: „Die erste Forschungsfrage (FF1) lautet, wie sich … unterscheiden."
- Erhalten: Wortlaut der Frage.
- Sicherheit: sicher; optional.

**K1-07 · Niedrig · 1.3, S. 5 · Z. 238–240**
- Original: „Interpretierbarkeit, Fairness und Robustheit gegenüber adversarialen Eingaben sind ebenso wenig Gegenstand der Arbeit wie die Erhebung und Annotation eines neuen Datensatzes."
- Befund: Generische Ausschlussliste. Nichts im Text davor lässt erwarten, dass diese Themen behandelt würden; die übrigen Abgrenzungen in 1.3 sind dagegen konkret auf die Arbeit bezogen.
- Vorschlag: Satz streichen oder auf den zweiten Teil beschränken („Ein neuer Datensatz wird nicht erhoben oder annotiert.").
- Erhalten: bei Streichung geht keine Aussage über die durchgeführte Arbeit verloren.
- Sicherheit: sicher; optional. Kein Fehler.

Keine Änderung nötig: Einstieg in 1.1 (konkret statt allgemein), Zahlenvergleiche aus der Literatur in 1.2, die Abgrenzungen zu Regimen, Sprache und Metrik in 1.3, Aufbau in 1.4.

## Kapitel 2: Grundlagen

**K2-01 · Mittel · 2.3, S. 9 · Z. 205–207**
- Original: „Ein Decoder verarbeitet die Sequenz kausal und gibt in der hier verwendeten Pipeline die Entitäten als Text aus; da der ganze Eingabesatz im Prompt liegt, kann er den rechten Satzkontext nutzen."
- Befund: „Den rechten Satzkontext" ist doppeldeutig (rechts von der Entität oder der richtige Kontext). Außerdem fehlt das „dennoch", das den Gegensatz zu „kausal" trägt.
- Vorschlag: „… als Text aus. Da der ganze Eingabesatz im Prompt liegt, kann er dabei dennoch auch den Kontext rechts von einer Entität nutzen."
- Erhalten: kausale Verarbeitung; der Satz steht vollständig im Prompt; daraus folgt der Zugriff auf den Kontext nach der Entität.
- Sicherheit: sicher.

**K2-02 · Mittel · 2.5, S. 13 · Z. 365–369**
- Original: „Viele LLMs werden nur mit frei herunterladbaren Gewichten und einer Lizenz veröffentlicht; weil die Open-Source-AI-Definition zusätzlich Angaben zu den Trainingsdaten und den vollständigen Code verlangt, gelten sie als Modelle mit offenen Gewichten (open weights)."
- Befund: Die Begründung ist nur verständlich, wenn man den ausgelassenen Schritt ergänzt (sie gelten deshalb nicht als Open Source). So wirkt das „weil" unlogisch.
- Vorschlag: „… veröffentlicht. Die Open-Source-AI-Definition verlangt zusätzlich Angaben zu den Trainingsdaten und den vollständigen Code; solche Modelle gelten deshalb nicht als Open Source, sondern als Modelle mit offenen Gewichten (open weights)."
- Erhalten: Anforderungen der Definition, Begriff „offene Gewichte", Quelle.
- Sicherheit: fachlich kurz zu prüfen, weil der Vorschlag den bisher nur mitgemeinten Schritt ausspricht.

**K2-03 · Mittel · 2.8, S. 17 · Z. 586–590**
- Original: „Ein F₁-Score misst damit die Übereinstimmung mit der Referenz und nur mittelbar die Richtigkeit; bei einer automatisch erzeugten Referenz (Silberreferenz) wie der von MultiNERD (Abschnitt 3.1) ist diese Lesart zu berücksichtigen."
- Befund: „Ist diese Lesart zu berücksichtigen" sagt nicht, was daraus folgt. Die Folgerung steht klarer in 1.3. Der Begriff „positive spezifische Übereinstimmung" im Satz davor bleibt unerklärt, ist aber als Fachbegriff der Quelle vertretbar.
- Vorschlag: „… und nur mittelbar die Richtigkeit. Bei einer automatisch erzeugten Referenz (Silberreferenz) wie der von MultiNERD (Abschnitt 3.1) geben die Werte also an, wie gut ein Modell diese Referenz trifft, nicht wie richtig es gegenüber einer geprüften Annotation liegt."
- Erhalten: Unterscheidung Übereinstimmung/Richtigkeit, Begriff Silberreferenz, Verweis.
- Sicherheit: sicher (dieselbe Aussage steht in 1.3).

**K2-04 · Mittel · 2.8, S. 17 · Z. 592–596**
- Original: „Neben der Erkennungsleistung erfasst diese Arbeit den Ressourcenbedarf; der Anteil trainierbarer Parameter quantifiziert dabei nur die Parameterreduktion durch parametereffiziente Verfahren, während Inferenzspeicher, Latenz und Trainingsdauer gesondert gemessen werden."
- Befund: Die Kernaussage (der Anteil trainierbarer Parameter ist kein Maß für den Ressourcenbedarf) ist nur indirekt über „dabei nur" zu erschließen. „Parameterreduktion durch parametereffiziente Verfahren" ist eine Substantivkette.
- Vorschlag: „Neben der Erkennungsleistung erfasst diese Arbeit den Ressourcenbedarf. Der Anteil trainierbarer Parameter gibt dabei nur an, wie stark ein parametereffizientes Verfahren die Zahl der trainierten Parameter senkt; Inferenzspeicher, Latenz und Trainingsdauer werden deshalb gesondert gemessen."
- Erhalten: alle vier Größen und der Verweis auf 4.5.2.
- Sicherheit: sicher.

**K2-05 · Mittel · 2.5, S. 12 · Z. 328–330**
- Original: „Greedy-Decoding wählt stets das wahrscheinlichste Token und ist als Auswahlregel deterministisch; Gleitkommaarithmetik und Batching können hardwareseitig Abweichungen verursachen."
- Befund: „Hardwareseitig" passt nicht zu Batching, und der Gegensatz zur ersten Satzhälfte ist nicht markiert. Der Satz ist wichtig, weil 5.2 sich darauf beruft.
- Vorschlag: „… deterministisch. Trotzdem können die Ausgaben abweichen, weil Gleitkommaarithmetik und Batching je nach Hardware zu leicht anderen Zahlenwerten führen."
- Erhalten: Regel deterministisch, Ausgabe nicht zwingend; beide genannten Ursachen.
- Sicherheit: fachlich zu prüfen (der Satz hat keine Quelle; die Zuordnung von Ursache und Hardware sollte der Verfasser bestätigen).

**K2-06 · Niedrig · 2.3, S. 9 · Z. 195–200**
- Original: „… eignen sie sich zur Texterzeugung Token für Token [Quelle]; Encoder-Decoder-Modelle entsprechen dem Original."
- Befund: Die dritte der angekündigten „drei Modellfamilien" hängt als Nachsatz hinter der Quellenangabe der zweiten.
- Vorschlag: eigener Satz: „Encoder-Decoder-Modelle behalten den Aufbau des Originals bei."
- Sicherheit: sicher; optional.

**K2-07 · Niedrig · 2.8, S. 16 · Z. 553–555**
- Original: „In der Praxis werden Metriken auf Entitätsebene aus Tagfolgen berechnet, und die Bibliothek bestimmt dabei, wie Spans aus den Tags gewonnen werden."
- Befund: „Die Bibliothek" steht mit bestimmtem Artikel, bevor eine Bibliothek eingeführt ist.
- Vorschlag: „… aus Tagfolgen berechnet; wie dabei Spans aus den Tags gewonnen werden, hängt von der verwendeten Bibliothek ab."
- Sicherheit: sicher; optional.

**K2-08 · Niedrig · 2.5, S. 13 · Z. 370–376**
- Original: „… folgten Modelle ohne Mixture-of-Experts-Komponenten, darunter die hier verwendeten …"
- Befund: „Mixture-of-Experts" wird nirgends erklärt, und es bleibt offen, warum die Angabe hier wichtig ist.
- Vorschlag: entweder in einem Halbsatz erklären oder die Angabe streichen („… folgten weitere Größen, darunter die hier verwendeten …").
- Sicherheit: fachlich zu prüfen (ob die Angabe an anderer Stelle gebraucht wird).

Keine Änderung nötig: 2.1 und 2.2 (klare Definitionen, gutes Beispiel), die Beschreibung von Self-Attention, DeBERTa und QLoRA, 2.7 (zwei Grenzen der Skalierungsgesetze, knapp und präzise), die Herleitung der strikten Zählweise in 2.8.

## Kapitel 3: Verwandte Arbeiten

**K3-01 · Mittel · 3.3.1, S. 23 · Z. 306–310**
- Original: „Nach Lu et al. entsteht die Latenz generativer NER vor allem durch das sequenzielle Decoding langer Ausgaben, die Inline-Formate durch die Wiederholung des Eingabetexts weiter verlängern; ihr paralleles Decoding beschleunigt die Inferenz um den Faktor 1,76 bis 10,22 bei unveränderter Erkennungsleistung."
- Befund: Im Relativsatz ist nicht sofort klar, was Subjekt und was Objekt ist („die Inline-Formate … verlängern"). „Ihr" meint die Autoren, steht aber direkt hinter „Inline-Formate" und „Ausgaben".
- Vorschlag: „… durch das sequenzielle Decoding langer Ausgaben; Inline-Formate verlängern die Ausgaben zusätzlich, weil sie den Eingabetext wiederholen. Das von den Autoren vorgeschlagene parallele Decoding beschleunigt die Inferenz um den Faktor 1,76 bis 10,22 bei unveränderter Erkennungsleistung."
- Erhalten: Ursache der Latenz, Rolle der Inline-Formate, beide Faktoren, „unveränderte Erkennungsleistung".
- Sicherheit: sicher.

**K3-02 · Mittel · 3.3.1, S. 22 · Z. 273–278**
- Original: „Im Vergleich mit vier per Prompting genutzten offenen Modellen zwischen 8 und 70 Mrd. Parametern, unter anderem auf dem mehrsprachigen UNER-Benchmark (dem Datensatz Universal NER von Mayhew et al., nicht dem gleichnamigen Modell von Zhou et al.), erweist sich Tagging als die robusteste, aber auch tokenintensivste Strategie."
- Befund: Die Hauptaussage kommt nach 35 Wörtern Vorfeld und einer Klammer mit Namensklärung.
- Vorschlag: „Die Autoren vergleichen die Strategien an vier per Prompting genutzten offenen Modellen zwischen 8 und 70 Mrd. Parametern, unter anderem auf dem mehrsprachigen UNER-Benchmark (dem Datensatz Universal NER von Mayhew et al., nicht dem gleichnamigen Modell von Zhou et al.). Tagging erweist sich dabei als die robusteste, aber auch tokenintensivste Strategie."
- Erhalten: Modellzahl und -größe, Benchmark samt Abgrenzung, Befund.
- Sicherheit: sicher.

**K3-03 · Mittel · 3.3.1, S. 23 · Z. 321–329**
- Original: „Erwartet wird erstens, dass die explizit in der Eingabe stehenden Indizes den … Einbruch vermeiden. Zweitens hängt die Ausgabelänge von Zahl und Spanlänge der Entitäten sowie der Serialisierung ab; … Drittens erlauben die Indizes eine eindeutige Rückabbildung …"
- Befund: „Erwartet wird erstens" kündigt drei Erwartungen an; der zweite und dritte Punkt sind aber Eigenschaften des Formats, keine Erwartungen. Die Aufzählung hat damit keinen gemeinsamen Nenner.
- Vorschlag: Einen Leitsatz voranstellen und den ersten Punkt als Erwartung kennzeichnen: „Für dieses Format sprechen drei Gründe. Erstens ist zu erwarten, dass die explizit in der Eingabe stehenden Indizes den … Einbruch vermeiden. Zweitens …"
- Erhalten: Unterschied zwischen Erwartung (Punkt 1) und Eigenschaft (Punkte 2 und 3); alle Quellen.
- Sicherheit: sicher.

**K3-04 · Mittel · 3.3.4, S. 25 · Z. 434–436**
- Original: „Ihre Stärke zeigen sie dort, wo wenige Beispiele vorliegen oder Domänen wechseln, bei CodeIE auch mit Prompts in Form von Python-Code"
- Befund: CodeIE kommt im Fließtext hier zum ersten Mal vor (sonst nur in Tabelle 3.2). Der Einschub setzt eine Einführung voraus, die es nicht mehr gibt.
- Vorschlag: entweder kurz einführen („bei CodeIE, das die Aufgabe als Python-Code formuliert,") oder den Einschub streichen und die Quelle in der Sammelangabe belassen.
- Erhalten: die Quellenangabe `li2023codeie`.
- Sicherheit: fachlich zu prüfen (Wahl zwischen beiden Varianten).

**K3-05 · Mittel · 3.4, S. 26 · Z. 524–525**
- Original: „Einen konstruktiven Umgang mit genau diesem Unterschied zeigen Dukić und Šnajder."
- Befund: „Genau diesem Unterschied" hat kein klares Bezugswort. Gemeint ist vermutlich die im vorigen Absatz genannte Schwäche (kausale Decoder gegenüber tokenbasierter NER); dazwischen steht aber der Satz zu LoRA und vollständigem Fine-Tuning.
- Vorschlag: „Einen konstruktiven Umgang mit der kausalen Maske der Decoder zeigen …"
- Erhalten: Anschluss an die von Lu et al. genannten Gründe.
- Sicherheit: fachlich zu prüfen (welcher Unterschied gemeint ist).

**K3-06 · Mittel · 3.4, S. 26 · Z. 514–517**
- Original: „Als Gründe für die Schwierigkeiten nennen sie die Abhängigkeit tokenbasierter NER von Einzeltoken-Repräsentationen, das darauf nicht ausgerichtete Vortrainingsziel der Decoder und die für die Modelle ungesehenen IOB2-Symbole."
- Befund: Drei schwere Nominalgruppen in Folge; „für die Modelle ungesehen" ist eine Lehnübersetzung von „unseen".
- Vorschlag: „Als Gründe nennen sie drei Punkte: Tokenbasierte NER ist auf Repräsentationen einzelner Tokens angewiesen, das Vortrainingsziel der Decoder ist darauf nicht ausgerichtet, und die IOB2-Symbole kommen im Vortraining der Modelle nicht vor."
- Erhalten: alle drei Gründe als Aussage der zitierten Autoren.
- Sicherheit: fachlich kurz zu prüfen (Formulierung des dritten Grundes gegen die Quelle).

**K3-07 · Mittel · 3.6, S. 30 · Z. 772–774**
- Original: „Aus der Literatur folgen drei Erwartungen als Lesehilfe ohne Hypothesentest, die Kapitel 6 je Forschungsfrage mit den Befunden abgleicht."
- Befund: Drei Aussagen in einem Satz, „als Lesehilfe ohne Hypothesentest" ist stark verkürzt, und im Relativsatz muss man Subjekt und Objekt erst sortieren.
- Vorschlag: „Aus der Literatur ergeben sich drei Erwartungen. Sie dienen als Lesehilfe und werden nicht als Hypothesen getestet; Kapitel 6 gleicht sie je Forschungsfrage mit den Befunden ab."
- Erhalten: Status der Erwartungen (kein Hypothesentest), Verweis auf Kapitel 6.
- Sicherheit: sicher.

**K3-08 · Mittel · 3.6, S. 29 · Z. 736–738**
- Original: „Leistung und Ressourcen werden in den direkten Vergleichen nur ausschnittsweise erfasst: Inferenzzeiten ohne Speicher, Trainingskosten ohne Inferenz oder gar keine Messung."
- Befund: Unpräzise gegenüber 1.2: Lückenhaft ist nach der Aufzählung nur die Ressourcenerfassung, nicht die Leistungsmessung.
- Vorschlag: „Der Ressourcenbedarf wird in den direkten Vergleichen nur ausschnittsweise erfasst: …"
- Erhalten: die drei Fälle der Aufzählung.
- Sicherheit: sicher.

**K3-09 · Mittel · 3.3.3, S. 24 · Z. 411–418**
- Original: „GNER … trainiert Flan-T5-Modelle bis 11 Mrd. Parameter und LLaMA-7B auf Pile-NER darauf, jedes Eingabewort mit seinem Label zu wiederholen …" und „gemessen werden Inferenzdurchsatz und GPU-Stunden des Trainings, aber kein Encoder in derselben Pipeline."
- Befund: „Auf Pile-NER darauf" stößt hart zusammen. Im zweiten Satz passt „kein Encoder" nicht zu „gemessen werden".
- Vorschlag: „GNER … trainiert Flan-T5-Modelle bis 11 Mrd. Parameter und LLaMA-7B auf Pile-NER. Die Modelle lernen, jedes Eingabewort mit seinem Label zu wiederholen, also auch Nicht-Entitäten als negative Instanzen mitzuführen." und „… ; die Autoren messen Inferenzdurchsatz und GPU-Stunden des Trainings, führen aber keinen Encoder in derselben Pipeline mit."
- Erhalten: Modelle, Trainingsdaten, Formatidee, gemessene Größen.
- Sicherheit: sicher.

**K3-10 · Mittel · 3.6, S. 28 · Z. 709–712**
- Original: „… ein Testumfang von 32 908 Beispielen gegenüber, etwa das Doppelte des englischen Testsplits; ihr Verhältnis zu den offiziellen Splits bleibt offen."
- Befund: „Ihr" hat kein eindeutiges Bezugswort (Beispiele, Angabe, Autoren).
- Vorschlag: „…; wie sich diese Testdaten zu den offiziellen Splits verhalten, bleibt offen."
- Sicherheit: sicher.

**K3-11 · Mittel (Grammatik) · 3.3.1, S. 22 · Z. 299**
- Original: „… werten für NER unter anderem OntoNotes 5.0 mit strikter Mikro-F₁ aus."
- Befund: Genus weicht vom übrigen Text ab („der Mikro-F₁", „einen Mikro-F₁").
- Vorschlag: „mit striktem Mikro-F₁"
- Sicherheit: sicher.

**K3-12 · Niedrig · Kapiteleinleitung, S. 18 · Z. 23–29**
- Original: „Die Auswahl beruht auf einer narrativen Recherche in Google Scholar, arXiv und der ACL Anthology (2019 bis 2025, ergänzt um … und eine gezielte Nachsuche vom 8. November 2026) mit Stichwörtern wie …"
- Befund: Die lange Klammer trennt „Recherche" von „mit Stichwörtern".
- Vorschlag: Zeitraum und Nachsuche in einen eigenen Satz nach der Stichwortliste stellen.
- Sicherheit: sicher; optional.

**K3-13 · Niedrig · mehrere Stellen**
- „Einschränkend haben Semin et al. …" (Z. 330): „einschränkend" als Satzadverb ist unüblich; „Allerdings haben …" sagt dasselbe.
- „Adapterverfahren werden nicht erwähnt" (Z. 379): Es bleibt offen, wer sie nicht erwähnt und warum das hier steht; klarer wäre „ob dabei Adapterverfahren eingesetzt wurden, geht aus der Arbeit nicht hervor" (fachlich zu prüfen).
- „Sie adressiert damit FF2" (Z. 754): Lehnübersetzung von „to address"; „Damit bearbeitet sie FF2" ist gebräuchlicher.
- „… die den Bereich der direkten Vergleiche nach unten und oben verlassen" (Z. 590–591): gemeint ist „über den Größenbereich der direkten Vergleiche nach unten und oben hinausgehen".
- „… belegen deshalb nur, dass …, nicht das … erwartbare Niveau" (Z. 451–453): Nebensatz und Substantiv sind ungleich gebaut; glatter: „…; welches Niveau auf dem reduzierten Typinventar zu erwarten ist, zeigen sie nicht."
- „mit übermäßiger Sicherheit" (Z. 347): Lehnübersetzung von „overconfident"; möglich wäre „auch dann Entitäten markiert, wenn die Eingabe keine des abgefragten Typs enthält".
- Alle sicher und optional, außer dem zweiten Punkt.

Keine Änderung nötig: Beschreibung der Datensätze in 3.1, die Zusammenfassungen der Einzelstudien mit Zahlen und Tabellenangaben, die Zwischenfazits am Ende von 3.2 und 3.4, die beiden Lückenabsätze in 3.6 in ihrer Substanz, die Formulierung der Erwartungen E1 bis E3.

## Kapitel 4: Methodik

**K4-01 · Hoch · 4.5.4, S. 49 · Z. 1124–1133**
- Original: „Die Auswahlregel ‚nicht parsebar' ist ein Stellvertreter für ‚abgeschnitten': Jede nicht parsebare Antwort der QLoRA-Läufe endet ohne schließende Klammer, und eine vollständige, aber aus anderem Grund nicht parsebare Antwort würde unter Greedy-Decoding identisch reproduziert; abgeschnittene, aber zufällig parsebare Antworten lassen sich in den gespeicherten Daten dagegen nicht erkennen. Bei identischen Logits und Greedy-Decoding bliebe eine Antwort, die vor dem Limit mit dem Ende-Token abschließt, unverändert; da die ursprünglichen Läufe den Stoppgrund nicht protokollierten, ist diese Bedingung für die übernommenen Antworten nicht vollständig prüfbar."
- Befund: Die schwierigste Stelle des Kapitels und zugleich die Begründung der zentralen Sensitivitätsanalyse. Vier Fälle (abgeschnitten und nicht parsebar; vollständig, aber nicht parsebar; abgeschnitten, aber parsebar; vollständig und parsebar) stehen in zwei Sätzen mit je einem Semikolon, ohne dass die Fälle als solche benannt werden. „Unter Greedy-Decoding" ist eine Lehnübersetzung.
- Vorschlag: „Die Auswahlregel ‚nicht parsebar' steht stellvertretend für ‚abgeschnitten'. Dafür spricht, dass jede nicht parsebare Antwort der QLoRA-Läufe ohne schließende Klammer endet. Eine vollständige, aber aus anderem Grund nicht parsebare Antwort würde mit Greedy-Decoding identisch reproduziert. Nicht erkennen lassen sich in den gespeicherten Daten dagegen Antworten, die abgeschnitten wurden und zufällig parsebar blieben. Für die übernommenen Antworten gilt: Schließt eine Antwort vor dem Limit mit dem Ende-Token ab, bliebe sie bei identischen Logits und Greedy-Decoding unverändert. Da die ursprünglichen Läufe den Stoppgrund nicht protokollierten, lässt sich diese Bedingung aber nicht vollständig prüfen."
- Erhalten: alle vier Fälle, der Konjunktiv („würde", „bliebe"), die Einschränkung zum fehlenden Stoppgrund.
- Sicherheit: fachlich zu prüfen, weil die Reihenfolge der Argumente geändert wird; inhaltlich kommt nichts hinzu.

**K4-02 · Mittel (Grammatik) · 4.2.2, S. 33 · Z. 169–171**
- Original: „Für die Familie sprachen neben der Lizenz die fein abgestufte Größenleiter mit gemeinsamem Tokenizer und Chat-Template, sodass Größeneffekte nicht mit Architekturwechseln vermischt werden."
- Befund: Das Subjekt („die Größenleiter") steht im Singular, das Verb im Plural. Der „sodass"-Satz hängt logisch nicht an „sprachen", sondern an der Größenleiter.
- Vorschlag: „Für die Familie sprach neben der Lizenz die fein abgestufte Größenleiter mit gemeinsamem Tokenizer und Chat-Template: Größeneffekte werden so nicht mit Architekturwechseln vermischt."
- Erhalten: beide Gründe und die Folgerung.
- Sicherheit: sicher.

**K4-03 · Mittel · 4.5.1, S. 46 · Z. 924–925**
- Original: „Das Bootstrap ist auf die vorhandenen trainierten Modelle und gespeicherten Vorhersagen bedingt."
- Befund: „Auf etwas bedingt sein" ist eine Lehnübersetzung von „conditional on" und im Deutschen nicht gebräuchlich. Derselbe Ausdruck steht im Kommentar zur Zero-Shot-Unsicherheit nicht noch einmal im Text, wohl aber sinngemäß in 5.7.5.
- Vorschlag: „Das Bootstrap hält die trainierten Modelle und die gespeicherten Vorhersagen fest und variiert nur die Auswahl der Testsätze."
- Erhalten: Aussage, dass nur die Stichprobenunsicherheit erfasst wird; die beiden folgenden Sätze bleiben.
- Sicherheit: sicher.

**K4-04 · Mittel · 4.4.5, S. 43 · Z. 733–740**
- Original: „LoRA wird gewählt, weil LLMs ohne aufgabenspezifisches Fine-Tuning bei NER auf Tokenebene nur eingeschränkt konkurrenzfähig sind und Adapter den Speicher- und Rechenbedarf gegenüber vollständigem Fine-Tuning erheblich senken, in den Vergleichen von Zhan et al. aber nicht hinter diesem zurückbleiben (Abschnitte 2.6 und 3.4)."
- Befund: Ein Satz mit rund 45 Wörtern und drei Begründungen; „diesem" bezieht sich über eine Quellenangabe hinweg auf „vollständigem Fine-Tuning".
- Vorschlag: „LoRA wird aus zwei Gründen gewählt. Ohne aufgabenspezifisches Fine-Tuning sind LLMs bei NER auf Tokenebene nur eingeschränkt konkurrenzfähig. Adapter senken zudem den Speicher- und Rechenbedarf gegenüber vollständigem Fine-Tuning erheblich, ohne in den Vergleichen von Zhan et al. dahinter zurückzubleiben (Abschnitte 2.6 und 3.4)."
- Erhalten: alle Quellenangaben an ihrer jeweiligen Aussage.
- Sicherheit: sicher.

**K4-05 · Mittel · 4.5.4, S. 49 · Z. 1119–1123**
- Original: „Um seinen Einfluss auf MultiNERD zu messen statt hochzurechnen, wurden für alle neun QLoRA-Läufe genau die Testsätze, deren gespeicherte Antwort nicht parsebar war, mit demselben Adapter, demselben Prompt und Greedy-Decoding, aber einem Limit von 1 024 Tokens erneut generiert; alle übrigen Antworten wurden unverändert übernommen."
- Befund: 45 Wörter; außerdem werden nicht die Testsätze erneut generiert, sondern die Antworten.
- Vorschlag: „Um seinen Einfluss auf MultiNERD zu messen statt hochzurechnen, wurden in allen neun QLoRA-Läufen die Antworten für genau die Testsätze neu erzeugt, deren gespeicherte Antwort nicht parsebar war. Adapter, Prompt und Greedy-Decoding blieben gleich, nur das Limit betrug 1 024 Tokens. Alle übrigen Antworten wurden unverändert übernommen."
- Erhalten: Auswahlregel, gleiche Bedingungen, einziger Unterschied Limit.
- Sicherheit: sicher.

**K4-06 · Mittel · 4.3.1, S. 35 · Z. 259–263**
- Original: „Verlustfunktion, Anpassungsumfang (vollständiges Fine-Tuning gegenüber Adaptern) und Checkpoint-Auswahl unterscheiden sich jedoch, und die gemeinsame Silberreferenz garantiert nicht, dass ihre Fehler für beide Modellfamilien neutral wirken; absolute Werte messen zudem die Übereinstimmung mit einer automatisch erzeugten Referenz."
- Befund: Drei verschiedene Vorbehalte in einem Satz; „neutral wirken" bleibt unbestimmt.
- Vorschlag: „Verlustfunktion, Anpassungsumfang (vollständiges Fine-Tuning gegenüber Adaptern) und Checkpoint-Auswahl unterscheiden sich jedoch. Die gemeinsame Silberreferenz garantiert zudem nicht, dass sich ihre Fehler auf beide Modellfamilien gleich auswirken. Absolute Werte messen ohnehin die Übereinstimmung mit einer automatisch erzeugten Referenz."
- Erhalten: alle drei Vorbehalte.
- Sicherheit: fachlich kurz zu prüfen („gleich auswirken" als Lesart von „neutral wirken").

**K4-07 · Mittel · 4.4.3, S. 41 · Z. 649–655**
- Original: „Die nach F₁ gewählte Epoche stimmt nur in einem der neun LLM-Läufe (4B, Seed 42) und in keinem der sechs Encoder-Läufe mit dem Minimum des Validierungsverlusts überein. Bei den LLMs liegt dieses Minimum in Epoche 1 oder 2. Da 12 der 15 Läufe … die letzte Epoche wählten, ist ein Untertraining nicht auszuschließen."
- Befund: Drei Befunde stehen nebeneinander, ohne dass der Absatz sagt, wie sie zusammenhängen. Der Verlust hat sein Minimum früh, das F₁-Kriterium wählt spät, und aus Letzterem wird ein mögliches Untertraining abgeleitet; ob der frühe Verlustanstieg dagegen spricht, bleibt offen.
- Vorschlag: einen verbindenden Satz ergänzen, der die beabsichtigte Lesart ausspricht. Den Wortlaut kann ich nicht vorgeben, weil er eine fachliche Einordnung enthält.
- Sicherheit: fachlich zu prüfen (Rückfrage an den Verfasser: Was soll der Leser aus dem Nebeneinander schließen?).

**K4-08 · Mittel · mehrere Stellen: Zeitformen im Methodenkapitel**
- Original (Beispiele): „Die Trainingsläufe der Hauptstudie laden … , setzen … und protokollieren …; die Zero-Shot-Läufe luden … dagegen ohne explizite Revisionsangabe" (Z. 496–501); „werden … trainiert und evaluiert (15 Trainingsläufe); die drei Zero-Shot-Konfigurationen wurden je einmal … ausgewertet" (Z. 79–81).
- Befund: Präsens und Präteritum wechseln innerhalb eines Satzes, ohne dass ein Bedeutungsunterschied erkennbar ist. Sonst hält das Kapitel die Trennung meist durch (Präsens für das Verfahren, Präteritum für einmalige Ereignisse wie Abbrüche).
- Vorschlag: an beiden Stellen eine Zeitform wählen („… laden dagegen …" bzw. „… werden je einmal … ausgewertet").
- Sicherheit: sicher.

**K4-09 · Mittel · 4.5.3, S. 48 · Z. 1045–1049**
- Original: „Diese erfasst Tabelle 4.7 über Kennzahlen der Validierungskette (Abschnitt 4.4.6) in vier Fehlerklassen: …"
- Befund: Beim ersten Lesen ist nicht klar, ob „Diese" Subjekt oder Objekt ist. Die Umstellung vermeidet offenbar einen Satzanfang mit „Tabelle" (siehe M-01).
- Vorschlag: „Sie werden über Kennzahlen der Validierungskette (Abschnitt 4.4.6) in vier Fehlerklassen erfasst (Tabelle 4.7): …"
- Sicherheit: sicher.

**K4-10 · Mittel · 4.4.4, S. 42 · Z. 681–683**
- Original: „Die Einordnung des Formats in die Indexing-Familie nach Semin et al. gibt Abschnitt 3.3.1."
- Befund: „Eine Einordnung geben" ist unüblich, dazu die Objekt-vor-Subjekt-Stellung.
- Vorschlag: „Abschnitt 3.3.1 ordnet das Format in die Indexing-Familie nach Semin et al. ein."
- Sicherheit: sicher.

**K4-11 · Mittel · 4.4.3, S. 41 · Z. 616–617**
- Original: „Für diese beiden Modelle erfolgte somit keine Konfigurationswahl anhand eigener Testergebnisse."
- Befund: Funktionsverbgefüge mit Substantivkette; die Aussage ist wichtig (geringerer Testbezug bei 4B und 27B) und geht in der Form unter.
- Vorschlag: „Die Einstellungen dieser beiden Modelle wurden also nicht anhand ihrer eigenen Testergebnisse gewählt."
- Sicherheit: sicher.

**K4-12 · Mittel · 4.5.2, S. 47 · Z. 986–988**
- Original: „Gemessen wird je Testsatz bei Batchgröße 1 die Dauer des Modellaufrufs (Vorwärtsdurchlauf bzw. Generierung), mit GPU-Synchronisation vor und nach dem Aufruf und nach einem nicht gewerteten Aufwärmdurchlauf."
- Befund: Das doppelte „nach" lässt sich so lesen, als werde nach dem Aufwärmdurchlauf synchronisiert; gemeint ist, dass die Messung nach dem Aufwärmdurchlauf beginnt.
- Vorschlag: „Gemessen wird je Testsatz bei Batchgröße 1 die Dauer des Modellaufrufs (Vorwärtsdurchlauf bzw. Generierung); die GPU wird vor und nach jedem Aufruf synchronisiert. Der Messung geht ein nicht gewerteter Aufwärmdurchlauf voraus."
- Sicherheit: sicher (entspricht dem Messcode).

**K4-13 · Niedrig · mehrere Stellen**
- „Einschränkend wurde der Testsplit bereits … ausgewertet" (Z. 376): wie K3-13; klarer „Eine Einschränkung: Der Testsplit wurde bereits …".
- „… unter dem Qwen3.5-Tokenizer 256 Tokens überschreiten" (Z. 589): „mit dem Qwen3.5-Tokenizer".
- „die Ergebnistabellen rechnen durch Division durch 1 024 in GiB um" (Z. 1004–1005): „für die Ergebnistabellen wird der Wert durch 1 024 geteilt und in GiB angegeben".
- „ihr Verhältnis quantifiziert die Reduktion der aktualisierten Parameter durch das Adaptertraining" (Z. 1014): „ihr Verhältnis zeigt, wie stark das Adaptertraining die Zahl der aktualisierten Parameter senkt".
- „er reagiert entsprechend stark auf die seltenen Typen" (Z. 947): „entsprechend" ohne Bezug; „deshalb" ist gemeint.
- „… unter diesen Annahmen …" (Z. 919): Die Annahmen werden erst im nächsten Absatz genannt; „unter den im Folgenden genannten Bedingungen" oder Umstellung.
- „Die Zero-Shot-Ergebnisse werden deshalb als explorative Ergebnisse der nach Testbeobachtungen gewählten Konfigurationen berichtet" (Z. 389–390): entzerrt „… werden deshalb als explorativ berichtet, weil ihre Konfiguration nach Testbeobachtungen gewählt wurde".
- Z. 1206–1211 (Inferenz auf CrossNER, 49 Wörter mit Klammer): Encoder und LLM je in einen eigenen Satz.
- Z. 638–642: „… Sätze des Validierungssplits, die wie in der Testinferenz validiert … werden": Bezug von „die" (Sätze oder Vorhersagen); neuer Satz mit „Die Vorhersagen werden …".
- Z. 924–943: Der Absatz wechselt mit „Für die Fehleranalyse werden …" das Thema; ein Absatzumbruch davor würde helfen.
- Alle sicher und optional.

Keine Änderung nötig: Kapiteleinleitung mit inhaltlichem Überblick, die drei Regime, der neue Absatz zu den drei Längengrenzen, Beschreibung der Encoder-Pipeline, die vier Schritte der Nachverarbeitung (klar und vollständig), Begründung für den Verzicht auf Constrained Decoding, die Absätze zu Trainingsdauer, GPU-Speicher und Ressourcenschwellen, der CrossNER-Abschnitt in seiner Substanz.

## Kapitel 5: Ergebnisse und Diskussion

**K5-01 · Mittel · 5.7.4, S. 71 · Z. 1095–1099**
- Original: „… ohne sie schrumpfte der Abstand von 2,1 auf 0,5 Punkte, sofern man auch Teilspans solcher Entitäten nicht als Fehler wertet."
- Befund: „Schrumpfte" ist als Konjunktiv gemeint, lautet aber wie das Präteritum. Der Satz lässt sich so lesen, als sei der Abstand in einer tatsächlichen Auswertung geschrumpft. An dieser Stelle hängt die Deutung des CrossNER-Ergebnisses.
- Vorschlag: „… ohne sie würde der Abstand von 2,1 auf 0,5 Punkte schrumpfen, sofern man auch Teilspans solcher Entitäten nicht als Fehler wertet."
- Erhalten: beide Zahlen, die Bedingung im „sofern"-Satz.
- Sicherheit: sicher.

**K5-02 · Mittel · 5.2, S. 56 · Z. 269–273**
- Original: „Die betroffenen Sätze … enthalten je Modellgröße im Seed-Mittel 656 bis 670 Referenzentitäten, rund 2,6 % aller Referenzentitäten, die in der Auswertung als übersehen zählen."
- Befund: Der Relativsatz hängt grammatisch an „aller Referenzentitäten"; so liest man „2,6 % aller Referenzentitäten, die als übersehen zählen". Gemeint ist, dass die 656 bis 670 Entitäten als übersehen zählen.
- Vorschlag: „… im Seed-Mittel 656 bis 670 Referenzentitäten. Das sind rund 2,6 % aller Referenzentitäten; sie zählen in der Auswertung als übersehen."
- Erhalten: alle Zahlen und die Zählweise.
- Sicherheit: sicher.

**K5-03 · Mittel · 5.2, S. 56 · Z. 274–278**
- Original: „… und die weiteren Fehlerzähler zeigen in Tabelle 5.2 null; dahinter stehen drei Einzelfälle in je einem Lauf …"
- Befund: Wirkt zunächst widersprüchlich (null, aber drei Fälle). Der Grund, die Rundung der Seed-Mittel, steht nur in der Tabellenlegende.
- Vorschlag: „… und die weiteren Fehlerzähler zeigen in Tabelle 5.2 gerundet null; tatsächlich gab es drei Einzelfälle in je einem Lauf …"
- Erhalten: die drei Einzelfälle mit Modell und Seed.
- Sicherheit: sicher.

**K5-04 · Mittel · 5.5, S. 62 · Z. 661–664**
- Original: „Auf den abgeschnittenen Sätzen erreicht DeBERTa-v3-large je LLM-Seed einen Recall von 93,8 bis 96,6 (bei jeweils gleichem Seed beider Modelle, Seeds 42, 123 und 456: 96,6, 93,8 und 95,1), das LLM dort null bei 256 und 97,8 bis 98,4 bei 1 024 Tokens."
- Befund: Abgeschnitten sind die Antworten, nicht die Sätze; „je LLM-Seed" erschließt sich erst durch die Klammer, und der Vergleichswert des LLM hängt als Nachtrag an.
- Vorschlag: „Auf den Sätzen, deren LLM-Antwort abgeschnitten wurde, erreicht DeBERTa-v3-large einen Recall von 93,8 bis 96,6; die Satzmenge hängt vom LLM-Seed ab, verglichen wird jeweils der gleiche Seed beider Modelle (Seeds 42, 123 und 456: 96,6, 93,8 und 95,1). Das LLM erreicht dort null bei 256 und 97,8 bis 98,4 bei 1 024 Tokens."
- Erhalten: alle Werte, die Paarung nach Seed.
- Sicherheit: fachlich kurz zu prüfen (ob „je LLM-Seed" so gemeint ist).

**K5-05 · Mittel · 5.7.4, S. 70 · Z. 1061–1066**
- Original: „Nach den Ressourcenschwellen (…) lässt eine Latenzgrenze ab 8,9 ms DeBERTa-v3-base zu, ab 16,5 ms DeBERTa-v3-large, ab 1,9 s das erste LLM (Qwen3.5-0.8B, QLoRA) und ab 5,5 s alle LLMs; die Speicherschwellen der trainierten Konfigurationen liegen bei …"
- Befund: 50 Wörter mit neun Zahlen; „eine Latenzgrenze ab 8,9 ms lässt … zu" ist verdreht (nicht die Grenze beginnt bei 8,9 ms, sondern das Modell ist ab dieser Grenze zulässig).
- Vorschlag: „Nach den Ressourcenschwellen (…) ist DeBERTa-v3-base ab einer Latenzgrenze von 8,9 ms zulässig, DeBERTa-v3-large ab 16,5 ms, das erste LLM (Qwen3.5-0.8B, QLoRA) ab 1,9 s, und ab 5,5 s sind es alle LLMs. Die Speicherschwellen der trainierten Konfigurationen liegen bei …"
- Erhalten: alle Schwellen und ihre Zuordnung.
- Sicherheit: sicher.

**K5-06 · Mittel · 5.7.2, S. 69 · Z. 975–979**
- Original: „Allerdings berichten Tam et al. Leistungseinbußen durch Formatzwang, Semin et al. dagegen Gewinne durch LogitMatch, das die Ausgabe auf gültige Spans der Eingabe beschränkt und so den Abgleich zwischen Text und Position übernimmt, der bei den Zero-Shot-Modellen hier misslingt."
- Befund: „Allerdings" kündigt einen Einwand gegen den vorigen Satz an; der erste Teil (Tam et al.) stützt ihn aber, nur der zweite widerspricht. Dazu zwei gestaffelte Relativsätze.
- Vorschlag: „Tam et al. berichten zudem Leistungseinbußen durch Formatzwang. Semin et al. erzielen dagegen Gewinne mit LogitMatch: Das Verfahren beschränkt die Ausgabe auf gültige Spans der Eingabe und übernimmt so den Abgleich zwischen Text und Position, der den Zero-Shot-Modellen hier misslingt."
- Erhalten: beide Quellenaussagen und ihre gegensätzliche Richtung.
- Sicherheit: sicher.

**K5-07 · Mittel · 5.4, S. 61 · Z. 484–486**
- Original: „Auch die Testinferenz über die 16 454 Sätze ist aufwendig: Sie dauerte für die Encoder rund 3 bzw. 5,5 Minuten, für die QLoRA-Modelle 8,6 bis 25,2 h."
- Befund: Die Ankündigung „ist aufwendig" passt nicht zum ersten Beleg (drei Minuten). Gemeint ist der Unterschied.
- Vorschlag: „Auch bei der Testinferenz über die 16 454 Sätze ist der Unterschied groß: Sie dauerte …"
- Sicherheit: sicher.

**K5-08 · Mittel · 5.7.1, S. 65 · Z. 831–866**
- Original: Absatz von „Das Ergebnis weicht von Obeidat et al. ab …" bis „… kein isolierter Effekt der Modellarchitektur."
- Befund: Ein Absatz von rund 35 Zeilen mit vier Aufgaben: Vergleich mit drei Arbeiten, vier mögliche Erklärungen, die Werte mit 1 024 Tokens, die Einordnung als Konfigurationsunterschied. Die Kernaussage tritt nicht hervor. „Das Ergebnis" am Absatzanfang ist nach zwei Ergebnissen (256 und 1 024 Tokens) und einer Tabelle nicht eindeutig. In der Aufzählung der Erklärungen ist ein Satz kreisförmig gebaut: „Der Nutzen von Vorwissen bei deutlich weniger Trainingsdaten, den … vor allem für solche Bedingungen untersuchen, wird hier nicht geprüft."
- Vorschlag: zwei Absatzumbrüche (vor „Die Encoder wurden vollständig feinjustiert …" und vor „Mit vollständigen Antworten …"); am Anfang benennen, welches Ergebnis gemeint ist; den Satz zu Vorwissen entzerren: „Ob Vorwissen bei deutlich weniger Trainingsdaten stärker nützt, wie es Bogdanov et al. und Zaratiana et al. untersuchen, wird hier nicht geprüft."
- Erhalten: alle Vergleiche, alle vier Erklärungen mit ihrem Status („ungeprüft", „nicht untersucht").
- Sicherheit: Umbrüche und der entzerrte Satz sind sicher; welches Ergebnis „Das Ergebnis" meint, muss der Verfasser festlegen. Achtung Seitenumbruch: Kapitel 5 endet am Seitenende.

**K5-09 · Mittel · 5.6, S. 64 · Z. 749–754**
- Original: „Auf CrossNER betrugen Latenz je Satz und GPU-Spitzenspeicher für DeBERTa-v3-large 25,7 ms und 1,7 GiB, das 1,6-Fache seiner MultiNERD-Latenz auf demselben GPU-Typ. Für Qwen3.5-27B waren es 14 691 ms (Median 13 763 ms) und 18,6 GiB, das 2,7-Fache seiner MultiNERD-Latenz; …"
- Befund: Die Apposition „das 1,6-Fache" folgt auf zwei Größen, bezieht sich aber nur auf die Latenz.
- Vorschlag: „… 25,7 ms und 1,7 GiB; die Latenz ist das 1,6-Fache seiner MultiNERD-Latenz auf demselben GPU-Typ. Für Qwen3.5-27B waren es 14 691 ms (Median 13 763 ms) und 18,6 GiB; hier ist die Latenz das 2,7-Fache des MultiNERD-Werts, der allerdings von der H100 NVL stammt."
- Sicherheit: sicher.

**K5-10 · Mittel · 5.7.1, S. 68 · Z. 899–902**
- Original: „… ein Teil der betrachteten Typfehler folgt Referenzentscheidungen wie der Annotation einer Band als PER."
- Befund: „Typfehler folgt Referenzentscheidungen" ist schwer zu deuten: Folgt das Modell der Referenz, oder geht der gezählte Fehler auf eine fragwürdige Referenz zurück?
- Vorschlag (Lesart 2): „… ein Teil der betrachteten Typfehler geht darauf zurück, dass die Referenz selbst anders entscheidet als erwartet, etwa eine Band als PER annotiert."
- Sicherheit: fachlich zu prüfen (Lesart).

**K5-11 · Mittel · 5.2, S. 56 · Z. 261–264 und 282–286**
- Original: „Im QLoRA-Regime ist die Parse-Fehlerrate mit rund 0,4 % gering; alle nicht parsebaren Antworten enden ohne schließende Klammer, was auf das Limit von 256 Ausgabetokens hinweist; die Nachinferenz stützt das (siehe unten)." Ebenso Z. 282–286 (drei Hauptsätze an zwei Semikolons).
- Befund: Ketten aus drei Aussagen mit zwei Semikolons; „(siehe unten)" ist unbestimmt.
- Vorschlag: „Im QLoRA-Regime ist die Parse-Fehlerrate mit rund 0,4 % gering. Alle nicht parsebaren Antworten enden ohne schließende Klammer, was auf das Limit von 256 Ausgabetokens hinweist; die Nachinferenz am Ende dieses Abschnitts stützt das." Z. 282–286 entsprechend in drei Sätze teilen.
- Sicherheit: sicher.

**K5-12 · Mittel · 5.7.2, S. 69 · Z. 987–991**
- Original: „Bei Batchgröße 1 hängt die Latenz vermutlich von der Zahl sequenzieller Decoding-Schritte und von der Dequantisierung der NF4-Gewichte ab, die beide bei jeder Entität anfallen, unabhängig davon, wie kompakt sie kodiert ist; ihr jeweiliger Anteil wurde nicht getrennt gemessen."
- Befund: Unklar, worauf sich „sie" bezieht (die Entität) und in welchem Sinn die Dequantisierung „bei jeder Entität" anfällt. Der Absatz wechselt außerdem ab Z. 980 ohne Umbruch vom Zero-Shot-Regime zur Latenz.
- Vorschlag: Absatzumbruch vor „Das Format ändert nichts daran …"; den Satz erst nach fachlicher Klärung umformulieren.
- Sicherheit: fachlich zu prüfen.

**K5-13 · Niedrig · mehrere Stellen**
- „Die Kontrollen stützen, dass ein höheres Limit …" (Z. 332): „stützen" verlangt ein Objekt; „stützen die Annahme, dass …".
- „… der Aufwand dürfte vor allem durch die Vorwärts- und Rückwärtsdurchläufe durch das vollständige Basismodell … entstehen" (Z. 479–481): dreimal „durch"; möglich „… dürfte vor allem daher rühren, dass jeder Schritt das vollständige Basismodell vorwärts und rückwärts durchläuft und die Sequenzen mit Prompt länger sind".
- „… weil seltene Typen auf wenigen Sätzen beruhen" (Z. 1130): genauer „weil die Werte seltener Typen auf wenigen Sätzen beruhen".
- „verrauschte Typen" (Z. 90, 896): Jargon; an anderer Stelle heißt dieselbe Gruppe „häufig verwechselte oder unvollständig verlinkte Typen".
- „… sodass der Vergleich nur die Größenordnung einordnet" (Z. 881–882), „dienen bei anderen Modellen und Datensätzen aber nur der Einordnung" (Z. 950–952 und 985–986): dreimal dieselbe Wendung in einem Abschnitt; die elliptische Form „bei anderen Modellen" meint „wegen anderer Modelle".
- „Vortrainingswissen und die manuelle Annotation von CrossNER lassen sich davon nicht trennen" (Z. 1100–1101): „davon" ohne klares Bezugswort; gemeint ist der Effekt der Typabbildung.
- „bei rund dem 500-Fachen von dessen Latenz" (Z. 1080): „bei rund der 500-fachen Latenz".
- „Die Empfehlung eines Encoders trotz kleiner Qualitätsunterschiede diskutieren auch Obeidat et al." (Z. 1074–1075): offen, ob die Autoren sie aussprechen oder nur erörtern (fachlich zu prüfen).
- „der Abstand base/large" (Z. 1132): Schrägstrich-Kürzel im Fließtext; „der Abstand zwischen DeBERTa-v3-base und -large".
- Z. 166–172 (Mikro gegenüber Makro): „gewichtet die seltenen Typen gleich" meint „gewichtet alle Typen gleich, also auch die seltenen"; zwei Semikolons.
- Alle sicher und optional, außer dem gekennzeichneten Punkt.

Keine Änderung nötig: Kapiteleinleitung mit Rundungsregel, die Darstellung der Gesamtergebnisse, die Absätze zur Nachinferenz und zur Kontrollstichprobe (trotz hoher Zahlendichte gut geführt), 5.3, die Absätze zu Latenz und Speicher in 5.4, die Fehleranalyse in 5.5, der Absatz „Datenkontamination" (klar gegliedert, Reichweite jeder Überlegung benannt), die Empfehlungen in 5.7.4 von „Auf MultiNERD ist damit …" bis „… unter keiner Grenze die beste Wahl".

## Kapitel 6: Fazit und Ausblick

**K6-01 · Hoch · 6.1, S. 74 · Z. 83–86**
- Original: „Für MultiNERD und vergleichbar annotierte Wikipedia-Texte mit festem Typinventar ist danach unter jedem Latenz- und Speicherbudget, das überhaupt eine Konfiguration zulässt, ein Encoder die beste gemessene Wahl …"
- Befund: Die Formulierung reicht weiter als Abstract und 5.7.4, die beide nur „auf MultiNERD" sagen. „Vergleichbar annotierte Wikipedia-Texte" wurden nicht gemessen, und „beste gemessene Wahl" passt nicht zu Texten, für die es keine Messung gibt. Ein Leser kann darin außerdem einen Widerspruch zum CrossNER-Ergebnis sehen, das ebenfalls auf Wikipedia-Texten beruht.
- Vorschlag: „Auf MultiNERD ist danach unter jedem Latenz- und Speicherbudget, das überhaupt eine Konfiguration zulässt, ein Encoder die beste gemessene Wahl …" Falls die Übertragung gewollt ist, als eigene, erkennbar vorsichtige Aussage anschließen.
- Erhalten: die Empfehlung für MultiNERD mit ihrer Bedingung.
- Sicherheit: fachlich zu prüfen (Entscheidung über die Reichweite liegt beim Verfasser).

**K6-02 · Mittel · 6.1, S. 74 · Z. 71–72**
- Original: „Hardwaregleich liegt die Latenz von Qwen3.5-0.8B um etwa das 210-Fache über der von DeBERTa-v3-base."
- Befund: „Um das 210-Fache über" heißt streng genommen das 211-Fache. Kapitel 5 und der Abstract sagen korrekt „das 210-Fache" bzw. „Latenzfaktor von etwa 210".
- Vorschlag: „Hardwaregleich beträgt die Latenz von Qwen3.5-0.8B etwa das 210-Fache der Latenz von DeBERTa-v3-base."
- Sicherheit: sicher.

**K6-03 · Mittel · 6.1, S. 74 · Z. 102–104**
- Original: „Das LLM rechtfertigt seinen Mehraufwand damit nur unter ähnlichen Transferbedingungen und bei einer Latenztoleranz von mehreren Sekunden je Satz."
- Befund: „Mehrere Sekunden" ist schwächer als der eigene Messwert zwei Sätze davor (ab rund 12,9 s je Satz; im Abstract „ab rund 13 s").
- Vorschlag: „… und bei einer Latenztoleranz von rund 13 Sekunden je Satz."
- Sicherheit: sicher (Zahl aus demselben Absatz).

**K6-04 · Mittel · 6.2, S. 75 · Z. 140–143**
- Original: „… bei den LLMs 200 Sätze ohne BIO- und INST-Entitäten, deren Typanteile um höchstens 3,1 Prozentpunkte vom Split abweichen; …"
- Befund: „Deren" bezieht sich grammatisch auf die BIO- und INST-Entitäten; gemeint sind die 200 Sätze.
- Vorschlag: „… bei den LLMs 200 Sätze, deren Typanteile um höchstens 3,1 Prozentpunkte vom Split abweichen und die keine BIO- und INST-Entitäten enthalten; …"
- Sicherheit: sicher.

**K6-05 · Mittel · 6.3, S. 77 · Z. 236–240**
- Original: „Eine manuell geprüfte Stichprobe wie die 1 000 Sätze mit mindestens 20 Instanzen je Typ, die Tedeschi und Navigli für ihre Qualitätsmessung nutzen, würde, sofern sie zugänglich ist und nicht mit dem Trainingssplit überlappt, für die häufigen, verrauschten Typen ANIM, FOOD und PLANT zwischen Modell- und Annotationsfehlern trennen."
- Befund: 45 Wörter; zwischen „würde" und „trennen" stehen ein Bedingungssatz und eine lange Präpositionalgruppe, davor ein Relativsatz.
- Vorschlag: „Eine manuell geprüfte Stichprobe könnte für die häufigen, verrauschten Typen ANIM, FOOD und PLANT zwischen Modell- und Annotationsfehlern trennen. In Frage kommen die 1 000 Sätze mit mindestens 20 Instanzen je Typ, die Tedeschi und Navigli für ihre Qualitätsmessung nutzen, sofern sie zugänglich sind und nicht mit dem Trainingssplit überlappen."
- Erhalten: beide Bedingungen, die drei Typen, die Quelle.
- Sicherheit: sicher.

**K6-06 · Mittel · 6.1, S. 74 · Z. 62–64**
- Original: „… und geht mit großen Unterschieden in der Formtreue einher; ein isolierter Anteil wurde nicht bestimmt."
- Befund: Es bleibt offen, wessen Anteil woran gemeint ist.
- Vorschlag: „…; welchen Anteil die Formtreue daran hat, wurde nicht bestimmt."
- Sicherheit: sicher.

**K6-07 · Niedrig · mehrere Stellen**
- „Die zweite Anschlussfrage ist ein vollständiger Neulauf" und „Die dritte Anschlussfrage ist ein kontrollierter Vergleich" (Z. 244, 252): Eine Frage ist kein Neulauf; wie bei der ersten: „Die zweite betrifft einen vollständigen Neulauf …".
- „… die Nachinferenz bessert allein dessen abgeschnittene Antworten nachträglich nach" (Z. 30–31): „nachträglich" und „nach-" doppeln sich; „ersetzt nachträglich allein dessen abgeschnittene Antworten".
- „Der MultiNERD-Testsplit war vor der Hauptstudie nicht unberührt" (Z. 125): doppelte Verneinung; „war vor der Hauptstudie bereits ausgewertet worden".
- „Mit dem Wechsel des GPU-Typs und dem nicht protokollierten Stoppgrund … ersetzt auch das keinen vollständigen Neulauf" (Z. 158–159): „mit" steht für „wegen".
- „der allokierte Inferenzspeicher begründet ohne Test keine Eignung für kleinere GPUs" (Z. 206–207): klarer „aus dem allokierten Inferenzspeicher folgt ohne Test nicht, dass ein Modell auf eine kleinere GPU passt".
- „das gepaarte Bootstrap ist auf die vorhandenen Vorhersagen bedingt" (Z. 213): wie K4-03.
- „Zweitens zeigt sie, dass das Wortindex-Format im QLoRA-Regime nahezu formtreu ist" (Z. 112–113): formtreu sind die Ausgaben der Modelle, nicht das Format; „dass die QLoRA-Modelle das Wortindex-Format nahezu vollständig einhalten".
- „neue Seeds stellen keine Unabhängigkeit her" (Z. 145): Bezug zu den drei Seeds der Hauptstudie fehlt.
- Alle sicher und optional.

Keine Änderung nötig: die Beantwortung von FF1 und FF2 mit Intervallen, der Absatz „Beitrag" in seiner dritten Aussage (methodische Lehre zum Ausgabelimit, konkret und belegt), der Aufbau von 6.2 mit der Schlusszeile „Betroffen sind …" je Absatz (schematisch, aber als Orientierung nützlich), die drei Anschlussfragen in 6.3 in ihrer Substanz.

## Anhang

Der Anhang wird nicht gekürzt. Alle Vorschläge erhalten den vollständigen Inhalt.

**AH-01 · Hoch · A.1, S. 89 · Z. 136–139**
- Original: „Die drei Zero-Shot-Läufe und der Greedy-Kontrolllauf entstanden am 1. August bzw. 29. Juli 2026 (27B auf einer H100 NVL, 0.8B und 4B auf einer A100) aus einem Commit ohne lokale Änderungen; ihre Job-IDs sind in den Laufartefakten nicht protokolliert."
- Befund: Der Satz sagt für alle vier Läufe „aus einem Commit ohne lokale Änderungen". Für den Greedy-Kontrolllauf widerspricht das Tabelle A.1 (Fußnote c: Arbeitskopie mit lokalen, nicht erhaltenen Änderungen) und den Laufmetadaten (`dirty: true` bei Commit `4679bf1`; die drei Zero-Shot-Läufe: `dirty: false` bei `137bffd`). Die Ergänzung stammt aus der Überarbeitung vom 21.11.
- Vorschlag: „Die drei Zero-Shot-Läufe entstanden am 1. August 2026 aus einem Commit ohne lokale Änderungen (27B auf einer H100 NVL, 0.8B und 4B auf einer A100), der Greedy-Kontrolllauf am 29. Juli 2026 aus einer Arbeitskopie mit lokalen Änderungen (Tabelle A.1); die Job-IDs dieser Läufe sind in den Laufartefakten nicht protokolliert."
- Erhalten: beide Daten, die GPU-Zuordnung, der Hinweis zu den Job-IDs.
- Sicherheit: fachlich zu prüfen (Zuordnung der Daten und der GPU-Angabe zum Greedy-Lauf durch den Verfasser).

**AH-02 · Niedrig · A.2, S. 91 · Z. 269–276**
- Original: „Er behandelt die explorativen Vorläufe und den Abbruch des Thinking-Vorlaufs, Architekturkennzahlen der Qwen3.5-Modelle, … die Vorstudie zum Ausgabelimit und die Kontrollstichprobe."
- Befund: Ein Satz mit 51 Wörtern, der das Inhaltsverzeichnis der zwölf Unterabschnitte nacherzählt.
- Vorschlag: denselben Inhalt als Aufzählung setzen (ein Punkt je Unterabschnitt), ohne etwas zu streichen.
- Sicherheit: sicher; optional.

**AH-03 · Niedrig · mehrere Stellen**
- „Unter dem Qwen3.5-Tokenizer sind die Referenzantworten …" (Z. 392): „Mit dem Qwen3.5-Tokenizer …".
- „… ist bei den hier verwendeten Sequenzlängen nicht limitierend" (Z. 336–337): „begrenzt die hier verwendeten Sequenzlängen nicht".
- „Die Hyperparameter aus Kapitel 4 ergänzt Tabelle A.4 um Einstellungen …" (Z. 202–204): Subjekt und Objekt sind erst spät zu unterscheiden; „Zu den Hyperparametern aus Kapitel 4 kommen in Tabelle A.4 Einstellungen hinzu, die …".
- „… um Precision und Recall ergänzt die Typwerte Tabelle A.10" (Z. 624–626): siehe M-01.
- „… ebenso wenig Sequence Packing" (Z. 450): Begriff wird nicht erklärt; ein Halbsatz genügt (fachlich zu prüfen).
- „… und ist mit den übrigen Läufen nur näherungsweise vergleichbar" (Z. 542–543, ebenso 4.5.2): verglichen wird die Dauer, nicht der Lauf; „mit der Dauer der übrigen Läufe".

Keine Änderung nötig: Beschreibung der Artefakte und Codestände, A.2.1 bis A.2.12 in ihrer Substanz (sachlich, knapp, mit Zahlen belegt), die Tabellenlegenden in A.3, die Beispiele in Tabelle A.15.

**V-04 · Niedrig · Abkürzungsverzeichnis, Eintrag IOB**
- Original: „Inside, Outside, Beginning (Tagging-Schema; IOB1 und IOB2 = BIO)"
- Befund: Liest sich, als hießen beide Varianten BIO; nach 2.2 wird nur IOB2 so genannt.
- Vorschlag: „Inside, Outside, Beginning (Tagging-Schema mit den Varianten IOB1 und IOB2; IOB2 auch BIO)"
- Sicherheit: sicher.


## 4. Gesamteinschätzung

- **Überwiegend verständlich und natürlich?** Ja. Der Text ist sachlich, konkret und fast frei von Leerformeln; Sätze sind im Mittel 20 Wörter lang. Der Ton ist dem Gegenstand angemessen und nicht künstlich formell.
- **Wiederkehrende schematische Formulierungen?** Drei: die Verweissätze mit vorangestelltem Objekt (M-01), die Hauptsatzketten an Semikolons (M-02) und die wiederholten Vorbehalte (M-03). Keines davon ist falsch; die Häufung macht den Text stellenweise gleichförmig.
- **Zu verschachtelte Sätze?** Selten. Wirklich schwer ist nur die Begründung der Auswahlregel in 4.5.4 (K4-01). Daneben stehen rund ein Dutzend überladene Sätze, vor allem in 3.3.1, 4.4.5, 4.5.4, 5.2, 5.7.4 und 6.3.
- **Kernaussage nicht klar genug?** An vier Stellen: 4.4.3 zur Checkpoint-Auswahl (K4-07), der lange Absatz in 5.7.1 (K5-08), der Schluss von 5.7.2 (K5-12) und der letzte Absatz von 2.8 (K2-04).
- **Präzise und konzise?** Weitgehend. Auffällig ist, dass das Fazit an drei Stellen ungenauer ist als Kapitel 5 und der Abstract: Reichweite der Empfehlung (K6-01), „um das 210-Fache über" (K6-02), „mehrere Sekunden" statt rund 13 (K6-03). Dazu kommen der als Präteritum lesbare Konjunktiv in 5.7.4 (K5-01) und der Widerspruch im Anhang (AH-01).
- **Welche Teile brauchen am ehesten Überarbeitung?** Kapitel 5 (5.2 und 5.7) wegen der Semikolonketten und der unklaren Bezüge; 4.5.4; 6.1. Kapitel 1 und 2 und der Abstract brauchen am wenigsten.
- **Worauf besonders achten?** Bezüge von „sie", „ihr", „das", „davon", „dieser" über eine Quellenangabe oder einen Einschub hinweg; Relativsätze, die am falschen Substantiv hängen (K5-02, K6-04); Appositionen nach zwei Größen (K5-09); Lehnübersetzungen (M-04).

**Grenzen dieser Einschätzung.** Sie beruht auf einer vollständigen Lektüre der Quelltexte in einem Durchgang. Feinheiten der Zeichensetzung und der Typografie waren nicht Gegenstand; dafür gab es den Durchgang vom 12.11. Nicht gefunden habe ich Rechtschreibfehler. Die beiden Grammatikfehler sind K4-02 und K3-11.

## 5. Vorschlag für die Freigabe

1. **Zuerst klären (fachlich):** K6-01, AH-01, K4-01, K1-02, K4-07, K5-08 (welches Ergebnis), K5-10, K5-12, K3-04, K3-05.
2. **Sicher und lohnend:** K4-02 und K3-11 (Grammatik), K5-01, K6-02, K6-03, K5-02, K5-03, K6-04, K3-01, K3-10, K5-09, K4-03, K4-12, dazu die vier Beispiele aus M-01.
3. **Sicher, aber mit Umbruchprüfung:** alle übrigen Mittel-Befunde in Kapitel 3 und 5 sowie V-01.
4. **Nach Geschmack:** die Niedrig-Befunde.

## 6. Nachtrag: Umsetzung vom 21.11.2026 (nach Freigabe)

Freigabe des Verfassers: KI-Angabe um Claude Fable 5.2 ergänzen und die Änderungen des Berichts umsetzen. Umgesetzt sind 94 Ersetzungen in zehn Dateien. Prüfbuild und endgültiger Build: 112 Seiten, keine Warnungen; alle Kapitel, Abschnitte, Abbildungen und Tabellen stehen auf denselben Seiten wie zuvor, mit einer Ausnahme (Tabelle 4.8 jetzt auf S. 51 statt S. 50). Zahlen, Quellenangaben und Querverweise sind gegen den Stand `b9de051` abgeglichen: keine Quelle und kein Verweis entfallen; neu sind nur „5.2" (KI-Angabe), „13" (K6-03) sowie Datum, GPU und Tabellenverweis in AH-01.

**Umgesetzt wie vorgeschlagen:** V-01, V-02, V-03 (Fable 5.2), K1-01, K1-03, K1-04, K1-06, K2-01, K2-03, K2-04, K2-06, K2-07, K3-01 bis K3-03, K3-06, K3-08 bis K3-11, K4-02, K4-04 bis K4-06, K4-08 bis K4-12, K5-01 bis K5-07, K5-09, K5-11, K6-02 bis K6-06, AH-01, die genannten Beispiele aus M-01 (5.2, 5.3, 4.4.1, A.3) und M-02 (5.1.2, 5.6, 5.7.3) sowie die sicheren Punkte aus den Sammelbefunden K3-13, K4-13, K5-13, K6-07 und AH-03.

**Umgesetzt in einer kürzeren Variante**, weil die vorgeschlagene Fassung eine Seite gekostet hätte (eine zusätzliche Zeile in 4.5.1 verschob über die Absatzkontrolle den Umbruch bis zum Kapitelende):
- K4-01: vier statt sechs Sätze, je Fall ein Satz; die beiden Schlussaussagen bleiben an einem Semikolon.
- K4-03: „Das Bootstrap hält die vorhandenen trainierten Modelle und die gespeicherten Vorhersagen fest." (ohne den erläuternden Halbsatz); ebenso in 6.2.
- K3-07: „Sie dienen als Lesehilfe, nicht als Hypothesentest".
- V-04: „(Tagging-Schema; IOB1, IOB2 = BIO)", damit das Abkürzungsverzeichnis auf einer Seite bleibt.

**Umgesetzt mit einer Entscheidung über die Bedeutung; bitte gegenlesen:**
- K6-01: Die Empfehlung in 6.1 gilt jetzt wie im Abstract und in 5.7.4 nur „auf MultiNERD"; der Zusatz „und vergleichbar annotierte Wikipedia-Texte mit festem Typinventar" ist entfallen.
- K2-02: „solche Modelle gelten deshalb nicht als Open Source, sondern als Modelle mit offenen Gewichten".
- K3-05: „Umgang mit der kausalen Maske der Decoder" statt „mit genau diesem Unterschied".
- K3-04: „das Verfahren CodeIE formuliert seine Prompts dabei als Python-Code".
- K1-02: „…, darunter eine in derselben Pipeline trainierte Encoder-Referenz"; der Ausdruck „in derselben Pipeline" selbst ist unverändert.
- K5-04: „die Satzmenge hängt vom LLM-Seed ab, verglichen wird jeweils der gleiche Seed beider Modelle" (nach `encoder_recall_on_truncated` im Aggregationsskript).
- K5-06: ohne „Allerdings" und ohne Wertung: „… berichten Leistungseinbußen durch Formatzwang, … dagegen Gewinne durch LogitMatch. Dieses Verfahren …".
- AH-01: nach den Laufmetadaten berichtigt (Zero-Shot-Läufe: 1. August, Commit ohne lokale Änderungen; Greedy-Kontrolllauf: 29. Juli, A100, Arbeitskopie mit lokalen Änderungen).

**Nicht umgesetzt, weil die gemeinte Bedeutung offen ist:** K1-05 („effizienter"), K2-05 („hardwareseitig"), K2-08 (Mixture-of-Experts), K4-07 (Zusammenhang von Verlustminimum und gewählter Epoche), K5-08 (welches „Ergebnis" am Absatzanfang gemeint ist), K5-10 („folgt Referenzentscheidungen"), K5-12 (Latenzsatz am Ende von 5.7.2), aus den Sammelbefunden: „Adapterverfahren werden nicht erwähnt" (3.3.3), „unter diesen Annahmen" (4.5.1), „diskutieren auch Obeidat et al." und „lassen sich davon nicht trennen" (5.7.4), „neue Seeds" (6.2), „Sequence Packing" (A.2.4).

**Bewusst nicht umgesetzt:**
- die beiden Absatzumbrüche in 5.7.1 (K5-08): Sie kosten eine Seite. Der kreisförmige Satz zu Vorwissen ist entzerrt.
- K1-07 (Streichung eines Satzes), K3-12 und AH-02 (Umstellungen, im Anhang als Aufzählung), M-03 (wiederholte Vorbehalte), M-05 (einheitliche Bezeichnungen), „verrauschte Typen", die übrigen Verweissätze aus M-01 und die dreifache „Einordnung" in 5.7: Entscheidungen des Verfassers oder ohne klaren Gewinn.
- „mit übermäßiger Sicherheit" (3.3.2): bleibt, weil es die Aussage der Quelle (Overconfidence) trägt.

`thesis/main.pdf` ist neu gebaut; die Änderungen sind zusammen mit diesem Bericht committet.
