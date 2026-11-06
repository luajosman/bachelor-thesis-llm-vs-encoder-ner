# Neubewertung der gesamten Arbeit nach den methodischen Korrekturen

Bewertet wird die aktuelle Arbeitskopie vom 6. November 2026 einschließlich der Änderungen aus `06_methodische_korrekturen_2026-11-06.md`. Das aktuelle PDF hat 110 physische Seiten, davon 89 Seiten Haupttext, acht Seiten Literatur und fünf Seiten Anhang; hinzu kommt der Vorspann. Die Kapitel beginnen auf den gedruckten Seiten 1, 7, 24, 38, 62 und 84. Diese Einschätzung ersetzt die Bewertungen der entsprechenden Textteile im älteren Gesamtgutachten `05_gesamtanalyse_2026-11-06.md`.

**Geschätzte Gesamtnote im jetzigen Zustand: 1,7.** Als plausiblen Beurteilungsspielraum sehe ich etwa 1,3 bis 2,0, abhängig insbesondere von der Gewichtung des Versuchsdesigns und der Erwartungen an wissenschaftliche Verdichtung. Das ist eine fachliche Einschätzung, keine verbindliche Hochschulbewertung. Ein konkretes Bewertungsraster und Betreuerfeedback liegen dieser Prüfung nicht vor. Eine 1,0 halte ich weiterhin für möglich, aber der jetzige Stand rechtfertigt sie aus meiner Sicht noch nicht.

**Gesamte Abgabereife: 8,5/10.** Die Skala bewertet den verbleibenden Überarbeitungsbedarf: 10 bedeutet ohne wesentliche offene Arbeit abgabereif, 9 nur noch kleinere Korrekturen, 8 eine tragfähige Fassung mit klar umrissenem inhaltlichem oder strukturellem Verbesserungsbedarf. Abgabereife und Note sind unterschiedliche Größen: Auch eine vollständig abgegebene Arbeit kann methodische Grenzen besitzen und eine andere Note als 1,0 erhalten. Die Gesamtbewertung ist kein arithmetischer Mittelwert der Kapitelwerte.

## Kapitelübersicht

| Teil | Gedruckte Seiten | Abgabereife | Wichtigster verbleibender Schritt |
|---|---:|---:|---|
| 1 Einleitung | 1–6 | 8,5/10 | Forschungslücke enger und widerspruchsfrei formulieren; Wiederholungen reduzieren |
| 2 Grundlagen | 7–23 | 8/10 | Einzelne technische Verallgemeinerungen präzisieren; Umfang stärker am Experiment ausrichten |
| 3 Verwandte Arbeiten | 24–37 | 8/10 | Auswahlweg und Forschungslücke schärfen; Vergleich und Synthese ausbauen |
| 4 Methodik | 38–61 | 8,5/10 | Letzte pauschale Fairness-/Effizienzaussagen begrenzen; Haupttext und Implementierungsdetails stärker trennen |
| 5 Ergebnisse und Diskussion | 62–83 | 8,5/10 | Verbliebene kausale Überdeutungen entfernen; Hauptbefunde kompakter gemeinsam darstellen |
| 6 Fazit und Ausblick | 84–89 | 8,5/10 | Forschungsfragenantworten verdichten; Wiederholung der Diskussion reduzieren |
| Abstract | Vorspann | 9/10 | Abschließender Sprach- und Konsistenzdurchgang |
| Anhang und Reproduzierbarkeit | 98–102 | 8/10 | Konkreten Zugang zu Code/Artefakten und reproduzierbaren Auswertungsstand nennen |
| Literaturapparat | 90–97 | 8/10 | Tragende Quellen, Publikationsfassungen und ihre Nutzung in der Argumentation abschließend prüfen |
| Formale Darstellung | gesamte Arbeit | 8,5/10 | Umfang, Titel und Erklärung finalisieren; vollständige Sichtkontrolle |

## Grundlage und Grenzen der Prüfung

Die Bewertung berücksichtigt alle sechs Kapitel, Abstract, Anhang, Frontmatter, Literaturapparat und die aktuelle Ergebnisaggregation. Die gerade abgeschlossene technische Prüfung wird übernommen: vollständige Neuberechnung aus gespeicherten Vorhersagen, unveränderte bisherige Leistungs-/Ressourcenwerte und MultiNERD-Intervalle, neue CrossNER-Intervalle sowie fünf bestandene Auswertungstests. Diese Prüfungen wurden für die Neubewertung nicht ohne Anlass erneut ausgeführt. Der aktuelle Build wurde anhand des Logs und der Dateistände kontrolliert: keine LaTeX-Fehler, offenen Zitate/Verweise, Overfull- oder Underfull-Meldungen. Abstract und neue CrossNER-Tabelle wurden im unmittelbar vorhergehenden Arbeitsschritt visuell geprüft.

Der Quellenabgleich ist gezielt, kein neuer Volltextaudit aller 101 BibTeX-Einträge. Im aktuellen Build sind 79 unterschiedliche Quellen verwendet. Die Zahl der Quellen ist kein Qualitätsmaß; ungenutzte BibTeX-Einträge sind ohne Ausgabe im Literaturverzeichnis kein Mangel. Historische GPU-Ausführung und individuelle Eigenleistung des Verfassers wurden nicht unabhängig rekonstruiert. Die mündliche Verteidigung, sofern vorgesehen, ist nicht Gegenstand dieser Note.

## Gesamturteil

Die Arbeit besitzt eine substanzielle eigene empirische Untersuchung: zwei Encoder, drei Größen einer LLM-Familie, mehrere Anpassungsregime, drei Trainingsseeds, eine gemeinsame strikte Auswertung, Ressourcenmessungen, Fehleranalysen und einen zusätzlichen Transfervergleich. Besonders relevant ist die Untersuchung des Ausgabelimits. Sie zeigt, dass ein zunächst deutlich erscheinender Qualitätsabstand auf MultiNERD weitgehend mit der Ausgabekonfiguration zusammenhängt und dass sich auf CrossNER die Rangfolge verändert. Dieser Befund ist aussagekräftiger als eine pauschale Entscheidung zwischen „Encoder“ und „LLM“.

Die methodische Überarbeitung hat den Text verbessert: Die SD-Heuristik ist entfernt, CrossNER wird mit gepaarten Intervallen eingeordnet, nicht nachgewiesene Gleichwertigkeit wird nicht mehr behauptet, und zentrale Empfehlungen berücksichtigen das höhere Ausgabelimit. Methodik und Fazit benennen nun klar, welche Faktoren gemeinsam variieren und welche Unsicherheiten verbleiben.

Für eine 1,0 fehlen vor allem eine konsequent zugespitzte Forschungslücke, eine stärker vergleichende Literaturargumentation, die Bereinigung einiger Restüberdeutungen und eine kompaktere wissenschaftliche Darstellung. Die frühere Testnutzung bleibt daneben ein echter Vorbehalt des Versuchsdesigns. Sie ist jetzt offen dokumentiert, wird dadurch aber nicht rückwirkend beseitigt. Deshalb wäre es unangebracht, den Weg zur 1,0 als bloße sprachliche Endpolitur zu beschreiben.

## 1. Einleitung – 8,5/10

**Stärken.** Motivation, Gegenstand und Forschungsfragen sind verständlich. FF1–FF3 decken Erkennungsleistung/Formtreue, Modellgröße/Ressourcen und praktische Empfehlungen ab und werden später wieder aufgegriffen. Die Abgrenzung zwischen tatsächlichem Konfigurationsvergleich und isolierten Kausaleffekten ist sinnvoll. Der Zusammenhang zwischen angemeldetem Titel und DeBERTa/QLoRA wird erklärt.

**Offen.** Die Aussage, ein allgemeiner feingranularer Datensatz sei für den direkten Encoder-LLM-Vergleich bislang nicht herangezogen worden, ist zu weit. Kapitel 3 beschreibt selbst eine MultiNERD-Variante mit UniNER- und BERT-Werten. Der eigenständige Beitrag liegt in der Kombination aus offiziellen Splits, selbst ausgeführter gemeinsamer Pipeline, Größenstaffelung, Formtreueanalyse und Ressourcenmessung. Genau diese Kombination sollte die Einleitung nennen. [Kapitel 1, ab Zeile 109](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/01_einleitung.tex:109)

Die sechs Seiten enthalten außerdem mehrfach Modellwahl, Regime, Messgrößen und Abgrenzung. Eine Kürzung um etwa eine Seite wäre möglich, ohne Substanz zu verlieren. Der DFKI-Kontext ist legitim, sollte aber knapp bleiben. Einzelne Literaturzahlen gehören eher in Kapitel 3, wenn sie in Kapitel 1 nicht unmittelbar eine Forschungsfrage begründen.

**Weg zur Abgabereife.** Forschungslücke auf die belegte Kombination begrenzen; Ziel und Beitrag in einem präzisen Absatz formulieren; jede Forschungsfrage mit dem tatsächlich folgenden Versuchsaufbau verbinden.

## 2. Grundlagen – 8/10

**Stärken.** NER, IOB-Schemata, Transformer, DeBERTa, generative Ausgabe, LoRA/QLoRA, Daten und Metriken sind nachvollziehbar aufgebaut. Die Unterscheidung von strikter Entity-Level-Auswertung, Mikro-/Makro-F₁ und Silberreferenz bereitet das Experiment gut vor. Die Abbildungen und Formeln haben eine erkennbare Funktion.

**Offen.** Einzelne technische Aussagen sind zu pauschal:

- Beim Hinweis auf fehlenden rechten Kontext des Decoders muss erkennbar bleiben, dass die generierte Antwort den vollständigen, zuvor eingegebenen Satz im Prompt sehen kann. Kausale Verarbeitung bedeutet in dieser Pipeline nicht, dass der rechte Teil des Eingabesatzes bei der Antwort unbekannt wäre. [Kapitel 2, ab Zeile 284](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/02_grundlagen.tex:284)
- Die Aussage, Inferenzzeit wachse vor allem mit der Ausgabe und nicht mit der Eingabe, sollte Prefill und autoregressives Decoding unterscheiden. Für kurze Sätze mag die Ausgabe dominieren; eine allgemeine Unabhängigkeit von Eingabelänge folgt daraus nicht. [Kapitel 2, Zeile 455](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/02_grundlagen.tex:455)
- Constrained Decoding garantiert nur die jeweils erzwungenen Regeln. Eine JSON-Grammatik allein stellt keine semantisch richtigen Entitäten oder korrekten Bezug zwischen Text und Indizes sicher. [Kapitel 2, Zeile 543](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/02_grundlagen.tex:543)

Die 17 Seiten sind fachlich tragfähig, aber ausführlich. Vor allem die Herleitung von Skalierungsgesetzen steht in einem Spannungsverhältnis dazu, dass später ausdrücklich keine solchen Gesetze geschätzt werden. Ein knapper Überblick mit den relevanten Grenzen reicht möglicherweise aus. Modellhistorie und allgemeine NLP-Einführung können ebenfalls gestrafft werden.

**Weg zur Abgabereife.** Die genannten Aussagen präzisieren und Theorie bevorzugt dort vertiefen, wo sie die spätere Bewertung von Ausgabeformat, Metrik, Anpassung oder Ressourcen unmittelbar erklärt.

## 3. Verwandte Arbeiten – 8/10

**Stärken.** Die Auswahl ist breit und thematisch passend. Die Arbeit berücksichtigt spezialisierte Encoder, Encoder-Generalisten, Prompting, generative Formate und PEFT. Unterschiede zwischen Aufgaben, Typinventaren und Datenaufbereitungen werden häufig korrekt kenntlich gemacht. Die Übersichtstabelle ist hilfreich, und die Quellen werden bereits mehrfach in Zwischenfazits zusammengeführt.

**Offen.** Der Auswahlweg bleibt nur teilweise nachvollziehbar: Es gibt Einschlusskriterien, aber keinen knappen Bericht über tatsächlich verwendete Suchwege, Suchbegriffe und Recherchezeitpunkt. Eine Bachelorarbeit benötigt hier keinen nachträglich behaupteten systematischen Review. Einige ehrliche Sätze über das tatsächliche Vorgehen würden genügen.

Die Forschungslücke wird erneut zu weit gefasst. Die Aussage zu bislang fehlenden MultiNERD-Vergleichen widerspricht dem zuvor beschriebenen Vergleich mit BERT-Werten. Auch „direkter Vergleich“, „gemeinsame Pipeline“ und „übernommene Baseline“ sollten begrifflich sauber getrennt werden. [Kapitel 3, ab Zeile 651](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:651)

Die Literatur kann die eigene Argumentation stärker tragen. Beispielsweise enthält Zhan et al. in Anhang D einen Vergleich von LoRA und vollständigem Fine-Tuning; in den dort geprüften Fällen bringt vollständiges Fine-Tuning keinen höheren F₁. Das ist für die Einordnung der unterschiedlichen Anpassungsverfahren unmittelbar relevant. Es belegt keine Gleichwertigkeit für Qwen3.5/MultiNERD, relativiert aber die pauschale Vermutung, vollständiges Fine-Tuning müsse besser sein. [Zhan et al., Anhang D](https://arxiv.org/html/2601.17898v3#A4)

GNER wäre eine gezielte Ergänzung, weil die Arbeit negative Trainingsinstanzen und die Überführung generierter Ausgaben in strukturierte Entitäten untersucht. Dafür genügt eine kurze Einordnung; ein zusätzliches eigenes Experiment folgt daraus nicht. [Ding et al., 2024](https://aclanthology.org/2024.findings-acl.206/)

**Weg zur Abgabereife.** Auswahlweg nennen, Forschungslücke präzisieren, lange Abschnitte durch Unterabschnitte gliedern und an den Abschnittsenden stärker vergleichen: Welche Ergebnisse lassen sich übertragen, welche nicht, und welche konkrete Designentscheidung folgt daraus? Hier liegt einer der größten verbleibenden Hebel für eine bessere Note.

## 4. Methodik – 8,5/10

**Stärken.** Experimentmatrix, Daten, Modellrevisionen, Prompt, Parser, Hyperparameter und Messgrößen sind ungewöhnlich detailliert dokumentiert. Ausgewertete historische und kanonische Läufe werden unterschieden. Die neu beschriebene Unsicherheitsauswertung stimmt mit dem Code überein. Testnutzung, asymmetrische Checkpoint-Auswahl und Abweichungen der QLoRA-Umsetzung werden nachvollziehbar behandelt.

**Historisch verbleibende Grenzen.** Testbeobachtungen beeinflussten die Zero-Shot-Entwicklung; die LLM-Checkpoint-Auswahl nutzt die ersten 200 Validierungssätze; Anpassung, Prompt, Decoding und Präzision variieren gemeinsam; Hardware und Auslastung erschweren reine Ressourcenvergleiche. Diese Punkte begrenzen mögliche Schlussfolgerungen und können bei strenger Bewertung die Note beeinflussen. Sie sind keine automatisch tödlichen Fehler für die gesamte Untersuchung, aber auch nicht durch Dokumentation ungeschehen gemacht.

**Noch zu korrigierende Formulierungen.** „Der Vergleich bleibt intern fair“ ist zu stark: Dieselbe Silberreferenz schafft eine gemeinsame Messgrundlage, garantiert aber nicht die Neutralität ihrer Fehler gegenüber unterschiedlichen Modellfamilien. Ebenso quantifiziert der Anteil trainierbarer Parameter unmittelbar die Reduktion aktualisierter Parameter, nicht automatisch einen entsprechenden Vorteil bei Trainingsdauer oder Gesamtressourcen. [Kapitel 4, Zeile 323](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/04_methodik.tex:323), [Parameterzahlen](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/04_methodik.tex:1076)

Mit 24 Seiten ist die Methodik sehr ausführlich. Entscheidungen und ihre wissenschaftlichen Konsequenzen sollten im Haupttext stehen; technische Details der Ausführung, Wiederaufnahme und Artefaktverwaltung können stärker in den Anhang. Die nötigen Informationen dürfen dabei nicht verloren gehen.

**Weg zur Abgabereife.** Die letzten Verallgemeinerungen begrenzen, Entscheidung und Konsequenz jeweils knapp zusammenführen und Implementierungsdetails verdichten. Ein pauschales Neutraining ist dafür nicht erforderlich.

## 5. Ergebnisse und Diskussion – 8,5/10

**Stärken.** Das empirisch stärkste Kapitel. Es verbindet Gesamtwerte, Typwerte, Formtreue, Ressourcen, qualitative Fehleranalyse, Transfer und Unsicherheitsintervalle. Die Untersuchung des Ausgabelimits liefert einen eigenen Erkenntnisbeitrag. Die neue CrossNER-Auswertung korrigiert die domänenweise Aussage und stützt den gepoolten Vorsprung bei höherem Limit. Die grundlegenden Messwerte sind reproduziert.

Die Kernbefunde sind jetzt angemessen eingeordnet:

| Bedingung | DeBERTa-large | Qwen-27B QLoRA | Aussage |
|---|---:|---:|---|
| MultiNERD, 256 Tokens beim LLM | 95,82 | 94,48 | Deutlicher Abstand unter dieser Ausgabebedingung |
| MultiNERD, selektive Nachinferenz mit 1.024 Tokens | 95,82 | 95,64 | Verbleibender Abstand 0,18 Punkte; Intervall [0,02; 0,33] |
| CrossNER, 256 Tokens, drei Grobtypen | 84,99 | 71,69 | Starke Begrenzung des LLM durch die Ausgabebedingung |
| CrossNER, vollständige Inferenz mit 1.024 Tokens | 84,99 | 87,12 | Gepoolter LLM-Vorsprung 2,13 Punkte; Intervall [1,47; 2,80] |

**Verbliebene Restüberdeutungen.** Zwei Stellen sind trotz der bereits erfolgten Überarbeitung noch zu korrigieren:

1. Der Anstieg von 0,5 auf 6,2 F₁ beim kleinen Zero-Shot-Modell wird allein dem Sampling zugeschrieben. Tatsächlich wurden Prompt und Decoding gleichzeitig verändert. Die Stelle muss den Unterschied beider Konfigurationen beschreiben. [Kapitel 5, Zeile 392](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/05_ergebnisse_und_diskussion.tex:392)
2. Aus den Latenzfaktoren gegenüber Encodern wird gefolgert, das Wortindexformat habe den erhofften Effizienzeffekt nicht erbracht. Ein Vergleich mit anderen Studien isoliert den Formateffekt nicht, da Modelle, Hardware, Daten und Implementierung abweichen. Festgestellt werden kann die hohe gemessene Latenz dieser Pipeline. Auch der Qualitätsvergleich zu Zeichenoffsetformaten auf anderen Benchmarks ist eine Einordnung, kein eigener kontrollierter Formatnachweis. [Kapitel 5, ab Zeile 1025](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/05_ergebnisse_und_diskussion.tex:1025), [Effizienzaussage, Zeile 1070](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/05_ergebnisse_und_diskussion.tex:1070)

Die Ergebnisdarstellung ist außerdem zahlenreich. Eine kompakte Übersicht der vier obigen Bedingungen würde den Hauptbefund schneller sichtbar machen. Lange Wiederholungen von Tabellenwerten können reduziert werden. Die qualitative Fehleranalyse sollte klar zwischen beobachteter Abweichung und vermuteter Ursache unterscheiden: Eine Text-/Indexabweichung allein beweist beispielsweise nicht, welches Feld falsch ist.

**Weg zur Abgabereife.** Die Reststellen berichtigen, Hauptbefunde visuell oder tabellarisch bündeln, reine Tabellenparaphrasen kürzen und Diskussion nach Erklärung, Gegenbefunden und Konsequenzen strukturieren. Dafür werden überwiegend vorhandene Daten genutzt.

## 6. Fazit und Ausblick – 8,5/10

**Stärken.** Das Kapitel beantwortet die drei Forschungsfragen jetzt wesentlich konsistenter. Das Ausgabelimit erscheint in der zentralen Antwort, die fehlende kausale Isolierung wird benannt, Ressourcenempfehlungen sind eingeschränkt und Testnutzung ist als eigener methodischer Vorbehalt sichtbar. Die frühere Behauptung eines nachgewiesenen Makro-Gleichstands ist entfallen.

**Offen.** Sechs Seiten sind für Schlussfolgerungen, Grenzen und Ausblick vertretbar, aber die Zusammenfassung wiederholt viele Detailwerte aus Kapitel 5. Die eigenen Erkenntnisse könnten knapper priorisiert werden: geringe Encoderlatenz, starke Abhängigkeit des Qualitätsvergleichs vom Ausgabelimit, begrenzte Übertragbarkeit der Rangfolge. Anschließend sollte die Reichweite dieser Erkenntnisse klar bleiben. Zu viele Wiederholungen der Grenzen können den tatsächlichen Beitrag verdecken.

Der Ausblick sollte drei Prioritäten hervorheben: eine unabhängige beziehungsweise goldgeprüfte Bestätigung, kontrollierte Variation von Prompt/Decoding/Anpassung und einheitliche Ressourcenmessung. Zusätzliche Sprachen, Modelle und Adaptervarianten können knapper folgen.

**Weg zur Abgabereife.** Die Forschungsfragen jeweils mit einer kurzen Antwort eröffnen, anschließend nur die dafür nötigen Zahlen nennen; den Ausblick nach Erkenntnisgewinn priorisieren. Keine neue empirische Untersuchung ist dafür erforderlich.

## Abstract, Anhang, Literatur und Darstellung

**Abstract: 9/10.** Er ist nun eine stimmige Zusammenfassung von Aufgabe, Verfahren, Ergebnissen und Grenzen. Der Ausgabelimitbefund und die CrossNER-Rangfolge sind enthalten; unbelegte allgemeine Encoderüberlegenheit ist entfernt. Die verbleibende Arbeit ist vor allem sprachlich: hohe Informationsdichte prüfen und Zahlen-/Begriffskonsistenz mit der endgültigen Fassung sichern.

**Anhang/Reproduzierbarkeit: 8/10.** Revisionen, Laufnamen und Artefakte sind detailliert dokumentiert. Ein konkreter Fundort oder Zugangsweg für Code und Ergebnisse fehlt jedoch im Manuskript. Ein Prüfer braucht eine tatsächlich nutzbare URL, Abgabeablage oder beschriebene Zugangsregel und den dazugehörigen Stand. Öffentlichkeit oder DOI sind nicht pauschal erforderlich. Die inzwischen ergänzte Auswertung muss im finalen Stand enthalten sein. Bei Anhangstabellen sollte das Ausgabelimit klar erkennbar sein. Die Beispieltabelle verspricht Beispiele je Fehlerklasse, enthält aber keinen kombinierten Grenz-und-Typfehler; Überschrift begrenzen oder ein belegtes Beispiel ergänzen. Satz-IDs würden die Nachprüfbarkeit verbessern.

**Literaturapparat: 8/10.** Umfang und Mischung der Quellen sind für das Thema ausreichend. Priorität hat die korrekte Nutzung der tragenden Studien, nicht zusätzliche Quellenzahl. Insbesondere sind neuere Fassungen und Ergänzungen zentraler Vergleichsarbeiten in der Diskussion zu berücksichtigen. Keraghel ist in der geprüften Fassung v3 ein Preprint; diese Kennzeichnung ist nicht pauschal ein Fehler. [Keraghel et al., v3](https://arxiv.org/abs/2401.10825v3)

**Formale Darstellung: 8,5/10.** Der Build ist technisch sauber. 110 PDF-Seiten beziehungsweise 89 Haupttextseiten sind ein erheblicher Umfang; ohne Fachgebietsvorgabe ist dies kein festgestellter Formverstoß. Die Leseführung leidet aber teilweise unter Wiederholung und dichtem Tabellen-/Zahlentext. Kürzungen sollten durch inhaltliche Verdichtung erfolgen. Exakter angemeldeter Titel, Abgabedatum, endgültige Erklärung und die tatsächliche Liste verwendeter KI-Hilfsmittel sind noch abzugleichen. Die vorhandene Werkzeugangabe ist ausdrücklich als Entwurf kommentiert und muss auch die tatsächliche Nutzung bei der aktuellen Überarbeitung korrekt abbilden.

## Warum derzeit 1,7 und nicht 1,0?

Für eine sehr gute Bewertung sprechen technische Substanz, nachvollziehbare Artefakte, eigene Baselines, wiederholte Trainings, differenzierte Auswertung und der Erkenntnisgewinn zum Ausgabelimit. Gegen eine sichere 1,0 sprechen derzeit die zu weit formulierte Forschungslücke, einige verbleibende Überdeutungen, ein noch ausbaufähiger Literaturvergleich und die fehlende Verdichtung. Dazu kommt die historische Testnutzung als echter, durch Textarbeit nicht vollständig lösbarer Vorbehalt.

Die vorhandene Arbeit ist aus meiner Sicht klar im guten Bereich und kann nach gezielter Überarbeitung in den sehr guten Bereich gelangen. Eine 1,3 ist ein plausibles Ziel. Eine 1,0 bleibt ein ambitioniertes Ziel, dessen Erreichbarkeit wesentlich davon abhängt, wie der Gutachter den Erkenntnisbeitrag und die sauber begrenzte Aussagekraft gegenüber den Designschwächen gewichtet. Mehr Trainingsläufe allein würden diese Probleme nicht automatisch lösen.

## Prioritäten für die verfügbaren rund 60 Stunden

| Aufgabe | Zeitansatz | Ergebnis |
|---|---:|---|
| Forschungslücke in Kapitel 1/3 und gezielter Abgleich tragender Literatur | 12 h | Belegter Beitrag, nachvollziehbare Auswahl, stärkere Synthese |
| Restüberdeutungen in Kapitel 2/4/5 korrigieren und kapitelübergreifend prüfen | 8 h | Einheitliche Grenzen der Aussagen |
| Kapitel 4/5 verdichten und zentrale Ergebnisübersicht erstellen | 12 h | Klarere Trennung von Messwert, Erklärung und Empfehlung |
| Kapitel 1/2/6 und Abstract sprachlich straffen | 8 h | Durchgehender roter Faden, weniger Wiederholung |
| Anhang, Quellenfassungen, Artefaktzugang und Frontmatter finalisieren | 6 h | Nachvollziehbare und vollständige Abgabeunterlagen |
| Rückmeldungen einarbeiten, vollständiger Schlussdurchgang und PDF-Kontrolle | 8 h | Konsistente Endfassung |
| Reserve | 6 h | Puffer für konkrete Rückmeldungen |
| **Gesamt** | **60 h** | |

Die Stunden sind Planungsschätzungen, keine Notengarantie. Die Prioritäten unterscheiden sich vom älteren Gutachten: Statistik und zentrale methodische Einordnung sind bereits weitgehend erledigt; Literaturargumentation, Restpräzision und Verdichtung stehen jetzt vorn. Eine neue große Trainingsmatrix, neue Modellfamilien oder eine breit angelegte Hyperparametersuche sind für diesen Überarbeitungsschritt nicht angezeigt.
