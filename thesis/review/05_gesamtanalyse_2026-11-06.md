**Gesamtanalyse der Bachelorarbeit – 6. November 2026**

Bewerteter Stand: Git-Commit `fa8fff9`, vollständige Kapitel 1–6, Frontmatter, Anhang, eingebundene Ergebnistabellen und vorhandenes PDF mit 109 Seiten. Zeitbudget des Verfassers: ungefähr 20 Stunden pro Woche, insgesamt 60 Stunden. In `thesis/STATUS.md` ist der 26. November als Abgabetermin vermerkt; der Plan sieht deshalb eine fertige Abgabefassung am 25. November vor. Eine gegebenenfalls geänderte offizielle Frist geht vor.

**Gesamturteil: 7,5/10 Abgabereife.** Die Arbeit besitzt die experimentelle Substanz für eine sehr gute Bachelorarbeit. Eine 1,0 ist ein plausibles, ambitioniertes Ziel. Der aktuelle Text ist dafür noch nicht hinreichend präzise: Mehrere Schlussfolgerungen gehen über die Evidenz hinaus, die statistische Entscheidungsregel ist problematisch, und die Empfehlungen verarbeiten die Ergebnisse der Sensitivitätsanalysen noch nicht konsequent. Mit 60 konzentrierten Stunden ist eine erhebliche Verbesserung realistisch. Eine sichere Notenprognose ist ohne Bewertungsraster und Betreuerfeedback nicht möglich.

Die Skala bewertet den Überarbeitungsbedarf, nicht die spätere Hochschulnote: 10 bedeutet ohne wesentliche offene Punkte abgabereif; 9 bedeutet nur noch kleinere Korrekturen; 8 bedeutet inhaltlich tragfähig mit klar begrenztem Überarbeitungsbedarf; 7 bedeutet substanzielle Überarbeitung wichtiger Aussagen; 6 bedeutet, dass die wissenschaftliche Zusammenfassung noch keine verlässliche Abgabefassung ist. Halbe Punkte dienen nur der Priorisierung. Das Gesamturteil ist eine qualitative Einschätzung, kein rechnerischer Notendurchschnitt.

| Teil | Abgabereife | Wichtigster Schritt zur Abgabereife |
|---|---:|---|
| 1 Einleitung | 8,5/10 | Beitrag und Reichweite knapper formulieren, Wiederholungen abbauen |
| 2 Grundlagen | 8/10 | Technische Verallgemeinerungen präzisieren, für die Untersuchung wenig benötigte Details kürzen |
| 3 Verwandte Arbeiten | 8/10 | Rechercheweg nachvollziehbar machen, Forschungslücke enger begrenzen, stärker synthetisieren |
| 4 Methodik | 7,5/10 | SD-Heuristik ersetzen, Auswahl- und Testnutzung ohne Entlastungsbehauptungen beschreiben |
| 5 Ergebnisse und Diskussion | 7/10 | 256-/1.024-Token-Befunde zusammenführen, kausale Überdeutungen und Zahlenfehler korrigieren |
| 6 Fazit und Ausblick | 6,5/10 | Forschungsfragen evidenzgerecht beantworten, Empfehlungen und Limitationen konsistent machen |
| Abstract | 6,5/10 | Hauptaussage an die vollständige Evidenz anpassen, ungetestete Vorteile herausnehmen |
| Anhang und Reproduzierbarkeit | 8/10 | Tatsächlichen Zugang zum Repository und zu den Artefakten dokumentieren |
| Literaturapparat | 8/10 | Gezielter Quellen- und Versionscheck der tragenden Aussagen |
| Formales und Darstellung | 8/10 | Umfang klären, Einheiten und Beschriftungen vereinheitlichen, Schlusskontrolle des PDFs |

**Prüfumfang und belastbare positive Ergebnisse.** Alle sechs Kapitel wurden im aktuellen Quelltext gelesen. Zusätzlich wurden die generierten Tabellen, die zentralen Auswertungsroutinen, ausgewählte Implementierungsstellen und die vorhandenen Vorhersageartefakte geprüft. Der Literaturabgleich erfolgte gezielt anhand ausgewählter Primärquellen; nicht jede Zahl aus allen zitierten Veröffentlichungen wurde erneut verifiziert. Die historische GPU-Ausführung wurde nicht wiederholt. Für die Zugänglichkeit der Rohdaten auf dem Cluster und die tatsächliche Autorenleistung kann diese Prüfung keinen unabhängigen Nachweis liefern.

Die Ergebnisaggregation wurde in ein separates Build-Verzeichnis erneut ausgeführt. Alle **18 Dateien** – Tabellen, Abbildungsdaten und `summary.json` – stimmen bytegenau mit `thesis/generated/` überein. Das schließt die erneute Berechnung der MultiNERD-Metriken und der vorhandenen Bootstrap-Intervalle ein. Die CrossNER-1.024-Vorhersagen wurden zusätzlich direkt ausgewertet: Der gepoolte F₁ beträgt über die drei Seeds **87,1196 ± 0,3342**. Die zentralen Ergebniszahlen sind damit nachvollziehbar; die Hauptprobleme liegen in ihrer Interpretation.

Der vorhandene Build protokolliert 109 Seiten, keine Overfull-Box und keine offenen Verweise; es gibt eine Underfull-Warnung. Quelltextprüfungen fanden keine fehlenden Zitationsschlüssel, doppelten Labels oder offenen `\PH{...}`-Platzhalter. Ausgewählte PDF-Seiten mit der Literaturübersicht, dem Trade-off-Plot und den Anhangstabellen wurden visuell geprüft. Es wurde kein neuer LaTeX-Build ausgeführt.

Die Literaturdatei enthält 99 Einträge, davon werden nach dem Zitationsabgleich 78 verwendet. Ungenutzte BibTeX-Einträge sind kein wissenschaftlicher Mangel und werden ohne `\nocite{*}` nicht automatisch mitgedruckt. Das Aufräumen dieser Einträge hat deutlich niedrigere Priorität als die Schlussfolgerungen.

**Was wissenschaftlich bereits stark ist.** Die Arbeit verbindet eigene Encoder-Baselines mit drei Größen derselben LLM-Familie, fünf trainierten Konfigurationen mit je drei Seeds und einer gemeinsamen strikten Metrik. Sie dokumentiert ungewöhnlich offen tatsächliche Abweichungen von der ursprünglich beabsichtigten Umsetzung. Die Trennung historischer und kanonischer Läufe, Revisionsangaben, gespeicherten Vorhersagen und automatischen Tabellen ist eine gute Grundlage für Reproduzierbarkeit. Die Silberreferenz wird ausdrücklich eingeordnet. Besonders wertvoll sind die Analysen zum Ausgabelimit: Sie erklären einen erheblichen Teil des ursprünglichen Qualitätsabstands und verändern die Transferbewertung. Das ist ein eigenständiger Erkenntnisbeitrag, der im Text stärker hervortreten sollte.

**1. Einleitung – 8,5/10; Seiten 1–6.** Motivation, Problemstellung, drei Forschungsfragen und Abgrenzung greifen bereits gut ineinander. Der Beitrag ist als empirischer Vergleich erkennbar. Positiv sind die ausdrückliche Beschränkung auf konkrete Konfigurationen und die Erläuterung der Abweichung zwischen angemeldetem Titel und eingesetzten Verfahren.

Die Einleitung ist für ihre Funktion ausführlich: Literaturzahlen, Modellwahl und Ausschlüsse werden später nochmals behandelt. Die Darstellung „zwei Lücken mit insgesamt vier Aspekten“ ist erklärungsbedürftiger als nötig. Die Größe des englischen Datensatzes und das feingranulare Inventar sind gute Gründe für die Studie, tragen aber allein keine umfassende Neuheitsbehauptung. Bereits in Kapitel 3 werden andere MultiNERD-Vergleiche genannt.

Überarbeitung: Den Beitrag auf die konkrete Kombination aus offiziellen Splits, eigener gemeinsamer Evaluation, Größenstaffelung, Ressourcenmessung und Ausgabelimit-Analyse zuspitzen. Die Forschungsfragen dürfen bleiben. FF3 sollte als szenariobezogene Ableitung aus den Messungen gelesen werden; es wurde keine allgemeine Optimierung unter beliebigen Ressourcenbudgets durchgeführt. Die wiederholten Erläuterungen zu DeBERTa, QLoRA und Ausschlüssen zusammenziehen. Der DFKI-Kontext braucht nur wenige Sätze. Etwa eine Seite Kürzung erscheint erreichbar, ohne Inhalt zu verlieren.

Abgabekriterium: Ein Leser versteht nach Kapitel 1, was genau verglichen wird, welche drei Fragen beantwortet werden und wie weit die Schlussfolgerungen reichen. Er erwartet weder einen isolierten Architekturvergleich noch einen umfassenden NER-Benchmark.

**2. Grundlagen – 8/10; Seiten 7–23.** Die notwendigen Konzepte sind weitgehend vollständig und sinnvoll angeordnet: NER und IOB2, Transformer, DeBERTa, generative Ausgabe, LoRA/QLoRA, Datensätze und Metriken. Die Unterscheidung zwischen Token- und Entity-Level sowie Mikro- und Makro-F₁ ist für den Ergebnisteil hilfreich. Die neu gezeichneten Abbildungen und die RoFormer-Quelle sind bereits vorhanden; entsprechende ältere Reviewpunkte sind erledigt.

Einige Aussagen sollten präziser werden. Bei der Erklärung der kausalen Maske muss ausdrücklich stehen, dass das LLM während der Generierung seiner Antwort den vollständigen vorangestellten Eingabesatz berücksichtigen kann. Der fehlende rechte Kontext innerhalb einer autoregressiven Eingaberepräsentation ist nicht gleichbedeutend mit fehlendem Zugriff auf den rechten Satzkontext bei der späteren Extraktion. Die Formulierung, ein Decoder „müsse“ Entitäten als Text ausgeben, beschreibt die gewählte Pipeline; Kapitel 3 behandelt selbst Decoder mit Klassifikationskopf.

Die Aussage, Constrained Decoding schließe die zuvor genannten Fehler aus, ist zu weit. Eine JSON-Grammatik kann Syntax und gegebenenfalls ein festes Schema sichern; sie garantiert weder semantisch richtige Typen noch korrekte Zuordnung zum Eingabespan. Ein Teil der Inferenzzeit entfällt außerdem auf die Eingabeverarbeitung; der Zusammenhang mit der Ausgabelänge sollte nicht als generelle Unabhängigkeit von der Eingabelänge erscheinen.

Die ausführliche Darstellung von Skalierungsgesetzen ist nur begrenzt anschlussfähig, weil später ausdrücklich keine solchen Gesetze geprüft werden. Die zentrale Einschränkung ist wichtig; Details zu Exponenten, Konstanten und zusätzlicher theoretischer Erklärung können gekürzt werden. Weitere Kürzungsmöglichkeiten liegen in historischer NLP-Einführung und mehrfachen Vorgriffen auf die eigene Methodik. Die LoRA-Abbildung und ihre Beschriftung sollten dieselbe Konvention für den Skalierungsfaktor α/r wie die Gleichung verwenden.

Abgabekriterium: Jede größere Passage unterstützt entweder eine konkrete Designentscheidung oder die Interpretation einer Ergebnisgröße. Die Theorie erklärt die Untersuchung, ohne einen umfangreichen allgemeinen Lehrbuchteil zu bilden.

**3. Verwandte Arbeiten – 8/10; Seiten 24–37.** Dieses Kapitel enthält bereits mehr als bloße Literaturzusammenfassungen. Besonders gut sind die Unterscheidungen zwischen Prompting und Anpassung, eigenen und übernommenen Encoder-Werten sowie nicht direkt vergleichbaren MultiNERD- und CrossNER-Protokollen. Die Überblickstabelle macht die Positionierung nachvollziehbar. Die zentralen Arbeiten von Zhan und Semin sind aktuell eingeordnet; die geprüften arXiv-Fassungen bestätigen die im Manuskript genannten Versionsstände. [Zhan et al., Version 3](https://arxiv.org/abs/2601.17898v3), [Semin et al., Version 2](https://arxiv.org/abs/2601.16946v2).

Offen ist eine kurze nachvollziehbare Beschreibung der Recherche: tatsächlich verwendete Suchorte, Suchbegriffe, zeitlicher Abschluss und Auswahlprinzip. Für diese Arbeit ist keine nachträglich erfundene systematische Literaturstudie nötig. Wenn die historische Suche nicht protokolliert wurde, sollte sie als narrative Auswahl beschrieben und durch einen dokumentierten abschließenden Suchlauf ergänzt werden.

Die Forschungslücke sollte konsequent auf die konkrete Kombination des Versuchsdesigns begrenzt bleiben. Die absolute Formulierung, allgemeine feingranulare Daten seien bislang nicht für Encoder-LLM-Vergleiche herangezogen worden, ist angesichts der zuvor besprochenen MultiNERD-Ergebnisse zu weit. Tragfähiger ist: In den ausgewerteten Studien fehlt die hier untersuchte Kombination aus offiziellen englischen Splits, gemeinsamer selbst ausgeführter Pipeline, Größenstaffelung und Ressourcenmessung.

GNER verdient eine kurze Einordnung, weil die Arbeit negative Instanzen und generatives NER adressiert und damit direkt zu den eigenen Formtreue- und Auslassungsbefunden passt. Ein Absatz genügt; ein zusätzlicher experimenteller Vergleich ist daraus nicht abzuleiten. [Ding et al., Rethinking Negative Instances for Generative Named Entity Recognition](https://aclanthology.org/2024.findings-acl.206/).

Strukturell helfen Unterabschnitte innerhalb 3.1 und 3.2. Die Absätze zur eigenen Modellwahl und zu den bekannten Abweichungen sollten zu einem großen Teil in Kapitel 4 bleiben. Nach jedem Themenblock sollte eine kurze Synthese stehen: Was ist bekannt, unter welchen Bedingungen, und was bleibt für die eigene Untersuchung offen? Die sehr dichte Tabelle 3.1 ist lesbar, könnte aber auf die für die Forschungslücke wichtigsten Arbeiten konzentriert werden.

Abgabekriterium: Die Forschungslücke wird durch die Literaturauswahl getragen, ohne Vollständigkeit oder erstmalige Untersuchung zu behaupten, die der Suchprozess nicht belegt.

**4. Methodik – 7,5/10; Seiten 38–60.** Der größte Pluspunkt ist die konkrete Dokumentation der tatsächlich ausgeführten Studie. Datenaufbereitung, Ausgabeformat, Parser, Hardware, Checkpoint-Auswahl und Abweichungen sind nachvollziehbar. Die drei Seeds, festen Revisionen und Auswertung derselben Testreferenzen sind eine brauchbare Vergleichsgrundlage. Die Sensitivitätsanalysen sind im aktuellen Text bereits beschrieben.

Der wichtigste offene Punkt ist die Regel, einen Unterschied nur dann anzuerkennen, wenn er die Summe zweier Seed-Standardabweichungen übersteigt. Sie ist weder ein kalibrierter Signifikanztest noch ein Äquivalenzkriterium. Drei Seeds und eine kleine SD sagen zudem wenig über die Unsicherheit seltener Testtypen aus. Die vorhandenen gepaarten Bootstrap-Intervalle sollten die zentrale Unsicherheitsdarstellung liefern; die Seed-SD bleibt eine deskriptive Ergänzung. Die Intervalle sind bedingt auf die vorhandenen trainierten Modelle und erfassen nicht sämtliche Unsicherheit durch erneutes Training, Datenauswahl oder Artikelabhängigkeiten. [Methodik, ab Zeile 943](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/04_methodik.tex:943).

Die Aussage, die Auswahl anhand der ersten 200 Validierungssätze sei in acht Läufen „folgenlos“ geblieben, weil die letzte Epoche gewählt wurde, ist unbelegt. Ohne Vergleich mit einer repräsentativeren Validierung ist unbekannt, welche Epoche diese gewählt hätte. Die beobachtete Auswahl darf berichtet werden; die behauptete Folgenlosigkeit muss entfallen. Die Verteilung der 200 Sätze kann, falls lokal verfügbar, ohne Training mit dem vollständigen Validierungssplit verglichen werden. Das wäre eine überschaubare Diagnose, kein Ersatz für einen unabhängigen Modellvergleich. [Methodik, ab Zeile 664](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/04_methodik.tex:664).

Die Testnutzung in Vorläufen und zur Festlegung der Zero-Shot-Einstellungen ist ein echter methodischer Vorbehalt. Er wird erfreulich offen genannt, verschwindet aber nicht durch die spätere Neuberechnung mit neuen Seeds. Insbesondere bleibt der Vergleich Zero-Shot/QLoRA durch Prompt, Decoding und Präzision konfundiert. Die Unterschiede dürfen als Ergebnisse der beiden Regime berichtet werden; ein isolierter Anpassungseffekt wurde nicht bestimmt. Die Änderungen der 27B-Epochenzahl sollten in einer knappen Chronologie mit ihrer Entscheidungsgrundlage erscheinen. Ein neuer Lauf auf demselben bereits verwendeten Testsplit würde die frühere Testnutzung nicht rückwirkend beheben.

Die teilweise Nachinferenz auf MultiNERD ist zweckmäßig, aber kein vollständig identischer Neulauf aller Sätze: 594 von 604 Präfixen wurden reproduziert; zehn nicht. Die fünf Kontrollsätze prüfen nur einen kleinen Ausschnitt. Die Ergebnisse sind als zusammengeführte Sensitivitätsauswertung mit überwiegend reproduzierten Präfixen zu berichten. Die bestehenden Bedingungen der Äquivalenzbehauptung müssen in der Interpretation erhalten bleiben.

Die Dokumentation von Full-Sequence-Loss, fehlender Double Quantization, nicht adaptierten DeltaNet-Projektionen und FP32-Adaptern nach Resume ist wichtig. Daraus folgt jedoch keine Pflicht, in den letzten drei Wochen neu zu trainieren. Zu erläutern sind jeweils wahrscheinlicher Einflussbereich, nicht gemessene Wirkung und verbleibende Reichweite der Aussagen. Konfigurationsdetails, die der Leser nicht für diese Beurteilung benötigt, können in den Anhang.

Abgabekriterium: Alle Aussagen zu kausalen Effekten, Vergleichbarkeit und Unsicherheit passen zum tatsächlich ausgeführten Design. Der Text benennt, was offenbleibt, und entlastet diese Einschränkungen nicht mit unbelegten Behauptungen.

**5. Ergebnisse und Diskussion – 7/10; Seiten 61–82.** Die empirische Basis dieses Kapitels ist stärker als seine aktuelle sprachliche Absicherung. Tabellen, Fehleranalyse und Nachinferenz sind vorhanden und die zentralen Werte reproduzierbar. Der Erkenntnisgewinn zum Ausgabelimit trägt die Arbeit. Die Diskussion bezieht relevante Literatur ein und benennt viele Einschränkungen.

Der Text startet allerdings mehrfach neu mit der ursprünglichen Rangfolge bei 256 Tokens, während die entscheidenden 1.024-Token-Befunde erst danach relativierend folgen. So bleibt beim Leser leicht die pauschale Botschaft eines allgemeinen Encoder-Vorsprungs hängen. Die Ergebnisse sollten unmittelbar gemeinsam lesbar sein:

| Bedingung | DeBERTa-base | DeBERTa-large | Qwen-27B QLoRA | Aussage |
|---|---:|---:|---:|---|
| MultiNERD, 256 Ausgabetokens beim LLM | 95,50 | 95,82 | 94,48 | Beide Encoder liegen im Punktwert höher |
| MultiNERD, zusammengeführte Nachinferenz mit 1.024 Tokens | 95,50 | 95,82 | 95,64 | Large liegt 0,18 vorn; Unterschied zu base nicht eindeutig abgesichert |
| CrossNER, 256 Tokens, PER/ORG/LOC | – | 84,99 | 71,69 | Deutlicher Encoder-Vorsprung unter diesem Limit |
| CrossNER, 1.024 Tokens, PER/ORG/LOC | – | 84,99 | 87,12 | LLM liegt im gepoolten Ergebnis 2,13 Punkte vorn |

Die Hauptauswertung bei 256 Tokens kann aus Gründen der Transparenz bestehen bleiben. Sie sollte überall entsprechend beschriftet sein. Die Sensitivitätsbefunde gehören unmittelbar an die zentrale Ergebnisaussage und in die praktischen Empfehlungen. Die Aussage zur Pareto-Front muss die konkrete Messbedingung nennen. Der kleine Encoder dominiert das 27B-Modell bei 1.024 Tokens nicht mehr im F₁-Punktwert; DeBERTa-large bleibt auf MultiNERD vorn. Daraus folgt nicht automatisch, dass sich die gesamte Pareto-Front ändern muss. Ressourcenwerte des 256er-Laufs dürfen aber nicht stillschweigend als vollständig gemessene 1.024er-Konfiguration behandelt werden.

Eine besonders wichtige Korrektur betrifft den Satz, von 1,3 Punkten Abstand entfielen 1,2 auf das Limit und 0,2 auf die Modellklasse. Die Nachinferenz quantifiziert eine Änderung des Limits unter den beschriebenen Bedingungen. Der Restunterschied ist kein isolierter Architektureffekt: Modellfamilie, Vortraining, Anpassung, Präzision und weitere Faktoren bleiben verschieden. Geeigneter wäre: „Nach der Nachinferenz verbleibt zwischen den untersuchten Konfigurationen ein Mikro-F₁-Unterschied von 0,18 Punkten; sein Ursprung wird durch das Versuchsdesign nicht isoliert.“ [Diskussion, ab Zeile 933](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/05_ergebnisse_und_diskussion.tex:933).

Ein Intervall, das null einschließt, belegt keinen Gleichstand. „Makro-Gleichstand belastbar“ sollte durch „kein eindeutiger Unterschied nachgewiesen“ ersetzt werden. Für einen Nachweis praktischer Äquivalenz wäre ein eigenes begründetes Kriterium erforderlich. Ein solches Verfahren ist hier nicht nötig, wenn die Aussage entsprechend begrenzt wird. [Lakens, Equivalence Tests](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/).

Weitere konkret zu korrigierende Stellen:

| Stelle | Problem | Korrektur |
|---|---|---|
| 5.1.1, etwa Zeile 144 | Zero-Shot-Werte als „obere Abschätzung“ | Potenziell optimistisch beeinflusste Einzelbeobachtungen; keine mathematische oder empirische Obergrenze |
| 5.1.1, etwa Zeile 129 | Validierungs-/Testdifferenz vollständig auf Typverteilung zurückgeführt | Als plausible Erklärung kennzeichnen; kein isolierter Nachweis des gesamten Unterschieds |
| 5.2, etwa Zeile 383 | Verbesserung von 0,5 auf 6,2 allein Sampling zugeschrieben | Prompt und Decoding wurden gemeinsam geändert; Beiträge nicht getrennt |
| 5.4, etwa Zeile 606 | 61 % erzeugte leere Arrays | Rohdaten: 8.389/16.454 = 50,98 % tatsächlich `[]`; 10.021/16.454 = 60,90 % erst nach Verarbeitung ohne Span |
| Tabelle 5.6, Abb. 5.3 und Fließtext | Speicher als GB beschriftet | Code rechnet Bytes → MiB → GiB; Werte sind GiB, sofern nicht neu in Dezimaleinheiten umgerechnet |
| Abb. 5.3, etwa Zeile 703 | „beide Achsen logarithmisch“ | Beide x-Achsen logarithmisch; F₁-Achsen linear |
| 5.7.1, etwa Zeile 979 | Niedrige Zero-Shot-Werte entlasten Kontaminationsverdacht | Niedrige Gesamtleistung schließt einen Nutzen bekannten Texts nicht aus; auch FT kann betroffen sein |
| 5.7.2 | Erfolg des Wortindexformats und Effizienzvergleich gegenüber fremden Formaten | Tragfähigkeit im eigenen Setup ist gezeigt; Überlegenheit oder fehlender Effizienzgewinn gegenüber Alternativformaten nicht kontrolliert geprüft |
| 5.7.4, etwa Zeile 1125 | CEL als belastbarer LLM-Vorteil | Widerspricht 5.1.2; höchstens numerischen Unterschied mit Unsicherheit berichten |
| 5.7.4 | Neues Typinventar per Prompt als Empfehlung | Nicht untersucht; der aktuelle Parser akzeptiert nur das feste Inventar. Potenzielle Erweiterung klar vom gemessenen Nutzen trennen |
| 5.7.4 und 6.1 | 24-GB-GPU und gemessene H100-Latenz eng verbunden | Allokierter Inferenzspeicher ist kein vollständiger Speichernachweis; Verbrauch, Latenz auf anderer GPU und Trainingsspeicher getrennt behandeln |
| 5.7.4 und 6.1 | Batching verschiebe Faktoren sicher zugunsten der LLMs | Nicht gemessen; Richtung und Ausmaß offenlassen |
| Ende 5.7.5 | Keine Einschränkung kehre eine Rangfolge um | Ausgabelimit tut dies auf CrossNER; Richtung anderer ungemessener Einflüsse nicht gesichert |

Die Empfehlungen sollten drei konkrete Situationen abdecken: ähnliche annotierte Daten mit strengen Latenzanforderungen; CrossNER-artiger Transfer auf die drei geprüften Grobtypen mit ausreichendem Ausgabebudget; fehlende Trainingsannotation unter den Grenzen des untersuchten Zero-Shot-Protokolls. Die Vorteile bei sehr seltenen Typen sind interessante Hinweise, aber mit 8, 12 oder 32 Referenzinstanzen keine ausreichende Grundlage für allgemeine Einsatzempfehlungen.

Abgabekriterium: Die Hauptbefunde bleiben auch dann widerspruchsfrei, wenn man ausschließlich Tabellen, Abstract, Empfehlungen und Fazit liest. Jede starke Erklärung ist als gemessen, als plausible Deutung oder als offene Frage erkennbar.

**Zusätzliche statistische Nachrechnung für CrossNER.** Für diesen Review wurden die gespeicherten Vorhersagen von DeBERTa-large und Qwen-27B bei 1.024 Tokens gepaart ausgewertet. Verwendet wurden 2.000 Bootstrap-Replikate mit Zufallsseed 42, identischen Satzstichproben für alle sechs Checkpoints und Mittelung der F₁-Werte über die drei Seeds. Die gepoolte Auswertung erhält die Satzanzahl jeder Domäne durch Ziehen innerhalb der Domänen. Es handelt sich um eine diagnostische Ergänzung, die noch nicht im Manuskript steht; die Intervalle resampeln keine Trainingsseeds und sind nicht für Mehrfachvergleiche korrigiert.

| Vergleich: Qwen-27B (1.024) minus DeBERTa-large | Δ Mikro-F₁ | 95-%-Perzentilintervall |
|---|---:|---|
| Künstliche Intelligenz | +0,72 | [−1,27; +2,71] |
| Literatur | +0,90 | [−0,37; +2,26] |
| Musik | +2,99 | [+1,40; +4,62] |
| Politik | +1,40 | [+0,41; +2,42] |
| Naturwissenschaften | +4,26 | [+2,28; +6,32] |
| Gepoolt, nach Domäne geschichtet | +2,13 | [+1,47; +2,80] |

Der gepoolte Vorteil bleibt deutlich. Auf Domänenebene schließen jedoch **drei** Intervalle null aus. Die in Kapitel 5 hervorgehobenen vier „unterscheidbaren“ Domänen beruhen auf der anderen, nicht kalibrierten SD-Regel. Die Zahlen zeigen konkret, warum beide Unsicherheitsbegriffe nicht gleichgesetzt werden sollten. Die Diagnose ist unter [CrossNER-Bootstrap-Ergebnis](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/build/review_2026_11_06_crossner_bootstrap.json) gespeichert. Vor einer Übernahme in die Arbeit sollte sie in die dokumentierte Aggregation integriert und als explorative Ergänzung bezeichnet werden.

**6. Fazit und Ausblick – 6,5/10; Seiten 83–88.** Die Gliederung entlang der Forschungsfragen ist gut. Die Reihenfolge der Limitationen wurde bereits verbessert. Inhaltlich ist das Kapitel aber besonders bewertungsrelevant, weil es mehrere Überdeutungen aus Kapitel 5 verdichtet: isolierter Modellklasseneffekt, Gleichwertigkeit, universelle Encoder-Empfehlung, Typänderung per Prompt und sichere Batching-Vorteile.

Das Fazit sollte zuerst die belastbaren Antworten geben: Bei gleicher MultiNERD-Referenz ist DeBERTa-large geringfügig besser als das 27B-Modell nach Nachinferenz und erheblich schneller unter den gemessenen Bedingungen. Der ursprüngliche größere Abstand hängt überwiegend mit dem Ausgabelimit zusammen. Im eingeschränkten CrossNER-Transfer kehrt sich die Rangfolge bei ausreichendem Ausgabebudget zugunsten des LLM um. Die untersuchten QLoRA-Regime liegen deutlich über den untersuchten Zero-Shot-Regimen; die genaue Größe eines isolierten Trainingseffekts wird nicht bestimmt.

Die Aussage „ein Datensatz, eine Sprache und ein Genre“ ist intern widersprüchlich: MultiNERD umfasst hier Wikipedia und Wikinews, außerdem gibt es einen ergänzenden CrossNER-Datensatz. Präzise wäre „ein Hauptdatensatz in englischer Sprache mit zwei verwandten Textgenres sowie ein auf drei Typen beschränkter Transfertest“. CrossNER erweitert die Domänenabdeckung innerhalb von Wikipedia, nicht die Genrevielfalt. Die MultiNERD-Originalarbeit bestätigt ausdrücklich zwei Genres. [Tedeschi und Navigli](https://aclanthology.org/2022.findings-naacl.60/).

Die Limitationen sollten keine unbelegten Garantien enthalten, bestimmte Faktoren beträfen nur den Betrag und niemals die Richtung. Die statistische Unsicherheit ist auch für kleine Mikro-Unterschiede relevant. Die Testnutzung ist ein Vorbehalt der internen Validität und sollte entsprechend eingeordnet werden, statt am Ende des Statistikabschnitts zu verschwinden.

Für den Ausblick reichen wenige priorisierte Schritte: unabhängige/goldgeprüfte Evaluation, kontrollierter Vergleich von Prompt/Decoding/Anpassung, einheitliche Effizienzmessung. Zusätzliche Modelle, Sprachen und Adaptervarianten können knapp folgen. Nicht jede denkbare Anschlussstudie benötigt einen eigenen ausführlichen Absatz.

Abgabekriterium: Kapitel 6 beantwortet die drei Fragen knapp und eindeutig; keine seiner Aussagen geht über Kapitel 4 und 5 hinaus. Eine Kürzung um etwa ein bis zwei Seiten ist sinnvoll, sofern die wichtigsten Limitationen erhalten bleiben.

**Abstract – 6,5/10.** Aufgabe, Datensatz, Modelle und wichtigste Messgrößen sind vorhanden. Die Nachinferenz wird erwähnt. Die Schlussfolgerung „bessere und zugleich effizientere Wahl“ bleibt jedoch zu pauschal, während der LLM-Vorsprung auf CrossNER nur eingeschoben wirkt. „Ohne das Ausgabelimit“ ist ungenau: Es wurde auf 1.024 erhöht, und in diesen Läufen wurde dieses Limit nicht erreicht. Der Makro-Gleichstand ist nicht nachgewiesen. Die angenommene Änderbarkeit des Typinventars ist kein eigenes Ergebnis und sollte nicht die zentrale Einschränkung der Empfehlung bilden. Den Abstract nach der Überarbeitung von Kapitel 5 und 6 abschließend neu formulieren.

Eine mögliche sachliche Kernaussage für die Überarbeitung lautet: „Die Unterschiede hängen deutlich von den Inferenzbedingungen ab. Auf MultiNERD verringert die Nachinferenz abgeschnittener Antworten den Abstand zwischen DeBERTa-large und Qwen-27B von 1,34 auf 0,18 F₁-Punkte. Im Transfer auf die drei gemeinsamen CrossNER-Typen erreicht Qwen-27B bei 1.024 Ausgabetokens höhere Werte. Die Encoder bieten unter den gemessenen Bedingungen einen erheblichen Latenzvorteil.“ Das ist ein Formulierungsvorschlag, kein bereits eingearbeiteter Abstract.

**Anhang, Quellen und Formalia.** Der Anhang dokumentiert viele Details, nennt aber keinen konkreten Fundort beziehungsweise Zugangsweg für das Repository und die Ergebnisartefakte. Ein Verzeichnisname allein ermöglicht keine Reproduktion. Zu ergänzen sind der tatsächlich zugängliche Stand, eine Versionsangabe und – bei nicht öffentlichen Artefakten – eine ehrliche Zugangsregel. Ein DOI ist nützlich, aber keine pauschale Pflicht. Ergebnisse und Gewichte sind getrennt zu behandeln; es muss nicht alles öffentlich sein, damit der Zugang korrekt beschrieben werden kann.

Die Anhangstabellen sollten das jeweilige Ausgabelimit ausdrücklich nennen. Eine Ergänzung um die 1.024er-Ergebnisse ist sinnvoll, soweit sie zentrale Aussagen im Haupttext trägt. Tabelle A.8 könnte um Satz-IDs und ein Beispiel eines Grenz-und-Typfehlers ergänzt werden; derzeit verspricht sie Beispiele je Fehlerklasse, zeigt aber diese kombinierte Klasse nicht. Ihr qualitativer Charakter muss erkennbar bleiben: einzelne ausgewählte Beispiele belegen nicht, wie häufig ein vermuteter Annotationsfehler insgesamt vorkommt.

Beim Quellencheck haben die acht bis zehn tragenden Studien Vorrang: DeBERTa, MultiNERD, CrossNER, LoRA/QLoRA, die direkten Vergleichsstudien und die Arbeiten zum Ausgabeformat. Titel, Publikationsfassung, Zitierjahr und die konkret verwendeten Tabellenwerte prüfen. Insbesondere ist die bloße Anzahl an Literaturquellen kein Qualitätsziel. Bestehende, bereits korrekte LoRA-/DeBERTa-Konferenzfassungen müssen nicht erneut umgebaut werden.

109 PDF-Seiten beziehungsweise 88 Seiten Haupttext sind ein erheblicher Umfang für eine Bachelorarbeit. Ohne Fachgebietsvorgaben ist das kein festgestellter Formverstoß. Inhaltlich fallen aber die 37 Seiten vor der Methodik und die Wiederholungen zwischen Diskussion und Fazit auf. Zuerst Umfang mit dem Betreuer klären, dann durch Zusammenziehen kürzen. Keine pauschale Seitenzahl durch kleinere Schrift oder geänderte Ränder erzwingen.

Deckblatt, exakter angemeldeter Titel, Datum, Eigenständigkeitserklärung und tatsächliche KI-Nutzung sind vor Abgabe gegen die aktuellen Vorgaben zu prüfen. Die vorhandene KI-Angabe ist ein Entwurf und muss die wirkliche Nutzung korrekt abbilden. Das Ziel der sprachlichen Überarbeitung ist eine selbst verantwortete Argumentation, die der Verfasser erklären kann; bloßes Umformulieren zur Veränderung des Stileindrucks genügt nicht.

**Priorisierung für das Ziel 1,0.** Die entscheidenden Arbeiten sind in dieser Reihenfolge: Schlussfolgerungen korrigieren; Statistik vereinheitlichen; Empfehlungen aus vollständiger Evidenz ableiten; Forschungslücke und Methodik straffen; formale Abgabequalität herstellen. Nicht jede dokumentierte methodische Einschränkung muss durch ein neues Experiment geschlossen werden. Testnutzung und fehlende historische Messungen lassen sich zudem nicht nachträglich wegformulieren. Sie bleiben Grenzen der Studie, mit denen die Argumentation umgehen muss.

Eine neue Trainingsmatrix, weitere große Modelle, eine breite Hyperparametersuche, vollständiges Fine-Tuning des 27B-Modells oder ein neu annotierter großer Goldtest passen nicht in dieses Zeitbudget. Zusätzliche Berechnungen sollten vorhandene Vorhersagen nutzen. Der CrossNER-Bootstrap liegt bereits als Diagnose vor. Ein weiterer kleiner Datencheck ist sinnvoll, falls er eine konkrete Unsicherheit beantwortet und nicht mehrere Tage Infrastrukturarbeit auslöst.

**Arbeitsplan mit insgesamt 60 Stunden.** Die Stunden sind Schätzungen für konzentrierte Arbeit einschließlich Lesen und Prüfen, nicht nur Schreibzeit. Kapitelübergreifende Korrekturen werden einmal gerechnet.

| Zeitraum | Aufgabe | Stunden | Prüffähiges Ergebnis |
|---|---|---:|---|
| 6.–12.11. | Review in Aufgaben übersetzen, betreuerfähigen Entwurf und drei gezielte Rückfragen zu Umfang, Vergleichsdesign und zentraler Schlussfolgerung vorbereiten | 3 | Konkrete Rückmeldung angefragt; kein Warten auf eine perfekte Fassung |
| 6.–12.11. | Zentrale Behauptungen in Kapitel 5, 6 und Abstract korrigieren; 256/1.024 gemeinsam darstellen | 8 | Konsistente Kernaussage in allen drei Teilen |
| 6.–12.11. | Statistik vereinheitlichen, CrossNER-Nachrechnung nachvollziehbar integrieren | 4 | Keine SD-Schwelle als Entscheidungsregel, keine unbelegte Äquivalenz |
| 6.–12.11. | Methodik: Auswahlverfahren, Testnutzung und Nachinferenz präzisieren | 3 | Grenzen ohne unbelegte Entlastung beschrieben |
| 6.–12.11. | Konkrete Daten- und Einheitenfehler korrigieren | 2 | GiB, leere Arrays, Beschriftungen, CEL und Genreangaben konsistent |
| 13.–19.11. | Kapitel 5 straffen und in eigene Argumentation bringen; Szenarioempfehlungen und Abbildungen anpassen | 8 | Klare Ergebnisdarstellung und eigenständige Diskussion |
| 13.–19.11. | Kapitel 6 und Abstract abschließend formulieren | 4 | Knappes, widerspruchsfreies Fazit |
| 13.–19.11. | Literaturauswahl dokumentieren, GNER einordnen, zentrale Quellen prüfen | 4 | Belastbar eingegrenzte Forschungslücke |
| 13.–19.11. | Kapitel 1 und 2 präzisieren und kürzen | 4 | Fokus auf tatsächlich benötigte Inhalte |
| 20.–25.11. | Betreuerfeedback einarbeiten | 6 | Inhaltliche Freigabefragen bearbeitet |
| 20.–25.11. | Gesamtdurchgang und externes Korrekturlesen auswerten | 4 | Zahlen, Verweise, Begriffe und Argumentation durchgehend konsistent |
| 20.–25.11. | Anhang, Repositoryzugang, Titel und Erklärung finalisieren | 3 | Vollständige Abgabeunterlagen |
| 20.–25.11. | Finaler Build und Sichtkontrolle des gesamten PDFs | 3 | Fertige Abgabefassung bis 25.11. |
| 20.–25.11. | Reserve für unerwartete Korrekturen | 4 | Zeitpuffer bleibt bis zur Schlussphase frei |
| **Gesamt** | | **60** | |

Als erste Betreuerschleife bietet sich der 8.–10. November an. Erforderlich sind vor allem Rückmeldungen zur akzeptierten Reichweite des Konfigurationsvergleichs, zur Darstellung von Haupt- und Sensitivitätsanalysen und zum Umfang. Feedback möglichst bis zum 19. November einholen, damit die letzte Woche nicht nur aus Warten besteht. Der Review selbst hat keine Nachricht an den Betreuer versendet.

Falls Zeit verloren geht, zuerst die Ausweitung der Literatur, zusätzliche Theorieabbildungen und größere Strukturumbauten reduzieren. Nicht kürzen sollte man die Korrektur unbelegter Schlussfolgerungen, die statistische Präzision, die Abgabeformalitäten und den vollständigen Schlussdurchgang. Rein inhaltlich ist die Konzentration auf die Kapitel 4–6 der größte Hebel.

**Kontrolle vor der Abgabe.** Ein Entwurf ist aus Sicht dieses Reviews bereit für die letzte formale Prüfung, wenn folgende Punkte erfüllt sind:

- [ ] 256- und 1.024-Token-Ergebnisse sind auf einen Blick unterscheidbar und werden gemeinsam interpretiert.
- [ ] Kein Restunterschied wird als isolierter Modellklasseneffekt ausgegeben.
- [ ] Kein Intervall mit null wird als Nachweis von Gleichwertigkeit bezeichnet.
- [ ] Die SD-Summenregel steuert keine belastbaren Unterschiedsaussagen mehr.
- [ ] Empfehlungen zu seltenen Typen, Batching und veränderten Typinventaren gehen nicht über die Messungen hinaus.
- [ ] Testnutzung und asymmetrische Checkpoint-Auswahl sind klar beschrieben; ihre Wirkung wird nicht als erwiesen harmlos dargestellt.
- [ ] Abstract, Forschungsfragenantworten und Ergebnistabellen erzählen dieselbe wissenschaftliche Aussage.
- [ ] Speicherwerte und Einheiten, tatsächlich leere Ausgaben und durch Verarbeitung geleerte Vorhersagen sind korrekt getrennt.
- [ ] Zugang zu Code und relevanten Artefakten ist für den vorgesehenen Leserkreis konkret beschrieben.
- [ ] Betreuerfeedback, Umfang, Titel, Erklärung und finale PDF-Darstellung sind geprüft.

Die Aussicht auf eine 1,0 hängt hier vor allem davon ab, ob aus dem umfangreichen Experimentbericht eine präzise, eigenständig begründete wissenschaftliche Argumentation wird. Die vorhandenen Experimente liefern dafür eine gute Grundlage. Die höchste Priorität haben die Aussagen, die ein Prüfer aus Abstract, Diskussion und Fazit mitnimmt.
