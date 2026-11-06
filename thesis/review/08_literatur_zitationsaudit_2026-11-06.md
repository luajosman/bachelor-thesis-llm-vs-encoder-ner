# Literatur-, Zitations- und Übernahmeprüfung

Stand: 6. November 2026. Geprüft wurde der aktuelle Arbeitsstand einschließlich der vorherigen methodischen Korrekturen. Dieser Bericht verändert das Manuskript nicht. Zeilennummern beziehen sich auf diesen Stand.

## Ergebnis

Die Arbeit besitzt eine umfangreiche, überwiegend nachvollziehbar zugeordnete Literaturbasis. Viele zentrale Zahlen sind richtig übernommen. **Die Zitierungen tragen aber noch nicht sämtliche Aussagen.** Insbesondere Kapitel 3 enthält mehrere sachlich falsche oder zu weit reichende Zusammenfassungen. Diese wirken auf die Forschungslücke in Kapitel 1 und 3 sowie auf die Interpretation in Kapitel 5 und 6 zurück.

**Im geprüften Material wurde kein konkreter Fall einer unkenntlich übernommenen Textpassage oder Abbildung nachgewiesen.** Das ist eine begrenzte Prüfbeobachtung, keine Bescheinigung der Plagiatsfreiheit. Die wesentlichen festgestellten Probleme sind fehlerhafte Quellenwiedergabe, ungenaue Zuschreibung und Schlussfolgerungen, die über die zitierten Befunde hinausgehen. Dafür ist eine gezielte Überarbeitung erforderlich, keine neue Versuchsreihe allein zur Behebung dieser Zitierprobleme.

## Umfang und Nachvollziehbarkeit

- 101 Einträge in `references.bib`; davon werden 79 verschiedene Quellen im Manuskript verwendet. Die 22 unbenutzten Einträge wurden inventarisiert, aber nicht inhaltlich gegen nicht vorhandene Textaussagen geprüft.
- 360 Einzelverweise auf Literatur, wobei bei Mehrfachzitaten jede enthaltene Quelle einzeln zählt. Alle verwendeten Schlüssel sind in der Bibliografie vorhanden.
- Kapitel 1–6 sowie Abstract, Anhang, Eigenständigkeitserklärung und Quellenkennzeichnung der Grundlagenabbildungen wurden einbezogen.
- Alle 19 verschiedenen Quellen des Literaturkapitels wurden anhand zugänglicher Volltexte, einschließlich relevanter Anhänge und Ergebnistabellen, geprüft. Bei gesperrten Verlagsseiten wurden Autorenfassungen verwendet; dies ist in der Quellenmatrix ausgewiesen.
- Für 72 der 79 verwendeten Quellen waren relevante Volltextpassagen, Buchkapitel oder offizielle Dokumentationen zugänglich. Bei sieben Quellen bleibt die Prüfung auf Metadaten, Abstract/Auszüge oder ergänzende offizielle Belege beschränkt. Das sind **keine sieben nachgewiesenen Fehler**.
- Ein automatischer Wortlautabgleich des Manuskripts mit 74 lokal verfügbaren Quelltextdateien ergab keine identische Folge von mindestens acht alphabetischen Wörtern nach der beschriebenen Normalisierung. Die Zahl bezeichnet Dateien, nicht 74 unterschiedliche zitierte Publikationen: einzelne Buchkapitel und eine ergänzte Primärquelle sind gesonderte Dateien. Dieser Test ist wegen Deutsch/Englisch und PDF-Extraktionsfehlern nur ein ergänzendes Screening.
- Vier kurze deutsche Formulierungen wurden zusätzlich als exakte Suchphrasen recherchiert. Daraus ergab sich kein zuordenbarer Übernahmebeleg. Es wurde kein vollständiges Manuskript an einen externen Plagiatsdienst hochgeladen.

Die [Quellenmatrix](08a_quellenmatrix_2026-11-06.md) enthält alle 79 Quellen mit Prüfstatus, benutzter Fassung, Kapitelzuordnung und Kurzbefund. Die [CSV-Fassung](08b_quellenmatrix_2026-11-06.csv) enthält zusätzlich sämtliche Zitatfundstellen. Downloads, Zugriffsergebnisse, Inventar und Wortlaut-Screening liegen lokal unter `thesis/build/literature_audit_2026_11_06/`; der Build-Ordner ist kein dauerhaft versioniertes Review-Artefakt.

## Befunde mit hoher Priorität

### F01 — GoLLIE: Ressourcenmessung fälschlich verneint

**Fundstellen:** [Kapitel 3:365](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:365), Tabellenzeile ab [625](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:625), Größenargument ab [679](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:679).

Die Aussage, GoLLIE messe keinen Trainings- oder Inferenzaufwand, ist falsch. Anhang D.3, Tabelle 10 enthält Hardware, FLOPs, Trainingsdauer und geschätzte CO₂-Äquivalente für mehrere Größen. Für 7B stehen dort beispielsweise 44,5 Stunden auf einer A100. Die Tabelle und die Aussage zur Größenstaffelung müssen das berücksichtigen. **Korrektur:** „Trainingsressourcen über mehrere Größen berichtet; kein gemeinsamer Encodervergleich mit Inferenzmessung.“ [GoLLIE, Autorenfassung v5](https://arxiv.org/html/2310.03668v5#A4.SS3).

### F02 — GoLLIE: Splitangabe zu pauschal als fehlend bezeichnet

**Fundstelle:** [Kapitel 3:347](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:347).

Anhang D.2 nennt Standard-Splits; Tabelle 9 führt 32.908 MultiNERD-Evaluationsbeispiele auf. Damit ist „welcher Split … nennt die Arbeit nicht“ zu absolut. Die genaue Beziehung zu deinen 16.454 englischen Testsätzen bleibt klärungsbedürftig. **Korrektur:** Die vorhandenen Angaben nennen und die ungeklärte Aufbereitung als Grund für eingeschränkte Vergleichbarkeit benennen. [GoLLIE, Anhang D.2](https://arxiv.org/html/2310.03668v5#A4.SS2).

### F03 — Zhan: eigenes Baseline-Training übersehen

**Fundstellen:** [Kapitel 3:402](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:402), [Übersichtstabelle:628](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:628).

Die zitierte aktuelle Fassung v3 sagt in Anhang G ausdrücklich, dass LLaMA3.1-8B und BERT-MRC für einen Kostenvergleich auf einer L40 trainiert wurden. Die pauschale Einstufung als ausschließlich „berichtete Werte“ ist daher nicht haltbar. Daraus folgt allerdings nicht, dass sämtliche anderen Baselines neu trainiert wurden. **Korrektur:** „Eigener Trainingskostenvergleich mit BERT-MRC; Herkunft weiterer Baselinewerte separat behandeln.“ Die angegebenen F₁-Werte 93,85/93,04 stimmen. [Zhan et al., v3, Anhang G und Tabelle 13](https://arxiv.org/html/2601.17898v3#A7).

### F04 — NuNER: Vergleichsbedingungen und GPT-4-Aussage vermischt

**Fundstellen:** [Kapitel 3:150](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:150), [Kapitel 3:520](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:520).

Abschnitt 6 trennt einen Vergleich mit GPT-Modellen auf zwei Datensätzen und verkleinerten Testsätzen von einem zusätzlichen NuNER/UniversalNER-Vergleich auf vier vollständigen Testsets. Dabei unterscheiden sich auch Stichprobendefinition und Metrik. Der Text macht daraus eine zu einheitliche Aussage und formuliert für GPT-4 einen nicht ausreichend abgesicherten Schwellenwert. Außerdem bedeutet ähnliches Vortraining nicht identische Daten: NuNER verwendet C4, UniversalNER Pile. Die später behauptete Gleichheit der LLM-erzeugten Daten trifft auf dieses Paar nicht zu.

**Korrektur:** Die zwei Vergleiche getrennt darstellen; den GPT-4-Gleichstand vorsichtig und ohne allgemeine „mehr als ein Dutzend“-Schwelle formulieren; „dieselben Daten“ für NuNER/UniversalNER streichen. Die 4,38 Millionen Annotationen, rund 200.000 Konzepte und der Größenfaktor 56 sind dagegen belegt. [NuNER, Abschnitt 6, Tabellen 2 sowie 11–12](https://aclanthology.org/2024.emnlp-main.660.pdf).

### F05 — GPT-NER: „ohne jedes aufgabenspezifische Training“ beschreibt nicht das ganze System

**Fundstelle:** [Kapitel 3:263](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:263).

GPT-3 wird bei GPT-NER nicht für NER feinjustiert. Die leistungsfähige Retrieval-Variante verwendet aber einen feinjustierten NER-Tagger für die Entitätsrepräsentationen; das steht in Abschnitt 3.2.2. Das ist für die Einordnung als rein trainingsfreies System relevant.

**Korrekturvorschlag:** „Diese Ansätze verzichten auf eine NER-spezifische Anpassung der LLM-Gewichte. GPT-NER verwendet in seiner stärksten Retrieval-Variante zusätzlich einen feinjustierten NER-Tagger zur Auswahl der Demonstrationen.“ Die 90,91 Punkte mit Few-Shot-Selbstverifikation und 92,8 der BERT-Referenz sind korrekt. [GPT-NER, Abschnitt 3.2.2 und Tabelle 1](https://aclanthology.org/2025.findings-naacl.239.pdf).

### F06 — Forschungslücke größer formuliert, als die Literatur hergibt

**Fundstellen:** [Kapitel 1:109](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/01_einleitung.tex:109), [Kapitel 3:480](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:480), [Kapitel 3:651](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:651), [Kapitel 4:295](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/04_methodik.tex:295).

Die pauschale Aussage, MultiNERD sei bislang nicht für einen Encoder-LLM-Vergleich herangezogen worden, widerspricht der eigenen Darstellung von UniversalNER: Tabelle 2 enthält UniNER und eine übernommene BERT-Referenz auch für MultiNERD. Ein neu trainierter Vergleich in derselben Pipeline ist eine engere und tragfähigere Abgrenzung. [UniversalNER, Tabelle 2](https://arxiv.org/html/2308.03279).

Auch „allgemeine Datensätze mit wenigen groben Typen“ ist als Beschreibung der gesamten Vorarbeit zu grob: Zhan untersucht unter anderem OntoNotes mit 18 Typen. Das macht OntoNotes nicht in jeder Hinsicht feingranularer als MultiNERD; die unterschiedliche fachliche Ausdifferenzierung sollte beschrieben werden, statt die Lücke nur aus der Typenzahl herzuleiten. [Zhan et al., Tabelle 2](https://arxiv.org/html/2601.17898v3#S3).

**Belastbare Neufassung des Beitrags:** „Die Arbeit ergänzt bestehende Encoder-LLM-Vergleiche um eine gemeinsame Auswertung auf den offiziellen englischen MultiNERD-Splits, eine Größenstaffelung beider Modellfamilien sowie Messungen von Erkennungsleistung, Formtreue und Ressourcen. Bereits veröffentlichte MultiNERD-Ergebnisse unterscheiden sich hinsichtlich Aufbereitung, Vergleichssystemen oder erfassten Messgrößen.“

Das ist weiterhin ein sinnvoller empirischer Beitrag. Die behauptete Neuheit wird präziser.

## Weitere inhaltliche Belegprobleme

### F07 — Studienübergreifender Vergleich beweist keinen Formateffekt

**Fundstellen:** [Kapitel 5:1023](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/05_ergebnisse_und_diskussion.tex:1023), [Kapitel 5:1070](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/05_ergebnisse_und_diskussion.tex:1070).

Deine Ergebnisse zeigen, dass die gewählte Wortindex-Pipeline in den trainierten Konfigurationen funktioniert. Der Vergleich mit Zeichenoffset-Zahlen einer anderen Studie isoliert jedoch keinen Vorteil der Wortindizierung. Entsprechend belegen deine Latenzfaktoren gegenüber den von Obeidat berichteten Faktoren nicht, dass das Format den erhofften Effizienzgewinn verfehlt hat: Modelle, Daten, Ausgabelängen und Hardwarebedingungen unterscheiden sich. Obeidats Ressourcenvergleich ist für die Einordnung geeignet, nicht als Kontrollgruppe deines Formats. [Obeidat et al., Tabellen III–IV](https://arxiv.org/html/2504.00664).

**Korrekturvorschlag:** „Das Wortindex-Format ist unter den untersuchten Bedingungen praktisch nutzbar. Seine Qualitäts- und Latenzvorteile gegenüber anderen Formaten wurden nicht isoliert untersucht. Auch mit diesem Format bleibt die gemessene Inferenz erheblich langsamer als die der Encoder.“

### F08 — Constrained Decoding wird zu weitreichend dargestellt

**Fundstelle:** [Kapitel 2:539](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/02_grundlagen.tex:539).

Nach der Erwähnung ungültiger Indizes und Typen steht, eine formale Grammatik schließe „solche Fehler“ aus. Die Quelle zu Outlines stützt die Garantie für die **explizit modellierten strukturellen Einschränkungen**. Eine JSON-Grammatik allein garantiert weder richtige Spangrenzen noch die Übereinstimmung von Text und Indizes. Typen lassen sich beispielsweise über eine explizite Aufzählung einschränken; andere Beziehungen benötigen zusätzliche Bedingungen. [Willard und Louf, strukturierte Generierung](https://arxiv.org/html/2307.09702v4).

**Korrekturvorschlag:** „Constrained Decoding kann syntaktische und ausdrücklich im Schema kodierte Fehler verhindern. Die semantische Richtigkeit der Entitäten und ihre Übereinstimmung mit der Eingabe sind dadurch nicht allgemein garantiert.“

### F09 — LogitMatch im Ausblick falsch beschrieben

**Fundstelle:** [Kapitel 6:247](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/06_fazit_und_ausblick.tex:247).

LogitMatch leitet nicht einfach Wortindizes aus den Logits ab. Es beschränkt die Generierung von Spantext auf gültige Eingabespans und gehört zur Matching-Familie. **Korrektur:** „Ein spanbeschränktes Matching-Verfahren wie LogitMatch könnte die freie Generierung numerischer Wortindizes ersetzen.“ Auch die weitergehende Beschreibung in Kapitel 5 sollte konsistent bleiben. [Semin et al., Abschnitt 4](https://arxiv.org/html/2601.16946v2#S4).

### F10 — Zusammenfassung heterogener Verfahren zu pauschal

**Fundstelle:** [Kapitel 3:473](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:473).

Die vier besprochenen Arbeiten unterstützen nicht gemeinsam die Aussage, ihre Ausgabeformate reproduzierten alle den Eingabetext und der Qualitätsgewinn sei klein gegenüber einem gemessenen Inferenzmehraufwand. Insbesondere Dukić und Šnajder verwenden hier einen Klassifikationskopf; das ist kein generatives NER-Verfahren mit sequenzieller Ausgabe. Die Einzelbeschreibung erkennt diesen Unterschied bereits richtig. In der Zusammenfassung muss er erhalten bleiben. [Dukić und Šnajder, Methode und Tabelle 2](https://aclanthology.org/2024.findings-acl.843.pdf).

**Korrektur:** Die Synthese nach generativer Ausgabe, Klassifikationskopf und tatsächlich gemessenen Ressourcen aufteilen. Aussagen über Inferenzmehrkosten nur den Studien zuschreiben, die diese erhoben haben. BIO-Tagfolgen skalieren mit der Eingabelänge; nicht jede BIO-Ausgabe muss aber den Wortlaut des Eingabetexts enthalten.

### F11 — Herkunft des Wortindex-Verfahrens genauer nennen

**Fundstellen:** [Kapitel 1:173](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/01_einleitung.tex:173), [Kapitel 3:226](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/03_verwandte_arbeiten.tex:226), [Kapitel 4:700](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/04_methodik.tex:700).

Semin ist als Quelle der Systematisierung passend und damit keine fehlende Zuschreibung. Das dort besprochene Verfahren mit nummerierten Wörtern wird aber bereits Ramponi et al. zugeschrieben. Deren veröffentlichte Promptvorlage enthält nummerierte Eingabetokens und numerische Spangrenzen. Die Primärquelle sollte ergänzt werden, während das konkrete JSON-Schema und die Nachverarbeitung als eigene Umsetzung beschrieben werden. [Ramponi et al., Anhang, Tabelle 9](https://aclanthology.org/2025.naacl-long.34.pdf).

**Einordnung:** Verbesserung der Quellenkette, kein nachgewiesenes Ideenplagiat. Die zugrunde liegende Literatur wird bereits genannt.

### F12 — Methodenbeleg für Bootstrap und Softwaredetails ergänzen

**Fundstellen:** [Kapitel 4:966](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/04_methodik.tex:966), [Kapitel 4:193](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/04_methodik.tex:193), [Kapitel 4:662](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/chapters/04_methodik.tex:662).

Das konkrete Bootstrap-Verfahren ist transparent beschrieben. Lakens belegt aber die Unterscheidung zwischen fehlendem Unterschied und Äquivalenz, nicht den Satz-Bootstrap selbst. Ergänze einen passenden Methodenbeleg; eine einschlägige Einordnung zur Testwahl in NLP bietet [Dror et al.](https://aclanthology.org/P18-1128/). Eine exakte Übereinstimmung mit deren Verfahren sollte erst nach Prüfung des gewählten Verfahrens behauptet werden.

Das Transformers-Paper von 2020 ist außerdem kein passgenauer Beleg für die aktuelle MIT-Lizenz von DeBERTa-v3. Dafür sind die jeweilige Modellkarte bzw. Lizenzdatei die richtige Quelle. Für versionsabhängiges TRL-Verhalten sind die verwendete Version und ihre Dokumentation oder der konkrete Code belastbarer als allein die Repository-Startseite. Das sind Nachweispräzisierungen; daraus folgt hier kein Beleg für eine falsche Laufbeschreibung.

### F13 — KI-Hilfsmittelangabe ist im aktuellen Stand unvollständig

**Fundstelle:** [Eigenständigkeitserklärung:18](/Users/luajosman/bachelor-thesis-llm-vs-encoder-ner/thesis/frontmatter/eigenstaendigkeitserklaerung.tex:18).

Die Erklärung nennt derzeit nur Claude Code. In dieser Sitzung wurde zusätzlich Codex von OpenAI für Analyse, Änderungen an Auswertungscode und Manuskript sowie Literaturprüfung eingesetzt. Nach dem Anspruch der **eigenen Erklärung**, die verwendeten Werkzeuge und ihre Nutzungsart zu benennen, muss diese Angabe vor der Abgabe ergänzt werden. Auch die bereits eingetragene Modell-/Versionsbezeichnung sollte anhand der tatsächlichen Nutzung überprüft werden; eine unbekannte Version darf nicht erfunden werden.

Dies ist ein Dokumentationspunkt. KI-Unterstützung oder ein bestimmter Schreibstil beweisen für sich genommen kein Plagiat. Ob und wie die Nutzung zulässig ist, richtet sich nach den konkret geltenden Vorgaben; deren vollständige Prüfung war nicht Gegenstand dieses Audits.

## Präzisierungen mit mittlerer oder niedriger Priorität

| Stelle | Beobachtung | Überarbeitung |
|---|---|---|
| Kapitel 2:286–293 | Die Erklärung zur kausalen Maske kann den Eindruck erwecken, generative NER sehe den rechten Kontext des zu extrahierenden Wortes nicht. | Ergänzen, dass bei der nachgelagerten Generierung der vollständige Eingabesatz im vorangehenden Prompt steht. Unterschied zur tokenweisen bidirektionalen Repräsentation beibehalten. |
| Kapitel 2:455 | „… nicht der Eingabe“ ist eine zu absolute Aussage über Inferenzzeit. | Prefill und kontextabhängige Decoding-Kosten berücksichtigen; Ausgabelänge als wesentlichen, nicht einzigen Faktor nennen. |
| Kapitel 2:588–589 | LoRA-Skalierung wird als Garantie dargestellt, dass keine Lernratenanpassung nötig sei. | Die Quelle spricht von einer Verringerung des erneuten Abstimmungsbedarfs. „Kann den Abstimmungsbedarf verringern“ ist genauer. [LoRA, Abschnitt 4.1](https://arxiv.org/html/2106.09685). |
| Kapitel 2:640, 650–659 | Präzision und Speichermanagement des QLoRA-Rezepts werden sehr allgemein dargestellt. | Gespeicherte Adapterpräzision, Rechenpräzision und die tatsächlich getesteten Aufgaben unterscheiden; Paged Optimizers verhindern nicht unter beliebigen Bedingungen jeden Speichermangel. Die eigene Abweichung wird in Kapitel 4 bereits transparent benannt. |
| Kapitel 2:940; Kapitel 4:1077 | Der Anteil trainierbarer Parameter wird mit einem allgemeinen Effizienzvorteil gleichgesetzt. | Er quantifiziert die Parameterreduktion; Speicher, Durchsatz und Trainingszeit sind gesonderte Größen. |
| Kapitel 3:527 | Obeidat halte trainierbare Parameter „konstant“. | „Auf eine ähnliche Größenordnung abgestimmt“: ungefähr 341/353 Mio. für Adapter gegenüber 336/435 Mio. bei Encodern. [Obeidat et al., Abschnitt II-B](https://arxiv.org/html/2504.00664). |
| Kapitel 3:165–167 | „Überwiegend grobe Typen“ verdeckt die unterschiedlichen Typinventare der Generalisten-Benchmarks. | Die tatsächlich verglichenen Datensätze und Regime nennen; insbesondere CrossNER nicht pauschal als Grobtypentest darstellen. |
| Kapitel 3:597, NuNER-Zeile | Die Ressourcenübersicht verzeichnet keine Messung. | NuNER enthält zumindest Kostenangaben; „Kostenangaben, keine systematische Latenz-/Speichermessung“ ist genauer als ein unkommentierter Strich. |
| Kapitel 3:245; Kapitel 5:1071 | Das eigene Ausgabeformat enthält zusätzlich Spantext. | Seine Länge hängt nicht ausschließlich von der Anzahl der Entitäten ab, sondern auch von deren Textlängen und Serialisierung. Es vermeidet weiterhin die Reproduktion aller Nicht-Entitätstokens. |
| Kapitel 4:281 | Die eigene Einteilung Small/Medium/Large erhält Skalierungsgesetz-Zitate. | Als eigene organisatorische Definition kennzeichnen; die Literaturbelege an Aussagen über Skalierung und deren Grenzen platzieren. |
| Kapitel 5:1050 | Nicht übereinstimmende Text- und Indexfelder werden sicher als Zählfehler bezeichnet. | „Inkonsistenz zwischen Text und Indizes“; ohne Prüfung jedes Falls ist auch ein falsches Textfeld möglich. |
| Kapitel 5:392 | Eine Verbesserung gegenüber dem Vorlauf wird dem Sampling zugeschrieben, obwohl zugleich der Prompt geändert wurde. | Den kombinierten Konfigurationswechsel beschreiben. Eine Quelle kann den fehlenden kontrollierten Vergleich nicht ersetzen. |
| Literaturübersicht, Einstieg | Auswahlkriterien sind genannt; Suchwege, Suchstand und tatsächlicher Auswahlprozess bleiben knapp. | Einen kurzen, wahrheitsgemäßen Rechercheabsatz ergänzen. Kein nachträglich erfundenes systematisches Review oder erfundene Trefferzahlen. |

## Was bereits gut belegt ist

Die folgende Auswahl enthält bewusst auch positive Gegenprüfungen. Die vollständige Zuordnung steht in der Quellenmatrix.

- **MultiNERD:** 15 Kategorien, automatische Annotation, manuelle Kontrolle von 1.000 englischen Sätzen sowie die typbezogenen Qualitätsprobleme sind durch die Originalarbeit gedeckt. Der dortige Token-F₁ von 0,85 darf nicht als unmittelbar vergleichbarer strikter Entity-F₁ der eigenen Studie gelesen werden; die Arbeit unterscheidet die Ebenen bereits. [Tedeschi und Navigli, Tabelle 5](https://aclanthology.org/2022.findings-naacl.60.pdf).
- **GLiNER:** Der Wert 55,4 für das mittlere Modell sowie der Vergleich mit UniNER-13B stehen in Tabelle 1. Die Quellenherkunft der fremden Baselinewerte wird dort ausdrücklich genannt. [GLiNER](https://aclanthology.org/2024.naacl-long.300.pdf).
- **UniversalNER:** 41,7/43,4 gegenüber 34,9 im Zero-Shot-Vergleich und 93,73/91,25 für die MultiNERD-Variante sind belegt. Die sekundäre Herkunft der BERT-Werte aus InstructUIE wird im Manuskript korrekt offengelegt. [UniversalNER](https://arxiv.org/html/2308.03279).
- **Tan/MultiCoNER:** Englischer Makro-F₁ 85,53 sowie die mehrsprachigen Validierungs-Mikro-F₁-Werte 14,78 und 92,77 passen zu den betreffenden Tabellen. Der Unterschied zwischen Makro/Mikro und Test/Validierung wird im Einzelabschnitt richtig dargestellt. [U-RaNER](https://aclanthology.org/2023.semeval-1.277.pdf).
- **CodeIE:** 82,32 im angegebenen Few-Shot-Setting und die voll überwachte Referenz 92,99 sind richtig zugeordnet. [CodeIE, Tabelle 3](https://aclanthology.org/2023.acl-long.855.pdf).
- **Dagdelen:** Der ungefähr 99-prozentige Parserfolg und die fehlende erzwungene JSON-Formatierung werden von der Quelle getragen. [Originalartikel](https://www.nature.com/articles/s41467-024-45563-x).
- **PaDeLLM:** Die Beschleunigungsspanne 1,76–10,22 ist richtig wiedergegeben; sie gehört zu dessen eigenem parallelen Decoding-Verfahren. [PaDeLLM-NER](https://proceedings.neurips.cc/paper_files/paper/2024/file/d5aed68fde8e934d0ae4aadb57acc6c0-Paper-Conference.pdf).
- **Grundlagen und Formeln:** Attention, LoRA, QLoRA und die Grundidee der Skalierungsgesetze besitzen passende Belege. Die vier aus der Literatur abgeleiteten Grundlagenabbildungen nennen jeweils ihre Vorlage und sind als eigene Darstellungen gekennzeichnet. Eine eigene Zeichnung hebt die Pflicht zur Angabe der konzeptionellen Vorlage nicht auf; diese Angabe ist hier vorhanden.
- **Methodische Einschränkungen:** Der Bezug auf Cawley zur Auswahlverzerrung und auf Lakens zur fehlenden Äquivalenzaussage ist sachlich passend. Bei Lakens konnte hier nur auf zugängliche Originalauszüge zugegriffen werden; der vollständige Artikelabruf blieb gesperrt.

## Übernahme- und Plagiatsprüfung

Geprüft wurden drei verschiedene Fragen:

1. **Wortlaut:** Der lokale Test suchte identische Achtwortfolgen nach Entfernung von LaTeX-Kommandos, Zitatschlüsseln, Kommentaren und einfachen Inline-Formeln sowie Normalisierung der Groß-/Kleinschreibung. Es wurden keine Treffer gefunden. Das ist angesichts deutschsprachigen Manuskripts und überwiegend englischer Literatur wenig aussagekräftig für Übersetzungsübernahmen. Eine Prozentzahl würde eine Genauigkeit vortäuschen, die dieses Verfahren nicht bietet.
2. **Inhaltliche und übersetzte Nähe:** Manuell abgeglichen wurden insbesondere NER/BIO, Attention, DeBERTa, LoRA/QLoRA, Skalierung sowie die Darstellungen der 19 Quellen in Kapitel 3. Die Quellenabhängigkeit wird regelmäßig durch Autorennennung oder Zitat kenntlich gemacht. Im geprüften Material ließ sich keine längere, unmarkierte Übernahme konkret belegen. Eine ähnliche technische Definition oder dieselbe notwendige Fachterminologie allein reicht dafür nicht aus.
3. **Abbildungen und Ideen:** NER-Prinzip, Transformer, LoRA-Adapter und Skalierungsdarstellung haben sichtbare Quellenangaben. Die Formeln sind in belegte Erläuterungen eingebettet. Beim Wortindex-Verfahren ist die Quellenkette ausbaubar (F11), aber die Abhängigkeit von Vorarbeiten wird nicht verschwiegen. Ergebnisabbildungen und Tabellen werden als eigene Auswertung präsentiert; ihre Datenprovenienz wurde in der vorausgehenden Methodenprüfung behandelt.

Die Prüfung deckt keine geschlossenen Hochschularchive, unveröffentlichten Abschlussarbeiten, sämtliche Bücher oder jede denkbare Internetquelle ab. Ebenso wurde keine umfassende Herkunftsanalyse des gesamten Programmcodes durchgeführt. Bei den sieben nur teilweise zugänglichen Literaturquellen bleibt ein zusätzlicher manueller Abgleich sinnvoll. Ein institutionelles Ähnlichkeitswerkzeug kann weitere Kandidaten liefern; auch dessen Treffer müssten im Kontext bewertet werden.

Eine falsche Quellenzusammenfassung ist wissenschaftlich zu korrigieren, aber nicht automatisch ein Plagiat. Umgekehrt würde ein Quellenverweis allein eine längere wörtliche oder sehr enge übersetzte Übernahme nicht automatisch korrekt kennzeichnen. Diese beiden Fragen wurden hier getrennt bewertet.

## Bewertung nach Kapitel für diesen Prüfauftrag

| Teil | Befund zur Beleglage | Nächster Schritt |
|---|---|---|
| Einleitung | Grundlagen und zentrale Zahlen überwiegend belegt; Forschungslücke zu weit. | F06 und Herkunft des Wortindex-Verfahrens präzisieren. |
| Grundlagen | Gute Belegdichte, Formeln und Bildvorlagen zugeordnet; einige technische Aussagen zu absolut. | F08 und die technischen Präzisierungen einarbeiten. |
| Verwandte Arbeiten | Hauptzahlen überwiegend korrekt; mehrere wichtige Fehler bei Trainingsbedingungen, Ressourcen und Synthese. | F01–F06, F10–F11 und Übersichtstabelle zuerst korrigieren. |
| Methodik | Eigene Entscheidungen meist erkennbar und beschrieben; vereinzelt unpassende oder zu allgemeine Belege. | Bootstrap-Quelle und konkrete Software-/Modellquellen ergänzen. |
| Ergebnisse/Diskussion | Eigene Ergebnisse brauchen keine fremde Ergebnisquelle; Literaturvergleich vorhanden, Formateffekte aber überinterpretiert. | F07 und die kausalen Formulierungen überarbeiten. |
| Fazit/Ausblick | Überwiegend aus eigener Untersuchung abgeleitet; LogitMatch falsch charakterisiert. | F09; anschließend Konsistenz mit der bereinigten Diskussion prüfen. |
| Anhang/Erklärung | Messdaten- und Konfigurationsbezug nachvollziehbar; KI-Angabe unvollständig. | F13 und genaue Herkunft/IDs von Fehlerbeispielen vor Abgabe prüfen. |

Dies ist eine Beleg- und Übernahmeprüfung, keine neue Gesamtnote. Es besteht kein Anlass, aus den gefundenen Zitierfehlern eine erfundene Note oder eine automatische Notenobergrenze abzuleiten.

## Konkreter Arbeitsumfang

1. **3–4 Stunden:** F01–F06 korrigieren; Tabelle der verwandten Arbeiten und Forschungslücke gemeinsam aktualisieren.
2. **2–3 Stunden:** F07–F10 und die betroffenen Aussagen in Grundlagen, Diskussion und Fazit bereinigen.
3. **1–2 Stunden:** Primärquelle zum Wortindex-Verfahren, Methodenbeleg, genauere Tabellen-/Abschnittsangaben und feste arXiv-Versionen ergänzen.
4. **1–2 Stunden:** Sieben Zugriffs-/Nachweisgrenzen soweit verfügbar schließen und Hilfsmittelangabe anhand der tatsächlichen Nutzung vervollständigen.
5. **1 Stunde:** Gesamtdurchlauf auf wiederholte Behauptungen, Literaturtabelle, Querverweise und PDF-Build.

**Schätzung: 8–12 konzentrierte Stunden**, sofern die eingeschränkt erreichbaren Quellen verfügbar sind. Neue Trainingsläufe sind für diese Korrekturen nicht erforderlich: Wo ein isolierter Effekt nicht untersucht wurde, muss die Aussage angemessen begrenzt werden. Experimente wären erst nötig, wenn weitergehende kausale Behauptungen erhalten bleiben sollen.
