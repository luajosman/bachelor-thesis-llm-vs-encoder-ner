# Prüfbericht: Quellenverwendung, Literaturverzeichnis und Übernahmerisiken

Stand: 21.11.2026. Es wurde nichts an der Arbeit, an `references.bib` oder an anderen Dateien der Arbeit geändert. Dieser Bericht ist die einzige neue Datei.

## 0. Geprüfte Fassung, verfügbare Quellen und Vorgehen

### Geprüfte Fassung

- **Quellstand:** Branch `lektorat-2026-11-25`, Commit `b4898f1` (12.11.2026, 15:41), identisch mit `origin/lektorat-2026-11-25`. Das ist der neueste Stand der Arbeit im Repository.
- **Andere Stände, nicht geprüft:** `main` (`53415e6`) liegt davor. `korrekturen-2026-11-25` (`a70c01b`, nicht gepusht, nicht eingemischt) enthält zwei Textänderungen in Kapitel 5 und das Datum der Erklärung, aber keine Änderung an Zitaten oder am Verzeichnis. `fassung-lang-2026-11-12` ist die ältere Langfassung.
- **PDF:** `thesis/main.pdf` (12.11., 14:49) ist älter als der letzte Commit. Ich habe den Stand `b4898f1` deshalb in einer Arbeitskopie neu gebaut: 111 Seiten, keine offenen Verweise oder Zitate, keine Biber-Warnung. Der Text des Neubaus ist mit `thesis/main.pdf` zeichengleich; der letzte Commit hat nur das Deckblatt-Layout verschoben. `thesis/main_long12112026.pdf` ist die Langfassung und wurde nicht geprüft.
- **Seitenangaben** in diesem Bericht sind die arabischen Seitenzahlen der Arbeit (Haupttext S. 1–77, Literaturverzeichnis S. 78–86, Anhang ab S. 87).
- **Bibliografie:** einzige Datei ist `thesis/references.bib` (86 Einträge). Gesetzt wird mit biblatex, Stil `apa`, deutsche Sprachanpassung; `\cite` ist auf `\parencite` umgelegt. Das Verzeichnis nimmt nur zitierte Einträge auf.
- **Vollständigkeit:** Die gesamte Arbeit lag vor und wurde gelesen: Erklärung, Abstract, Kapitel 1 bis 6, Anhang A.1 bis A.3, alle Tabellenlegenden und Bildunterschriften. Fußnoten mit Quellenangaben gibt es nicht. Die erzeugten Tabellenkörper (`generated/`), die Abbildungsdateien und das Abkürzungsverzeichnis enthalten keine Zitate; ich habe sie nur daraufhin durchsucht. Eigene Messwerte der Arbeit waren nicht Gegenstand dieser Prüfung.

### Verfügbare Originalquellen

Von den 85 zitierten Einträgen lagen vor:

| Art | Anzahl | Bemerkung |
|---|---|---|
| Volltext in der zitierten Fassung | 56 | darunter Modellkarten, Dokumentationsseiten und arXiv-Einträge, die als solche zitiert sind; Zhao et al. nur als arXiv v19 mit Zeitschriftenverweis; Mayhew et al. lag vor, wurde aber nur über Semin et al. gegengeprüft |
| Volltext, aber andere Fassung als zitiert | 18 | 17-mal arXiv-Fassung statt Konferenz- oder Zeitschriftenfassung: `he2020deberta`, `he2021debertav3`, `hu2021lora`, `han2024parameter`, `liu2021crossner`, `li2018survey`, `zhou2024universalner`, `sainz2024gollie`, `obeidat2025llm`, `hoffmann2022training`, `lu2024llmstruggle`, `bengio2013representation`, `wei2022emergent`, `holtzman2020curious`, `clark2020electra`, `yang2025gated`, `goldberg2016primer`; dazu `lafferty2001crf` als Vorlesungskopie |
| nur Auszug | 5 | Jurafsky & Martin (Kapitel 1 und 18), van Rijsbergen (Kapitel 7), OntoNotes (Katalogseite), seqeval (README und PyPI), Schuster & Nakajima (PDF nur dekodiert lesbar) |
| nur bibliografisch | 2 | Hripcsak & Rothschild (Volltext nicht abrufbar), PEFT (Release-Metadaten) |
| nicht verfügbar | 4 | Manning & Schütze 1999, Manning et al. 2008, Efron & Tibshirani 1993, Qwen3.5-Blog (Seite nur mit JavaScript lesbar) |

Die Texte stammen aus dem Ordner `thesis/build/literature_audit_2026_11_06/` (Abruf vom 6.11.) und aus eigenen Abrufen vom 21.11. (ACL Anthology, arXiv, Hugging Face, PMC, Dokumentationsseiten).

### Vorgehen

1. Abgleich aller 313 Zitierbefehle mit den Schlüsseln der Literaturdatei (Skript).
2. Abgleich der bibliografischen Angaben mit Crossref, DataCite, arXiv, ACL Anthology, PubMed und den Seiten von JMLR und NeurIPS. OpenReview und die ACM Digital Library verweigern automatische Abrufe; DBLP war nicht erreichbar.
3. Prüfung jeder Zitatstelle gegen den Quelltext. Diese Einzelprüfung haben fünf nur lesende Prüf-Agenten kapitelweise vorgenommen. Alle Befunde der Stufen „Wichtig“ in Abschnitt B habe ich selbst am Quelltext nachgeprüft; zwei Agentenurteile habe ich dabei verworfen (siehe B, „Geprüft und nicht beanstandet“). Die Urteile „passend“ der Agenten habe ich nicht einzeln wiederholt.
4. Suche nach wörtlichen Überschneidungen: Folgen von 8, 6 und 5 gleichen Wörtern zwischen der Arbeit und 97 Quelltextdateien.
5. Durchsicht des gesetzten PDF auf Zitierform und Verzeichnis.

### Einschränkungen

- Wo nur die arXiv-Fassung vorlag, sind Abschnitts-, Tabellen- und Anhangsnummern nur für diese Fassung bestätigt.
- Tabellenwerte wurden aus Textfassungen der PDFs gelesen. Bei Tabellen mit zerfasertem Text ist die Zuordnung von Zeile und Spalte rekonstruiert.
- Aussagen, die auf Abbildungen der Quellen beruhen, konnten nicht geprüft werden (betrifft Lu et al., Abb. 3).
- Ein Abgleich mit einer Ähnlichkeitsdatenbank hat nicht stattgefunden. Es wurde kein Text der Arbeit an einen externen Plagiats- oder Suchdienst gegeben.

## A. Kurzfazit

**Literaturverzeichnis.** Das Verzeichnis ist in gutem Zustand. Jeder zitierte Schlüssel ist vorhanden, es gibt keine Dubletten, und für 56 Einträge stimmen Autoren, Jahr, Titel, Ort, Band und Seiten mit einem Register überein. Zwei Einträge brauchen eine inhaltliche Korrektur: ein vertauschter Autorenname (Ding et al.) und ein Preprint, der inzwischen begutachtet erschienen ist und in der Arbeit 23-mal zitiert wird (Zhan et al.). Dazu kommen Kleinigkeiten bei Datumsangaben und Online-Einträgen.

**Tragen die Quellen die Aussagen?** Überwiegend ja. Keine einzige aus der Literatur wiedergegebene Zahl ist falsch; alle geprüften Werte (über 80 Zahlen in Kapitel 1, 3 und 5) stehen so in den Quellen. Die Schwächen liegen woanders: In 13 Befunden (W3 bis W15, rund 20 Textstellen) ist die Formulierung stärker oder genauer als die Quelle, oder eine Quelle steht als Beleg bei einer Aussage, die sie nicht enthält. Am deutlichsten ist das in Kapitel 4, wo die Vergleichsstudie von Keraghel et al. zweimal für Dinge zitiert wird, die sie nicht behandelt, und wo drei Quellen eine Aussage über „komplementäre Kapazität“ stützen sollen, die in keiner steht.

**Übernahmerisiken.** Ich habe keinen Anhaltspunkt für eine nicht gekennzeichnete Übernahme gefunden. Die Arbeit enthält keine wörtlichen Zitate aus der Literatur; alles ist Paraphrase mit Beleg. Die Wortfolgensuche ergab keine inhaltlichen Treffer. Einige deutsche Sätze folgen einem englischen Quellsatz eng, sind aber jeweils der Quelle zugeschrieben (Abschnitt F). Eine Garantie ist das nicht: Geprüft wurde nur gegen die zitierten Quellen, nicht gegen fremde Texte, die nicht im Verzeichnis stehen.

**Nicht geprüft.** Vier Quellen ganz (drei Lehrbücher, ein Blog), eine nur bibliografisch (Hripcsak & Rothschild), 18 nur in einer anderen als der zitierten Fassung (17-mal arXiv, einmal Vorlesungskopie). Elf Zitatstellen hängen an den drei Lehrbüchern und an Hripcsak & Rothschild und sind damit inhaltlich ungeprüft.

## B. Befunde nach Gewicht

### Kritisch

Kein Befund. Keine zentrale Aussage der Arbeit steht ohne tragfähigen Beleg, und es gibt keinen begründeten Verdacht auf eine nicht gekennzeichnete Übernahme. Die eigenen Ergebnisse und Schlussfolgerungen hängen an keinem der unten genannten Punkte.

### Wichtig

Für jeden Befund: Fundstelle, Quelle, Problem, Vorschlag, Sicherheit, und ob der Originaltext geprüft wurde.

**W1. Autorenname bei Ding et al. vertauscht**
- Fundstelle: Literaturverzeichnis S. 79, Eintrag `ding2024gner`; im Text Abschnitt 3.2.3 (S. 24) und Tabelle 3.1 (S. 29).
- Bisher: `Bowen, Yan` (gesetzt als „Bowen, Y.“).
- Geprüft: Das PDF der ACL Anthology nennt die Autoren als „Yuyang Ding, Juntao Li, Pinzheng Wang, Zecheng Tang, Bowen Yan, Min Zhang“, mit der Mailadresse `yanbw@mail.tsinghua.edu.cn`; arXiv 2402.16602 führt ebenfalls „Bowen Yan“. Die Metadaten von ACL Anthology und Crossref führen den Namen vertauscht („Bowen, Yan“).
- Vorschlag: `Yan, Bowen`.
- Sicherheit: hoch. Original geprüft: ja (PDF).

**W2. Zhan et al. ist inzwischen begutachtet erschienen**
- Fundstelle: Eintrag `zhan2026generativeNER` (S. 86); 23 Zitatstellen in Kapitel 1, 3, 4, 5 und 6. Abschnitt 3 der Arbeit sagt, begutachtete Fassungen würden bevorzugt (S. 19).
- Bisher: arXiv-Preprint, Version 3 vom 9.10.2026, mit dem Hinweis „zur Veröffentlichung in Findings of EMNLP 2026 angenommen (Stand 7. November 2026)“.
- Geprüft: Crossref und ACL Anthology führen den Beitrag in *Findings of the Association for Computational Linguistics: EMNLP 2026*, S. 36475–36493, DOI `10.18653/2026.findings-emnlp.1706`, URL `https://aclanthology.org/2026.findings-emnlp.1706/`, Autoren unverändert.
- Vorschlag: Eintrag auf die Findings-Fassung umstellen und die Fundstellenangaben (Tab. 3; Anh. D, Tab. 11; Anh. G, Tab. 13) an dieser Fassung bestätigen. Der Satz in Kapitel 3, 2026 seien nur „zwei Preprints“ einbezogen, wäre anzupassen.
- Sicherheit: hoch für die Veröffentlichung. Für die Fundstellen: Die Kennzahlen 93,85, 93,04, 4,02, 0,14 und 0,48 sowie „Table 11“ und „Table 13“ stehen auch in der Verlagsfassung; „Table 3“ konnte ich in der zweispaltigen Textfassung nicht sicher zuordnen.
- Original geprüft: arXiv v3 vollständig, Verlagsfassung nur auf diese Kennzahlen.

**W3. Keraghel et al. als Beleg für Mikro-F₁ als primäre Kennzahl**
- Fundstelle: Abschnitt 4.5.1, S. 45: „Primäre Kennzahl ist der mikrogemittelte F₁-Wert (Manning et al., 2008; Keraghel et al., 2024)“.
- Problem: Keraghel et al. berichten ihre Ergebnisse als Makro-F₁ („Table 5 provides a comparison … based on their macro-averaged F1 scores“) und erklären Mikro-Mittelung nur allgemein. Die Arbeit selbst gibt den Keraghel-Wert in 1.2 und 3.1.2 richtig als Makro-F₁ an.
- Vorschlag: Keraghel hier streichen. Für die Definition genügt Manning et al.; dass Mikro-F₁ die übliche Hauptkennzahl ist, belegt die CoNLL-Aufgabe (`sang2003conll`).
- Sicherheit: hoch. Original geprüft: ja.

**W4. Keraghel et al. als Beleg für das Label-Alignment**
- Fundstelle: Abschnitt 4.3.2, S. 35: erstes Subwort erhält das Label, Folge-Subwörter und Sondertokens den Wert −100 (Devlin et al., 2019; Keraghel et al., 2024). Ähnlich 4.1, S. 31 (Encoder mit linearem Klassifikationskopf).
- Problem: Keraghel et al. behandeln weder Subwörter noch den Wert −100. Devlin et al. belegen nur, dass die Repräsentation des ersten Subworts klassifiziert wird. Der Wert −100 und das Ignorieren in Verlust und Evaluation sind eine Konvention der Transformers-Bibliothek.
- Vorschlag: Keraghel an beiden Stellen streichen; für die Konvention `wolf2020transformers` oder die Dokumentation der Bibliothek nennen, oder den Satz als Beschreibung der eigenen Umsetzung ohne Fremdbeleg stehen lassen.
- Sicherheit: hoch. Original geprüft: ja.

**W5. „Komplementäre Kapazität“ unterschiedlicher Linearschichten**
- Fundstelle: Abschnitt 4.4.5, S. 43: Die Einbeziehung der Feed-Forward-Schichten „folgt Befunden, nach denen Adapter in unterschiedlichen Linearschichten komplementäre Kapazität bereitstellen“ (Dettmers et al., 2023; Han et al., 2024; L. Wang et al., 2024).
- Problem: Die Aussage steht in keiner der drei Quellen. Dettmers et al. schreiben, LoRA auf allen Schichten sei nötig, um die 16-Bit-Leistung zu erreichen („Using LoRA on all transformer layers is critical to match 16-bit performance“; „add LoRA modules on all linear layers“). In beiden Übersichtsarbeiten war kein passender Befund zu finden.
- Vorschlag: auf das formulieren, was Dettmers et al. zeigen, und die beiden Übersichtsarbeiten hier streichen. Damit wäre `wang2024parameter` im Verzeichnis unbenutzt.
- Sicherheit: hoch für Dettmers; mittel für die Übersichtsarbeiten (Stichwortsuche in 170 und 180 Tsd. Zeichen, kein vollständiges Lesen).
- Original geprüft: ja (Dettmers: Verlagsfassung; Han: arXiv; Wang: arXiv v1).

**W6. CoNLL-2003 „bis heute der am häufigsten verwendete NER-Benchmark“**
- Fundstelle: Abschnitt 2.8, S. 16, Beleg `li2018survey`.
- Problem: Die Quelle sagt nur, viele neuere Arbeiten berichteten auf CoNLL03 und OntoNotes. Ein Superlativ steht dort nicht, und „bis heute“ kann eine Übersicht mit Stand 2020 nicht tragen.
- Vorschlag: „einer der am häufigsten verwendeten“ ohne „bis heute“, oder einen aktuellen Beleg ergänzen.
- Sicherheit: hoch für die arXiv-Fassung; die TKDE-Fassung lag nicht vor.

**W7. Lu et al.: „bleibt aber auf allen Typen darunter“**
- Fundstelle: Abschnitt 3.3, S. 26.
- Problem: Der Text der Quelle sagt nur, das mit LoRA angepasste Modell komme dem feinjustierten BioClinicalBERT nahe („closely rivaling“, „approaching the levels of“). Werte je Typ stehen nur in Abbildung 3, die ich nicht auswerten konnte.
- Vorschlag: an der Abbildung der AMIA-Fassung prüfen. Lässt sich „auf allen Typen“ dort nicht ablesen, auf die Textaussage der Autoren zurückgehen.
- Sicherheit: mittel. Original geprüft: Text der arXiv-Fassung ja, Abbildung nein.

**W8. Erwartung E3 nennt „Zero-Shot“, die Belege arbeiten mit Beispielen im Prompt**
- Fundstelle: Abschnitt 3.5, S. 30: „E3 (zu FF1): Im Zero-Shot-Regime, also ohne Anpassung der Gewichte, bleiben per Prompting genutzte LLMs deutlich hinter feinjustierten Modellen zurück (Keraghel et al., 2024; Tan et al., 2023).“ Folgestelle 6.1, S. 73 („das stützt Erwartung E3“).
- Problem: Bei Keraghel et al. enthält der Prompt Beispiele; bei Tan et al. ist die zitierte beste Variante (14,78) die mit Demonstrationsbeispielen. Abschnitt 2.5 der Arbeit definiert Zero-Shot dagegen als Prompt ohne Beispiele. Kapitel 1 und 3.1.2 geben beide Quellen richtig wieder; nur E3 setzt „Zero-Shot“ mit „ohne Anpassung der Gewichte“ gleich.
- Vorschlag: E3 als „ohne Anpassung der Gewichte, auch mit Beispielen im Prompt“ fassen und im Zusatz sagen, dass die Belege Few-Shot-Prompts verwenden. Der Schluss wird dadurch eher stärker.
- Sicherheit: hoch. Original geprüft: ja.

**W9. Sammelaussage zu den direkten Vergleichen ist weiter gefasst als die Quellen**
- Fundstelle: Abschnitt 3.3, S. 27: „Drei der vier Arbeiten stimmen in einem Punkt überein: Mit Adaptern feinjustierte LLMs erreichen auf etablierten Benchmarks das Niveau feinjustierter Encoder oder übertreffen es leicht.“
- Problem: Bei Zhan et al. liegt das beste LLM auf GENIA 4,02 Punkte zurück (die Arbeit nennt das eine Seite vorher selbst). Bei Dukić & Šnajder gilt der Vorsprung für CoNLL-2003-NER; auf zwei weiteren Aufgaben derselben Studie liegt RoBERTa-large vorn (Rest14 0,474 gegenüber 0,402).
- Vorschlag: auf „flache, allgemeine NER-Benchmarks wie CoNLL-2003“ eingrenzen.
- Sicherheit: mittel bis hoch. Original geprüft: ja.

**W10. Ravikumar et al.: „mehr als 40 %“**
- Fundstelle: Abschnitt 3.2.1, S. 22. Der Satz folgt unmittelbar auf die Angabe, ausgewertet werde unter anderem OntoNotes 5.0.
- Problem: Die 40 % nennt die Quelle für den biomedizinischen Datensatz BC5CDR („particularly pronounced for BC5CDR, reaching over 40% in certain instances“); auf OntoNotes sind die Unterschiede klein. Die Quelle lässt außerdem offen, ob Prozent oder Prozentpunkte gemeint sind.
- Vorschlag: BC5CDR nennen.
- Sicherheit: hoch. Original geprüft: ja.

**W11. GoLLIE und UniversalNER: Deutung als Befund der Quellen ausgegeben**
- Fundstelle: Abschnitt 5.7.3, S. 69: Das Muster ähnle den Beobachtungen bei GoLLIE (Abschn. 5.2) und UniversalNER (Abschn. 5.2), „wo größere Backbones nur wenig hinzugewinnen und der Gewinn durch die Anpassung den Gewinn durch die Größe übersteigt“.
- Problem: Der erste Teil ist belegt (GoLLIE 55,3/56,0/57,2; UniNER 41,7/43,4). Der Vergleich „Anpassung größer als Größe“ steht in keiner der beiden Quellen; er ist eine eigene Ableitung aus den Tabellenwerten und bei UniversalNER nicht ohne Weiteres herzuleiten. Abschnitt 3.4 formuliert dasselbe für GoLLIE vorsichtiger und nur für GoLLIE.
- Vorschlag: als eigene Ableitung kennzeichnen und auf GoLLIE beschränken; die beiden Abschnittsangaben „5.2“ an den ICLR-Fassungen bestätigen.
- Sicherheit: mittel. Original geprüft: arXiv-Fassungen.

**W12. Wei et al. (2022) und das „Befolgen komplexer Formatvorgaben“**
- Fundstelle: Abschnitt 5.7.3, S. 70.
- Problem: Die Quelle beschreibt, dass das Befolgen von Anweisungen nach Instruction-Tuning erst bei großen Modellen die Leistung hebt, und schränkt das selbst ein (kleinere T5-Modelle zeigen es ebenfalls). Von Formatvorgaben ist nicht die Rede.
- Vorschlag: „Befolgen von Anweisungen“ schreiben und die Übertragung auf Formatvorgaben als eigene Vermutung kennzeichnen. Der Satz ist schon als Vermutung markiert.
- Sicherheit: hoch für die arXiv-Fassung.

**W13. Mehrere Aussagen zu Qwen3.5 stehen nicht in den zitierten Modellkarten**
- (a) Abschnitt 2.5, S. 12: Dialogformat mit den Rollen System, Nutzer, Assistent und Chat-Template mit Steuertokens, belegt mit der 27B-Karte. Die Karte zeigt nur API-Beispiele mit der Rolle `user`. Passender Beleg wäre der technische Bericht zu Qwen3 (`qwen2025qwen3`), der Chat-Template und Steuertokens beschreibt.
- (b) Abschnitt 2.5, S. 13: „Modelle ohne Mixture-of-Experts-Komponenten“. Die Karten der drei Modelle weisen im Schichtaufbau ein dichtes Feed-Forward-Netz aus, sagen das aber nicht ausdrücklich, und ihre Merkmalsliste nennt für die Familie „sparse Mixture-of-Experts“. Die Aussage ist eine richtige eigene Ableitung aus den Architekturangaben und sollte so formuliert sein.
- (c) Abschnitt 4.2.2, S. 33: Qwen3.5 „baut auf der Architektur- und Trainingstradition von Qwen2.5 und Qwen3 auf“, belegt mit beiden technischen Berichten und dem Blog. Die Berichte können über Qwen3.5 nichts sagen; der Blog war nicht lesbar.
- (d) Abschnitt 4.2.2, S. 34: „Grouped-Query-Attention“ und „Ausgabe-Gating“ stehen nicht in den Karten (dort: Kopfzahlen für Q und KV sowie „Gated Attention“). Der Satz beruft sich auf die Konfigurationsdateien, zitiert aber die Karten.
- Vorschlag: (a) Beleg tauschen; (b) und (d) als Ableitung aus Karte und Konfigurationsdatei der angegebenen Revision formulieren; (c) im Browser am Blog prüfen oder auf „Nachfolger von Qwen3“ verkürzen.
- Sicherheit: hoch für (a), (b), (d); (c) nicht beurteilbar. Original geprüft: Karten ja (zitierte Revisionen), Blog nein.

**W14. Dagdelen et al. für Positionsangaben und Nachverarbeitung**
- Fundstelle: Abschnitt 2.5, S. 13 („strukturierte Liste im JSON-Format mit Angaben zu Position und Typ“); Abschnitt 4.1, S. 33 (JSON-Array, das eine regelbasierte Nachverarbeitung in eine IOB2-Sequenz überführt; Dagdelen et al. und Semin et al.).
- Problem: Dagdelen et al. geben Entitäten als Zeichenketten in JSON-Objekten aus, ohne Positionen, und kennen keine Rückführung auf Tags. Semin et al. bestimmen Positionen beim Matching nachträglich per Zeichenkettenabgleich. Die beschriebene Pipeline ist eine Eigenleistung der Arbeit.
- Vorschlag: in 2.5 „mit Angaben zum Typ, je nach Verfahren auch zur Position“ und Semin et al. als zweiten Beleg; in 4.1 die Quellen als Hintergrund für JSON-Ausgaben kennzeichnen oder den Beleg nach dem ersten Satzteil setzen.
- Sicherheit: hoch. Original geprüft: ja.

**W15. IOB2 mit zwei Quellen belegt, die IOB1 beschreiben**
- Fundstelle: Abschnitt 4.5.1, S. 45: strikte Metriken „nach IOB2-Schema“ (Manning et al., 2008; Tjong Kim Sang & De Meulder, 2003; Ramshaw & Marcus, 1995).
- Problem: Abschnitt 2.2 der Arbeit stellt selbst fest, dass Ramshaw & Marcus das Schema einführten, das später IOB1 hieß, und dass die CoNLL-2003-Daten im IOB1-Schema vorliegen. Die Bezeichnung IOB2 belegt dort `tjong1999representing`.
- Vorschlag: für IOB2 `tjong1999representing` zitieren; `sang2003conll` nur für die exakte Übereinstimmung.
- Sicherheit: hoch. Original geprüft: ja.

**W16. Publikationsstand zweier weiterer Preprints**
- `wang2024parameter` (S. 84) ist als Zeitschriftenartikel erschienen: Wang, L., Chen, S., Jiang, L., Pan, S., Cai, R., Yang, S., & Yang, F. (2025). Parameter-efficient fine-tuning in large language models: a survey of methodologies. *Artificial Intelligence Review*, 58(8), Artikel 227. DOI `10.1007/s10462-025-11236-4` (Crossref). Wird der Beleg nach W5 behalten, wäre diese Fassung zu zitieren. Sicherheit: hoch.
- Zu `keraghel2024recent` sagt Abschnitt 3.1.2 (S. 20), die Studie liege „bislang nur als Preprint“ vor. Crossref führt von zwei der drei Autoren eine Konferenzarbeit ähnlichen Themas: Keraghel, I., & Nadif, M. (2025). Named Entity Recognition in the Era of Large Language Models: A Comparative Study. *2025 International Conference on Advanced Machine Learning and Data Science (AMLDS)*, 617–623. DOI `10.1109/amlds63918.2025.11159373`. Ob sie die zitierte Tabelle 5 enthält, konnte ich nicht prüfen. Vorschlag: nachsehen; andernfalls den Satz auf „die hier zitierte Fassung ist ein Preprint“ beschränken. Sicherheit: niedrig, nur Metadaten.

### Geringfügig

| Nr. | Fundstelle | Quelle | Befund | Vorschlag | Original geprüft |
|---|---|---|---|---|---|
| G1 | 1.1, S. 1 | Jurafsky & Martin; Keraghel et al. | „Relationsextraktion, Aufbau von Wissensgraphen, Frage-Antwort-Systeme“: Frage-Antwort-Systeme sind gedeckt, die beiden anderen nicht wörtlich (Jurafsky: „linking text to … structured knowledge sources“) | `li2018survey` ergänzen (nennt „knowledge base construction“) oder Wortlaut angleichen | ja (Jurafsky nur Kap. 1, 18) |
| G2 | 1.1, S. 1 | `he2021debertav3` | „einige hundert Millionen Parameter“: Der Aufsatz nennt Backbone-Größen (86 und 304 Mio.); die Gesamtzahlen stehen in den Modellkarten | Modellkarten mitzitieren | ja (arXiv) |
| G3 | 1.1, S. 2 | `devlin2019bert` | „in einem einzigen Vorwärtsdurchlauf“ ist eine Folgerung, kein Befund der Quelle | Beleg vor den Halbsatz ziehen oder Obeidat et al. nennen („parallel simultaneous token classification“) | ja |
| G4 | 2.2, S. 8 | `lafferty2001crf` | „Lange Zeit bestimmten CRF den Stand der Technik“ steht nicht in der Quelle | zusätzlich `li2018survey` oder Satz teilen | ja (Kopie) |
| G5 | 2.4, S. 11 | DeBERTa-Modellkarten | 512 Positionen stehen in `config.json`, nicht in den Karten; für large nicht eingesehen | Konfigurationsdatei der Revision nennen | ja (base) |
| G6 | 2.5, S. 12 | `brown2020language` | „darauf trainiert, das nächste Token vorherzusagen“ steht dort nicht wörtlich („autoregressive language model“) | Zhao et al. mitzitieren | ja |
| G7 | 2.5, S. 12 | `ouyang2022instructgpt` | Der Begriff „Post-Training“ kommt in der Quelle nicht vor (er steht in den Qwen-Karten) | Begriff nicht der Quelle zuschreiben | ja |
| G8 | 2.6, S. 14 | `dettmers2022llmint8` | LLM.int8() normiert zeilen- bzw. spaltenweise; blockweise Konstanten gehören zu QLoRA | `dettmers2023qlora` mitzitieren oder „Block“ durch „Gruppe von Gewichten“ ersetzen | ja |
| G9 | 2.9, S. 17 | `segura2013semeval` | „nur den Typ“: Der Typ-Modus verlangt zusätzlich eine Überlappung | „den Typ bei überlappenden Grenzen“ | ja |
| G10 | 3.2.1, S. 23 | `lu2024padellm` | „bei unveränderter Erkennungsleistung“; Quelle: „on par with the state-of-the-art approaches“ | „bei vergleichbarer Erkennungsleistung“ | ja |
| G11 | 3.2.2, S. 23 | `ashok2023promptner` | „weniger als 2 %“; Quelle: „only 1%−2% of the data“ | „1 bis 2 %“ | ja |
| G12 | 3.3, S. 26 | Obeidat et al.; Lu et al. | Beide Quellen schreiben „BIO“, nicht „IOB2“ | unkritisch (2.2 erklärt die Gleichsetzung); bei Lu et al. „BIO-Tags“ übernehmen | ja (arXiv) |
| G13 | Tab. 3.1, S. 29 | Semin; Li et al. 2023; Dukić | „Ausgabetokens“ sind bei Semin geschätzt (Anh. B.2), nicht gemessen; bei CodeIE sind nur die Full-Data-Werte aus UIE übernommen; bei Dukić fehlt CoNLL-Chunking | Zellen präzisieren | ja |
| G14 | 1.2, 3.3, 3.5, 5.7.2, 6.1 (S. 2, 26, 30, 69, 74) | `obeidat2025llm` | Die Faktoren 40 (JNLPBA) und knapp 220 (BioRED) stehen im Fließtext der Abschnitte III und IV; Tab. IV liefert die Zeiten | Fundstelle um den Abschnitt ergänzen | ja (arXiv) |
| G15 | 3.4, S. 27 | `bogdanov2024nuner` | NuNER wird mit UniNER-7B verglichen, nicht mit 13B | „7 Mrd. (NuNER) bzw. 7 bis 13 Mrd. (GLiNER)“ | ja |
| G16 | 3.5, S. 28 | `zaratiana2024gliner` | 10 000 Testbeispiele stimmen (Tab. 6); die Herkunft aus der InstructUIE-Aufbereitung ist nur indirekt belegt („following previous works“) | „nach eigenen Angaben dem Benchmark von Wang et al. und Zhou et al. folgend“ | ja |
| G17 | 4.2.1, S. 33 | Keraghel et al. | „dienen häufig als starke Baselines“ steht dort nicht | zweiten Halbsatz streichen oder eigenen Beleg | ja |
| G18 | 4.4.5, S. 42 | Obeidat et al. | als Beleg für „Adapter senken Speicher- und Rechenbedarf gegenüber vollständigem Fine-Tuning“ schwach: kein solcher Vergleich in der Quelle | Obeidat hier streichen | ja |
| G19 | 4.4.5, S. 43 | `dettmers2022llmint8` | als Beleg für das Laden im NF4-Format: die Quelle behandelt 8 Bit und nennt nur die Bibliothek | als Bibliotheksbeleg lassen, NF4 mit QLoRA belegen | ja |
| G20 | 4.5.3, S. 48 | Tedeschi; X. Wei et al.; Dagdelen | Belege an den Fehlerklassen tragen wenig: Tedeschi definiert das Inventar, nicht „halluzinierte Kategorien“; ChatIE sagt nichts zu Fehlerklassen | Klassen als eigene Festlegung kennzeichnen, Belege reduzieren | ja |
| G21 | 4.1, 4.3.1 (S. 32, 34) | `tedeschi2022multinerd` | Dass es offizielle Splits gibt, ergibt sich aus dem veröffentlichten Datensatz, nicht klar aus dem Aufsatz | Datensatzkarte mitzitieren | ja |
| G22 | 2.9, S. 17 | van Rijsbergen | Die Quelle definiert das E-Maß; „F-Maß“ steht dort nicht | „geht auf das E-Maß … zurück“ (der Eintrag im Verzeichnis sagt das schon) | ja (Kap. 7) |

### Geprüft und nicht beanstandet

Diese Punkte waren in einer der Teilprüfungen angemerkt und haben sich bei der Nachprüfung erledigt:

- **Tam et al.:** Der Titel im Verzeichnis stimmt mit dem PDF überein. Nur die Metadaten von ACL Anthology und Crossref führen einen anderen Wortlaut.
- **TRL-Dokumentation:** `assistant_only_loss` ist dort ausdrücklich mit „defaults to False“ dokumentiert.
- **Obeidat et al., Faktor 40 bis 220:** Aus Tab. IV ergeben sich für „bester Encoder gegenüber bestem LLM“ Faktoren von etwa 38 bis 219. Die Spanne ist gedeckt; es bleibt nur der Hinweis G14 zur Fundstelle.
- **Hoffmann et al.:** Der Titel im Verzeichnis entspricht den NeurIPS-Proceedings; der abweichende Preprint-Titel ist im Eintrag vermerkt.
- **Tan et al., 92,77:** Der Wert gehört zum eigenen System der Autoren. Die Arbeit sagt das in 3.1.2 ausdrücklich.

## C. Quellen-Audit

Lesehilfe: „Volltext, aber arXiv-Fassung“ heißt, dass die zitierte Verlags- oder Konferenzfassung nicht vorlag. „Metadaten“ nennt das Register, mit dem der Eintrag verglichen wurde. Die Kürzel W und G verweisen auf Abschnitt B, F auf die formalen Punkte in Abschnitt E.

| Schlüssel | Literatureintrag (gekürzt auf Erstautor; vollständig im Verzeichnis S. 78–86) | Fundstellen (Abschnitte) | Gestützte Aussagen | Art der Prüfung | Metadaten | Ergebnis | Korrektur oder offene Prüfung |
|---|---|---|---|---|---|---|---|
| `cawley2010selectionbias` | Cawley & Talbot (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. Journal of Machine Learning Research, 11(70), 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html | 4.3.3 | Auswahlentscheidungen können optimistisch verzerren | Volltext (JMLR-PDF) | JMLR-Seite: stimmt | passend | – |
| `lakens2017equivalence` | Lakens (2017). Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses. Social Psychological and Personality Science, 8(4), 355–362. DOI 10.1177/1948550617697177 | 4.5.1 | fehlende Signifikanz belegt keine Gleichwertigkeit | Volltext (PMC) | Crossref: stimmt | passend | – |
| `jurafsky2024speech` | Jurafsky & Martin (2026). Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition, with Language Models. https://web.stanford.edu/~jurafsky/slp3/ | 1.1 (2×), 2.1 | Definition NLP; Anwendungen von NER | teilweise (Kap. 1 und 18 der Online-Fassung) | Autorenseite: Fassung 19.08.2026 stimmt | teilweise passend | „Wissensgraphen“ und „Relationsextraktion“ dort nicht wörtlich (G1) |
| `manning1999foundations` | Manning & Schütze (1999). Foundations of Statistical Natural Language Processing. MIT Press. https://nlp.stanford.edu/fsnlp/ | 1.1, 2.1 (2×) | Definition NLP; handentworfene Merkmale | nicht geprüft (kein Text) | nicht über Register geprüft | unklar | am Buch nachsehen; drei Stellen ungeprüft |
| `goldberg2016primer` | Goldberg (2016). A Primer on Neural Network Models for Natural Language Processing. Journal of Artificial Intelligence Research, 57, 345–420. DOI 10.1613/jair.4992 | 2.1 | Wortvektoren aus Kontexten | Volltext, aber arXiv-Fassung | Crossref: stimmt | passend | – |
| `keraghel2024recent` | Keraghel et al. (2024). Recent Advances in Named Entity Recognition: A Comprehensive Survey and Comparative Study. arXiv:2401.10825v3. DOI 10.48550/arXiv.2401.10825 | 1.1, 1.2 (2×), 1.3, 2.9, 3.1.2 (2×), 3.1.3, 3.4, 3.5 (3×), 4.1, 4.2.1, 4.3.2, 4.5.1, 4.5.4, 5.7.1, 5.7.3 (2×) | NER-Anwendungen; GPT-4o 65,17 vs. DeBERTa-v3-large 90,98 (Tab. 5); DeBERTa am effektivsten; base/large-Rangfolge; F1 üblich; in Kap. 4 auch Label-Alignment und Mikro-F1 als Primärmaß | Volltext (arXiv v3 = zitierte Fassung) | arXiv: stimmt; verwandte IEEE-Arbeit 2025 gefunden | teilweise passend | Zahlen stimmen; als Beleg für Label-Alignment (W4) und Mikro-F1 (W3) ungeeignet; „bislang nur Preprint“ prüfen (W16) |
| `li2018survey` | Li et al. (2022). A Survey on Deep Learning for Named Entity Recognition. IEEE Transactions on Knowledge and Data Engineering, 34(1), 50–70. DOI 10.1109/TKDE.2020.2981314 | 1.1, 2.2 (4×), 2.8 (2×), 3.5 | NER-Definition, Formalisierung, Sequenzlabeling; CoNLL-2003 „am häufigsten verwendet“; OntoNotes 18 Typen | Volltext, aber arXiv-Fassung (zitiert: TKDE 2022) | Crossref: stimmt | teilweise passend | Superlativ zu CoNLL-2003 nicht belegt (W6) |
| `ravikumar2026lostformatting` | Ravikumar et al. (2026). Lost in Formatting: How Output Formats Skew LLM Performance on Information Extraction. Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), 5498–5513. DOI 10.18653/v1/2026.eacl-long.256 | 3.2.1 | 3–32 Mrd., QLoRA, sechs Formate, OntoNotes 5.0, „mehr als 40 %“ | Volltext (ACL-PDF) | Crossref und ACL: stimmt | teilweise passend | 40 % gehören zu BC5CDR; Einheit klären (W10) |
| `lample2016neural` | Lample et al. (2016). Neural Architectures for Named Entity Recognition. Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 260–270. DOI 10.18653/v1/N16-1030 | 2.2 | BiLSTM-CRF ersetzt Handmerkmale | Volltext | Crossref und ACL: stimmt | passend | – |
| `ma2016end` | Ma & Hovy (2016). End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 1064–1074. DOI 10.18653/v1/P16-1101 | 2.2 | wie Lample et al. | Volltext | Crossref und ACL: stimmt | passend | – |
| `tedeschi2022multinerd` | Tedeschi & Navigli (2022). MultiNERD: A Multilingual, Multi-Genre and Fine-Grained Dataset for Named Entity Recognition (and Disambiguation). Findings of the Association for Computational Linguistics: NAACL 2022, 801–812. DOI 10.18653/v1/2022.findings-naacl.60 | 1.1, 1.2, 1.3, 2.8 (4×), 3.1.1 (2×), 3.5, 4.1, 4.3.1 (4×), 4.3.3, 4.5.3, 5.1.2, 6.2, 6.3, A.3 | 15 Typen, 10 Sprachen, zwei Genres, Silberannotation, 1 000 geprüfte Sätze, Tab. 5 (0,85; Typwerte), verrauschte Typen; offizielle Splits | Volltext | Crossref und ACL: stimmt | passend | „offizielle Splits“ zusätzlich über die Datensatzkarte belegen (G21) |
| `liu2021crossner` | Liu et al. (2021). CrossNER: Evaluating Cross-Domain Named Entity Recognition. Proceedings of the AAAI Conference on Artificial Intelligence, 35(15), 13452–13460. DOI 10.1609/aaai.v35i15.17587 | 1.2, 2.8 (3×), 4.6 | fünf Domänen, 9–17 Typen, 100–200 Trainingssätze, CoNLL-2003 als Quelldomäne | Volltext, aber arXiv-Fassung | Crossref: stimmt | passend | – |
| `sang2003conll` | Tjong Kim Sang & De Meulder (2003). Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003, 142–147. DOI 10.3115/1119176.1119195 | 1.1, 1.3, 2.2 (2×), 2.8 (2×), 2.9 (2×), 4.5.1 | vier Typen, EN/DE, IOB1-Daten, flache Annotation, exakte Übereinstimmung; in 4.5.1 für IOB2 | Volltext | ACL-Seite und Crossref: stimmt | teilweise passend | für „nach IOB2-Schema“ unpassend (W15) |
| `zhan2026generativeNER` | Zhan et al. (2026). Assessment of Generative Named Entity Recognition in the Era of Large Language Models. arXiv:2601.17898v3. DOI 10.48550/arXiv.2601.17898 | 1.1 (2×), 1.2, 3, 3.2.1 (2×), 3.2.4, 3.3 (6×), 3.5 (4×), 4.4.5, 5.7.1 (2×), 5.7.2 (2×), 6.3 | 93,85 vs. 93,04 (Tab. 3); acht LLMs, fünf Formate; Inline ≈92, Offsets ≈54; GENIA −4,02; Kosten (Anh. G, Tab. 13); LoRA vs. Full-FT 0,14–0,48 (Anh. D, Tab. 11); Fehlerprofil; CleanCoNLL | Volltext arXiv v3 (zitierte Fassung); Verlagsfassung auf Kennzahlen gegengeprüft | arXiv: stimmt; inzwischen in Findings of EMNLP 2026 erschienen | passend | auf Verlagsfassung umstellen, Fundstellen dort bestätigen (W2); Sammelaussage in 3.3 eingrenzen (W9) |
| `semin2026spanLabeling` | Semin et al. (2026). Strategies for Span Labeling with Large Language Models. arXiv:2601.16946v2. DOI 10.48550/arXiv.2601.16946 | 1.1, 1.2, 2.5, 3, 3.2.1 (3×), 3.2.4, 3.5, 4.1, 4.3.2, 4.4.4, 4.5.3, 5.7.2 (2×), 6.3 | drei Strategiefamilien, vier Modelle 8–70 Mrd., UNER, Befunde, Wortindex-Variante nicht evaluiert, Formfehler, LogitMatch (Abschn. 4), Abschn. 3.2 | Volltext arXiv v2 (zitierte Fassung) | arXiv: stimmt | passend | Tabellenzelle „Ausgabetokens“: Schätzung, keine Messung (G13) |
| `dagdelen2024structuredIE` | Dagdelen et al. (2024). Structured information extraction from scientific text with large language models. Nature Communications, 15(1), Art. 1418. DOI 10.1038/s41467-024-45563-x | 2.5, 3.2.1, 4.1, 4.4.4, 4.5.3 (2×), 5.7.2 | JSON-Ausgabe „mit Position und Typ“; ≈99 % parsebar ohne Formatzwang; API-Szenario; Fehlerklasse JSON-Validität | Volltext (Nature-HTML) | Crossref: stimmt | teilweise passend | keine Positionsangaben und keine IOB2-Rückkonvertierung in der Quelle (W14) |
| `vaswani2017attention` | Vaswani et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30, 5998–6008. https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html | 2.3 (4×) | Self-Attention, Encoder/Decoder, quadratische Kosten, Abb. 2.2 | Volltext | NeurIPS-Seite: stimmt; Seitenzahlen dort nicht ausgewiesen | passend | – |
| `devlin2019bert` | Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 4171–4186. DOI 10.18653/v1/N19-1423 | 1.1 (2×), 1.3, 2.1 (2×), 2.3, 2.4 (3×), 4.1 (2×), 4.3.2, 4.4.2 | Encoder + Klassifikator, MLM/NSP, erstes Subwort, Fine-Tuning aller Parameter, bidirektional | Volltext | Crossref und ACL: stimmt | passend | „ein Vorwärtsdurchlauf“ (G3) und Wert −100 (W4) stehen nicht in der Quelle |
| `wolf2020transformers` | Wolf et al. (2020). Transformers: State-of-the-Art Natural Language Processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38–45. DOI 10.18653/v1/2020.emnlp-demos.6 | 4.1, 4.2.1, 4.4.2 | Transformers-Bibliothek | Volltext | Crossref und ACL: stimmt | passend | geeigneter Beleg für die −100-Konvention wäre die Bibliotheksdokumentation |
| `liu2019roberta` | Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692v1. DOI 10.48550/arXiv.1907.11692 | 2.4 | Byte-Paar-Kodierung auf Byte-Ebene | Volltext (arXiv v1 = zitierte Fassung) | arXiv: stimmt | passend | – |
| `he2020deberta` | He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=XPZIaotutsD | 2.4 | Disentangled Attention, Enhanced Mask Decoder | Volltext, aber arXiv-Fassung | ICLR 2021 über DBLP-Kennung bestätigt; OpenReview nicht abrufbar | passend | – |
| `he2021debertav3` | He et al. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=sE7-XhLxHA | 1.1 (2×), 1.2, 2.4 (4×), 4.1 | RTD-Vortraining, 128 000 Einheiten, 160 GB (Abschn. 4.1), Schichten und Backbone-Größen; „einige hundert Millionen Parameter“ | Volltext, aber arXiv-Fassung | ICLR 2023 nicht unabhängig bestätigt (OpenReview gesperrt) | passend | Gesamtparameter stehen in den Modellkarten, nicht im Aufsatz (G2) |
| `brown2020language` | Brown et al. (2020). Language Models are Few-Shot Learners. Advances in Neural Information Processing Systems, 33, 1877–1901. https://proceedings.neurips.cc/paper_files/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html | 1.1 (2×), 2.1, 2.3, 2.5 (3×), 4.1 | LLM-Definition, Zero-/Few-Shot, In-Context Learning, autoregressives Modell | Volltext | NeurIPS-Seite: stimmt | passend | „nächstes Token vorhersagen“ dort nicht wörtlich (G6) |
| `zhao2023survey` | Zhao et al. (2026). A Survey of Large Language Models. Frontiers of Computer Science, 20(12), Art. 2012627. DOI 10.1007/s11704-026-60308-3 | 1.1 (2×), 2.5, 4.1 | LLMs mit Milliarden bis Hunderten Milliarden Parametern; breite Aufgaben; Zero-Shot-Nutzung | Volltext arXiv v19 (trägt den Zeitschriftenverweis); Springer-Satz nicht eingesehen | Crossref: stimmt (FCS 20(12), 2012627) | passend | – |
| `kaplan2020scaling` | Kaplan et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361v1. DOI 10.48550/arXiv.2001.08361 | 2.7, 4.5.4 | Potenzgesetze für den Verlust | Volltext (arXiv v1 = zitierte Fassung) | arXiv: stimmt | passend | – |
| `hoffmann2022training` | Hoffmann et al. (2022). An Empirical Analysis of Compute-Optimal Large Language Model Training. Advances in Neural Information Processing Systems, 35, 30016–30030. DOI 10.52202/068431-2176 | 4.5.4 | Skalierungsgesetze (Sammelbeleg) | Volltext, aber arXiv-Fassung | NeurIPS-Seite: Titel und Seiten stimmen; Crossref führt den Preprint-Titel | passend | – |
| `qwen2024qwen25` | Yang et al. (2024). Qwen2.5 Technical Report. arXiv:2412.15115v1. DOI 10.48550/arXiv.2412.15115 | 4.2.2 | Qwen3.5 „baut auf Qwen2.5 und Qwen3 auf“ | Volltext (arXiv v1 = zitierte Fassung) | arXiv: stimmt (v2 existiert) | unklar | trägt die Aussage über Qwen3.5 nicht selbst (W13) |
| `hu2021lora` | Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=nZeVKeeFYf9 | 1.1, 1.3, 2.6 (6×), 4.1, 4.4.5, 4.5.2, 6.3 | LoRA-Formel, α/r, Initialisierung, Query/Value, Ergebnisse, Speicherbedarf | Volltext, aber arXiv-Fassung | ICLR 2022 nicht unabhängig bestätigt (OpenReview gesperrt) | passend | „Abschn. 4.1“ stimmt in der arXiv-Fassung; ICLR-Fassung nicht gegengeprüft |
| `han2024parameter` | Han et al. (2024). Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey. Transactions on Machine Learning Research. https://openreview.net/forum?id=lIsCS8b6zj | 1.1, 2.6, 4.4.5 (2×), 4.5.2 | PEFT-Definition; Adapter senken Aufwand; „komplementäre Kapazität“; Effizienz zentral | Volltext, aber arXiv-Fassung | TMLR 2024 nicht unabhängig bestätigt | teilweise passend | für 4.4.5 („komplementäre Kapazität“) kein Beleg gefunden (W5) |
| `wang2024parameter` | Wang et al. (2024). Parameter-Efficient Fine-Tuning in Large Models: A Survey of Methodologies. arXiv:2410.19878v1. DOI 10.48550/arXiv.2410.19878 | 4.4.5 | „komplementäre Kapazität“ unterschiedlicher Linearschichten | Volltext (arXiv v1 = zitierte Fassung) | arXiv: stimmt; als Zeitschriftenartikel erschienen (Artif. Intell. Rev. 58(8), Art. 227, 2025) | problematisch | Aussage nicht gefunden; Beleg streichen oder ersetzen (W5); sonst Verlagsfassung zitieren (W16) |
| `weischedel2013ontonotes` | Weischedel et al. (2013). OntoNotes Release 5.0. DOI 10.35111/xmhb-2b84 | 2.8 (2×), 3.5 | OntoNotes 5.0: Sprachen, Genres | teilweise (LDC-Katalogseite) | DataCite: stimmt | passend | 18 Typen über Li et al. belegt |
| `manning2008introduction` | Manning et al. (2008). Introduction to Information Retrieval. Cambridge University Press. DOI 10.1017/CBO9780511809071 | 1.3, 2.9 (2×), 4.5.1 (2×) | P/R/F aus dem IR; Mikro-/Makro-Mittelung; strikte Metriken | nicht geprüft (kein Text) | Crossref: stimmt | unklar | am Buch nachsehen (Kap. 8 und 13); fünf Stellen ungeprüft |
| `vanrijsbergen1979information` | van Rijsbergen (1979). Information Retrieval. Butterworths. https://www.dcs.gla.ac.uk/Keith/Chapter.7/Ch.7.html | 2.9 (2×) | Herkunft des F-Maßes | teilweise (Kap. 7 der Online-Fassung) | nicht über Register geprüft | teilweise passend | Quelle definiert das E-Maß; „F-Maß“ steht dort nicht (G22) |
| `hripcsak2005agreement` | Hripcsak & Rothschild (2005). Agreement, the F-Measure, and Reliability in Information Retrieval. Journal of the American Medical Informatics Association, 12(3), 296–298. DOI 10.1197/jamia.M1733 | 2.9 | F-Maß = positive spezifische Übereinstimmung | nur bibliografisch (Volltext nicht abrufbar) | Crossref und PubMed: stimmt | unklar | am Aufsatz nachsehen |
| `zaratiana2024gliner` | Zaratiana et al. (2024). GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer. Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 5364–5376. DOI 10.18653/v1/2024.naacl-long.300 | 3.1.3 (2×), 3.4, 3.5 (2×), 5.7.1, 5.7.3, 6.3 | GLiNER: Aufbau, Pile-NER, 7 OOD-Datensätze, 55,4 bei 90 Mio., Vergleich UniNER/ChatGPT; MultiNERD mit 10 000 Testbeispielen | Volltext | Crossref und ACL: stimmt | passend | Herkunft der Aufbereitung aus InstructUIE nur indirekt belegt (G16) |
| `bogdanov2024nuner` | Bogdanov et al. (2024). NuNER: Entity Recognition Encoder Pre-training via LLM-Annotated Data. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 11829–11841. DOI 10.18653/v1/2024.emnlp-main.660 | 3.1.3 (4×), 3.4, 3.5, 5.7.1, 5.7.3, 6.3 | NuNER: RoBERTa-base, 4,38 Mio. Annotationen, 200 000 Konzepte, 56-mal, Tab. 2, Abschn. 6, Kosten | Volltext | Crossref und ACL: stimmt | passend | Vergleich betrifft UniNER-7B, nicht 13B (G15) |
| `tan2023damo` | Tan et al. (2023). DAMO-NLP at SemEval-2023 Task 2: A Unified Retrieval-augmented System for Multilingual Named Entity Recognition. Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023), 2014–2028. DOI 10.18653/v1/2023.semeval-1.277 | 3.1.2 (2×), 3.1.3, 3.5 (3×), 5.7.1 | MultiCoNER II, 9 von 13 Tracks, System, ChatGPT 14,78 vs. 92,77 (Tab. 4) | Volltext | ACL: stimmt (Crossref führt nur zehn Autoren) | passend | als Beleg für „Zero-Shot“ in E3 ungenau (W8) |
| `wang2025gptner` | Wang et al. (2025). GPT-NER: Named Entity Recognition via Large Language Models. Findings of the Association for Computational Linguistics: NAACL 2025, 4257–4275. DOI 10.18653/v1/2025.findings-naacl.239 | 3.2.2 (3×), 3.2.4, 3.5 | GPT-NER: Marker, kNN über Tagger (3.2.2), Selbstverifikation, 90,91 vs. 92,8 (Tab. 1) | Volltext | Crossref und ACL: stimmt | passend | – |
| `ashok2023promptner` | Ashok & Lipton (2023). PromptNER: Prompting For Named Entity Recognition. arXiv:2305.15444v2. DOI 10.48550/arXiv.2305.15444 | 3.2.2, 3.2.4, 3.5 | PromptNER: +4/+9 Punkte, CrossNER 3 von 5, Music 84,26, Datenanteil | Volltext (arXiv v2 = zitierte Fassung) | arXiv: stimmt | passend | Quelle: „1–2 %“, Arbeit: „weniger als 2 %“ (G11) |
| `li2023codeie` | Li et al. (2023). CodeIE: Large Code Generation Models are Better Few-Shot Information Extractors. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 15339–15353. DOI 10.18653/v1/2023.acl-long.855 | 3.2.4, 3.5 | Code-Prompts; Tabellenzeile | Volltext | Crossref und ACL: stimmt | passend | nur Full-Data-Werte sind aus UIE übernommen (G13) |
| `zhou2024universalner` | Zhou et al. (2024). UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition. International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=r65xfUb76p | 3.2.1, 3.2.3 (2×), 3.2.4, 3.4, 3.5 (3×), 5.7.1, 5.7.3 | UniversalNER: Pile, 7B/13B, 43 Datensätze, 41,7/43,4/34,9, 84,78 vs. 80,09, MultiNERD 93,73 vs. 91,25 (Tab. 2); Abschn. 5.2 | Volltext, aber arXiv-Fassung | ICLR 2024 nicht unabhängig bestätigt | passend | Fundstellen an der ICLR-Fassung prüfen; Deutung in 5.7.3 als eigene kennzeichnen (W11) |
| `sainz2024gollie` | Sainz et al. (2024). GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction. International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=Y3wpuxd7u9 | 3.2.3 (4×), 3.2.4 (2×), 3.4, 3.5 (3×), 5.7.1 (2×), 5.7.3 | GoLLIE: Modelle, QLoRA, +13 auf 55,3, CrossNER 55,5–67,8, MultiNERD 76,0 (Tab. 3), acht Typen (Tab. 6), Tab. 9 (32 908), Tab. 10; Abschn. 5.2 | Volltext, aber arXiv-Fassung | ICLR 2024 über DBLP-Kennung bestätigt | passend | Fundstellen an der ICLR-Fassung prüfen; W11 |
| `lu2024llmstruggle` | Lu et al. (2025). Large Language Models Struggle in Token-Level Clinical Named Entity Recognition. AMIA Annual Symposium Proceedings, 2024, 748–757. https://pubmed.ncbi.nlm.nih.gov/40417588/ | 3.3 (3×), 3.5, 4.4.5, 5.7.1, 5.7.2 | RareDis, 0,149/0,576, LoRA nähert sich BioClinicalBERT, „auf allen Typen darunter“, drei Gründe | Volltext, aber arXiv-Fassung (zitiert: AMIA) | PubMed: stimmt | teilweise passend | Typweise Aussage nur aus Abb. 3 ableitbar (W7); Quelle sagt „BIO“ |
| `obeidat2025llm` | Obeidat et al. (2025). Do LLMs Surpass Encoders for Biomedical NER?. 2025 IEEE 13th International Conference on Healthcare Informatics (ICHI), 352–358. DOI 10.1109/ICHI64645.2025.00048 | 1.2, 3.3 (7×), 3.4, 3.5 (5×), 4.4.5, 4.5.2, 4.5.4, 5.7.1 (2×), 5.7.2, 5.7.4 | fünf Datensätze, Modelle, QLoRA, 2–8 Punkte, Precision, 40- bis 220-fach (Tab. IV), zwei H100, Adapterparameter, Encoder-Empfehlung | Volltext, aber arXiv-Fassung (zitiert: IEEE ICHI) | Crossref: stimmt | passend | 40 und 220 stehen im Text (Abschn. III/IV), Tab. IV liefert die Zeiten (G14); als Beleg in 4.4.5 schwach (G18) |
| `qwen2025qwen3` | Yang et al. (2025). Qwen3 Technical Report. arXiv:2505.09388v1. DOI 10.48550/arXiv.2505.09388 | 2.5, 4.2.2 | Thinking-Modus; Vorgängerfamilie | Volltext (arXiv v1 = zitierte Fassung) | arXiv: stimmt | passend | wäre auch der passende Beleg für Chat-Template und Rollen |
| `qwen2026qwen35blog` | Qwen Team (2026). Qwen3.5: Towards Native Multimodal Agents. https://qwen.ai/blog?id=qwen3.5 | 2.5, 4.2.2 | erstes Qwen3.5-Modell im Februar 2026; Familienbezug | nicht geprüft (Seite nur mit JavaScript lesbar) | Titel/Monat/URL über den Zitierblock der Modellkarte bestätigt | unklar | im Browser prüfen, ob der Blog beide Aussagen trägt (W13) |
| `mangrulkar2022peft` | Hugging Face (2026). PEFT Version 0.19.1: Software Release. https://github.com/huggingface/peft/tree/v0.19.1 | 4.4.3 | verwendete Bibliothek PEFT 0.19.1 | nur bibliografisch (Release-Metadaten) | GitHub: v0.19.1 vom 16.04.2026 | passend | – |
| `wei2023chatie` | Wei et al. (2023). ChatIE: Zero-Shot Information Extraction via Chatting with ChatGPT. arXiv:2302.10205v2. DOI 10.48550/arXiv.2302.10205 | 4.4.3, 4.5.3 | Multi-Turn-Ansatz; Fehlerklassen | Volltext (arXiv v2 = zitierte Fassung) | arXiv: stimmt | teilweise passend | für die Fehlerklassen in 4.5.3 ohne Beitrag (G20) |
| `geng2023grammar` | Geng et al. (2023). Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 10932–10952. DOI 10.18653/v1/2023.emnlp-main.674 | 2.5, 4.4.4 | Constrained Decoding bindet an Grammatik | Volltext | Crossref und ACL: stimmt | passend | – |
| `tam2024speakfreely` | Tam et al. (2024). Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, 1218–1236. DOI 10.18653/v1/2024.emnlp-industry.91 | 4.4.4, 5.7.2 | Formatzwang kann Leistung mindern | Volltext | Crossref und ACL: Seiten stimmen; Registertitel weicht vom PDF-Titel ab | passend | Eintrag folgt dem PDF-Titel; keine Änderung nötig |
| `dettmers2023qlora` | Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. Advances in Neural Information Processing Systems, 36, 10088–10115. DOI 10.52202/075280-0441 | 1.1, 1.3, 2.6 (3×), 4.1, 4.4.5 (3×), 4.5.2 | QLoRA: NF4, Double Quantization, Paged Optimizers, 65 Mrd. auf 48 GB, 0,37 Bit; Adapter auf allen Schichten | Volltext | Crossref: stimmt | passend | trägt „alle Schichten adaptieren“, nicht „komplementäre Kapazität“ (W5) |
| `dettmers2022llmint8` | Dettmers et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. Advances in Neural Information Processing Systems, 35, 30318–30332. DOI 10.52202/068431-2198 | 2.6, 4.4.5 | Quantisierung mit Skalierungskonstante; bitsandbytes | Volltext | Crossref: stimmt | teilweise passend | blockweise Konstanten und NF4 stammen aus anderen Arbeiten (G8, G19) |
| `ramshaw1995text` | Ramshaw & Marcus (1995). Text Chunking using Transformation-Based Learning. Third Workshop on Very Large Corpora, 82–94. https://aclanthology.org/W95-0107/ | 2.2, 4.5.1 | I/O/B-Tagset; in 4.5.1 für IOB2 | Volltext | ACL-Seite: stimmt | teilweise passend | für IOB2 unpassend (W15) |
| `nakayama2018seqeval` | Nakayama (2018). seqeval: A Python framework for sequence labeling evaluation. https://pypi.org/project/seqeval/1.2.2/ | 2.9 (2×), 4.5.1 | seqeval: zwei Modi, Schemata | teilweise (README, PyPI) | PyPI: 1.2.2 vom 24.10.2020; Zitiervorschlag des Projekts stimmt | passend | – |
| `segura2013semeval` | Segura-Bedmar et al. (2013). SemEval-2013 Task 9 : Extraction of Drug-Drug Interactions from Biomedical Texts (DDIExtraction 2013). Second Joint Conference on Lexical and Computational Semantics (*SEM), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval 2013), 341–350. https://aclanthology.org/S13-2056/ | 2.9, 4.5.1 (2×) | strikter Modus und weniger strenge Modi | Volltext | ACL: stimmt | passend | „nur den Typ“ ungenau: Typ-Modus verlangt Überlappung (G9) |
| `babelscape2023multinerdcard` | Babelscape (o. J.). Babelscape/multinerd: Dataset Card. https://huggingface.co/datasets/Babelscape/multinerd | 4.3.1 | Lizenz CC BY-NC-SA 4.0, nichtkommerzielle Forschung | Volltext (Karte, Stand 21.11.2026) | – | passend | Eintrag ohne Datum (F3) |
| `dukic2024lookingright` | Dukic & Snajder (2024). Looking Right is Sometimes Right: Investigating the Capabilities of Decoder-only LLMs for Sequence Labeling. Findings of the Association for Computational Linguistics: ACL 2024, 14168–14181. DOI 10.18653/v1/2024.findings-acl.843 | 3.3 (3×), 3.4, 3.5 | Maskenentfernung, 0,744 → 0,908, RoBERTa-large 0,900, 68 Mio. | Volltext | Crossref und ACL: stimmt | passend | Vorsprung nur auf CoNLL-2003-NER (W9); Tabellenzelle ohne Chunking (G13) |
| `lu2024padellm` | Lu et al. (2024). PaDeLLM-NER: Parallel Decoding in Large Language Models for Named Entity Recognition. Advances in Neural Information Processing Systems, 37, 117853–117880. DOI 10.52202/079017-3743 | 1.1, 3.2.1, 3.2.4, 3.5 | Latenz durch sequenzielles Decoding, Faktor 1,76–10,22 | Volltext | Crossref: stimmt | passend | Quelle: „on par with state of the art“, Arbeit: „unverändert“ (G10) |
| `xie2023zeroshot` | Xie et al. (2023). Empirical Study of Zero-Shot NER with ChatGPT. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 7935–7956. DOI 10.18653/v1/2023.emnlp-main.493 | 3.2.2, 3.2.4, 3.5 | Zerlegung je Typ, sieben Benchmarks | Volltext | Crossref und ACL: stimmt | passend | – |
| `bengio2013representation` | Bengio et al. (2013). Representation Learning: A Review and New Perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8), 1798–1828. DOI 10.1109/TPAMI.2013.50 | 2.1 | Representation Learning, verteilte Repräsentationen | Volltext, aber arXiv-Fassung | Crossref: stimmt | passend | – |
| `wei2022emergent` | Wei et al. (2022). Emergent Abilities of Large Language Models. Transactions on Machine Learning Research. https://openreview.net/forum?id=yzkSU5zdwD | 2.5, 5.7.3 | emergente Fähigkeiten; in 5.7.3 „Formatvorgaben“ | Volltext, aber arXiv-Fassung | TMLR 2022 nicht unabhängig bestätigt | teilweise passend | Quelle nennt Instruction Following, keine Formatvorgaben (W12) |
| `lafferty2001crf` | Lafferty et al. (2001). Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data. Proceedings of the Eighteenth International Conference on Machine Learning (ICML 2001), 282–289. https://dl.acm.org/doi/10.5555/645530.655813 | 2.2 | CRF modelliert bedingte Wahrscheinlichkeit der Labelfolge; „lange Stand der Technik“ | Volltext (Vorlesungskopie) | ACM-Seite nicht abrufbar | teilweise passend | historische Einordnung nicht in der Quelle (G4) |
| `tjong1999representing` | Tjong Kim Sang & Veenstra (1999). Representing Text Chunks. Ninth Conference of the European Chapter of the Association for Computational Linguistics, 173–179. https://aclanthology.org/E99-1023/ | 2.2 | Bezeichnungen IOB1/IOB2 | Volltext | ACL: stimmt | passend | wäre der passende Beleg für IOB2 in 4.5.1 |
| `schuster2012japanese` | Schuster & Nakajima (2012). Japanese and Korean Voice Search. 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 5149–5152. DOI 10.1109/ICASSP.2012.6289079 | 2.4 | WordPiece | teilweise (PDF nur dekodiert lesbar) | Crossref: stimmt | passend | – |
| `kudo2018sentencepiece` | Kudo & Richardson (2018). SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 66–71. DOI 10.18653/v1/D18-2012 | 2.4 | SentencePiece | Volltext | Crossref und ACL: stimmt | passend | – |
| `kudo2018subword` | Kudo (2018). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 66–75. DOI 10.18653/v1/P18-1007 | 2.4 | SentencePiece (Unigram-Modell) | Volltext | Crossref und ACL: stimmt | passend | – |
| `clark2020electra` | Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=r1xMH1BtvB | 2.4 | Replaced Token Detection, Signal an allen Positionen | Volltext, aber arXiv-Fassung | ICLR 2020 nicht unabhängig bestätigt | passend | – |
| `yang2025gated` | Yang et al. (2025). Gated Delta Networks: Improving Mamba2 with Delta Rule. International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=r8H7xhYPwz | 2.3 | lineare Attention mit Zustand fester Größe | Volltext, aber arXiv-Fassung | ICLR 2025 nicht unabhängig bestätigt | passend | – |
| `ainslie2023gqa` | Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 4895–4901. DOI 10.18653/v1/2023.emnlp-main.298 | 2.3 | Grouped-Query-Attention | Volltext | Crossref und ACL: stimmt | passend | – |
| `ouyang2022instructgpt` | Ouyang et al. (2022). Training Language Models to Follow Instructions with Human Feedback. Advances in Neural Information Processing Systems, 35, 27730–27744. DOI 10.52202/068431-2011 | 2.5 | SFT auf menschlichen Antworten; „Post-Training“ | Volltext | Crossref: stimmt | passend | Begriff „Post-Training“ steht nicht in der Quelle (G7) |
| `holtzman2020curious` | Holtzman et al. (2020). The Curious Case of Neural Text Degeneration. International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=rygGQyrFvH | 2.5 | Top-k, Top-p, Temperatur | Volltext, aber arXiv-Fassung | ICLR 2020 nicht unabhängig bestätigt | passend | – |
| `osi2024osaid` | Open Source Initiative (2024). The Open Source AI Definition -- 1.0. https://opensource.org/ai/open-source-ai-definition | 2.5 | Definition verlangt Angaben zu Trainingsdaten und vollständigen Code | Volltext (Webseite) | Titel/Version stimmen | passend | – |
| `qwen2026card08b` | Qwen Team (2026). Qwen3.5-0.8B: Model Card. https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md | 1.2, 1.3, 2.5, 4.2.2 (3×) | Modell, Lizenz, Trainingsstufen, Schichtaufbau | Volltext (Karte, zitierte Revision) | HF: Revision vom 02.03.2026 | passend | „Grouped-Query“, „Ausgabe-Gating“, „dicht“ stehen nicht in der Karte (W13) |
| `qwen2026card4b` | Qwen Team (2026). Qwen3.5-4B: Model Card. https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md | 1.2, 1.3, 2.5, 4.2.2 (3×) | wie 0.8B | Volltext (Karte, zitierte Revision) | HF: Revision vom 02.03.2026 | passend | wie 0.8B |
| `qwen2026card27b` | Qwen Team (2026). Qwen3.5-27B: Model Card. https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/README.md | 1.2, 1.3, 2.3, 2.5 (2×), 4.2.2 (3×), 4.4.3, 5.7.1, A.2.5 | wie 0.8B; hybride Attention; Chat-Template/Rollen; Sampling-Empfehlungen; keine Angaben zu Vortrainingsdaten | Volltext (Karte, zitierte Revision) | HF: zitierte Revision vom 24.04.2026, Eintrag nennt 24.02.2026 | teilweise passend | Rollen und Steuertokens nicht in der Karte (W13); Datum des Eintrags angleichen (F2) |
| `qwen2026card397b` | Qwen Team (2026). Qwen3.5-397B-A17B: Model Card. https://huggingface.co/Qwen/Qwen3.5-397B-A17B | 1.1 | LLMs mit Hunderten Milliarden Parametern | Volltext (Karte, Stand 21.11.2026) | HF: angelegt 16.02.2026; URL ohne Revision | passend | Revision ergänzen (F2) |
| `su2024roformer` | Su et al. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing, 568, 127063. DOI 10.1016/j.neucom.2023.127063 | keine | – | nicht geprüft | Crossref: stimmt | nicht verwendet | Eintrag ohne Zitat; erscheint nicht im Verzeichnis (F1) |
| `wang2023instructuie` | Wang et al. (2023). InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction. arXiv:2304.08085v1. DOI 10.48550/arXiv.2304.08085 | 3.2.3 | Herkunft der BERT-Werte bei UniversalNER | Volltext (arXiv) | arXiv: stimmt | passend | – |
| `ramponi2025fallacy` | Ramponi et al. (2025). Fine-grained Fallacy Detection with Human Label Variation. Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 762–784. DOI 10.18653/v1/2025.naacl-long.34 | 1.2, 3.2.1 (2×), 4.3.2, 5.7.2 | Tab. 9: nummerierte Eingabetokens, Ausgabe über Indizes; Tab. 3 | Volltext | Crossref und ACL: stimmt | passend | Quelle spricht von Tokens; Aufgabe ist Fallacy-Erkennung (im Text einordnen) |
| `mayhew2024uner` | Mayhew et al. (2024). Universal NER: A Gold-Standard Multilingual Named Entity Recognition Benchmark. Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 4322–4337. DOI 10.18653/v1/2024.naacl-long.243 | 3.2.1 | UNER-Benchmark (Abgrenzung zum Modell UniversalNER) | Volltext vorhanden; Aussage nur über Semin et al. geprüft | Crossref und ACL: stimmt | passend | – |
| `ding2024gner` | Ding et al. (2024). Rethinking Negative Instances for Generative Named Entity Recognition. Findings of the Association for Computational Linguistics: ACL 2024, 3461–3475. DOI 10.18653/v1/2024.findings-acl.206 | 3.2.3, 3.5 | GNER: Modelle, Pile-NER, +9 F1, Durchsatz, GPU-Stunden | Volltext | ACL/Crossref führen „Bowen, Yan“; PDF und arXiv: „Bowen Yan“ | passend | Autorenname korrigieren (W1) |
| `efron1993bootstrap` | Efron & Tibshirani (1993). An Introduction to the Bootstrap. Chapman & Hall. DOI 10.1201/9780429246593 | 4.5.1 | Bootstrap | nicht geprüft (kein Text) | Crossref: DOI gehört zur CRC-Ausgabe (1994) | unklar | Jahr/DOI abstimmen (F4) |
| `koehn2004bootstrap` | Koehn (2004). Statistical Significance Tests for Machine Translation Evaluation. Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing, 388–395. https://aclanthology.org/W04-3250/ | 4.5.1 | gepaartes Bootstrap mit denselben Stichproben | Volltext | ACL: stimmt | passend | – |
| `he2021debertav3card` | Microsoft (o. J.). microsoft/deberta-v3-base: Model Card. https://huggingface.co/microsoft/deberta-v3-base | 2.4, 4.2.1 | MIT-Lizenz, 86 Mio. Backbone, 512 Positionen | Volltext (Karte und config.json, Stand 21.11.2026) | – | teilweise passend | 512 steht in config.json, nicht in der Karte (G5); Eintrag ohne Datum (F3) |
| `he2021debertav3cardlarge` | Microsoft (o. J.). microsoft/deberta-v3-large: Model Card. https://huggingface.co/microsoft/deberta-v3-large | 2.4 (2×), 4.2.1 | MIT-Lizenz, 304 Mio. Backbone, 24 Schichten | Volltext (Karte); config.json nicht eingesehen | – | passend | wie base |
| `vonwerra2026trldocs` | Hugging Face (o. J.). TRL Documentation, Version 1.8.0: SFT Trainer. https://huggingface.co/docs/trl/v1.8.0/en/sft_trainer | 4.4.3 (2×) | SFTTrainer; assistant_only_loss Standard False | Volltext (Dokumentation v1.8.0) | – | passend | Eintrag ohne Datum (F3) |

## D. Anspruch-Beleg-Übersicht

Die wichtigsten fremden Aussagen, auf denen Forschungslücke, Erwartungen und Literaturvergleich aufbauen.

| Aussage und Fundstelle | Quelle | Trägt die Quelle? | Empfehlung |
|---|---|---|---|
| GPT-4o 65,17 gegenüber DeBERTa-v3-large 90,98 Makro-F₁ auf CoNLL-2003 (1.2, 3.1.2, 5.7.1) | Keraghel et al., Tab. 5 | ja | keine |
| LLaMA3.1-8B 93,85 gegenüber BERT-MRC 93,04 (1.2, 3.3) | Zhan et al., Tab. 3 | ja | Fundstelle an der Verlagsfassung bestätigen (W2) |
| LLMs 2 bis 8 Punkte vorn auf vier von fünf Datensätzen, 40- bis 220-fache Inferenzzeit (1.2, 3.3, E2, 5.7.2, 6.1) | Obeidat et al., Tab. III, IV | ja (+2,1; +2,3; +1,9; +7,9; −0,5) | Fundstelle der Faktoren ergänzen (G14) |
| MultiNERD: Originalarbeit berichtet keine Werte auf den eigenen Splits; Silberreferenz mit F₁ 0,85 gegenüber manueller Korrektur (3.1.1) | Tedeschi & Navigli, Tab. 5 | ja | keine |
| Veröffentlichte MultiNERD-Werte beruhen auf übernommenen Baselines oder eigener Aufbereitung (1.2, 3.5, 4.3.1) | Zhou et al., Tab. 2; Zaratiana et al., Tab. 6; Sainz et al., Tab. 9 | überwiegend ja | bei GLiNER die Herkunft vorsichtiger fassen (G16); „Werte auf den offiziellen Splits sind nicht auffindbar“ (4.3.1) ist ein eigener Suchbefund und sollte so heißen („nach der Recherche dieser Arbeit“) |
| GoLLIE: MultiNERD 76,0, acht der 15 Typen im Training gesehen, Testumfang 32 908 (3.2.3, 3.5, 5.7.1) | Sainz et al., Tab. 3, 6, 9 | ja (arXiv-Fassung) | Tabellennummern an der ICLR-Fassung bestätigen |
| Direkte Vergleiche: LLMs einer Größenklasse von 3 bis 9 Mrd.; Ressourcen nur ausschnittsweise gemessen (3.5) | Zhan; Obeidat; Lu; Dukić | ja | keine |
| E1: LoRA-angepasste LLMs erreichen Encoder-Niveau (3.3, 3.5) | Zhan; Obeidat | ja, für flache und biomedizinische Datensätze | Sammelaussage in 3.3 eingrenzen (W9) |
| E3: Prompting ohne Anpassung bleibt deutlich zurück (3.5) | Keraghel; Tan | ja, aber mit Beispielen im Prompt | E3 umformulieren (W8) |
| LoRA liegt 0,14 bis 0,48 Punkte über vollständigem Fine-Tuning (3.3, 4.4.5) | Zhan et al., Anh. D, Tab. 11 | ja (zwei Modelle, zwei Datensätze) | in 4.4.5 den Umfang nennen, wie in 3.3 |
| Wortindex-Variante bei Ramponi et al. eingesetzt, bei Semin et al. beschrieben, aber nicht evaluiert (1.2, 3.2.1, 5.7.2) | Ramponi et al., Tab. 9; Semin et al., Abschn. 3.2 | ja | Ramponi arbeitet mit Tokens und an Fallacy-Erkennung; einmal im Text sagen |
| „soweit bekannt, erstmals mit QLoRA und strikter Entitätsmetrik auf MultiNERD geprüft“ (5.7.2, S. 68) | keine (Neuheitsanspruch) | nicht belegbar, als „soweit bekannt“ markiert | so belassen; die Nachsuche vom 8.11. ist in Kapitel 3 genannt |
| JSON-Ausgaben feinjustierter Modelle zu rund 99 % parsebar (3.2.1, 5.7.2) | Dagdelen et al. | ja | keine |
| Constrained Decoding kann die Leistung mindern (4.4.4, 5.7.2) | Tam et al. | ja (für Reasoning-Aufgaben) | keine |
| Double Quantization spart rund 0,37 Bit je Parameter ohne Leistungseinbuße (4.4.5) | Dettmers et al. 2023 | ja | keine |
| Auswahl nach Testbeobachtungen kann optimistisch verzerren (4.3.3) | Cawley & Talbot | ja | keine |
| Intervall mit null belegt keine Gleichwertigkeit (4.5.1) | Lakens | ja | keine |
| Gepaartes Bootstrap mit gemeinsamen Satzstichproben (4.5.1) | Efron & Tibshirani; Koehn | Koehn ja; Efron nicht geprüft | Efron am Buch bestätigen |
| ANIM, BIO, FOOD, PLANT mit Annotations-F₁ 0,52 bis 0,71, INST 0,67 (2.8, 5.1.2, 6.2) | Tedeschi & Navigli, Tab. 5 | ja | keine |

**Aussagen ohne Beleg, bei denen einer sinnvoll wäre**

- 2.5, S. 12: Gleitkommaarithmetik und Batching können bei Greedy-Decoding Abweichungen verursachen. Die Arbeit stützt später eine Erklärung darauf (5.2).
- 2.9, S. 17: Die strenge Zählweise habe sich „mit CoNLL-2003 als Standard etabliert“.
- 2.6, S. 14: Dropout auf dem Adapterpfad als Regularisierung (bei Hu et al. nur als Hyperparameter erwähnt).
- 3.5, S. 29: Die Zusammensetzung der Qwen3.5-Trainingsdaten sei nicht vollständig offengelegt. In 5.7.1 steht dazu der Beleg (Modellkarte); hier fehlt er.
- 5.4, Legende der Tabelle zum Ressourcenbedarf (S. 60): Speichergrößen der GPUs „laut Herstellerangabe“, ohne Quelle.

Als Allgemeinwissen des Fachs ohne Beleg vertretbar: Adam speichert zwei Größen je Parameter; nicht definierte Precision wird als null gewertet; Aufbau der Self-Attention.

**Methoden, Datensätze und Software ohne eigenen Eintrag**

In Kapitel 3 werden viele Modelle und Datensätze nur über die besprochene Arbeit eingeführt (MultiCoNER II, Pile-NER, CleanCoNLL, BabelNet, XLM-RoBERTa, BERT-MRC, UIE). Das ist für einen Literaturüberblick üblich. Wo die Arbeit selbst Eigenschaften solcher Ressourcen behauptet, wäre die Primärquelle sauberer: MultiCoNER II („kurze, kontextarme Sätze“, 33 Typen, zwölf Sprachen; S. 20 und 28, derzeit über Tan et al.). Ohne Beleg genannt sind außerdem BIOES/BILOU (S. 8), AdamW (S. 38), rsLoRA und DoRA (S. 42), FlashAttention-2 (S. 93). Bei der eingesetzten Software haben Transformers, PEFT, TRL, bitsandbytes (über LLM.int8()) und seqeval einen Eintrag; PyTorch, Datasets und Accelerate werden mit Version genannt (S. 93), aber nicht zitiert. Das CrossNER-Repository ist mit Commit und Lizenz genannt (S. 50, 90), hat aber keinen eigenen Eintrag.

## E. Abgleich von Zitaten und Literaturverzeichnis

**Im Text zitiert, im Verzeichnis fehlend:** keine. Alle 85 zitierten Schlüssel sind in `references.bib` vorhanden; der Neubau meldet kein offenes Zitat.

**Im Verzeichnis, aber nicht verwendet:** `su2024roformer` (F1). Der Eintrag steht nur in der `.bib`-Datei und erscheint nicht im gesetzten Verzeichnis. Kein Fehler im PDF; kann bei Gelegenheit entfernt werden. Nach W5 käme gegebenenfalls `wang2024parameter` hinzu.

**Doppelte oder mehrdeutige Einträge:** keine Dubletten. Gleichnamige Erstautoren unterscheidet der Stil richtig durch Initialen (J. Lu/Q. Lu, L./S./X. Wang, J./P. Li, J./X. Wei, Y./Z. Liu, A./S. Yang), gleiche Autorenschaft im selben Jahr durch Buchstaben (Qwen Team 2026a bis 2026e; Microsoft o. J. a/b).

**Schlüssel, Namen, Jahre**

- Einige Schlüssel tragen ein anderes Jahr als der Eintrag (`li2018survey` → 2022, `zhao2023survey` → 2026, `jurafsky2024speech` → 2026, `he2020deberta` → 2021, `he2021debertav3` → 2023, `hu2021lora` → 2022, `lu2024llmstruggle` → 2025, `mangrulkar2022peft` und `vonwerra2026trldocs` mit institutionellem Autor). Im PDF ist das unsichtbar; die gesetzten Jahre sind jeweils die der zitierten Fassung.
- F2. Modellkarten: Die Einträge für 0.8B und 4B nennen das Datum der zitierten Revision (2.3.2026). Der Eintrag für 27B nennt den 24.2.2026 (Anlagedatum), die zitierte Revision `fc05daec…` ist laut Commit-Liste vom 24.4.2026. Der Eintrag für 397B hat keine Revision in der URL; die Karte wurde zuletzt am 24.4.2026 geändert.
- F3. Vier Online-Einträge haben kein Datum und erscheinen als „o. J.“ (Babelscape, zwei Microsoft-Karten, TRL-Dokumentation). Im Text entsteht daraus die uneinheitliche Form „(Microsoft, o.J. a; o.J., b)“ (S. 11 und 33). Ein Jahr im Eintrag behebt das; welches Datum (Abruf, letzte Änderung der Karte, Veröffentlichung der Version) gelten soll, ist eine Festlegung.
- F4. `efron1993bootstrap`: Jahr 1993 und Verlag Chapman & Hall entsprechen der Erstausgabe; der angegebene DOI `10.1201/9780429246593` gehört laut Crossref zur Ausgabe bei Chapman and Hall/CRC von 1994. Entweder den DOI weglassen oder die Ausgabe angleichen.
- F5. `hoffmann2022training`: Titel nach den Proceedings, Namensformen einzelner Autoren nach dem Preprint (Tom/Thomas Hennigan, Katie/Katherine Millican). Im gesetzten Verzeichnis nicht sichtbar.
- `obeidat2025llm`: Crossref führt den zweiten Autor als „Sultan Al Nahian, Md“, der Eintrag als „Al Nahian, Md Sultan“. Welche Zerlegung richtig ist, lässt sich aus den Metadaten nicht entscheiden.
- `tan2023damo`: Crossref führt zehn Autoren, ACL Anthology und der Eintrag elf. Der Eintrag folgt der Anthology.

**Nicht verifizierbare Angaben**

- ICLR- und TMLR-Einträge: OpenReview war nicht abrufbar. Über eine DBLP-Kennung bestätigt sind DeBERTa (ICLR 2021) und GoLLIE (ICLR 2024). Nicht unabhängig bestätigt, aber ohne Hinweis auf einen Fehler: DeBERTaV3 (ICLR 2023), LoRA (ICLR 2022), ELECTRA und Holtzman et al. (ICLR 2020), UniversalNER (ICLR 2024), Gated Delta Networks (ICLR 2025), Han et al. (TMLR 2024), J. Wei et al. (TMLR 2022).
- `lafferty2001crf`: ACM-Seite nicht abrufbar (Seiten 282–289, Morgan Kaufmann nicht bestätigt).
- `vaswani2017attention`: Die NeurIPS-Seite bestätigt Titel, Autoren, Band 30; Seitenzahlen weist sie nicht aus.
- `manning1999foundations`, `vanrijsbergen1979information`: nicht über ein Register geprüft.
- `qwen2026qwen35blog`: Seite nicht lesbar; Titel, Monat und URL stimmen mit dem Zitierblock der Modellkarten überein.

**Bestätigt (Auswahl):** alle 31 Einträge mit ACL-Anthology-Adresse (Titel, Autoren, Seiten, DOI); alle Zeitschriften- und IEEE-Einträge mit DOI; Zhao et al. (*Frontiers of Computer Science* 20(12), Artikel 2012627, 2026); Lu et al. (*AMIA Annual Symposium Proceedings* 2024, 748–757, erschienen 22.5.2025); Cawley & Talbot (JMLR 11(70), 2079–2107); Jurafsky & Martin (Fassung vom 19.8.2026); seqeval 1.2.2 (PyPI, 24.10.2020); PEFT v0.19.1 (16.4.2026); OntoNotes (DataCite). Die arXiv-Versionen stimmen mit den Einträgen überein; für `qwen2024qwen25` (v2) und `wang2024parameter` (v3) gibt es neuere Versionen als die zitierte v1.

## F. Übernahme- und Paraphrasenrisiken

**Korrekt gekennzeichnete wörtliche Zitate:** Die Arbeit enthält keine wörtlichen Zitate aus der Literatur. Anführungszeichen stehen nur bei Suchbegriffen, Beispielen aus den Daten und eigenen Bezeichnungen. Die Prompts in den Listings sind eigene Texte. Die Beispielsätze in Tabelle A.14 stammen aus MultiNERD und sind so ausgewiesen.

**Wortfolgensuche:** Bei 8, 6 und 5 Wörtern gab es gegen 97 Quelltextdateien nur Treffer in Formeln, Modellnamen, Tag-Namen, Projektionsnamen und Steuertokens des Chat-Templates. Da die Arbeit deutsch und die Quellen englisch sind, erfasst diese Suche nur unübersetzte Übernahmen; solche gibt es nicht.

**Plausible, korrekt belegte Paraphrasen:** der Regelfall. Kapitel 3 gibt die Arbeiten in eigener Gliederung wieder, mit Zahlen, Fundstellen und eigener Einordnung.

**Nah am Original, aber der Quelle zugeschrieben.** Diese Sätze folgen einem Quellsatz in Wortwahl und Aufbau eng. Sie sind belegt und als Aussage der Quelle erkennbar; ein Redlichkeitsproblem sehe ich nicht. Wer den Abstand vergrößern will, formuliert sie eigenständiger oder setzt den Kern als wörtliches Zitat in Anführungszeichen.

| Fundstelle | Arbeit | Quelle |
|---|---|---|
| 3.2.1, S. 22 | „… dass das Berechnen von Zeichenpositionen der generativen Arbeitsweise der Modelle schlecht entspricht.“ | Zhan et al.: „precise character-offset calculations are poorly suited to the text-generation nature of LLMs“ |
| 3.2.1, S. 22 | „… Tagging als die robusteste, aber auch tokenintensivste Strategie. Matching ist konkurrenzfähig und tokeneffizient; Indexing … nur dann konkurrenzfähig, wenn die Eingabe selbst mit Zeichenindizes angereichert wird“ | Semin et al.: „tagging-based methods are the most robust, but matching methods are competitive and more token-efficient“; „The indexing strategies can be competitive only if the input is enriched with indices.“ |
| 3.3, S. 26 | „… die für die Modelle ungesehenen IOB2-Symbole“ | Lu et al.: „special, previously unseen symbols, such as the diverse BIO tags“ |
| 4.4.5, S. 43 | „spart sie im Mittel rund 0,37 Bit je Parameter“ | Dettmers et al.: „saving an average of about 0.37 bits per parameter“ |
| 5.7.2, S. 68 | „dass feinjustierte Modelle JSON fast immer parsebar ausgeben“ | Dagdelen et al.: „Outputs are nearly always parsable“ |

Zum zweiten Eintrag: Der Ergebnisteil von Semin et al. schränkt stärker ein als ihr Fazitsatz („this method still underperforms other methods, especially with weaker models“). Die Arbeit übernimmt den Fazitsatz.

**Fehlende oder unklare Kennzeichnung:** keine gefunden. Die Abbildungen 2.1 bis 2.3 sind als eigene Darstellungen „in Anlehnung an“ die jeweilige Quelle ausgewiesen.

**Übersetzte Fachbegriffe:** einheitlich; englische Originalbegriffe stehen bei der Einführung in Klammern. Zwei Stellen, an denen die Übersetzung die Quelle verschiebt, sind G9 („nur den Typ“) und G10 („unverändert“).

**Nicht abschließend beurteilbar**

- Passagen, die sich auf die vier nicht verfügbaren Quellen und auf Hripcsak & Rothschild stützen (1.1, 2.1, 2.9, 4.5.1).
- Übernahmen aus Texten, die nicht im Verzeichnis stehen (Lehrbücher, andere Abschlussarbeiten, Webseiten, Dokumentation). Dagegen wurde nicht geprüft.
- Eigene frühere Arbeiten: Es lagen keine Vergleichstexte vor.
- Die Erklärung nennt die Nutzung von KI-Werkzeugen für Entwurf, Überarbeitung und Verdichtung von Textpassagen. Solche Werkzeuge können quellennahe Formulierungen erzeugen, ohne dass es auffällt. Die Prüfung gegen die zitierten Quellen hat dafür keinen Hinweis ergeben; gegen andere Texte ist das offen.

## G. Priorisierter Korrekturplan

### Zwingende Korrekturen

1. W1: Autorenname `Yan, Bowen` bei `ding2024gner`.
2. W2: `zhan2026generativeNER` auf die Findings-Fassung umstellen, Fundstellen dort bestätigen, Satz zu den „zwei Preprints“ in Kapitel 3 anpassen.
3. W3 und W4: Keraghel et al. als Beleg an drei Stellen in Kapitel 4 streichen oder ersetzen.
4. W5: Aussage zur „komplementären Kapazität“ auf den Befund von Dettmers et al. zurückführen.
5. W15: IOB2 in 4.5.1 mit `tjong1999representing` belegen.
6. W8: Erwartung E3 umformulieren.

### Wichtige Prüfungen an den Originalquellen

1. W7: Lu et al., Abbildung 3 der AMIA-Fassung (typweise Werte nach LoRA).
2. W13 (c): Qwen3.5-Blog im Browser öffnen; trägt er „erstes Qwen3.5-Modell im Februar 2026“ und den Bezug zu Qwen2.5/Qwen3?
3. W16: Konferenzarbeit von Keraghel & Nadif (2025) einsehen.
4. Manning et al. 2008 (Mikro-/Makro-Mittelung; P/R/F), Manning & Schütze 1999 (Definition NLP, Merkmale), Efron & Tibshirani 1993, Hripcsak & Rothschild 2005 am Text bestätigen. Zusammen elf Zitatstellen.
5. Fundstellenangaben an den Verlagsfassungen bestätigen, wo nur arXiv vorlag: Obeidat et al. (Abschn. II-B, IV; Tab. III, IV), Sainz et al. (Tab. 3, 6, 9, 10; Abschn. 5.2), Zhou et al. (Tab. 2; Abschn. 5.2), Hu et al. (Abschn. 4.1), He et al. 2023 (Abschn. 4.1).
6. W6, W9 bis W12, W14: Formulierungen an die Quellen angleichen.

### Formale Vereinheitlichungen

1. F2: Datum der 27B-Karte an die zitierte Revision angleichen; 397B-Karte mit Revision zitieren.
2. F3: Online-Einträge ohne Datum mit Jahr versehen.
3. F4: DOI oder Ausgabe bei Efron & Tibshirani.
4. F1: `su2024roformer` aus der Literaturdatei nehmen.
5. Drei Zitate in 4.3.1 stehen mit normalem statt geschütztem Leerzeichen vor `\cite` (Quelltextzeilen 223, 225, 238 in `04_methodik.tex`).
6. G-Punkte nach Wahl.

### Offene Fragen an dich oder den Betreuer

1. **Seitenangaben.** Die Arbeit nennt bei konkreten Zahlen Tabellen oder Abschnitte, aber nie Seiten; bei allgemeinen Paraphrasen steht nur Autor und Jahr. Das entspricht dem APA-Stil und der Praxis im Fach. Ein verbindlicher Leitfaden des Fachgebiets liegt mir nicht vor. Der allgemeine Leitfaden „AssisThesis“ der TU Berlin, den ich in deinem Download-Ordner gefunden habe, bezeichnet sich selbst als nicht verbindlich und behandelt Seitenangaben in Kurzbelegen als Regelfall. Frage an den Betreuer: Genügen Tabellen- und Abschnittsangaben?
2. **Datum für Online-Quellen ohne Datum:** Abrufdatum, Datum der letzten Änderung oder Erscheinen der Softwareversion?
3. **Software ohne Eintrag** (PyTorch, Datasets, Accelerate; CrossNER-Repository): ergänzen oder bei der Nennung mit Version belassen?
4. **Zitierempfehlungen der Projekte:** PEFT und TRL empfehlen Einträge mit Personenautoren (Mangrulkar et al.; von Werra et al.). Die Arbeit zitiert versionsgebunden mit „Hugging Face“ als Autor. Beides ist vertretbar; gewünscht?
5. **Ähnlichkeitsprüfung:** Soll vor der Abgabe ein Abgleich über einen von der TU bereitgestellten Dienst laufen? Ohne deine Zustimmung gebe ich keinen Text der Arbeit an einen externen Dienst.

## H. Abschlussbewertung

**Was geprüft und bestätigt wurde.** Alle 313 Zitierbefehle sind mit dem Verzeichnis abgeglichen. 56 der 85 zitierten Einträge sind gegen ein Register bestätigt, die übrigen gegen Verlags- oder Projektseiten oder gar nicht (Liste in E). Für 79 Quellen lag Text vor, für 56 in der zitierten Fassung. Jede Zitatstelle wurde gegen den vorhandenen Quelltext geprüft, die gewichtigen Abweichungen ein zweites Mal von mir. Alle wiedergegebenen Zahlen aus fremden Arbeiten stimmen.

**Was gefunden wurde.** Ein falscher Autorenname (W1), ein überholter Publikationsstand bei einer vielzitierten Quelle (W2), zwei weitere Preprints mit überholtem oder unklarem Stand (W16) und 13 Befunde, in denen eine Quelle die Aussage nicht oder nur teilweise trägt (W3 bis W15). Davon sind drei Fehlzuordnungen im engeren Sinn (W3, W4, W5); die übrigen sind Formulierungen, die über die Quelle hinausgehen. Hinzu kommen 22 kleinere Ungenauigkeiten und einige formale Punkte. Keiner dieser Befunde berührt die eigenen Ergebnisse oder die Schlussfolgerungen.

**Einschätzung.** Die Quellenverwendung wirkt sorgfältig und nachvollziehbar. Dafür sprechen die durchgängig genauen Fundstellen bei Zahlen, die Kennzeichnung von Preprints, die Trennung von Wiedergabe und eigener Einordnung in Kapitel 3, die offengelegten Grenzen der Vergleichbarkeit und die versionsgebundenen Belege für Software und Modelle. Die Schwächen liegen in Kapitel 4, wo Quellen mehrfach als Sammelbelege neben Beschreibungen der eigenen Umsetzung stehen, und bei den Qwen3.5-Angaben, die teils aus Konfigurationsdateien stammen, aber den Modellkarten zugeschrieben sind.

**Was offen bleibt.**
- Vier Quellen sind inhaltlich ungeprüft, eine nur bibliografisch; elf Zitatstellen hängen daran.
- 18 Quellen sind nur an einer anderen als der zitierten Fassung geprüft (17-mal arXiv); Fundstellennummern der zitierten Fassungen können abweichen.
- Ein Ähnlichkeitsvergleich mit einer autorisierten Datenbank (Turnitin, iThenticate oder Ähnliches) hat nicht stattgefunden. Übereinstimmungen mit Texten außerhalb des Literaturverzeichnisses kann diese Prüfung weder feststellen noch ausschließen.
- Die Urteile „passend“ der Teilprüfungen sind nicht einzeln von mir wiederholt worden.

Eine Aussage, die Arbeit sei frei von Plagiaten, lässt sich auf dieser Grundlage nicht treffen. Belegt ist: Gegen die zitierten Quellen gibt es keine wörtlichen Übernahmen und keine ungekennzeichneten Paraphrasen, und die gefundenen Mängel sind Ungenauigkeiten der Zuordnung, keine Übernahmen.

## Anhang zum Bericht: Hilfsdateien

Nicht im Repository, nur für diese Sitzung:

- Neubau des PDF (Seitentexte im selben Ordner): `/private/tmp/claude-501/-Users-luajosman-bachelor-thesis-llm-vs-encoder-ner/0a490e67-26c1-48d3-86fd-95233b8c9e8b/scratchpad/pdf/main_b4898f1_neubau.pdf`
- Registerdaten je Eintrag: `/private/tmp/claude-501/-Users-luajosman-bachelor-thesis-llm-vs-encoder-ner/0a490e67-26c1-48d3-86fd-95233b8c9e8b/scratchpad/meta/registry.json`
- Am 21.11. nachgeladene Quelltexte: `/private/tmp/claude-501/-Users-luajosman-bachelor-thesis-llm-vs-encoder-ner/0a490e67-26c1-48d3-86fd-95233b8c9e8b/scratchpad/src/`

## Nachtrag: Umsetzung der zwingenden Korrekturen (21.11.2026)

Nach Freigabe umgesetzt, nicht committet:

| Befund | Datei | Änderung |
|---|---|---|
| W1 | `references.bib` | `ding2024gner`: Autor `Yan, Bowen` |
| W2 | `references.bib` | `zhan2026generativeNER` als Beitrag in *Findings of the Association for Computational Linguistics: EMNLP 2026*, S. 36475–36493, DOI `10.18653/2026.findings-emnlp.1706`; Preprint-Vermerk entfällt |
| W2 | `03_verwandte_arbeiten.tex` | Kapiteleinleitung: „zwei Arbeiten aus 2026“ statt „zwei Preprints aus 2026“ |
| W3 | `04_methodik.tex`, 4.5.1 | Mikro-F₁ als primäre Kennzahl nur noch mit Manning et al. belegt |
| W4 | `04_methodik.tex`, 4.1 und 4.3.2 | Keraghel et al. gestrichen; Devlin et al. steht in 4.3.2 jetzt direkt bei der Regel zum ersten Subwort, der Satzteil zum Wert −100 ist Beschreibung der eigenen Umsetzung |
| W5 | `04_methodik.tex`, 4.4.5 | „folgt dem Befund von Dettmers et al. (2023), dass erst Adapter auf allen Schichten die Leistung des 16-Bit-Fine-Tunings erreichen“ |
| W8 | `03_verwandte_arbeiten.tex`, 3.5 | E3: „Ohne Anpassung der Gewichte bleiben per Prompting genutzte LLMs, in den zitierten Vergleichen selbst mit Beispielen im Prompt, deutlich hinter feinjustierten Modellen zurück“ |
| W15 | `04_methodik.tex`, 4.5.1 | IOB2 mit Tjong Kim Sang & Veenstra (1999) statt Ramshaw & Marcus (1995) belegt |

Prüfung: Neubau in einer Arbeitskopie ohne Fehler, ohne offene Zitate und ohne Overfull-Meldung; weiterhin 111 Seiten, alle Kapitelanfänge unverändert (Kapitel 3 endet weiter auf S. 30). Der Satzvergleich mit dem vorigen Bau zeigt nur die genannten Änderungen (S. 19, 30, 31, 35, 43, 45 und im Verzeichnis S. 79, 84 bis 86).

Zu W2: Tab. 3, Anh. D mit Tab. 11 und Anh. G mit Tab. 13 tragen in der Verlagsfassung dieselben Nummern; alle in der Arbeit genannten Kennzahlen stehen dort.

Folgen: `wang2024parameter` wird nicht mehr zitiert und erscheint nicht mehr im Verzeichnis (85 → 84 Einträge); der Eintrag steht wie `su2024roformer` noch in der Literaturdatei. `thesis/main.pdf` ist nicht neu gebaut.
