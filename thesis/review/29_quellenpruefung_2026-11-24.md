# Prüfbericht 29: Literaturverzeichnis und Quellenverweise

Stand: 24.11.2026. Reine Prüfung. An LaTeX-Dateien, an `references.bib` und an allen anderen Dateien der Arbeit wurde nichts geändert. Neu sind nur dieser Bericht und die zugehörige Matrix `29b_aussage_beleg_matrix_2026-11-24.md`; beide sind nicht committet.

Der Bericht ist vollständig, kein Zwischenstand: Alle 84 Literatureinträge und alle 298 Zitierbefehle sind erfasst und geprüft. Was nur eingeschränkt oder gar nicht geprüft werden konnte, steht in Abschnitt 6.

## 1. Kurzfazit

### Geprüfte Fassung

- **Quellstand:** Branch `lektorat-2026-11-25`, Commit `e82599f` (24.11.2026), identisch mit `main` und mit beiden Remote-Branches; Arbeitsbaum ohne offene Änderungen.
- **PDF:** `thesis/main.pdf` vom 22.11.2026, 11:58 Uhr, 110 Seiten. Die Quelldateien sind seit diesem Build unverändert, die Hilfsdateien in `thesis/build/` (bbl, bcf, Log) stammen aus demselben Lauf. Neu gebaut habe ich nicht, weil ein Build Dateien im Projekt überschreibt.
- **Nicht geprüft und nicht vermischt:** `thesis/main_long12112026.pdf` und der Branch `fassung-lang-2026-11-12` (ältere Langfassung).
- **Bibliografie:** einzige Datei ist `thesis/references.bib` mit 84 Einträgen. Gesetzt wird mit biblatex, Stil `apa`, Biber; `\cite` ist auf `\parencite` umgelegt. Ein verbindlicher Zitierleitfaden liegt im Repository nicht vor; ich habe deshalb gegen den tatsächlich verwendeten APA-Stil geprüft.
- **Seitenangaben** in diesem Bericht sind die gedruckten arabischen Seitenzahlen der Arbeit. Die PDF-Seite ist jeweils um 9 höher (Literaturverzeichnis: gedruckt S. 77–85, PDF-Seiten 86–94).
- **Umfang der Durchsicht:** Abstract, Kapitel 1 bis 6, alle Tabellenlegenden und Bildunterschriften, Literaturverzeichnis, Anhang A.1 bis A.3. Fußnoten gibt es nicht. Erklärung, Abkürzungsverzeichnis, Abbildungsdateien und die erzeugten Tabellenkörper (`generated/`) enthalten keine Zitate und keine fremden Quellenangaben; sie wurden nur daraufhin durchsucht.

### Gesamtbild

- **Literaturverzeichnis:** in gutem Zustand. Alle 84 Einträge sind eindeutig identifiziert; kein Zitierschlüssel ist ungelöst, keiner doppelt, keiner unbenutzt. Zwei Einträge sollten angepasst werden (Titelzusatz bei Hoffmann et al.; nicht erreichbare URL bei Manning und Schütze), einer ist nicht bestätigt (Qwen-Blog). Der Rest sind Hinweise.
- **Zahlen aus der Literatur:** Jede geprüfte Zahl, die die Arbeit aus einer Quelle wiedergibt, steht so in der Quelle (unter anderem 65,17/90,98; 93,85/93,04; 4,02; 14,78/92,77; 76,0; 93,73/91,25; 55,4; 40 bis 220; 0,14 bis 0,48; 1,76 bis 10,22; alle Werte aus Tab. 5 von Tedeschi und Navigli). Ich habe keinen Zahlenfehler gefunden.
- **Tragfähigkeit der Belege:** Von 508 geprüften Einzelaussagen tragen die Quellen 417 in der vorliegenden Form, 83 nur teilweise, 2 gar nicht, 1 widerspricht einer Tabellenzelle, 5 waren nicht prüfbar. „Teilweise“ reicht dabei von einer ungenauen Tabellennummer bis zu einer Sammelaussage, die weiter geht als die Quellen. Die gewichtigen Fälle sind die zehn Punkte A1 bis A10 in Abschnitt 5.
- **Fehlzuordnungen im engeren Sinn** (falsches Werk gemeint) gibt es keine. Es gibt Stellen, an denen die Quelle nur einen Teil der Aussage trägt und der Rest an einer anderen, teils schon vorhandenen Quelle hängt.
- **Übernahmen:** Die Arbeit enthält keine wörtlichen Zitate aus der Literatur. Die lokale Wortfolgensuche (sechs gleiche Wörter in Folge gegen 87 Quelltextdateien) ergab nur Modellnamen, Parameterwerte und eine Lizenzbezeichnung. Vollständige Plagiatsfreiheit kann ich damit nicht bescheinigen: Geprüft wurde nur gegen die zitierten Quellen, und eine deutsche Paraphrase englischer Texte fällt bei einer Wortfolgensuche nicht auf. An einen externen Prüf- oder Suchdienst wurde kein Text der Arbeit gegeben.

### Die wichtigsten Risiken

1. **Die Sammelaussage zu den direkten Vergleichen und Erwartung E1 (A1).** „Mit Adaptern feinjustierte LLMs erreichen auf etablierten Benchmarks das Niveau feinjustierter Encoder oder übertreffen es leicht“ gilt bei Zhan et al. nur für CoNLL-2003 und bei Dukić und Šnajder nur für CoNLL-2003 mit entfernter Maske. Das ist die folgenreichste Stelle, weil E1 daraus abgeleitet und in Kapitel 6 abgeglichen wird.
2. **Zwei Vergleichssätze in der Diskussion (A4, A5).** Tam et al. berichten Einbußen nur bei Schlussfolgerungsaufgaben und Gewinne bei Klassifikation; Semin et al. zeigen für NER keinen durchgehenden Gewinn durch LogitMatch; „Gewinn durch die Anpassung übersteigt den Gewinn durch die Größe“ ist eine eigene Ableitung, die als Befund von GoLLIE und UniversalNER dasteht.
3. **Belege, die die Aussage nicht enthalten (A2, A6).** Die 512 Positionen von DeBERTa-v3 stehen nicht in den zitierten Modellkarten, sondern in der Konfigurationsdatei. Dagdelen et al. werden viermal für Dinge zitiert, die dort nicht stehen (Positionsangaben, Rückführung auf IOB2, API-Szenario).
4. **Tabelle 3.2 (A7).** Acht Zellen passen nicht zur eigenen Legende der Tabelle, darunter eine, die der Quelle widerspricht (GLiNER nennt eine Trainingszeit).

### Zuerst angehen

A1, A2, A3 und A8 sind jeweils mit einem Satz oder einem Zusatz erledigt und beseitigen die sichtbarsten Angriffsflächen. Danach A4 bis A7 und A9, A10. Die bibliografischen Punkte C1 bis C3 kosten zusammen wenige Minuten. Die B-Punkte sind Präzisierungen, die du nach Zeit und Geschmack übernehmen kannst.

### Verhältnis zum Bericht 25 vom 21.11.

Diese Prüfung ist unabhängig neu durchgeführt, mit neu beschafften Quelltexten in den zitierten Fassungen. Die damals umgesetzten Punkte (W1 bis W5, W8, W15 und die formalen Korrekturen) sind in der aktuellen Fassung bestätigt. Von den damals offenen Punkten bestehen fort: W6 (jetzt A9), W9 (A1), W10 (A3), W11 (A5), W13 (A10), W14 (A6) sowie die meisten der damals als geringfügig eingestuften Punkte (jetzt unter B). **Erledigt hat sich W7:** Abbildung 3 bei Lu et al. zeigt das mit LoRA angepasste Modell bei allen fünf Typen unter BioClinicalBERT; „bleibt auf allen Typen darunter“ ist damit belegt. W12 ist durch die jetzige Formulierung („Befolgen von Instruktionen“) gedeckt. Neu sind unter anderem A4, A7, A8 und der Befund zu den OntoNotes-Typen in A9.

## 2. Bibliografieprüfung

### Vorgehen

Jeder Eintrag wurde mit mindestens einer maßgeblichen Metadatenquelle verglichen: Crossref (über den DOI), DataCite und die arXiv-Schnittstelle (für arXiv-Einträge mit Versionsstand), ACL Anthology (BibTeX-Export und PDF-Titelseite), die Proceedings-Seiten von NeurIPS, JMLR, AAAI und JAIR, PubMed, der TMLR-Index, die ICLR-Tagungsseite, Hugging Face (Commit-Historie und Datei der zitierten Revision), GitHub und PyPI. Verglichen wurden Autorennamen und Reihenfolge, Titel, Jahr, Ort der Veröffentlichung, Band, Heft, Seiten oder Artikelnummer, DOI und URL sowie, ob DOI und URL zum angegebenen Werk führen. Wo Register und PDF voneinander abweichen, ist das als Konflikt vermerkt und nicht nach Plausibilität aufgelöst.

Einschränkungen: OpenReview und DBLP verlangen eine Browserprüfung und waren automatisch nicht abrufbar; die zehn Einträge mit OpenReview-Link (ICLR, TMLR) sind über arXiv-Metadaten, den TMLR-Index, die ICLR-Tagungsseite und Semantic Scholar bestätigt, der Link selbst nicht. Die ACM Digital Library und IEEE Xplore liefern nur eine Sperrseite; dort stützt sich die Prüfung auf Crossref.

### Ergebnis

| Prüfstatus | Anzahl |
|---|---|
| verifiziert | 61 |
| verifiziert, mit Hinweis | 17 |
| teilweise verifiziert | 3 |
| Korrektur empfohlen | 2 |
| offen | 1 |

Befunde im Einzelnen (Maßnahmen in Abschnitt 5, Teil C):

- **Hoffmann et al. (2022), Titelkonflikt.** Die Proceedings-Seite listet den Beitrag als „An empirical analysis of compute-optimal large language model training“. Das dort hinterlegte PDF und der Crossref-Eintrag zum DOI tragen den Titel „Training Compute-Optimal Large Language Models“. Der Zusatz im Verzeichnis („Preprint-Fassung unter dem Titel …“) stimmt deshalb nicht ganz: Auch die veröffentlichte Fassung trägt diesen Titel.
- **Manning und Schütze (1999), URL.** `https://nlp.stanford.edu/fsnlp/` war am 24.11.2026 von zwei verschiedenen Netzen aus nicht erreichbar (Verbindung abgelehnt). Die Buchangaben selbst sind über die Archivkopie der Seite vom 16.10.2026 bestätigt.
- **Qwen-Blog (2026a).** Die Seite liefert ohne Browser nur eine leere JavaScript-Hülle. Titel und Monat decken sich mit der Zitierempfehlung in den Qwen3.5-Modellkarten („Qwen3.5: Towards Native Multimodal Agents“, February 2026); den Inhalt konnte ich nicht lesen.
- **Konflikte zwischen Register und PDF, bei denen der Eintrag richtig ist:** Ding et al. (ACL Anthology und Crossref führen „Bowen, Yan“; das PDF nennt Bowen Yan), Tam et al. (Register mit abweichendem Titelwortlaut), Dettmers et al. 2022 (Proceedings-Seite mit „GPT3.int8()“), Brown et al. und Wolf et al. (Kurzformen von Vornamen im Register).
- **Fassungen:** Alle neun als arXiv-Preprint zitierten Arbeiten sind in der angegebenen Version vorhanden; bei Keraghel et al., Semin et al., Ashok und Lipton sowie Wang et al. (InstructUIE) vermerkt arXiv keine begutachtete Fassung. Beim Qwen2.5-Bericht gibt es eine neuere Version 2 mit einem zusätzlichen Autor; zitiert und korrekt wiedergegeben ist Version 1. Zhan et al. ist richtig auf die Findings-Fassung umgestellt. Ein Rückzug oder eine Korrekturmeldung ist bei keinem Eintrag mit Crossref-Datensatz verzeichnet.
- **Software, Datensätze, Modelle:** Die sieben zitierten Hugging-Face-Revisionen existieren und tragen die angegebenen Daten; sechs davon sind die in Tabelle A.3 genannten Revisionen der verwendeten Modelle und des Datensatzes, die siebte gehört zur Karte des nicht verwendeten 397B-Modells. PEFT 0.19.1, TRL 1.8.0 und seqeval 1.2.2 stimmen mit den Angaben in Anhang A.2.3 überein und sind unter diesen Versionsnummern veröffentlicht; dasselbe gilt für die dort ohne Zitat genannten Versionen von Transformers, PyTorch, bitsandbytes, Accelerate und Datasets.
- **Gesetztes Verzeichnis:** keine Platzhalter, kein „o. J.“, keine abgeschnittenen Einträge. Gleichnamige Erstautoren werden im Text richtig mit Initialen unterschieden (A. Yang/S. Yang, J. Lu/Q. Lu, J. Wei/X. Wei); die Suffixe 2026a bis 2026e (Qwen Team) und 2026a/b (Hugging Face) stimmen zwischen Text und Verzeichnis überein. Zwei Schönheitsfehler: Bei OntoNotes fehlt im Verzeichnis der Herausgeber (Linguistic Data Consortium), und „LJ Miranda“ erscheint als „Miranda, L.“.

### Tabelle aller Einträge

Die Spalte „Zitierstellen“ zählt die Verwendungen des Schlüssels in der Arbeit.

| Schlüssel | Identifizierte Publikation | Nachweisquelle | Zitierstellen | Prüfstatus | Hinweis oder Korrektur |
|---|---|---|---|---|---|
| `ainslie2023gqa` | Ainslie et al. (2023), GQA, EMNLP 2023, S. 4895–4901 | Crossref (DOI), ACL Anthology | 1 | verifiziert | – |
| `ashok2023promptner` | Ashok & Lipton (2023), PromptNER, arXiv:2305.15444v2 | arXiv-API, DataCite | 3 | verifiziert | v2 (20.06.2023) ist die aktuelle Fassung; keine begutachtete Fassung im arXiv-Eintrag vermerkt |
| `babelscape2023multinerdcard` | Babelscape (2023), Datensatzkarte Babelscape/multinerd, Revision 2814b78 | Hugging Face (Commit-Historie, README der Revision) | 4 | verifiziert | Commit vom 20.04.2023; Revision stimmt mit Tab. A.3 überein; Lizenzangabe steht in der Karte |
| `bengio2013representation` | Bengio, Courville & Vincent (2013), TPAMI 35(8), 1798–1828 | Crossref (DOI) | 1 | verifiziert | – |
| `bogdanov2024nuner` | Bogdanov et al. (2024), NuNER, EMNLP 2024, S. 11829–11841 | Crossref, ACL Anthology | 9 | verifiziert | – |
| `brown2020language` | Brown et al. (2020), NeurIPS 33, S. 1877–1901 | NeurIPS-Proceedings (Seite und PDF) | 7 | verifiziert, Hinweis | Namensformen folgen dem PDF (Tom B. Brown, Daniel M. Ziegler, Christopher Hesse); die Proceedings-Seite führt Kurzformen. Kein Änderungsbedarf |
| `cawley2010selectionbias` | Cawley & Talbot (2010), JMLR 11(70), 2079–2107 | JMLR-Seite | 1 | verifiziert | – |
| `clark2020electra` | Clark et al. (2020), ELECTRA, ICLR 2020 | arXiv 2003.10555 (Kommentar „ICLR 2020“) | 1 | verifiziert, Hinweis | OpenReview-Link nicht maschinell prüfbar (Browserprüfung); im Browser kontrollieren |
| `dagdelen2024structuredIE` | Dagdelen et al. (2024), Nature Communications 15(1), Art. 1418 | Crossref (DOI), Verlagsseite | 7 | verifiziert | – |
| `dettmers2022llmint8` | Dettmers et al. (2022), LLM.int8(), NeurIPS 35, S. 30318–30332 | Crossref (DOI), NeurIPS-Proceedings, PDF | 2 | verifiziert, Hinweis | Die Proceedings-Seite führt den Titel als „GPT3.int8(): …“; PDF und Crossref haben „LLM.int8(): …“ wie im Eintrag. Kein Änderungsbedarf |
| `dettmers2023qlora` | Dettmers et al. (2023), QLoRA, NeurIPS 36, S. 10088–10115 | Crossref (DOI), NeurIPS-Proceedings | 9 | verifiziert | – |
| `devlin2019bert` | Devlin et al. (2019), BERT, NAACL-HLT 2019, S. 4171–4186 | Crossref, ACL Anthology | 12 | verifiziert | – |
| `ding2024gner` | Ding et al. (2024), GNER, Findings of ACL 2024, S. 3461–3475 | Crossref, ACL Anthology, PDF | 2 | verifiziert, Hinweis | Konflikt: ACL Anthology und Crossref führen den fünften Autor als „Bowen, Yan“; das PDF nennt „Bowen Yan“ (Mail yanbw@…). Der Eintrag („Yan, Bowen“) folgt dem PDF und ist richtig |
| `dukic2024lookingright` | Dukić & Šnajder (2024), Findings of ACL 2024, S. 14168–14181 | Crossref, ACL Anthology | 5 | verifiziert | – |
| `efron1993bootstrap` | Efron & Tibshirani (1993), An Introduction to the Bootstrap, Chapman & Hall | OpenAlex (Ausgabe 1993, DOI 10.1007/978-1-4899-4541-9 der digitalisierten Ausgabe) | 1 | teilweise verifiziert | Buch nicht eingesehen; Reihe und Nummer (Monographs on Statistics and Applied Probability 57) nur aus dem Gedächtnis plausibel, nicht am Register bestätigt. Kein DOI im Eintrag ist in Ordnung |
| `geng2023grammar` | Geng et al. (2023), EMNLP 2023, S. 10932–10952 | Crossref, ACL Anthology | 1 | verifiziert | – |
| `goldberg2016primer` | Goldberg (2016), JAIR 57, 345–420 | Crossref (DOI), JAIR-Seite | 1 | verifiziert | – |
| `han2024parameter` | Han et al. (2024), TMLR | TMLR-Index (jmlr.org/tmlr, BibTeX zum OpenReview-Schlüssel) | 3 | verifiziert | – |
| `he2020deberta` | He et al. (2021), DeBERTa, ICLR 2021 | arXiv 2006.03654, Semantic Scholar (DBLP conf/iclr/HeLGC21) | 1 | verifiziert, Hinweis | OpenReview-Link nicht maschinell prüfbar. Schlüsselname mit 2020 ist nur intern |
| `he2021debertav3` | He, Gao & Chen (2023), DeBERTaV3, ICLR 2023 | arXiv 2111.09543 (Kommentar „published as a conference paper at ICLR 2023“) | 8 | verifiziert, Hinweis | OpenReview-Link nicht maschinell prüfbar |
| `he2021debertav3card` | Microsoft (2022), Modellkarte microsoft/deberta-v3-base, Revision 8ccc9b6 | Hugging Face (Commit-Historie, README) | 2 | verifiziert | Commit 22.09.2022; Revision = Tab. A.3; Karte nennt MIT, 86M Backbone, 128K Vokabular |
| `he2021debertav3cardlarge` | Microsoft (2023), Modellkarte microsoft/deberta-v3-large, Revision 64a8c8e | Hugging Face (Commit-Historie, README) | 3 | verifiziert | Commit 19.03.2023; Revision = Tab. A.3; Karte nennt MIT, 304M Backbone, 128K Vokabular |
| `hoffmann2022training` | Hoffmann et al. (2022), NeurIPS 35, S. 30016–30030 | Crossref (DOI), NeurIPS-Proceedings (Seite und PDF) | 1 | Korrektur empfohlen | Titelkonflikt: Die Proceedings-Seite listet „An empirical analysis of compute-optimal large language model training“; das dort hinterlegte PDF und Crossref tragen „Training Compute-Optimal Large Language Models“. Der Zusatz im Eintrag („Preprint-Fassung unter dem Titel …“) ist deshalb ungenau: Auch die veröffentlichte Fassung trägt diesen Titel. Vorschlag: Titel des PDF verwenden und im Zusatz vermerken, dass die Proceedings-Seite den anderen Titel führt. Namensformen (Tom Hennigan, Katie Millican, Karen Simonyan, Jack W. Rae) folgen dem PDF |
| `holtzman2020curious` | Holtzman et al. (2020), ICLR 2020 | arXiv 1904.09751 (Kommentar „Published in ICLR 2020“) | 1 | verifiziert, Hinweis | OpenReview-Link nicht maschinell prüfbar |
| `hripcsak2005agreement` | Hripcsak & Rothschild (2005), JAMIA 12(3), 296–298 | Crossref (DOI), PubMed/PMC | 1 | verifiziert | Crossref führt nur den Erstautor; PMC beide |
| `hu2021lora` | Hu et al. (2022), LoRA, ICLR 2022 | ICLR-Tagungsseite (iclr.cc, Poster 2022), arXiv 2106.09685 | 11 | verifiziert, Hinweis | OpenReview-Link nicht maschinell prüfbar |
| `jurafsky2024speech` | Jurafsky & Martin (2026), Speech and Language Processing, 3. Aufl., Online-Manuskript vom 19.08.2026 | Autorenseite (Zitierempfehlung dort) | 3 | verifiziert, Hinweis | Titel, Jahr und Fassungsdatum stimmen mit der Zitierempfehlung überein. Keine der drei Zitierstellen nennt ein Kapitel; bei einem laufend geänderten Manuskript ist die Kapitelangabe sinnvoll |
| `kaplan2020scaling` | Kaplan et al. (2020), arXiv:2001.08361v1 | arXiv-API, DataCite | 2 | verifiziert | – |
| `keraghel2024recent` | Keraghel, Morbieu & Nadif (2024), arXiv:2401.10825v3 | arXiv-API, DataCite | 16 | verifiziert | v3 (20.12.2024) ist die aktuelle Fassung; im arXiv-Eintrag ist keine begutachtete Fassung vermerkt |
| `koehn2004bootstrap` | Koehn (2004), EMNLP 2004, S. 388–395 | ACL Anthology | 1 | verifiziert | – |
| `kudo2018sentencepiece` | Kudo & Richardson (2018), SentencePiece, EMNLP 2018 System Demonstrations, S. 66–71 | Crossref, ACL Anthology | 1 | verifiziert | – |
| `kudo2018subword` | Kudo (2018), Subword Regularization, ACL 2018, S. 66–75 | Crossref, ACL Anthology | 1 | verifiziert | – |
| `lafferty2001crf` | Lafferty, McCallum & Pereira (2001), ICML 2001, S. 282–289 | OpenAlex (Jahr, Seiten) | 1 | teilweise verifiziert | ACM-Link sperrt automatische Abrufe (403); Verlag Morgan Kaufmann und ISBN nicht am Register bestätigt |
| `lakens2017equivalence` | Lakens (2017), SPPS 8(4), 355–362 | Crossref (DOI) | 1 | verifiziert | – |
| `lample2016neural` | Lample et al. (2016), NAACL-HLT 2016, S. 260–270 | Crossref, ACL Anthology | 1 | verifiziert | – |
| `li2018survey` | Li, Sun, Han & Li (2022), IEEE TKDE 34(1), 50–70 | Crossref (DOI) | 8 | verifiziert | Online vorab 2020, Heft 2022; der Eintrag nennt zu Recht 2022 |
| `li2023codeie` | Li et al. (2023), CodeIE, ACL 2023, S. 15339–15353 | Crossref, ACL Anthology | 2 | verifiziert | – |
| `liu2019roberta` | Liu et al. (2019), RoBERTa, arXiv:1907.11692v1 | arXiv-API, DataCite | 1 | verifiziert | – |
| `liu2021crossner` | Liu et al. (2021), CrossNER, AAAI 35(15), 13452–13460 | Crossref (DOI), AAAI-Seite | 5 | verifiziert | – |
| `lu2024llmstruggle` | Lu, Q. et al. (2025), AMIA Annual Symposium Proceedings 2024, 748–757 | PubMed (PMID 40417588) | 7 | verifiziert, Hinweis | PubMed: Band 2024, online 22.05.2025. Das Jahr 2025 folgt dem Erscheinungsdatum, der Band der Tagung 2024; beides ist vertretbar und so im Eintrag angelegt |
| `lu2024padellm` | Lu, J. et al. (2024), PaDeLLM-NER, NeurIPS 37, S. 117853–117880 | Crossref (DOI), NeurIPS-Proceedings | 4 | verifiziert | – |
| `ma2016end` | Ma & Hovy (2016), ACL 2016, S. 1064–1074 | Crossref, ACL Anthology | 1 | verifiziert | – |
| `mangrulkar2022peft` | Hugging Face (2026), PEFT Version 0.19.1 | GitHub (Tag v0.19.1), PyPI (Release 16.04.2026) | 1 | verifiziert | Version stimmt mit Anhang A.2.3 überein |
| `manning1999foundations` | Manning & Schütze (1999), Foundations of Statistical NLP, MIT Press | Begleitseite des Buchs (Archivkopie vom 16.10.2026) | 3 | Korrektur empfohlen | Die im Verzeichnis gedruckte URL https://nlp.stanford.edu/fsnlp/ war am 24.11.2026 von zwei Netzen aus nicht erreichbar (Verbindung abgelehnt). Vor der Abgabe erneut prüfen; sonst URL streichen (für ein gedrucktes Buch nicht nötig) oder Archivlink angeben |
| `manning2008introduction` | Manning, Raghavan & Schütze (2008), Introduction to Information Retrieval, CUP | Crossref (DOI), Begleitseite (Archivkopie) | 5 | verifiziert | Im Verzeichnis erscheint nur der DOI; die nicht erreichbare Stanford-URL steht nur in der bib-Datei |
| `mayhew2024uner` | Mayhew et al. (2024), Universal NER, NAACL 2024, S. 4322–4337 | Crossref, ACL Anthology | 1 | verifiziert, Hinweis | Formal: „LJ Miranda“ wird im Verzeichnis zu „Miranda, L.“ gekürzt; die Autorin/der Autor führt „LJ“ als Namen. Optional {LJ} schützen |
| `nakayama2018seqeval` | Nakayama (2018), seqeval, verwendete Version 1.2.2 | PyPI (Version 1.2.2 vom 24.10.2020; Zitierempfehlung im README) | 2 | verifiziert | Titel, Autor und Jahr 2018 entsprechen der Zitierempfehlung des Projekts |
| `obeidat2025llm` | Obeidat, Al Nahian & Kavuluru (2025), IEEE ICHI 2025, S. 352–358 | Crossref (DOI), PubMed | 18 | verifiziert | – |
| `osi2024osaid` | Open Source Initiative (2024), The Open Source AI Definition – 1.0 | OSI-Seite (Veröffentlichungsdatum 28.10.2024 in den Metadaten) | 1 | verifiziert | – |
| `ouyang2022instructgpt` | Ouyang et al. (2022), NeurIPS 35, S. 27730–27744 | Crossref (DOI), NeurIPS-Proceedings | 1 | verifiziert | – |
| `qwen2024qwen25` | Yang, A. et al. (2024), Qwen2.5 Technical Report, arXiv:2412.15115v1 | arXiv-API, DataCite | 1 | verifiziert, Hinweis | Zitiert ist v1 (19.12.2024, 41 Autoren, stimmt). Es gibt v2 vom 03.01.2025 mit einem zusätzlichen Autor; nur relevant, falls auf v2 umgestellt wird |
| `qwen2025qwen3` | Yang, A. et al. (2025), Qwen3 Technical Report, arXiv:2505.09388v1 | arXiv-API, DataCite | 2 | verifiziert | – |
| `qwen2026card08b` | Qwen Team (2026), Modellkarte Qwen3.5-0.8B, Revision 2fc0636 | Hugging Face (Commit-Historie, README) | 5 | verifiziert | Commit 02.03.2026; Revision = Tab. A.3 |
| `qwen2026card27b` | Qwen Team (2026), Modellkarte Qwen3.5-27B, Revision fc05dae | Hugging Face (Commit-Historie, README) | 10 | verifiziert | Commit 24.04.2026; Revision = Tab. A.3 |
| `qwen2026card397b` | Qwen Team (2026), Modellkarte Qwen3.5-397B-A17B, Revision 8472618 | Hugging Face (Commit-Historie, README) | 1 | verifiziert | Commit 24.04.2026 |
| `qwen2026card4b` | Qwen Team (2026), Modellkarte Qwen3.5-4B, Revision 851bf6e | Hugging Face (Commit-Historie, README) | 5 | verifiziert | Commit 02.03.2026; Revision = Tab. A.3 |
| `qwen2026qwen35blog` | Qwen Team (2026), Blogbeitrag „Qwen3.5: Towards Native Multimodal Agents“ | keine: Seite liefert ohne Browser nur eine leere JavaScript-Hülle | 1 | offen | Titel, Datum (Februar 2026) und Inhalt konnten nicht bestätigt werden. Bitte im Browser prüfen und, wenn möglich, das genaue Datum eintragen |
| `ramponi2025fallacy` | Ramponi, Daffara & Tonelli (2025), NAACL 2025, S. 762–784 | Crossref, ACL Anthology | 4 | verifiziert | – |
| `ramshaw1995text` | Ramshaw & Marcus (1995), Third Workshop on Very Large Corpora, S. 82–94 | ACL Anthology (Seite W95-0107), PDF | 1 | teilweise verifiziert | Titel, Autoren, Jahr und Band bestätigt; die Seitenangabe 82–94 steht auf der Anthology-Seite nicht maschinenlesbar und wurde nicht bestätigt |
| `ravikumar2026lostformatting` | Ravikumar, Ibrahim & Batista-Navarro (2026), EACL 2026, S. 5498–5513 | Crossref, ACL Anthology | 1 | verifiziert | – |
| `sainz2024gollie` | Sainz et al. (2024), GoLLIE, ICLR 2024 | arXiv 2310.03668 (Kommentar „ICLR 2024“) | 13 | verifiziert, Hinweis | OpenReview-Link nicht maschinell prüfbar |
| `sang2003conll` | Tjong Kim Sang & De Meulder (2003), CoNLL-2003, S. 142–147 | Crossref (DOI), ACL Anthology | 9 | verifiziert | – |
| `schuster2012japanese` | Schuster & Nakajima (2012), ICASSP 2012, S. 5149–5152 | Crossref (DOI) | 1 | verifiziert | – |
| `segura2013semeval` | Segura-Bedmar, Martínez & Herrero-Zazo (2013), SemEval 2013, S. 341–350 | ACL Anthology | 2 | verifiziert | – |
| `semin2026spanLabeling` | Semin, Dušek & Kasner (2026), arXiv:2601.16946v2 | arXiv-API, DataCite | 14 | verifiziert | v2 (08.07.2026) ist die aktuelle Fassung; keine begutachtete Fassung vermerkt |
| `tam2024speakfreely` | Tam et al. (2024), EMNLP 2024 Industry Track, S. 1218–1236 | Crossref, ACL Anthology, PDF | 2 | verifiziert, Hinweis | Konflikt: ACL Anthology und Crossref führen den Titel als „… On Large Language Model Performance.“; das PDF trägt den Titel des Eintrags. Kein Änderungsbedarf |
| `tan2023damo` | Tan et al. (2023), SemEval-2023, S. 2014–2028 | Crossref, ACL Anthology | 7 | verifiziert | Crossref führt nur zehn Autoren; ACL Anthology elf wie im Eintrag |
| `tedeschi2022multinerd` | Tedeschi & Navigli (2022), MultiNERD, Findings of NAACL 2022, S. 801–812 | Crossref, ACL Anthology | 18 | verifiziert | – |
| `tjong1999representing` | Tjong Kim Sang & Veenstra (1999), EACL 1999, S. 173–179 | ACL Anthology | 2 | verifiziert | – |
| `vanrijsbergen1979information` | van Rijsbergen (1979), Information Retrieval, 2. Aufl., London: Butterworths | Online-Fassung des Autors (Vorwort zur 2. Auflage) | 2 | verifiziert | – |
| `vaswani2017attention` | Vaswani et al. (2017), NeurIPS 30, S. 5998–6008 | NeurIPS-Proceedings, OpenAlex (Seiten) | 4 | verifiziert | Die Proceedings-Seite nennt keine Seitenzahlen; 5998–6008 nach OpenAlex |
| `vonwerra2026trldocs` | Hugging Face (2026), TRL-Dokumentation 1.8.0, SFT Trainer | Dokumentationsseite (versionsgebundene URL), PyPI (Release 09.07.2026) | 2 | verifiziert | Version stimmt mit Anhang A.2.3 überein |
| `wang2023instructuie` | Wang, X. et al. (2023), InstructUIE, arXiv:2304.08085v1 | arXiv-API, DataCite | 1 | verifiziert | – |
| `wang2025gptner` | Wang, S. et al. (2025), GPT-NER, Findings of NAACL 2025, S. 4257–4275 | Crossref, ACL Anthology | 5 | verifiziert | – |
| `wei2022emergent` | Wei, J. et al. (2022), TMLR | TMLR-Index (BibTeX zum OpenReview-Schlüssel) | 2 | verifiziert | – |
| `wei2023chatie` | Wei, X. et al. (2023), ChatIE, arXiv:2302.10205v2 | arXiv-API, DataCite | 2 | verifiziert | v2 vom 27.05.2024 wie im Zusatz angegeben |
| `weischedel2013ontonotes` | Weischedel et al. (2013), OntoNotes Release 5.0, LDC2013T19 | DataCite (DOI), LDC-Katalog | 3 | verifiziert, Hinweis | Formal: Im gesetzten Verzeichnis fehlt der Herausgeber „Linguistic Data Consortium“ (Eintragstyp misc gibt publisher nicht aus). Vorschlag: Typ @dataset oder Angabe im Zusatzfeld |
| `wolf2020transformers` | Wolf et al. (2020), EMNLP 2020 System Demonstrations, S. 38–45 | Crossref, ACL Anthology, PDF | 2 | verifiziert | „Alexander M. Rush“ nach PDF |
| `xie2023zeroshot` | Xie et al. (2023), EMNLP 2023, S. 7935–7956 | Crossref, ACL Anthology | 3 | verifiziert | – |
| `yang2025gated` | Yang, S., Kautz & Hatamizadeh (2025), ICLR 2025 | arXiv 2412.06464 (Kommentar „ICLR 2025 camera ready“), Semantic Scholar | 1 | verifiziert, Hinweis | OpenReview-Link nicht maschinell prüfbar |
| `zaratiana2024gliner` | Zaratiana et al. (2024), GLiNER, NAACL 2024, S. 5364–5376 | Crossref, ACL Anthology | 8 | verifiziert | – |
| `zhan2026generativeNER` | Zhan, Wang & Huang (2026), Findings of EMNLP 2026, S. 36475–36493 | Crossref (DOI), ACL Anthology | 24 | verifiziert | Der DOI 10.18653/2026.findings-emnlp.1706 hat kein „v1/“; so steht er in ACL Anthology und Crossref, und er löst korrekt auf |
| `zhao2023survey` | Zhao et al. (2026), Frontiers of Computer Science 20(12), Art. 2012627 | Crossref (DOI), arXiv 2303.18223 (journal_ref) | 3 | verifiziert | Autorenliste (23) folgt der Zeitschriftenfassung; die arXiv-Fassung hat eine andere Reihenfolge |
| `zhou2024universalner` | Zhou et al. (2024), UniversalNER, ICLR 2024 | arXiv 2308.03279 (Kommentar „Accepted at ICLR 2024“) | 10 | verifiziert, Hinweis | OpenReview-Link nicht maschinell prüfbar |

## 3. Aussage-Beleg-Matrix

### Aufbau

Die vollständige Matrix steht in `29b_aussage_beleg_matrix_2026-11-24.md`: 508 Zeilen, je eine pro Einzelaussage und Quelle, mit der Aussage der Arbeit, der nachgeschlagenen Fundstelle im Original (Abschnitt, Tabelle oder Abbildung, mit Kurzzitat oder Werten), Status, Begründung und Maßnahme. Die folgende Tabelle fasst sie auf die 298 Zitierbefehle zusammen. Bei Befehlen mit mehreren Teilaussagen oder Quellen nennt die Statusspalte alle vorkommenden Urteile; die Spalte „Verweis“ führt zum Korrekturplan in Abschnitt 5.

Statusdefinitionen: **Trägt** (die Passage stützt die Aussage in der vorliegenden Form), **Trägt teilweise** (nur ein Teil oder eine engere Formulierung ist belegt), **Kein Beleg** (thematisch passend, stützt die konkrete Behauptung nicht), **Widerspricht**, **Nicht verifizierbar** (Passage nicht zugänglich).

### Ergebnis in Zahlen

| | Trägt | Trägt teilweise | Kein Beleg | Widerspricht | Nicht verifizierbar | Summe |
|---|---|---|---|---|---|---|
| Einzelaussagen (Matrixzeilen) | 417 | 83 | 2 | 1 | 5 | 508 |
| Zitierbefehle (schlechtestes Urteil zählt) | 224 | 67 | 1 | 1 | 5 | 298 |

Nach Kapiteln (Matrixzeilen, nicht „Trägt“ / gesamt): Kapitel 1, 6 und Anhang 8/55; Kapitel 2 26/85; Kapitel 3 30/274; Kapitel 4 20/62; Kapitel 5 7/32. Kapitel 3 hat viele Zeilen, weil dort jede Detailangabe zu einer Arbeit und jede Zelle von Tabelle 3.2 einzeln geprüft wurde.

### Zusammenfassung je Zitierbefehl

| ID | Abschnitt | S. | Quelle(n) | Belegte Aussage (Kurzform) | Status | Verweis |
|---|---|---|---|---|---|---|
| #1 | 1.1 | 1 | `jurafsky2024speech`, `manning1999foundations` | NLP umfasst Verfahren zur automatischen Verarbeitung und Analyse von Text | Nicht verifizierbar, trägt | E1 |
| #2 | 1.1 | 1 | `li2018survey` | NER: Erwähnungen benannter Entitäten lokalisieren und vorgegebenen Kategorien zuordnen | Trägt |  |
| #3 | 1.1 | 1 | `jurafsky2024speech`, `keraghel2024recent` | NER-Ergebnisse dienen als Grundlage für Relationsextraktion, Wissensgraphen, Frage-Antwort-Systeme | Trägt teilweise | B1 |
| #4 | 1.1 | 1 | `sang2003conll` | CoNLL-2003 unterscheidet Personen, Organisationen, Orte und Restkategorie | Trägt |  |
| #5 | 1.1 | 1 | `tedeschi2022multinerd` | MultiNERD: 15 Kategorien u. a. Krankheiten, Ereignisse, Medien | Trägt |  |
| #6 | 1.1 | 1 | `devlin2019bert`, `he2021debertav3` | Encoder berechnen kontextabhängige Token-Repräsentation, Klassifikator vergibt Label je Token … | Trägt |  |
| #7 | 1.1 | 1 | `brown2020language`, `zhao2023survey` | LLMs sind große generative Modelle, die ein breites Aufgabenspektrum mit einem Modell bearbeiten | Trägt |  |
| #8 | 1.1 | 1 | `zhan2026generativeNER`, `semin2026spanLabeling` | LLMs lösen NER, indem sie Entitäten als strukturierten Text ausgeben | Trägt |  |
| #9 | 1.1 | 1 | `brown2020language` | Zero-Shot: nur Aufgabenbeschreibung im Prompt, kein Training auf annotierten Daten | Trägt |  |
| #10 | 1.1 | 1 | `zhan2026generativeNER` | LLM wird alternativ mit annotierten Beispielen angepasst | Trägt |  |
| #11 | 1.1 | 2 | `he2021debertav3` | Encoder wie DeBERTa-v3 umfassen einige hundert Millionen Parameter | Trägt teilweise | B2 |
| #12 | 1.1 | 2 | `zhao2023survey`, `qwen2026card397b` | aktuelle LLMs haben Milliarden bis Hunderte Milliarden Parameter | Trägt |  |
| #13 | 1.1 | 2 | `lu2024padellm` | LLMs erzeugen Token für Token, Inferenzzeit wächst mit Ausgabelänge | Trägt |  |
| #14 | 1.1 | 2 | `devlin2019bert` | Encoder bestimmt Labels eines Satzes in einem einzigen Vorwärtsdurchlauf | Trägt teilweise | B3 |
| #15 | 1.1 | 2 | `hu2021lora` | Vollständiges Fine-Tuning hält Gewichte, Gradienten und Optimiererzustände im Speicher | Trägt |  |
| #16 | 1.1 | 2 | `han2024parameter` | PEFT trainiert nur kleine Zahl zusätzlicher oder ausgewählter Parameter | Trägt |  |
| #17 | 1.1 | 2 | `dettmers2023qlora` | QLoRA erlaubt Anpassung von Modellen mit Dutzenden Mrd. Parametern auf einer GPU | Trägt |  |
| #18 | 1.2 | 2 | `keraghel2024recent` | Vergleichsstudie von Keraghel et al. (Encoder vs. GPT-4o) | Trägt |  |
| #19 | 1.2 | 2 | `keraghel2024recent` | GPT-4o mit Beispielen: Makro-F1 65,17 auf CoNLL-2003, DeBERTa-v3-large 90,98 (Tab. 5) | Trägt |  |
| #20 | 1.2 | 2 | `zhan2026generativeNER` | LLaMA3.1-8B bestes Format 93,85 F1 auf CoNLL-2003 vs. 93,04 BERT-MRC; Format beeinflusst Leistung stark | Trägt |  |
| #21 | 1.2 | 2 | `obeidat2025llm` | fünf biomed. Datensätze, Encoder Full-FT vs. QLoRA-LLMs (7-8 Mrd.); LLMs auf vier Datensätzen 2-8 F1 vorn; … | Trägt |  |
| #22 | 1.2 | 3 | `tedeschi2022multinerd` | englischer Teil von MultiNERD mit 15 Entitätstypen | Trägt |  |
| #23 | 1.2 | 3 | `he2021debertav3` | DeBERTa-v3-base/-large als Modelle (Fine-Tuning) | Trägt |  |
| #24 | 1.2 | 3 | `qwen2026card08b`, `qwen2026card4b`, `qwen2026card27b` | Qwen3.5-0.8B (Modell) | Trägt |  |
| #25 | 1.2 | 3 | `ramponi2025fallacy` | JSON-Format mit Wortindizes: Eingabetokens durchnummeriert, Ausgabe über Indizes (Tab. 9) | Trägt |  |
| #26 | 1.2 | 3 | `semin2026spanLabeling` | Einordnung als Indexing-Variante | Trägt |  |
| #27 | 1.2 | 4 | `liu2021crossner` | CrossNER: Wikipedia-Texte aus fünf Domänen | Trägt |  |
| #28 | 1.3 | 4 | `keraghel2024recent` | DeBERTa-v3-base/-large als besonders leistungsfähig in vergleichenden NER-Studien | Trägt |  |
| #29 | 1.3 | 4 | `devlin2019bert` | Vollständiges Fine-Tuning: Encoder-Gewichte und Klassifikationskopf gemeinsam trainiert | Trägt |  |
| #30 | 1.3 | 4 | `hu2021lora`, `dettmers2023qlora` | LoRA: nur Low-Rank-Adapter trainiert, Basismodell eingefroren | Trägt |  |
| #31 | 1.3 | 4 | `tedeschi2022multinerd` | Englischer Teil, Texte aus Wikipedia und Wikinews; automatische Annotation | Trägt |  |
| #32 | 1.3 | 5 | `manning2008introduction`, `sang2003conll` | Precision, Recall, F1 | Trägt |  |
| #33 | 2.1 | 6 | `jurafsky2024speech`, `manning1999foundations` | NLP = Verfahren, mit denen Computer natürliche Sprache (geschrieben/gesprochen) verarbeiten, analysieren, … | Nicht verifizierbar, trägt | E1 |
| #34 | 2.1 | 6 | `manning1999foundations` | NLP-Systeme beruhten lange auf von Hand entworfenen Merkmalen (Wortformen, Wortarten, Namenslisten), die … | Nicht verifizierbar | E1 |
| #35 | 2.1 | 6 | `bengio2013representation` | Representation Learning lernt Repräsentationen aus Daten, als dichte reellwertige verteilte Vektoren | Trägt |  |
| #36 | 2.1 | 6 | `goldberg2016primer` | Wortvektoren aus großen Textmengen gelernt, ähnliche Kontexte = nahe Vektoren | Trägt |  |
| #37 | 2.1 | 6 | `devlin2019bert` | kontextualisierte Repräsentationen: Netz liest ganze Sequenz, Vektor je Token abhängig von umgebenden Tokens | Trägt |  |
| #38 | 2.1 | 6 | `devlin2019bert`, `brown2020language` | Vortrainieren auf nicht annotiertem Text, danach Fine-Tuning der Gewichte | Trägt |  |
| #39 | 2.2 | 7 | `li2018survey` | NER = Erwähnungen lokalisieren und vordefinierter Kategorie zuordnen | Trägt |  |
| #40 | 2.2 | 7 | `li2018survey` | Formalisierung als Tripel (Start-, Endindex, Typ), "eigene Formalisierung nach" | Trägt |  |
| #41 | 2.2 | 7 | `li2018survey` | Abb. 1 als Vorlage der Abbildung (Tokenfolge -> Tripel) | Trägt |  |
| #42 | 2.2 | 7 | `li2018survey` | NER wird "meist" als Sequenzlabeling formuliert: je Token genau ein Label, kodiert Span-Segmentierung und Typ | Trägt teilweise | B4 |
| #43 | 2.2 | 7 | `ramshaw1995text` | Tagset {I,O,B} für Chunking; I innerhalb, O außerhalb, B nur bei unmittelbar vorangehendem Chunk | Trägt |  |
| #44 | 2.2 | 8 | `tjong1999representing` | IOB1/IOB2-Bezeichnungen "prägen" Tjong Kim Sang/Veenstra; IOB2 = jeder Chunk beginnt mit B | Trägt |  |
| #45 | 2.2 | 8 | `sang2003conll` | CoNLL-2003 nimmt flache, nicht überlappende/verschachtelte Entitäten an | Trägt |  |
| #46 | 2.2 | 8 | `sang2003conll` | CoNLL-2003-Daten liegen im IOB1-Schema vor | Trägt |  |
| #47 | 2.2 | 8 | `lafferty2001crf` | CRFs bestimmten lange den Stand der Technik für NER; modellieren bedingte Wahrscheinlichkeit der ganzen … | Trägt teilweise | B4 |
| #48 | 2.2 | 8 | `lample2016neural`, `ma2016end` | BiLSTM-CRF ersetzte handentworfene Merkmale durch gelernte Repräsentationen | Trägt |  |
| #49 | 2.3 | 8 | `vaswani2017attention` | Transformer beruht vollständig auf Attention | Trägt |  |
| #50 | 2.3 | 8 | `vaswani2017attention` | Self-Attention: Query/Key/Value, Skalarprodukt, Softmax, gewichtete Summe der Values | Trägt |  |
| #51 | 2.3 | 9 | `vaswani2017attention` | Abb. in Anlehnung: Encoder (2 Teilschichten, Residual+LayerNorm), Decoder mit dritter Teilschicht und … | Trägt |  |
| #52 | 2.3 | 9 | `devlin2019bert` | Encoder-only-Modelle wie BERT: nur Encoder, Kontext in beide Richtungen | Trägt |  |
| #53 | 2.3 | 9 | `brown2020language` | Decoder-only-Modelle wie GPT-3: nur Decoder ohne Encoder-Attention; kausale Maske, Texterzeugung Token für … | Trägt teilweise | B5 |
| #54 | 2.3 | 10 | `vaswani2017attention` | Kosten der Self-Attention wachsen quadratisch mit Sequenzlänge | Trägt |  |
| #55 | 2.3 | 10 | `ainslie2023gqa` | GQA: mehrere Query-Köpfe teilen K/V-Kopf, verkleinert Zwischenspeicher | Trägt |  |
| #56 | 2.3 | 10 | `yang2025gated` | lineare Attention (z. B. Gated DeltaNet) ersetzt Relevanzmatrix durch Zustand fester Größe, Aufwand linear | Trägt |  |
| #57 | 2.3 | 10 | `qwen2026card27b` | "Die in dieser Arbeit verwendeten LLMs" wechseln lineare Schichten mit einzelnen Schichten voller Attention ab | Trägt teilweise | A10 |
| #58 | 2.4 | 10 | `devlin2019bert` | Transformer-Encoder seit BERT die Standardarchitektur für überwachte NER | Trägt teilweise | B4 |
| #59 | 2.4 | 10 | `schuster2012japanese` | WordPiece als verbreitetes Subwort-Verfahren | Trägt |  |
| #60 | 2.4 | 10 | `liu2019roberta` | RoBERTa nutzt Byte-Paar-Kodierung auf Byte-Ebene | Trägt |  |
| #61 | 2.4 | 10 | `kudo2018sentencepiece`, `kudo2018subword` | SentencePiece als verbreitetes Subwort-Verfahren | Trägt teilweise, trägt | B6 |
| #62 | 2.4 | 10 | `he2021debertav3` | DeBERTa-v3 nutzt SentencePiece-Vokabular mit 128 000 Einheiten | Trägt |  |
| #63 | 2.4 | 10 | `devlin2019bert` | Erstes-Subwort-Verfahren; übrige Subwörter ohne Label, gehen nicht in den Verlust ein | Trägt teilweise | B7 |
| #64 | 2.4 | 10 | `devlin2019bert` | BERT-Vortraining mit MLM und NSP; Fine-Tuning aller Gewichte mit aufgabenspezifischer Ausgabeschicht | Trägt |  |
| #65 | 2.4 | 11 | `he2021debertav3card`, `he2021debertav3cardlarge` | Eingabelänge von DeBERTa-v3 laut Modellkonfiguration auf 512 Positionen begrenzt | Kein Beleg | A2 |
| #66 | 2.4 | 11 | `he2020deberta` | DeBERTa verändert zwei Komponenten: Disentangled Attention (Inhalt/relative Position, drei Terme, ohne … | Trägt |  |
| #67 | 2.4 | 11 | `he2021debertav3` | DeBERTa-v3 behält Architektur bei und ersetzt MLM durch Replaced Token Detection (nach ELECTRA) | Trägt |  |
| #68 | 2.4 | 11 | `clark2020electra` | RTD: Signal an allen Positionen statt nur maskierten, dateneffizienter | Trägt |  |
| #69 | 2.4 | 11 | `he2021debertav3` | weitgehend dieselben Einstellungen wie DeBERTa; dieselben 160 GB wie DeBERTa-v2; Abschn. 4.1 | Trägt |  |
| #70 | 2.4 | 11 | `he2021debertav3`, `he2021debertav3cardlarge` | Base: 12 Schichten, 86 Mio. Backbone-Parameter; Large 24 Schichten | Trägt |  |
| #71 | 2.5 | 11 | `brown2020language`, `zhao2023survey` | LLM = Sprachmodell mit Milliarden Parametern, auf sehr großen Korpora vortrainiert, ein Modell für breites … | Trägt teilweise, trägt | B8 |
| #72 | 2.5 | 12 | `brown2020language` | Training: autoregressive Modellierung, Vorhersage des nächsten Tokens je Position | Trägt |  |
| #73 | 2.5 | 12 | `holtzman2020curious` | Greedy, Sampling, Temperatur, Top-k, Top-p (Nucleus) mit Definition (kleinste Menge mit kumulierter … | Trägt |  |
| #74 | 2.5 | 12 | `ouyang2022instructgpt` | Nachtraining ändert Befolgen von Anweisungen, u. a. durch SFT auf menschlich geschriebenen Antworten | Trägt |  |
| #75 | 2.5 | 12 | `qwen2026card27b` | Nachtrainierte Modelle verarbeiten Dialogformat mit Rollen System/Nutzer/Assistent, das ein Chat-Template … | Trägt teilweise | A10 |
| #76 | 2.5 | 12 | `qwen2025qwen3` | Qwen-Modelle ab Qwen3 können abgegrenzten Denkblock erzeugen; Thinking-Modus ein-/ausschaltbar | Trägt |  |
| #77 | 2.5 | 12 | `brown2020language` | In-Context Learning; Anweisung plus optionale Beispiele im Prompt, keine Parameteränderung; Zero-/Few-Shot | Trägt |  |
| #78 | 2.5 | 12 | `wei2022emergent` | Manche Fähigkeiten treten erst ab gewisser Modellgröße auf; Wei et al. nennen sie emergent | Trägt |  |
| #79 | 2.5 | 12 | `osi2024osaid` | OSAID verlangt zusätzlich Angaben zu Trainingsdaten und vollständigen Code; solche Modelle gelten nicht … | Trägt |  |
| #80 | 2.5 | 13 | `qwen2024qwen25`, `qwen2025qwen3` | Qwen-Familie von Alibaba mit Generationen Qwen2.5, Qwen3, Qwen3.5; Gruppe der Modelle mit offenen Gewichten | Trägt teilweise | A10 |
| #81 | 2.5 | 13 | `qwen2026qwen35blog` | Im Februar 2026 erschien das erste Qwen3.5-Modell | Nicht verifizierbar | A10, E3 |
| #82 | 2.5 | 13 | `qwen2026card27b`, `qwen2026card4b`, `qwen2026card08b` | Auf das erste Qwen3.5-Modell folgten Modelle ohne Mixture-of-Experts, darunter Qwen3.5-27B | Trägt teilweise | A10 |
| #83 | 2.5 | 13 | `dagdelen2024structuredIE` | NER generativ: Entitäten als strukturierte JSON-Liste mit Angaben zu Position und Typ | Trägt teilweise | A6 |
| #84 | 2.5 | 13 | `semin2026spanLabeling` | Ausgabe kann formal fehlerhaft sein: ungültige Indizes, Typen außerhalb des Inventars | Trägt |  |
| #85 | 2.5 | 13 | `geng2023grammar` | Constrained Decoding bindet Erzeugung an formale Grammatik | Trägt |  |
| #86 | 2.6 | 13 | `han2024parameter` | PEFT hält vortrainierte Gewichte eingefroren, trainiert kleine Zahl zusätzlicher oder ausgewählter Parameter | Trägt |  |
| #87 | 2.6 | 13 | `hu2021lora` | LoRA trainiert niedrigdimensionale Darstellung der Gewichtsänderung | Trägt |  |
| #88 | 2.6 | 13 | `hu2021lora` | Rang r bestimmt Kapazität und Zahl trainierbarer Parameter r(d+k) statt d*k; alpha skaliert den Beitrag … | Trägt teilweise | B9 |
| #89 | 2.6 | 14 | `hu2021lora` | A zufällig, B mit null initialisiert, Delta W anfangs null | Trägt |  |
| #90 | 2.6 | 14 | `hu2021lora` | Hu et al. adaptieren in ihren Experimenten Query- und Value-Projektionen | Trägt |  |
| #91 | 2.6 | 14 | `hu2021lora` | Auf RoBERTa, DeBERTa, GPT-2, GPT-3 erreicht LoRA mit Bruchteil der Parameter gleiche oder bessere Qualität … | Trägt |  |
| #92 | 2.6 | 14 | `hu2021lora` | Figurenunterschrift: Eigene Darstellung in Anlehnung an Hu et al. | Trägt |  |
| #93 | 2.6 | 14 | `dettmers2022llmint8` | Quantisierung: Gewichte auf Stufen abbilden (256 bei 8 Bit); Block durch Skalierungskonstante normiert, … | Trägt teilweise | B10 |
| #94 | 2.6 | 14 | `dettmers2023qlora` | QLoRA: Basismodell eingefroren, 4 Bit gespeichert, in 16 Bit gerechnet, Adapter in 16-Bit trainiert | Trägt |  |
| #95 | 2.6 | 14 | `dettmers2023qlora` | Drei Komponenten: NF4 (16 Stufen, gleich viele Werte je Stufe), Double Quantization … | Trägt |  |
| #96 | 2.6 | 15 | `dettmers2023qlora` | 65 Mrd. Parameter auf einer GPU mit 48 GB; auf geprüften Benchmarks (GLUE, Super-NaturalInstructions: … | Trägt |  |
| #97 | 2.7 | 15 | `kaplan2020scaling` | Skalierungsgesetze als Potenzgesetze: Verlust fällt als Potenzfunktion von Parameterzahl, Datenmenge, … | Trägt |  |
| #98 | 2.8 | 15 | `vanrijsbergen1979information`, `manning2008introduction` | Precision, Recall und F-Maß stammen aus dem Information Retrieval | Trägt teilweise, trägt | B11 |
| #99 | 2.8 | 15 | `vanrijsbergen1979information` | Das F-Maß fasst beide in einer Kennzahl zusammen | Trägt teilweise | B11 |
| #100 | 2.8 | 15 | `sang2003conll`, `keraghel2024recent` | F1 ist in der NER-Forschung üblicherweise berichtete Kennzahl (harmonisches Mittel) | Trägt |  |
| #101 | 2.8 | 15 | `sang2003conll` | Entitätsebene: korrekt nur bei exakter Übereinstimmung von Spangrenzen und Typ | Trägt |  |
| #102 | 2.8 | 16 | `segura2013semeval` | SemEval-2013 definiert neben streng weniger strenge Modi: nur Grenzen, nur Typ, oder Überlappung | Trägt teilweise | B12 |
| #103 | 2.8 | 16 | `nakayama2018seqeval` | seqeval hat zwei Modi; Standardmodus ahmt CoNLL-Skript nach und toleriert unzulässige Tagfolgen, ein I-Tag … | Trägt teilweise | B13 |
| #104 | 2.8 | 16 | `manning2008introduction` | Mikro: TP/FP/FN über Typen aggregiert, häufige Typen dominieren; Makro: Mittel der je Typ berechneten F1, … | Trägt |  |
| #105 | 2.8 | 16 | `hripcsak2005agreement` | F-Maß entspricht beim Vergleich zweier Annotationen der positiven spezifischen Übereinstimmung | Trägt | E4 |
| #106 | 3 | 17 | `zhan2026generativeNER`, `semin2026spanLabeling` | Zwei Arbeiten aus 2026 wurden in die Auswahl einbezogen | Trägt |  |
| #107 | 3.1 | 17 | `sang2003conll` | CoNLL-2003: englische und deutsche Nachrichtentexte (+1 Teilaussagen) | Trägt |  |
| #108 | 3.1 | 17 | `li2018survey` | CoNLL-2003 sei "bis heute der am häufigsten verwendete NER-Benchmark" | Trägt teilweise | A9 |
| #109 | 3.1 | 18 | `weischedel2013ontonotes`, `li2018survey` | OntoNotes 5.0: manuell annotiertes Korpus mit 18 Entitätstypen, mehrere Genres, Englisch, Chinesisch, Arabisch | Trägt teilweise, trägt | A9 |
| #110 | 3.1 | 18 | `tedeschi2022multinerd` | MultiNERD: zehn Sprachen (+3 Teilaussagen) | Trägt |  |
| #111 | 3.1 | 18 | `tedeschi2022multinerd` | Qualitätsmessung auf 1 000 zufällig gezogenen, manuell geprüften englischen Sätzen mit mindestens 20 … | Trägt |  |
| #112 | 3.1 | 18 | `tedeschi2022multinerd` | Zuverlässig annotiert: PER, ORG, LOC, CEL, DIS, EVE, MEDIA (+2 Teilaussagen) | Trägt |  |
| #113 | 3.1 | 18 | `tedeschi2022multinerd` | Über alle Klassen F1 0,85 der Silberreferenz gegenüber manueller Korrektur (+3 Teilaussagen) | Trägt |  |
| #114 | 3.1 | 18 | `liu2021crossner` | CrossNER wurde zur NER-Evaluation über Domänengrenzen entworfen | Trägt |  |
| #115 | 3.1 | 18 | `liu2021crossner` | Fünf Zieldomänen: Politik, Naturwissenschaften, Musik, Literatur, KI; Sätze aus Wikipedia, manuell … (+2 Teilaussagen) | Trägt |  |
| #116 | 3.1 | 19 | `sang2003conll` | Tab. 3.1 Zeile CoNLL-2003: EN, DE; Nachrichten; 4 Typen; manuell (IOB1) | Trägt teilweise | B14 |
| #117 | 3.1 | 19 | `weischedel2013ontonotes` | Tab. 3.1 Zeile OntoNotes: EN, ZH, AR; Nachrichten, Rundfunk, Gespräche, Web; 18 Typen; manuell | Trägt teilweise | A9 |
| #118 | 3.1 | 19 | `tedeschi2022multinerd` | Tab. 3.1 Zeile MultiNERD: 10 Sprachen; Wikipedia, Wikinews; 15 Typen; automatisch (Silberreferenz) + … | Trägt |  |
| #119 | 3.1 | 19 | `liu2021crossner` | Tab. 3.1 Zeile CrossNER: EN; Wikipedia, 5 Domänen; 9-17 je Domäne; manuell | Trägt |  |
| #120 | 3.2.1 | 19 | `tan2023damo` | MultiCoNER II: 33 feingranulare Typen, zwölf Sprachen (+2 Teilaussagen) | Trägt |  |
| #121 | 3.2.1 | 19 | `tan2023damo` | ChatGPT (gpt-3.5-turbo) mit drei Prompt-Varianten auf dem Validierungssplit (Z. 171-172) (+2 Teilaussagen) | Trägt |  |
| #122 | 3.2.1 | 19 | `keraghel2024recent` | Zehn NER-Datensätze aus allgemeinen und spezialisierten Domänen; MultiNERD nicht enthalten (+4 Teilaussagen) | Trägt |  |
| #123 | 3.2.1 | 20 | `keraghel2024recent` | GPT-4o bleibt auf neun von zehn Datensätzen deutlich hinter feinjustierten Encodern; CoNLL-2003 Makro-F1 … | Trägt |  |
| #124 | 3.2.2 | 20 | `zaratiana2024gliner` | GLiNER auf Basis von DeBERTa-v3; Typbezeichnungen und Text in gemeinsamer Sequenz; Training auf Pile-NER … (+4 Teilaussagen) | Trägt |  |
| #125 | 3.2.2 | 20 | `bogdanov2024nuner` | NuNER: Vortraining für anschließende Feinjustierung statt Zero-Shot; RoBERTa-base kontrastiv auf 4,38 Mio. … | Trägt |  |
| #126 | 3.2.2 | 20 | `bogdanov2024nuner` | Gegenüber UniNER-7B mit 56-mal mehr Parametern ähnliche Transferfähigkeiten im Few-Shot-Fine-Tuning auf … | Trägt |  |
| #127 | 3.2.2 | 20 | `bogdanov2024nuner` | Vergleich mit GPT-3.5/GPT-4 per ICL nur auf zwei verkleinerten Testsets; NuNER übertrifft GPT-3.5 bei … | Trägt |  |
| #128 | 3.2.2 | 20 | `tan2023damo`, `keraghel2024recent` | Vollständig feinjustierte Encoder bleiben Referenz auch bei feingranularen Typen; LLMs per Prompting ohne … | Trägt |  |
| #129 | 3.2.2 | 20 | `zaratiana2024gliner`, `bogdanov2024nuner` | Wissen aus LLMs über erzeugte Trainingsdaten in Encoder übertragen; Modelle mit Bruchteil der Parameter … | Trägt | B31 |
| #130 | 3.3.1 | 21 | `semin2026spanLabeling` | Drei Familien Tagging/Indexing/Matching; Tagging: Ausgabelänge linear in Eingabelänge; Indexing: nur von … (+4 Teilaussagen) | Trägt |  |
| #131 | 3.3.1 | 21 | `mayhew2024uner` | UNER = mehrsprachiger Datensatz Universal NER von Mayhew et al. | Trägt |  |
| #132 | 3.3.1 | 21 | `zhou2024universalner` | Abgrenzung: gleichnamiges Modell UniversalNER stammt von Zhou et al. | Trägt |  |
| #133 | 3.3.1 | 21 | `ramponi2025fallacy` | Indexing-Variante mit nummerierten Wörtern statt Zeichenpositionen geht auf Ramponi et al. (Tab. 9) zurück | Trägt |  |
| #134 | 3.3.1 | 21 | `zhan2026generativeNER` | Acht per LoRA trainierte LLMs, fünf Ausgabeformate; Format beeinflusst Leistung maßgeblich (+2 Teilaussagen) | Trägt |  |
| #135 | 3.3.1 | 21 | `ravikumar2026lostformatting` | Modelle 3 bis 32 Mrd. Parameter, QLoRA, sechs Ausgabeformate (+3 Teilaussagen) | Trägt teilweise, trägt | A3 |
| #136 | 3.3.1 | 21 | `dagdelen2024structuredIE` | Materialwissenschaftliche Extraktion mit feinjustierten Modellen (GPT-3, Llama-2) (+1 Teilaussagen) | Trägt |  |
| #137 | 3.3.1 | 21 | `lu2024padellm` | Latenz entsteht vor allem durch sequenzielles Decoding langer Ausgaben (+2 Teilaussagen) | Trägt teilweise, trägt | B15 |
| #138 | 3.3.1 | 22 | `semin2026spanLabeling` | Wortindex-Variante mit nummerierten Eingabetokens gehört zur Indexing-Familie; Semin beschreiben, aber … (+1 Teilaussagen) | Trägt |  |
| #139 | 3.3.2 | 22 | `wang2025gptner` | GPT-NER: GPT-3 erzeugt Eingabesatz mit Markern um Entitäten eines Typs; stärkste Variante wählt … | Trägt |  |
| #140 | 3.3.2 | 22 | `wang2025gptner` | Zusatz Abschn. 3.2.2: Demonstrationsauswahl über feinjustierten Tagger | Trägt |  |
| #141 | 3.3.2 | 22 | `wang2025gptner` | CoNLL-2003: 90,91 gegenüber 92,8 für feinjustierten BERT-Tagger | Trägt |  |
| #142 | 3.3.2 | 22 | `xie2023zeroshot` | Zero-Shot-NER mit ChatGPT in eine Abfrage je Entitätstyp zerlegt (+3 Teilaussagen) | Trägt |  |
| #143 | 3.3.2 | 22 | `ashok2023promptner` | PromptNER ergänzt Typdefinitionen und verlangt Kandidatenliste mit Begründungen (+2 Teilaussagen) | Trägt |  |
| #144 | 3.3.3 | 22 | `zhou2024universalner` | ChatGPT annotiert Pile-Passagen mit Entitäten frei gewählter Typen; Training von LLaMA-Modellen mit 7 und … (+2 Teilaussagen) | Trägt |  |
| #145 | 3.3.3 | 23 | `wang2023instructuie` | BERT-base-Werte (80,09 Mittel; MultiNERD 91,25) stammen aus der InstructUIE-Arbeit | Trägt |  |
| #146 | 3.3.3 | 23 | `zhou2024universalner` | Überwacht: UniNER-7B auf 20 Datensätzen im Mittel 84,78 gegenüber 80,09 für BERT-base (+2 Teilaussagen) | Trägt |  |
| #147 | 3.3.3 | 23 | `sainz2024gollie` | GoLLIE macht Annotationsrichtlinien nutzbar; Code-LLaMA 7/13/34 Mrd.; QLoRA; Schemata samt Richtlinien als … (+2 Teilaussagen) | Trägt |  |
| #148 | 3.3.3 | 23 | `sainz2024gollie` | CrossNER: GoLLIE-7B zwischen 55,5 (Science) und 67,8 (Music) (+1 Teilaussagen) | Trägt |  |
| #149 | 3.3.3 | 23 | `sainz2024gollie` | MultiNERD: 7B-Modell F1 76,0 | Trägt |  |
| #150 | 3.3.3 | 23 | `sainz2024gollie` | Acht der 15 MultiNERD-Typen kamen in anderer Benennung bereits im Training vor | Trägt |  |
| #151 | 3.3.3 | 23 | `ding2024gner` | GNER trainiert Flan-T5-Modelle bis 11 Mrd. und LLaMA-7B auf Pile-NER (+4 Teilaussagen) | Trägt |  |
| #152 | 3.3.4 | 23 | `semin2026spanLabeling`, `zhan2026generativeNER`, `lu2024padellm` | Sammelzitat: Formate mit Texteproduktion robust, aber teuer; Zeichenpositionen fehleranfällig | Trägt teilweise, trägt | B16 |
| #153 | 3.3.4 | 23 | `wang2025gptner`, `xie2023zeroshot`, `ashok2023promptner`, `li2023codeie` | Prompting-LLMs bleiben hinter überwachten Encodern zurück, auch mit Tagger-Demonstrationen; Stärke bei … | Trägt teilweise, trägt | B16 |
| #154 | 3.3.4 | 24 | `zhou2024universalner`, `sainz2024gollie` | Destillation/Fine-Tuning verringern Abstand, Vergleich mit Encodern stützt sich auf übernommene Werte oder … | Trägt |  |
| #155 | 3.4 | 24 | `zhan2026generativeNER` | Acht offene LLMs der Familien LLaMA, Qwen, Granite, Gemma mit 3 bis 9 Mrd. Parametern, LoRA (+2 Teilaussagen) | Trägt teilweise, trägt | B17 |
| #156 | 3.4 | 24 | `zhan2026generativeNER` | Im besten Format erreicht LLaMA3.1-8B auf CoNLL-2003 F1 93,85 gegenüber 93,04 für BERT-MRC (Tab. 3) (+1 Teilaussagen) | Trägt teilweise, trägt | A8 |
| #157 | 3.4 | 24 | `zhan2026generativeNER` | Kostenvergleich LLaMA3.1-8B vs. BERT-MRC, selbst trainiert auf einer einzelnen L40-GPU (+5 Teilaussagen) | Trägt |  |
| #158 | 3.4 | 24 | `obeidat2025llm` | Ressourcenaspekt im Mittelpunkt; fünf biomedizinische Datensätze, vollständig feinjustierte Encoder (+2 Teilaussagen) | Trägt teilweise, trägt | B18 |
| #159 | 3.4 | 24 | `obeidat2025llm` | Encoder mit 336 bis 435 Mio. Parametern (BERT-large, BiomedBERT-large, DeBERTa-v3-large), Abschn. II-B | Trägt |  |
| #160 | 3.4 | 24 | `obeidat2025llm` | Vier Datensätze LLMs 2 bis 8 F1-Punkte vorn, auf dem fünften 0,5 Punkte hinter dem besten Encoder (Tab. III) (+1 Teilaussagen) | Trägt |  |
| #161 | 3.4 | 25 | `obeidat2025llm` | Inferenzzeit je Testbeispiel das 40- bis 220-Fache des besten Encoders (Tab. IV) (+1 Teilaussagen) | Trägt |  |
| #162 | 3.4 | 25 | `lu2024llmstruggle` | RareDis-v1, fünf Typen zu seltenen Krankheiten; lokale 7-Mrd.-LLMs sowie GPT-3.5 und GPT-4 sollen … (+5 Teilaussagen) | Trägt teilweise, trägt | B19 |
| #163 | 3.4 | 25 | `zhan2026generativeNER` | In ihren Vergleichen auf CoNLL-2003 und GENIA liegt LoRA um 0,14 bis 0,48 F1-Punkte über vollständigem … (+1 Teilaussagen) | Trägt |  |
| #164 | 3.4 | 25 | `dukic2024lookingright` | Llama-2-7B und Mistral-7B mit Klassifikationskopf, QLoRA (+4 Teilaussagen) | Trägt |  |
| #165 | 3.4 | 25 | `zhan2026generativeNER`, `obeidat2025llm`, `dukic2024lookingright` | Sammelaussage: Mit Adaptern feinjustierte LLMs erreichen auf etablierten Benchmarks das Niveau … | Trägt teilweise, trägt | A1 |
| #166 | 3.4 | 25 | `lu2024llmstruggle` | Lu et al. finden im klinischen Setting mit LoRA einen bleibenden Rückstand | Trägt |  |
| #167 | 3.4 | 25 | `zhan2026generativeNER` | Zhan arbeiten generativ | Trägt |  |
| #168 | 3.4 | 25 | `obeidat2025llm` | Obeidat arbeiten generativ | Trägt |  |
| #169 | 3.4 | 25 | `lu2024llmstruggle` | Lu arbeiten generativ | Trägt |  |
| #170 | 3.4 | 25 | `dukic2024lookingright` | Dukić setzen Decoder mit Klassifikationskopf ein | Trägt |  |
| #171 | 3.4 | 25 | `obeidat2025llm` | Den Mehraufwand bei der Inferenz beziffern allein Obeidat (+1 Teilaussagen) | Trägt teilweise, trägt | B20 |
| #172 | 3.5 | 26 | `zhou2024universalner` | UniversalNER gewinnt vom 7B- zum 13B-Schülermodell im Zero-Shot leicht hinzu | Trägt |  |
| #173 | 3.5 | 26 | `sainz2024gollie` | GoLLIE 7/13/34 Mrd.: im Mittel steigende Zero-Shot-Werte bei stark unterschiedlichem Gewinn je Datensatz (+2 Teilaussagen) | Trägt |  |
| #174 | 3.5 | 26 | `zaratiana2024gliner`, `bogdanov2024nuner` | GLiNER mit Encodern weit unter 1 Mrd. Parametern erreicht das Niveau von LLMs mit 7 bis 13 Mrd. Parametern (+3 Teilaussagen) | Trägt | B31 |
| #175 | 3.5 | 26 | `obeidat2025llm` | Trainierbare Parameter über den LoRA-Rang an Encoder angeglichen (rund 341 bzw. 353 Mio. gegenüber 336 bis … (+1 Teilaussagen) | Trägt |  |
| #176 | 3.5 | 26 | `keraghel2024recent` | DeBERTa-v3-base liegt auf CoNLL-2003 und NCBI Disease knapp vor -large, auf den meisten anderen … | Trägt |  |
| #177 | 3.5 | 26 | `dukic2024lookingright` | Maskenentfernung wirkt erst ab einer gewissen Modellgröße | Trägt teilweise | B21 |
| #178 | 3.6 | 27 | `tedeschi2022multinerd` | Modelle: mBERT + BiLSTM-CRF (+4 Teilaussagen) | Trägt |  |
| #179 | 3.6 | 27 | `tan2023damo` | Modelle: XLM-R-large + Retrieval; ChatGPT (+4 Teilaussagen) | Trägt |  |
| #180 | 3.6 | 27 | `keraghel2024recent` | Modelle: BERT, XLM-R, DeBERTa-v3, GLiNER; GPT-4o (+4 Teilaussagen) | Trägt |  |
| #181 | 3.6 | 27 | `zaratiana2024gliner` | Modelle: DeBERTa-v3 (50–300 Mio.) (+4 Teilaussagen) | Widerspricht, trägt | A7 |
| #182 | 3.6 | 27 | `bogdanov2024nuner` | Modelle: RoBERTa-base; UniNER-7B; GPT-3.5/4 (+4 Teilaussagen) | Trägt |  |
| #183 | 3.6 | 27 | `semin2026spanLabeling` | Modelle: Qwen3, Mistral, Llama, Gemma (8–70 Mrd.) (+4 Teilaussagen) | Trägt teilweise, trägt | A7 |
| #184 | 3.6 | 27 | `lu2024padellm` | Modelle: Llama-2-7B, Baichuan2-7B (+4 Teilaussagen) | Trägt teilweise, trägt | A7 |
| #185 | 3.6 | 27 | `wang2025gptner` | Modelle: GPT-3 (Retrieval mit Tagger) (+4 Teilaussagen) | Trägt teilweise, trägt | A7 |
| #186 | 3.6 | 27 | `xie2023zeroshot` | Modelle: GPT-3.5 (+4 Teilaussagen) | Trägt |  |
| #187 | 3.6 | 27 | `ashok2023promptner` | Modelle: GPT-4, GPT-3.5, T5 (+4 Teilaussagen) | Trägt |  |
| #188 | 3.6 | 27 | `li2023codeie` | Modelle: Codex, GPT-3 (+4 Teilaussagen) | Trägt teilweise, trägt | A7 |
| #189 | 3.6 | 27 | `zhou2024universalner` | Modelle: LLaMA 7B/13B (+4 Teilaussagen) | Trägt |  |
| #190 | 3.6 | 27 | `sainz2024gollie` | Modelle: Code-LLaMA 7/13/34B (+4 Teilaussagen) | Trägt teilweise, trägt | A7 |
| #191 | 3.6 | 27 | `ding2024gner` | Modelle: Flan-T5 (bis 11 Mrd.), LLaMA-7B (+4 Teilaussagen) | Trägt |  |
| #192 | 3.6 | 27 | `zhan2026generativeNER` | Modelle: acht LLMs (3–9 Mrd.); BERT-MRC u. a. (+4 Teilaussagen) | Trägt teilweise, trägt | A7 |
| #193 | 3.6 | 27 | `obeidat2025llm` | Modelle: BERT-large, BiomedBERT, DeBERTa-v3-large; Mistral-7B, Llama-3.1-8B (+4 Teilaussagen) | Trägt |  |
| #194 | 3.6 | 27 | `lu2024llmstruggle` | Modelle: 7B-LLMs; GPT-3.5/4; BioClinicalBERT (+4 Teilaussagen) | Trägt |  |
| #195 | 3.6 | 27 | `dukic2024lookingright` | Modelle: Llama-2-7B, Mistral-7B; RoBERTa (+4 Teilaussagen) | Trägt |  |
| #196 | 3.6 | 27 | `weischedel2013ontonotes`, `li2018survey` | OntoNotes: 18 Typen aus mehreren Genres | Trägt teilweise, trägt | A9 |
| #197 | 3.6 | 27 | `obeidat2025llm` | Fachkorpora wie biomedizinische Datensätze | Trägt |  |
| #198 | 3.6 | 27 | `tan2023damo` | MultiCoNER II: 33 Typen, kurze kontextarme Sätze; nur ChatGPT per Prompting gegenüber Encodern | Trägt |  |
| #199 | 3.6 | 28 | `zhou2024universalner` | BERT-Werte aus InstructUIE auf dessen Datensatzvariante übernommen | Trägt |  |
| #200 | 3.6 | 28 | `zaratiana2024gliner` | GLiNER auf InstructUIE-Aufbereitung mit 10 000 Testbeispielen ausgewertet | Trägt teilweise | B22 |
| #201 | 3.6 | 28 | `sainz2024gollie` | Standardsplits zugesagt, Testumfang 32 908 | Trägt |  |
| #202 | 3.6 | 28 | `obeidat2025llm` | je Eingabetoken ein IOB2-Tag erzeugt | Trägt |  |
| #203 | 3.6 | 28 | `zhan2026generativeNER` | Inline-Markierungen wiederholen den Satz | Trägt |  |
| #204 | 3.6 | 28 | `zhou2024universalner`, `sainz2024gollie` | Größenstaffelung innerhalb von LLM-Familien | Trägt |  |
| #205 | 3.6 | 28 | `sainz2024gollie` | Trainingsressourcen je Größe (Anh. D.3, Tab. 10) | Trägt |  |
| #206 | 3.6 | 28 | `keraghel2024recent` | Encoder-Studien variieren die Größe nur innerhalb ihrer Klasse | Trägt |  |
| #207 | 3.6 | 28 | `zhan2026generativeNER` | weisen auf bekannte Lecks hin, ziehen CleanCoNLL heran | Trägt |  |
| #208 | 3.6 | 29 | `zhan2026generativeNER`, `obeidat2025llm` | E1: LoRA-LLM erreicht Encoder-Niveau oder übertrifft es leicht | Trägt teilweise, trägt | A1 |
| #209 | 3.6 | 29 | `obeidat2025llm` | E2: Faktoren von etwa 40 bis 220 (Tab. IV) | Trägt | B30 |
| #210 | 3.6 | 29 | `keraghel2024recent`, `tan2023damo` | E3: ohne Gewichtsanpassung deutlich hinter feinjustierten Modellen, selbst mit Beispielen | Trägt teilweise, trägt | B23 |
| #211 | 4 | 30 | `devlin2019bert`, `he2021debertav3` | Encoder sagen je Wort ein IOB2-Tag vorher | Trägt teilweise | B7 |
| #212 | 4 | 30 | `dagdelen2024structuredIE`, `semin2026spanLabeling` | LLMs erzeugen ein JSON-Array von Entitäten, regelbasierte Nachverarbeitung überführt es in IOB2 | Trägt teilweise | A6 |
| #213 | 4.1 | 32 | `tedeschi2022multinerd`, `babelscape2023multinerdcard` | englischer Teil von MultiNERD mit seinen offiziellen Splits | Trägt teilweise | B24 |
| #214 | 4.2.1 | 32 | `keraghel2024recent` | DeBERTa-Varianten zählen auf mehreren NER-Benchmarks zu den leistungsfähigsten Encodern und dienen häufig … | Trägt teilweise | B25 |
| #215 | 4.2.1 | 32 | `he2021debertav3card`, `he2021debertav3cardlarge` | DeBERTa-v3-base steht laut Modellkarte unter MIT-Lizenz | Trägt |  |
| #216 | 4.2.1 | 32 | `wolf2020transformers` | Modelle sind über die Transformers-Bibliothek verfügbar | Trägt |  |
| #217 | 4.2.2 | 32 | `qwen2026card08b`, `qwen2026card4b`, `qwen2026card27b` | Qwen3.5-0.8B unter Apache-2.0 laut Modellkarte | Trägt |  |
| #218 | 4.2.2 | 32 | `qwen2026card08b`, `qwen2026card4b`, `qwen2026card27b` | veröffentlichte Checkpoints sind vor- und nachtrainiert | Trägt |  |
| #219 | 4.2.3 | 33 | `kaplan2020scaling`, `hoffmann2022training` | Skalierungsgesetze im Sinne angepasster Potenzgesetze | Trägt |  |
| #220 | 4.3.1 | 33 | `tedeschi2022multinerd` | Datensatz kombiniert Wikipedia- und Wikinews-Artikel (zwei Genres) | Trägt |  |
| #221 | 4.3.1 | 33 | `tedeschi2022multinerd`, `babelscape2023multinerdcard` | offizielle train-/validation-/test-Splits stehen zur Verfügung | Trägt teilweise | B24 |
| #222 | 4.3.1 | 34 | `babelscape2023multinerdcard` | CC BY-NC-SA 4.0, auf nichtkommerzielle Forschung beschränkt | Trägt |  |
| #223 | 4.3.1 | 34 | `tedeschi2022multinerd` | typbezogene Aussagen auf der manuell geprüften Stichprobe sehr unsicher (Tab. 5) | Trägt |  |
| #224 | 4.3.2 | 34 | `devlin2019bert` | erstes Subwort jedes Wortes erhält das Wortlabel | Trägt |  |
| #225 | 4.3.3 | 35 | `tedeschi2022multinerd`, `babelscape2023multinerdcard` | Splits identisch übernommen; vergleichbar mit Arbeiten, die dieselben Splits verwenden | Trägt teilweise | B24 |
| #226 | 4.3.3 | 36 | `cawley2010selectionbias` | Auswahlentscheidungen auf dem Testsplit können zu optimistischer Verzerrung führen | Trägt |  |
| #227 | 4.4.2 | 37 | `devlin2019bert`, `wolf2020transformers` | Encoder werden mit Klassifikationskopf end-to-end vollständig feinjustiert | Trägt |  |
| #228 | 4.4.3 | 39 | `qwen2026card27b` | Sampling folgt den Empfehlungen der Modellkarte für Non-Thinking nur teilweise: Temperatur, top-k, min-p … | Trägt |  |
| #229 | 4.4.3 | 40 | `mangrulkar2022peft` | Anpassung mit peft | Trägt |  |
| #230 | 4.4.3 | 40 | `vonwerra2026trldocs` | SFTTrainer aus trl als Single-Turn-SFT | Trägt |  |
| #231 | 4.4.3 | 40 | `vonwerra2026trldocs` | assistant_only_loss Standardwert False; Verlust über gesamte Tokenfolge | Trägt |  |
| #232 | 4.4.4 | 41 | `semin2026spanLabeling` | Format gehört zur Indexing-Familie nach Semin Abschn. 3.2 in der angereicherten Variante auf Wortebene | Trägt teilweise | B26 |
| #233 | 4.4.4 | 41 | `ramponi2025fallacy` | nummerierte Eingabetokens und Wortindizes in der Ausgabe folgen Ramponi, Tab. 9 | Trägt |  |
| #234 | 4.4.4 | 41 | `zhan2026generativeNER` | bei Zeichenoffsets beobachteter Einbruch | Trägt |  |
| #235 | 4.4.4 | 42 | `tam2024speakfreely` | Constrained Decoding kann die Modellleistung beeinträchtigen | Trägt | A4 |
| #236 | 4.4.4 | 42 | `dagdelen2024structuredIE` | freie Generierung mit nachgelagerter Validierung entspricht Einsatzszenarien, in denen nur die Textausgabe … | Trägt teilweise | A6 |
| #237 | 4.4.5 | 42 | `lu2024llmstruggle` | ohne Fine-Tuning sind LLMs bei NER auf Tokenebene nur eingeschränkt konkurrenzfähig | Trägt teilweise | B19 |
| #238 | 4.4.5 | 42 | `hu2021lora`, `han2024parameter` | Adapter senken Speicher- und Rechenbedarf gegenüber vollständigem Fine-Tuning | Trägt |  |
| #239 | 4.4.5 | 42 | `zhan2026generativeNER` | LoRA bleibt in den Vergleichen nicht hinter vollem Fine-Tuning zurück (Anh. D, Tab. 11) | Trägt |  |
| #240 | 4.4.5 | 42 | `dettmers2023qlora` | Feed-Forward-Schichten einbeziehen: erst Adapter auf allen Schichten erreichen die Leistung des … | Trägt |  |
| #241 | 4.4.5 | 42 | `dettmers2022llmint8` | bitsandbytes lädt die Basismodelle im 4-Bit-Format NF4 | Trägt teilweise | B10 |
| #242 | 4.4.5 | 42 | `dettmers2023qlora` | QLoRA-Standardrezept inkl. Double Quantization und Paged Optimizer | Trägt |  |
| #243 | 4.4.5 | 42 | `dettmers2023qlora` | Double Quantization spart im Mittel rund 0,37 Bit je Parameter ohne Leistungseinbuße (Abschn. 3) | Trägt teilweise | B27 |
| #244 | 4.5.1 | 44 | `manning2008introduction`, `sang2003conll`, `tjong1999representing` | strikte Precision, Recall, F1 auf Entitätsebene nach IOB2-Schema (Quelle für Metriken) | Trägt teilweise, trägt | B28 |
| #245 | 4.5.1 | 44 | `nakayama2018seqeval` | Berechnung mit seqeval im strikten Modus | Trägt |  |
| #246 | 4.5.1 | 44 | `manning2008introduction` | primäre Kennzahl mikrogemittelter F1 | Trägt |  |
| #247 | 4.5.1 | 45 | `efron1993bootstrap` | gepaarter Satz-Bootstrap folgt dem Bootstrap nach Efron und Tibshirani | Nicht verifizierbar | E2 |
| #248 | 4.5.1 | 45 | `koehn2004bootstrap` | gepaartes Ziehen derselben Satzstichproben für alle Systeme | Trägt |  |
| #249 | 4.5.1 | 45 | `lakens2017equivalence` | Enthält das Intervall null, ist Gleichwertigkeit nicht nachgewiesen | Trägt |  |
| #250 | 4.5.1 | 45 | `segura2013semeval` | Zuordnung folgt Logik weniger strenger Evaluationsmodi | Trägt |  |
| #251 | 4.5.2 | 46 | `hu2021lora`, `dettmers2023qlora` | Verhältnis trainierbar/gesamt zeigt, wie stark Adapter die Zahl aktualisierter Parameter senken | Trägt |  |
| #252 | 4.5.3 | 47 | `dagdelen2024structuredIE` | Fehlerklasse JSON-Validität | Trägt |  |
| #253 | 4.5.3 | 47 | `semin2026spanLabeling` | Fehlerklasse Span-Alignment | Trägt |  |
| #254 | 4.5.3 | 47 | `dagdelen2024structuredIE`, `wei2023chatie` | LLMs müssen Ausgabeformat und Typinventar erst über Prompt und Fine-Tuning erlernen | Trägt teilweise | A6 |
| #255 | 4.6 | 49 | `liu2021crossner` | CrossNER als Grundlage des Ergänzungsversuchs | Trägt |  |
| #256 | 5.1.2 | 54 | `tedeschi2022multinerd` | ANIM, BIO, FOOD, PLANT sind in der Silberreferenz häufig verwechselt oder unvollständig verlinkt | Trägt |  |
| #257 | 5.7.1 | 65 | `obeidat2025llm` | Auf vier von fünf biomedizinischen Datensätzen Vorsprung des LLM von 2 bis 8 Punkten, vor allem aus der … | Trägt |  |
| #258 | 5.7.1 | 65 | `obeidat2025llm` | Tab. III als Beleg für Vorsprung und Precision-Ursache | Trägt |  |
| #259 | 5.7.1 | 65 | `zhan2026generativeNER` | LoRA-angepasste LLMs und spezialisierte Modelle liegen auf flachen Datensätzen knapp auseinander; … | Trägt |  |
| #260 | 5.7.1 | 65 | `lu2024llmstruggle` | Ein LoRA-angepasstes 7B-Modell bleibt hinter einem feinjustierten Encoder | Trägt |  |
| #261 | 5.7.1 | 66 | `bogdanov2024nuner` | NuNER untersucht den Nutzen von Vorwissen bei deutlich weniger Trainingsdaten | Trägt |  |
| #262 | 5.7.1 | 66 | `zaratiana2024gliner` | GLiNER untersucht den Nutzen von Vorwissen bei deutlich weniger Trainingsdaten | Trägt teilweise | B29 |
| #263 | 5.7.1 | 66 | `keraghel2024recent` | Makro-F1 65,17 (GPT-4o mit Beispielentitäten im Prompt) gegen 90,98 auf CoNLL-2003 | Trägt |  |
| #264 | 5.7.1 | 66 | `tan2023damo` | ChatGPT auf MultiCoNER II 14,78 gegenüber 92,77 | Trägt teilweise | B29 |
| #265 | 5.7.1 | 66 | `sainz2024gollie` | GoLLIE-7B Zero-Shot 76,0 auf einer MultiNERD-Fassung | Trägt |  |
| #266 | 5.7.1 | 66 | `sainz2024gollie` | GoLLIE hatte acht der 15 Typen bereits gesehen (Tab. 6) | Trägt |  |
| #267 | 5.7.1 | 66 | `zhou2024universalner` | 93,73 (UniNER-7B) und 91,25 (BERT-base) auf der MultiNERD-Variante des InstructUIE-Benchmarks; Abstand … | Trägt |  |
| #268 | 5.7.1 | 66 | `zhan2026generativeNER` | Auf CoNLL-2003 übersieht der Encoder (BERT-MRC) vor allem, das LLM (LLaMA3.1-8B) vergibt vor allem falsche … | Trägt |  |
| #269 | 5.7.1 | 67 | `qwen2026card27b` | Die Modellkarten von Qwen3.5 nennen keine Vortrainingsdaten | Trägt |  |
| #270 | 5.7.1 | 67 | `zhan2026generativeNER` | Ergebnisse ändern sich kaum, wenn Typbezeichnungen durch bedeutungslose Symbole ersetzt werden; nur für … | Trägt |  |
| #271 | 5.7.2 | 67 | `zhan2026generativeNER` | Für Zeichenoffsets beobachteter Einbruch | Trägt |  |
| #272 | 5.7.2 | 67 | `zhan2026generativeNER` | Rund 54 Punkte für JSON mit Zeichenoffsets | Trägt |  |
| #273 | 5.7.2 | 67 | `ramponi2025fallacy` | Nummerierte Eingabetokens für die Spanerkennung (Tab. 3, Tab. 9) | Trägt teilweise | B26 |
| #274 | 5.7.2 | 67 | `semin2026spanLabeling` | Beschreiben die Wortindex-Variante, ohne sie zu evaluieren | Trägt |  |
| #275 | 5.7.2 | 67 | `dagdelen2024structuredIE` | Feinjustierte Modelle geben JSON fast immer parsebar aus, auch ohne erzwungene Formatierung | Trägt |  |
| #276 | 5.7.2 | 68 | `lu2024llmstruggle` | Formatvorgabe und Typinventar begrenzen die Leistung ohne Training; für die Formatvorgabe beschreiben das … | Trägt teilweise | B19 |
| #277 | 5.7.2 | 68 | `tam2024speakfreely` | Berichten Leistungseinbußen durch Formatzwang | Trägt teilweise | A4 |
| #278 | 5.7.2 | 68 | `semin2026spanLabeling` | Berichten dagegen Gewinne durch LogitMatch | Trägt teilweise | A4 |
| #279 | 5.7.2 | 68 | `obeidat2025llm` | Faktoren von 40 bis 220 für IOB2-Ausgaben (Tab. IV) | Trägt | B30 |
| #280 | 5.7.3 | 68 | `keraghel2024recent` | Bei Keraghel wechselt die Rangfolge von DeBERTa-v3-base und -large zwischen den Datensätzen | Trägt |  |
| #281 | 5.7.3 | 68 | `keraghel2024recent` | Tab. 5 als Beleg | Trägt |  |
| #282 | 5.7.3 | 68 | `sainz2024gollie` | Größere Backbones gewinnen bei GoLLIE nur wenig hinzu, und der Gewinn durch die Anpassung übersteigt den … | Trägt teilweise | A5 |
| #283 | 5.7.3 | 68 | `zhou2024universalner` | Bei UniversalNER gewinnen größere Backbones wenig, der Gewinn durch die Anpassung übersteigt den durch die … | Trägt | A5 |
| #284 | 5.7.3 | 69 | `wei2022emergent` | Fähigkeiten wie das Befolgen von Instruktionen treten erst ab einer gewissen Größe zuverlässig auf | Trägt |  |
| #285 | 5.7.3 | 69 | `zaratiana2024gliner` | Architektur, Vortrainingsdaten und Anpassungsverfahren überlagern den Größenunterschied, wie GLiNER für … | Trägt |  |
| #286 | 5.7.3 | 69 | `bogdanov2024nuner` | Wie oben für NuNER | Trägt |  |
| #287 | 5.7.4 | 69 | `obeidat2025llm` | Die Empfehlung eines Encoders trotz kleiner Qualitätsunterschiede diskutieren auch Obeidat et al. | Trägt |  |
| #288 | 6.2 | 74 | `tedeschi2022multinerd` | Annotations-F1 0,52 bis 0,71 für ANIM, BIO, FOOD, PLANT; 0,67 für INST | Trägt |  |
| #289 | 6.3 | 76 | `tedeschi2022multinerd` | manuell geprüfte Stichprobe existiert, Zugang offen, wenige Dutzend Instanzen für seltene Typen | Trägt |  |
| #290 | 6.3 | 76 | `hu2021lora` | LoRA für Encoder als Anschlussfrage | Trägt |  |
| #291 | 6.3 | 76 | `zhan2026generativeNER` | Inline-Formate als Alternative zum Wortindex-Format | Trägt |  |
| #292 | 6.3 | 76 | `semin2026spanLabeling` | LogitMatch könnte freie Erzeugung numerischer Wortindizes ersetzen | Trägt |  |
| #293 | 6.3 | 76 | `zaratiana2024gliner` | GLiNER als Encoder-Generalist | Trägt |  |
| #294 | 6.3 | 76 | `bogdanov2024nuner` | NuNER als Encoder-Generalist | Trägt |  |
| #295 | A.2.2 | 91 | `qwen2026card08b`, `qwen2026card4b`, `qwen2026card27b` | Laut Konfiguration: dichter Decoder, 3 von 4 Schichten Gated DeltaNet, vierte Softmax-Attention mit … | Trägt teilweise | A10 |
| #296 | A.2.4 | 92 | `wei2023chatie` | ChatIE ist ein Multi-Turn-Ansatz | Trägt |  |
| #297 | A.2.5 | 93 | `qwen2026card27b` | Non-Thinking: T 0,7, top-p 0,8, top-k 20, min-p 0, Presence-Penalty 1,5; Reasoning: zwei Wertesätze … | Trägt |  |
| #298 | A.3 | 101 | `tedeschi2022multinerd` | Beispielsätze der Tabelle stammen aus dem MultiNERD-Testsplit | Trägt |  |

## 4. Fehlende, vertauschte oder doppelte Verweise

### 4.1 Abgleich von Text und Verzeichnis

- **Ungelöste Verweise:** keine. Das Build-Protokoll enthält keine Warnung zu Zitaten oder Verweisen; Biber meldet 84 Schlüssel, alle gefunden.
- **Einträge ohne Verwendung:** keine. Alle 84 Einträge der Datei werden zitiert und erscheinen im Verzeichnis; `\nocite` wird nicht verwendet. In `references.bib` steht kein Eintrag, der in der Arbeit nicht erscheint.
- **Dubletten:** keine. Kein Werk hat zwei Schlüssel, kein Eintrag führt zwei Publikationen zusammen. Die Paare He et al. 2021/2023, Kudo 2018 und Kudo und Richardson 2018, Dettmers et al. 2022/2023, Tjong Kim Sang 1999/2003 und die beiden DeBERTa-Karten sind jeweils verschiedene Werke und richtig getrennt.
- **Verwechslungen von Methode, Modell oder Datensatz:** keine gefunden. Die Abgrenzung des Datensatzes Universal NER (Mayhew et al.) vom Modell UniversalNER (Zhou et al.) in 3.3.1 stimmt.
- **Kurzbelege und Einträge:** Name, Jahr und Suffix stimmen überall überein.
- **Defekte Links:** `https://nlp.stanford.edu/fsnlp/` (im Verzeichnis gedruckt, siehe C2). Nicht maschinell prüfbar waren die zehn OpenReview-Links und `https://dl.acm.org/doi/10.5555/645530.655813` (Sperrseiten, kein Hinweis auf einen Fehler). Den Link auf das eigene Repository in Anhang A.1 habe ich nicht abgerufen; laut `STATUS.md` ist das Repository noch öffentlich zu schalten.
- **Schlüsselnamen mit abweichendem Jahr** (`li2018survey` für 2022, `jurafsky2024speech` für 2026, `zhao2023survey` für 2026, `he2020deberta` für 2021, `mangrulkar2022peft` für 2026, `lu2024llmstruggle` für 2025) sind nur intern und im PDF nicht sichtbar. Kein Handlungsbedarf.

### 4.2 Belege, die an eine andere Quelle gehören

Keine dieser Stellen meint ein falsches Werk; es fehlt jeweils die Quelle, die den genannten Teil trägt.

| Stelle | Aussage | Zitiert | Trägt tatsächlich |
|---|---|---|---|
| 2.4, S. 11 (#65) | 512 Positionen bei DeBERTa-v3 „laut Modellkonfiguration“ | beide Modellkarten | `config.json` der zitierten Revisionen (`max_position_embeddings: 512`, von mir geprüft) |
| 2.5, S. 13 (#83) | JSON-Liste „mit Angaben zu Position und Typ“ | Dagdelen et al. | JSON-Ausgabe: Dagdelen et al.; Positionsangaben: Semin et al., Abschn. 3.2 |
| 2.8, S. 15 (#98, #99) | „F-Maß“ | van Rijsbergen | van Rijsbergen definiert das E-Maß; das F-Maß nennt Manning et al., Abschn. 8.3 |
| 3.1, S. 18 und Tab. 3.1 (#109, #117), 3.6, S. 27 (#196) | OntoNotes: 18 Entitätstypen | LDC-Katalogeintrag (und Li et al.) | nur Li et al., Tab. 1; der Katalogeintrag nennt Sprachen und Genres, aber weder die Typzahl noch die NER-Annotation. In der Unterschrift von Tab. 3.1 fehlt Li et al. |
| 3.4, S. 24 (#156) | GENIA: 4,02 Punkte Rückstand | Zusatz „Tab. 3“ am vorangehenden Halbsatz | Zhan et al., Abschn. 3.2 und Tab. 4 (Tab. 3 enthält nur die flachen Datensätze) |
| 4.4.5, S. 42 (#241) | Laden im 4-Bit-Format NF4 mit bitsandbytes | Dettmers et al. 2022 | Bibliothek: Dettmers et al. 2022 (Fußnote); NF4: Dettmers et al. 2023, Abschn. 3 |
| 4.5.1, S. 44 (#244) | strikt, Entitätsebene, IOB2 | Sammelzitat aus drei Quellen | P/R/F: Manning et al.; exakte Übereinstimmung: Tjong Kim Sang und De Meulder; IOB2: Tjong Kim Sang und Veenstra. Inhaltlich vollständig, nur nicht zugeordnet |
| 4.5.3, S. 47 (#254) | Format und Typinventar „über Prompt und Fine-Tuning erlernen“ | Dagdelen et al.; X. Wei et al. | Fine-Tuning: Dagdelen et al.; Prompt: X. Wei et al. |

### 4.3 Belegpflichtige Aussagen ohne Nachweis

Aufgenommen sind nur fremde, überprüfbare Aussagen; Fachwissen und eigene Ergebnisse nicht. Wo ein vorhandener Eintrag passt und ich die Passage geprüft habe, ist er genannt.

| Stelle | Aussage | Einschätzung und Vorschlag |
|---|---|---|
| 2.4, S. 11 (Z. 261) | „Da die Positionskodierung von BERT auf 512 Positionen festgelegt ist“ | Beleg ergänzen: Devlin et al., Anh. A.2 (Vortraining mit Sequenzen bis 512 Tokens, gelernte Positionseinbettungen); geprüft |
| 2.6, S. 14 (Z. 424) | „Ein Dropout auf dem Adapterpfad dient als Regularisierung“ | steht nicht bei Hu et al. in dieser Form; entweder als Eigenschaft der verwendeten Implementierung formulieren oder streichen |
| 2.6, S. 14 (Z. 434) | „spätere Arbeiten beziehen auch die Feed-Forward-Schichten ein“ | Dettmers et al. 2023 nennen Adapter auf allen linearen Schichten; Beleg dort setzen (geprüft) |
| 2.8, S. 16 (Z. 537) | „Diese strenge Zählweise hat sich mit CoNLL-2003 als Standard etabliert“ | Tjong Kim Sang und De Meulder tragen die Definition, nicht die Verbreitung. Abschwächen („wird seit CoNLL-2003 üblicherweise verwendet“) oder mit Li et al. bzw. Keraghel et al. belegen (dort nicht eigens geprüft) |
| 2.8, S. 16 (Z. 569) | fehlende Vorhersagen: Precision „wird in der Praxis als null gewertet“ | Softwareverhalten; als Verhalten der verwendeten Bibliothek formulieren (4.5.1 tut das bereits) |
| 3.2.1, S. 19 (Z. 166) und 3.6 | MultiCoNER II (33 Typen, zwölf Sprachen) | nur über Tan et al. belegt. Das ist als Wiedergabe von Tan et al. erkennbar und zulässig; die Primärquelle des Wettbewerbs habe ich nicht eingesehen und empfehle sie deshalb nicht als Direktzitat |
| 3.3.3, S. 23 (Z. 366), 3.6, S. 28 (Z. 678, 680) und 5.7.1, S. 66 | Aussagen über InstructUIE (eigene Datenaufbereitung, Benchmark) ohne Zitierbefehl im Satz | `wang2023instructuie` ist vorhanden und trägt die Aufbereitung (Abschn. 2.3 und Anhang, geprüft); Zitat an diesen Stellen ergänzen |
| 3.6, S. 28 (Z. 736) | CleanCoNLL | nur über Zhan et al. wiedergegeben; als Sekundärwiedergabe erkennbar, in Ordnung |
| 3.6, S. 28 (Z. 731) und 5.7.1, S. 67 | Wikipedia- und Wikinews-Artikel seien „mit hoher Wahrscheinlichkeit“ Teil der Vortrainingskorpora | Herkunft der Texte: Tedeschi und Navigli (Beleg ergänzen). Die Wahrscheinlichkeitsaussage ist eine eigene Annahme; so kennzeichnen („ist anzunehmen“). Dass die Modellkarten keine Vortrainingsdaten nennen, ist bestätigt |
| 3.6, S. 28 (Z. 711) | direkte Vergleiche betrachten LLMs „einer einzigen Größenklasse zwischen 3 und 9 Mrd. Parametern“ | für die feinjustierten Modelle richtig. Lu et al. prompten zusätzlich GPT-3.5 und GPT-4, Dukić und Šnajder trainieren zusätzlich Modelle mit 68 Mio. Parametern (die Arbeit zitiert das selbst in 3.5). Auf „die mit Adaptern feinjustierten LLMs“ eingrenzen |
| 3.6, S. 28 (Z. 720) | „Keine der Arbeiten … untersucht eine Größenstaffelung auf beiden Seiten mit gemeinsamer Leistungs- und Ressourcenerfassung“ | hält bei enger Lesart. GNER staffelt Flan-T5 von 77 Mio. bis 11 Mrd. Parametern und misst den Durchsatz, GLiNER staffelt Encoder von 50 bis 300 Mio. gegen übernommene LLM-Werte. Die enge Lesart (Encoder und LLM selbst trainiert, beide gestaffelt, beide gemessen) im Satz nennen |
| 4.2.2, S. 33 (Z. 181) | Qwen3.5 multimodal, Vision-Encoder, MTP-Module | steht in den Modellkarten bzw. Konfigurationsdateien; Zitat der Karten an dieser Stelle ergänzen |
| 5.1.2, S. 54 (Z. 186) und 5.5, S. 62 (Z. 689) | Typdefinition TIME; „Richtlinien des Datensatzes“ | Tedeschi und Navigli bzw. Datensatzkarte ergänzen (Typdefinitionen dort nicht eigens geprüft) |
| 5.7.2, S. 67 (Z. 962) | „soweit bekannt, erstmals mit QLoRA und strikter Entitätsmetrik auf MultiNERD geprüft“ | Neuheitsbehauptung; in den 84 Quellen findet sich kein Gegenbeispiel. Sie ist eng gefasst und mit „soweit bekannt“ abgesichert. Belegen lässt sie sich nur über die in Kapitel 3 beschriebene Recherche; ein Verweis darauf würde genügen |
| 6.1, S. 73 (Z. 74) | „Literaturspanne von 40 bis 220“ | bezieht sich auf E2; Kurzbeleg `\cite{obeidat2025llm}` ergänzen. Beide Zahlen stammen aus einer einzigen Studie (40 bei JNLPBA, knapp 220 bei BioRED); „Spanne aus Obeidat et al.“ wäre genauer als „Literaturspanne“ |
| A.2.3, S. 92 (Z. 383) | Softwareversionen | für PEFT, TRL, seqeval und Transformers gibt es Einträge; Zitate optional ergänzen |

### 4.4 Manuelle Quellenangaben außerhalb von Zitierbefehlen

| Stelle | Angabe | Prüfung |
|---|---|---|
| 4.6, S. 49 und Tab. A.3 | CrossNER-Repository `zliucr/CrossNER`, Commit `2e7ba2a…`, MIT-Lizenz laut Lizenzdatei | bestätigt: Der Commit existiert, die Datei `LICENSE` enthält dort die MIT-Lizenz (Copyright 2020 Zihan Liu). Empfehlung: Repository mit Commit als Online-Quelle ins Verzeichnis aufnehmen, weil hier eine Lizenzaussage daran hängt |
| Tab. A.3, S. 89 | Revisionen von Datensatz und fünf Modellen | bestätigt; identisch mit den im Verzeichnis zitierten Revisionen |
| 4.1, 4.3.1, 4.3.3 | „offizielle Splits“ | das Repository enthält bei der zitierten Revision die Ordner `train`, `val` und `test`; die Karte selbst sagt nur „same among all splits“. Die Aussage stimmt (siehe B24) |
| A.2.3, S. 92 | Python, PyTorch, Transformers, PEFT, TRL, bitsandbytes, Accelerate, Datasets, seqeval mit Versionen | alle neun Versionsnummern existieren auf PyPI bzw. sind plausibel (Python 3.12.3 nicht geprüft) |
| A.2.5, S. 93 | Überschriften „Quickstart“ und „Best Practices“ der Qwen-Karte, Sampling-Werte | wörtlich und in allen Werten bestätigt |
| A.1, S. 86 | eigenes GitHub-Repository mit Tag | nicht abgerufen (siehe 4.1) |
| mehrfach | Systeme und Datensätze, die nur über eine zitierte Arbeit eingeführt werden (BabelNet, Pile-NER, C4, BERT-MRC, BINDER, RareDis-v1, FewNERD, GENIA, ACE2005, CleanCoNLL, MultiCoNER II, GLUE, MMLU, conlleval) | durchweg als Wiedergabe der jeweils zitierten Arbeit erkennbar; kein eigener Eintrag nötig |

### 4.5 Direkte Zitate, Paraphrasen, Abbildungen

- **Direkte Zitate:** keine aus der Literatur. Anführungszeichen stehen nur bei Suchbegriffen (Kapitel 3), eigenen Begriffen, Beispielspans aus den Daten und den beiden Überschriften der Qwen-Karte.
- **Eng am Original, aber zugeschrieben und belegt:** die Beschreibung des I/O/B-Tagsets in 2.2 (Ramshaw und Marcus, Abschn. 3), Disentangled Attention und Enhanced Mask Decoder in 2.4 (He et al. 2021, einschließlich des Arguments zur syntaktischen Rolle), die Typgruppen in 3.1 und 5.1.2 (Tedeschi und Navigli, Abschn. 5.3), die drei Kernbefunde in 3.3.1 (Semin et al., Abstract) und der Satz zu Zeichenpositionen (Zhan et al.). Das sind kurze, eindeutig zugeordnete Wiedergaben; ein Redlichkeitsproblem sehe ich nicht. Wer mehr Abstand möchte, formuliert die Passage zu DeBERTa freier.
- **Abbildungen 2.1 bis 2.3:** „Eigene Darstellung in Anlehnung an …“ ist jeweils angemessen. Es sind TikZ-Neuzeichnungen mit eigenem Beispiel und deutscher Beschriftung; die Vorlagen (Li et al., Abb. 1; Vaswani et al., Abb. 1; Hu et al., Abb. 1) sind genannt bzw. erkennbar. Bei Abb. 2.2 und 2.3 fehlt die Abbildungsnummer der Vorlage („Abb. 1“); bei Abb. 2.1 ist sie angegeben.
- **Tabelle 3.1:** „Eigene Zusammenstellung nach …“ nennt vier Quellen; für die Typzahl von OntoNotes fehlt Li et al. (siehe 4.2).

## 5. Priorisierter Korrekturplan

Jeder Punkt nennt Stelle, Befund mit Fundstelle im Original und eine konkrete Maßnahme. Formulierungsvorschläge sind Angebote, keine Vorgaben. Alle A-Punkte und alle Angaben mit „geprüft“ habe ich selbst am Quelltext nachvollzogen.

### Teil A: Zentrale Stellen (Aussage geht über die Quelle hinaus oder Beleg trägt nicht)

**A1. Sammelaussage zu den direkten Vergleichen und Erwartung E1**
- Stellen: 3.4, S. 25 (#165, Z. 512–515); 3.6, S. 29 (#208, E1); mittelbar 3.6, S. 26 (Z. 569: „in ihren Ergebnissen weitgehend übereinstimmen“) und der Abgleich von E1 in 6.1.
- Befund: Zhan et al., Tab. 3 und 4: Das beste LLM liegt nur auf CoNLL-2003 vorn (93,85 gegenüber 93,33 für BERT-MRC+DSC). Auf OntoNotes 5.0 liegt es 0,79 Punkte zurück (91,28 gegenüber 92,07), auf ACE2005 1,98 (88,02 gegenüber 90,00 für BINDER), auf GENIA 4,02. Dukić und Šnajder: Vorsprung nur auf CoNLL-2003 und nur mit entfernter Maske (0,908 gegenüber 0,900); mit Maske 0,744, und beim ACE05-Trigger-Labeling bleibt RoBERTa-large vorn. Obeidat et al. tragen die Aussage für vier von fünf Datensätzen. Abschnitt 5.7.1 gibt Zhan et al. bereits richtig wieder („auf CoNLL-2003 knapp vorn, auf OntoNotes 5.0 knapp dahinter“).
- Maßnahme: eingrenzen, etwa: „In drei der vier Arbeiten erreichen mit Adaptern feinjustierte LLMs auf einzelnen Benchmarks das Niveau feinjustierter Encoder oder übertreffen es leicht: auf CoNLL-2003 bei Zhan et al. und, mit entfernter Maske, bei Dukić und Šnajder, auf vier von fünf biomedizinischen Datensätzen bei Obeidat et al. Auf OntoNotes 5.0, ACE2005 und GENIA liegen die LLMs bei Zhan et al. zurück.“ E1 entsprechend („auf einzelnen Benchmarks“). In 3.6 „weitgehend übereinstimmen“ durch „in der Tendenz übereinstimmen“ ersetzen. Der Abgleich in Kapitel 6 wird dadurch eher schlüssiger, weil das eigene Ergebnis (knapper Rückstand) dann innerhalb der Literaturspanne liegt.

**A2. 512 Positionen bei DeBERTa-v3**
- Stelle: 2.4, S. 11 (#65, Z. 262–264).
- Befund: Die beiden zitierten Modellkarten enthalten die Zahl 512 nicht (sie nennen Schichten, Parameter, Vokabular; im Beispielaufruf `--max_seq_length 256`). In `config.json` beider zitierter Revisionen steht `"max_position_embeddings": 512` (geprüft). Die Aussage stimmt also, der Beleg nicht. He et al. (2021, Anh. A.5) beschreiben zudem relative Positionen, sodass 512 keine Grenze der Architektur ist, sondern eine der veröffentlichten Konfiguration; der Text sagt mit „laut Modellkonfiguration“ bereits das Richtige.
- Maßnahme: Quelle benennen, etwa „laut den Konfigurationsdateien (`config.json`) der verwendeten Revisionen (Tabelle A.3)“, und die Kartenzitate an dieser Stelle streichen oder durch einen Verweis auf Tabelle A.3 ersetzen.

**A3. Ravikumar et al.: „mehr als 40 %“**
- Stelle: 3.3.1, S. 21 (#135, Z. 294).
- Befund: Die Quelle nennt die 40 % für BC5CDR („particularly pronounced for BC5CDR, reaching over 40% in certain instances“); im Text steht der Satz direkt hinter OntoNotes 5.0, wo die Formate höchstens rund fünf Punkte auseinanderliegen. Nach Tab. 1 der Quelle sind es F₁-Punkte (Llama 3.1 8B auf BC5CDR: 45,00 gegenüber 88,17), nicht Prozent des Werts.
- Maßnahme: „Allein das Format verschiebt den F₁-Wert auf dem biomedizinischen Datensatz BC5CDR teils um mehr als 40 Punkte; auf OntoNotes 5.0 bleiben die Unterschiede klein.“

**A4. Tam et al. und Semin et al. in der Diskussion der Formtreue**
- Stellen: 5.7.2, S. 68 (#277, #278, Z. 981–983); verwandt 4.4.4, S. 42 (#235).
- Befund: Tam et al. finden Einbußen durch Formatvorgaben bei Schlussfolgerungsaufgaben (GSM8K, Last Letter, Shuffled Objects); bei Klassifikationsaufgaben verbessert der JSON-Modus die Genauigkeit teils. Informationsextraktion untersuchen sie nicht. Semin et al., Tab. 3 (hartes F₁, NER): MATCH 74,1/79,1/81,7/87,4 gegenüber LOGITMATCH 73,3/76,3/79,8/89,2; nur eines von vier Modellen gewinnt. Die Autoren schreiben, LogitMatch beseitige Zuordnungsfehler „while retaining their performance“; deutliche Gewinne zeigen sich bei der Grammatikkorrektur.
- Maßnahme: „Tam et al. berichten Leistungseinbußen durch Formatvorgaben bei Schlussfolgerungsaufgaben, bei Klassifikation dagegen teils Gewinne. Semin et al. zeigen, dass LogitMatch Zuordnungsfehler zwischen erzeugtem Text und Eingabe beseitigt und die NER-Leistung des freien Matchings dabei im Wesentlichen hält.“ In 4.4.4 genügt ein Zusatz: „… kann die Modellleistung beeinträchtigen, wie Tam et al. für Schlussfolgerungsaufgaben zeigen“.

**A5. GoLLIE und UniversalNER: Größe gegen Anpassung**
- Stelle: 5.7.3, S. 68 (#282, #283, Z. 1018–1020).
- Befund: Belegt sind die Werte (GoLLIE, Tab. 3: Baseline 42,3; 7B 55,3; 13B 56,0; 34B 57,2. UniversalNER: 41,7 und 43,4). „Der Gewinn durch die Anpassung übersteigt den Gewinn durch die Größe“ sagt keine der beiden Quellen. Bei GoLLIE ist der große Schritt der von der Baseline ohne Richtlinien zur Variante mit Richtlinien; beide sind feinjustiert. Für UniversalNER lässt sich ein solcher Vergleich aus den zitierten Werten nicht bilden. Abschnitt 3.5 formuliert dasselbe für GoLLIE schon richtig („Größengewinn bleibt deutlich hinter dem Gewinn durch das Richtlinientraining zurück“).
- Maßnahme: „Das Muster ähnelt den bei GoLLIE und UniversalNER berichteten Werten, nach denen größere Backbones nur wenig hinzugewinnen; bei GoLLIE ist der Gewinn durch das Richtlinientraining (42,3 auf 55,3) nach eigener Rechnung aus Tab. 3 deutlich größer als der durch die Größe (55,3 auf 57,2).“ Den Zusatz „Abschn. 5.2“ habe ich nur an den arXiv-Fassungen bestätigt.

**A6. Dagdelen et al. als Beleg für vier Aussagen, die dort nicht stehen**
- Stellen: 2.5, S. 13 (#83); Einleitung zu Kapitel 4, S. 30 (#212); 4.4.4, S. 42 (#236); 4.5.3, S. 47 (#254).
- Befund: Dagdelen et al. geben Entitäten und Relationen als JSON-Objekte mit Zeichenketten aus, ohne Positionsangaben und ohne Rückführung auf Tags. Sie beschreiben die Nutzung proprietärer Modelle über eine API, begründen damit aber nicht freie Generierung gegenüber Constrained Decoding. Dass das Format erst durch Fine-Tuning gelernt wird, tragen sie; „über Prompt“ trägt X. Wei et al.
- Maßnahmen: (a) 2.5: „etwa als strukturierte Liste im JSON-Format (Dagdelen et al.), je nach Verfahren mit Angaben zur Position (Semin et al., Abschn. 3.2)“. (b) Kapitel 4, Einleitung: Beleg hinter „JSON-Array von Entitäten“ setzen und die Überführung in IOB2 als eigene Umsetzung stehen lassen (4.4.4 sagt das bereits ausdrücklich). (c) 4.4.4: den Halbsatz zum API-Szenario als eigene Begründung ohne Fremdbeleg führen oder Dagdelen et al. nur für „Modelle, die über eine API genutzt werden“ zitieren. (d) 4.5.3: Zitate trennen, wie in 4.2 angegeben.

**A7. Tabelle 3.2: acht Zellen passen nicht zur eigenen Legende**
- Stelle: 3.6, S. 27 (#181, #183, #184, #185, #188, #190, #192).
- Befund und Maßnahme je Zeile:
  - GLiNER, „Ressourcen: –“: Die Quelle nennt eine Trainingszeit („GLiNER-L takes 5 hours to train on an A100 GPU“, Abschn. 3.2; im Anhang rund 3 h 50 min). Zelle auf „Trainingszeit“ ändern.
  - Semin et al., „Ressourcen: Ausgabetokens“: Die Ausgabelänge ist dort geschätzt (Abb. 3, Anh. B.2), nicht gemessen. „Ausgabetokens (geschätzt)“.
  - Semin et al., „Anpassung: Prompting“: ergänzen um das eingeschränkte Decoding (LogitMatch).
  - PaDeLLM und GoLLIE, „Vergleich: –“: Beide stellen eigene LLM-Werte neben übernommene Werte von Encoder-Systemen (PaDeLLM: BINDER, W2NER; GoLLIE: Spalte mit Bestwerten aus der Literatur). Nach der Legende wäre das „berichtete Werte“.
  - GPT-NER, „berichtete Werte“: Im Experiment mit wenig Trainingsdaten trainieren die Autoren die Encoder-Baseline selbst („we train the ACE model“). „berichtete Werte; im Low-Resource-Vergleich selbst trainiert“.
  - CodeIE, „berichtete Werte (UIE)“: Nur die Werte mit vollen Daten stammen aus UIE; die Few-Shot-Baselines (T5, UIE) sind selbst trainiert. Diese Baselines sind zudem Encoder-Decoder-Modelle.
  - Zhan et al., „teilweise – nur BERT-MRC selbst trainiert“: Belegt ist, dass BERT-MRC für den Kostenvergleich selbst trainiert wurde; dass alle anderen Werte übernommen sind, sagt die Quelle nicht ausdrücklich. „nur“ streichen.

**A8. GENIA-Rückstand ohne passenden Beleg**
- Stelle: 3.4, S. 24 (#156, Z. 445).
- Befund: Der Zusatz „Tab. 3“ steht am ersten Halbsatz (CoNLL-2003) und stimmt dort. Der GENIA-Wert steht in Abschn. 3.2 und Tab. 4 der Findings-Fassung („The best LLM result is 4.02 F1 points lower than the best specialized model BERT-MRC“; 83,75 gegenüber 79,73).
- Maßnahme: nach dem GENIA-Halbsatz `\cite[Abschn.~3.2, Tab.~4]{zhan2026generativeNER}` ergänzen.

**A9. CoNLL-2003 „bis heute der am häufigsten verwendete“ Benchmark; OntoNotes mit 18 Typen**
- Stellen: 3.1, S. 17 (#108); 3.1, S. 18 und Tab. 3.1 (#109, #117); 3.6, S. 27 (#196).
- Befund: Li et al. schreiben nur, viele neuere Arbeiten berichteten auf CoNLL03 und OntoNotes; ein Superlativ steht dort nicht, und „bis heute“ kann eine Übersicht mit Stand 2020 nicht tragen. Die Typzahl 18 trägt nur Li et al. (Tab. 1), nicht der LDC-Eintrag.
- Maßnahme: „und gehört zu den am häufigsten verwendeten NER-Benchmarks“. Bei OntoNotes das Sammelzitat so stellen, dass die Typzahl bei Li et al. steht, und Li et al. in die Unterschrift von Tabelle 3.1 aufnehmen. Hinweis: „manuell (IOB1)“ in Tabelle 3.1 ist durch Abschnitt 2.2 gedeckt; Tjong Kim Sang und De Meulder selbst sprechen nur vom „IOB scheme“ nach Ramshaw und Marcus.

**A10. Aussagen zu Qwen3.5, die nicht in den zitierten Karten stehen**
- Stellen: 2.3, S. 10 (#57); 2.5, S. 12 (#75) und S. 13 (#80 bis #82); A.2.2, S. 91 (#295).
- Befund: (a) Die 27B-Karte belegt Nachrichten mit Rollen und ein Chat-Template, nennt aber weder die Rollen System und Assistent noch Steuertokens. (b) „Modelle ohne Mixture-of-Experts-Komponenten“: Die Karten der drei Modelle weisen im Schichtaufbau dichte Feed-Forward-Netze aus, ihre Merkmalsliste nennt für die Familie aber „sparse Mixture-of-Experts“. Die Aussage ist eine richtige Ableitung aus dem Schichtaufbau, steht aber nicht wörtlich dort. (c) Die Reihenfolge („folgten“) steht in keiner zitierten Quelle. Sie stimmt nach den Commit-Historien der Repositorien: erste Commits am 16.02.2026 (397B-A17B), 24.02. (27B), 27.02. (4B) und spätestens 01.03. (0.8B). (d) „Grouped-Query-Attention“ und „Ausgabe-Gating“ stehen nicht in den Karten (dort: Kopfzahlen für Q und KV, „Gated Attention“); der Satz beruft sich richtig auf die Konfigurationsdateien, zitiert aber die Karten. (e) In 2.3 steht für „die in dieser Arbeit verwendeten LLMs“ nur die 27B-Karte. (f) „Qwen-Familie von Alibaba“ und die Einordnung als offene Gewichte statt Open Source sind eigene Einordnungen; die Berichte zu Qwen2.5 und Qwen3 nennen ihre Modelle „open-weight“, der Qwen3-Bericht an einer Stelle auch „open-source“.
- Maßnahme: (a) auf „Nachrichten mit Rollen, die ein Chat-Template in eine Tokenfolge überführt“ verkürzen oder zusätzlich den Qwen3-Bericht nennen, sofern du die Passage dort nachschlägst (von mir für diese Aussage nicht geprüft). (b) und (d) als Ableitung kennzeichnen: „nach dem Schichtaufbau der Modellkarten und den Konfigurationsdateien der Revisionen in Tabelle A.3“. (c) Reihenfolge streichen („neben dem im Februar 2026 vorgestellten ersten Modell umfasst die Familie …“) oder auf die Veröffentlichungsdaten der Repositorien stützen. (e) alle drei Karten zitieren. (f) als eigene Einordnung nach der OSI-Definition formulieren; die Definition selbst ist richtig wiedergegeben.

### Teil B: Präzisierungen (Formulierung eingrenzen, Beleg zuordnen)

| Nr. | Stelle | Befund (Fundstelle im Original) | Maßnahme |
|---|---|---|---|
| B1 | 1.1, S. 1 (#3) | Jurafsky und Martin, Kap. 18, nennen Frage-Antwort-Systeme und die Verknüpfung mit Wissensquellen; Keraghel et al., Abschn. 4, nennen Frage-Antwort-Systeme und Informationsextraktion. „Aufbau von Wissensgraphen“ steht in keiner der beiden | „Informationsextraktion, die Verknüpfung von Text mit Wissensquellen und Frage-Antwort-Systeme“; bei Jurafsky und Martin „Kap. 18“ ergänzen |
| B2 | 1.1, S. 2 (#11) | He et al. nennen nur Backbone-Größen (86 Mio.; 304 Mio. steht nur in der Large-Karte); „einige hundert Millionen“ ergibt sich erst mit den Embeddings | beide Modellkarten mitzitieren |
| B3 | 1.1, S. 2 (#14) | „in einem einzigen Vorwärtsdurchlauf“ ist eine Folgerung aus der Architektur, kein Satz bei Devlin et al. | Beleg vor den Halbsatz ziehen; Obeidat et al. beschreiben den Punkt ausdrücklich (parallele Klassifikation aller Tokens) |
| B4 | 2.2, S. 7 und 8 (#42, #47); 2.4, S. 10 (#58) | „meist als Sequenzlabeling“, „Lange Zeit bestimmten CRF den Stand der Technik“ und „seit BERT die Standardarchitektur“ sind Aussagen zum Forschungsstand; Lafferty et al. und Devlin et al. tragen nur das jeweilige Verfahren | Zeitaussagen abschwächen („waren lange verbreitet“) oder mit Li et al. belegen; Lafferty et al. nur für die Modellbeschreibung |
| B5 | 2.3, S. 9 (#53) | Brown et al. nennen GPT-3 ein autoregressives Sprachmodell mit der Architektur von GPT-2; „Decoder ohne Encoder-Attention“ und kausale Maske stehen dort nicht | unkritisch; wer es genau will, zitiert Brown et al. nur für GPT-3 als Beispiel |
| B6 | 2.4, S. 10 (#61) | Kudo (2018) trägt das Unigram-Sprachmodell, Kudo und Richardson das Werkzeug SentencePiece | „SentencePiece (Kudo und Richardson) mit dem Unigram-Modell nach Kudo“ |
| B7 | 2.4, S. 10 (#63); Kap. 4, S. 30 (#211) | Devlin et al., Abschn. 5.3, tragen die Klassifikation auf dem ersten Subwort. Dass die übrigen Subwörter kein Label erhalten und nicht in den Verlust eingehen, und „IOB2“, stehen dort nicht | Halbsatz als übliche Umsetzung bzw. als eigene Umsetzung formulieren; IOB2 ist in 2.2 belegt |
| B8 | 2.5, S. 11 (#71) | Zhao et al. sprechen von „tens or hundreds of billions of parameters“; die Arbeit nennt auch das 0.8B-Modell ein LLM | ein Halbsatz genügt: „die kleinsten hier verwendeten Modelle liegen mit 0,8 Mrd. Parametern am unteren Rand dessen, was als LLM bezeichnet wird“ |
| B9 | 2.6, S. 13 (#88) | Hu et al., Abschn. 4.1, tragen die Skalierung mit α/r; die Parameterzahl r(d+k) ist eine einfache eigene Rechnung, „Kapazität“ steht dort nicht | Zusatz „Abschn. 4.1“ an den Halbsatz zur Skalierung setzen |
| B10 | 2.6, S. 14 (#93); 4.4.5, S. 42 (#241) | LLM.int8() normiert tensor- bzw. vektorweise im 8-Bit-Bereich; blockweise Konstanten und NF4 gehören zu QLoRA (Dettmers et al. 2023, Abschn. 2 und 3) | in 2.6 „Block“ durch „Gruppe von Gewichten“ ersetzen oder Dettmers et al. 2023 mitzitieren; in 4.4.5 NF4 mit Dettmers et al. 2023 belegen |
| B11 | 2.8, S. 15 (#98, #99) | van Rijsbergen definiert das E-Maß; „F-Maß“ steht dort nicht | „Das F-Maß, das auf das E-Maß bei van Rijsbergen zurückgeht (F = 1 − E), …“ und Manning et al. mitzitieren |
| B12 | 2.8, S. 16 (#102) | Segura-Bedmar et al., Abschn. 3: vier Modi; der Typ-Modus verlangt zusätzlich eine Überlappung | „die nur die exakten Grenzen, eine teilweise Überlappung der Grenzen oder den Typ bei überlappenden Grenzen werten“ |
| B13 | 2.8, S. 16 (#103) | Die PyPI-Seite nennt die zwei Modi und die Kompatibilität mit conlleval; das beschriebene Verhalten zeigt sie an genau einem Beispiel (I-NP ohne B-NP: Standard 1,00, strikt 0,00) | unkritisch; optional „wie das Beispiel der Dokumentation zeigt“ |
| B14 | Tab. 3.1, S. 19 (#116) | „IOB1“ steht nicht bei Tjong Kim Sang und De Meulder (dort „IOB scheme“) | „IOB“ schreiben oder auf Abschnitt 2.2 verweisen |
| B15 | 3.3.1, S. 21 (#137) | PaDeLLM: „on par with the state-of-the-art“, „without sacrificing prediction accuracy“; der Faktor 10,22 gilt für mehrere GPUs | „bei vergleichbarer Erkennungsleistung“ |
| B16 | 3.3.4, S. 23 (#152, #153) | Sammelzitate im Zwischenfazit: Lu et al. (PaDeLLM) tragen nur „teuer“, Zhan et al. nur „robust“ und „fehleranfällig“, Xie et al. nur den Rückstand im Zero-Shot-Regime | Zitate an die Satzteile setzen |
| B17 | 3.4, S. 24 (#155) | „Den breitesten Vergleich“ ist eine eigene Wertung über die gesichtete Literatur | „Den breitesten der hier gesichteten Vergleiche“ |
| B18 | 3.4, S. 24 (#158); 3.6, S. 28 | Obeidat et al. und Lu et al. schreiben „BIO“, nicht „IOB2“; der Satz in 3.4 hat keinen eigenen Beleg | unkritisch (2.2 erklärt die Gleichsetzung); Beleg „Abschn. II“ ergänzen |
| B19 | 3.4, S. 25 (#162); 4.4.5, S. 42 (#237); 5.7.2, S. 68 (#276) | Lu et al. untersuchen klinische NER mit lokalen 7B-Modellen. Llama2-MedTuned ist medizinisch instruktions-feinjustiert, nicht „vortrainiert“. Die ungewohnten BIO-Symbole nennen die Autoren als einen von drei möglichen Gründen, nicht als Befund | „medizinisch angepasste“; in 4.4.5 „in der klinischen Studie von Lu et al.“; in 5.7.2 „nennen ungewohnte Tag-Symbole als möglichen Grund“ |
| B20 | 3.4, S. 25 (#171) | Obeidat et al.: Vorsprung klein (rund 2 Punkte) auf drei Datensätzen, rund 8 Punkte auf Reddit-Impacts | „fällt der Leistungsgewinn auf drei der vier Datensätze klein aus“ |
| B21 | 3.5, S. 26 (#177) | Dukić und Šnajder zeigen: kein Gewinn bei 68 Mio., Gewinn bei 7 Mrd. Parametern; eine Schwelle ist daraus nicht ableitbar | „bei einem Modell mit 68 Mio. Parametern nicht wirkt, bei den 7B-Modellen dagegen schon“ |
| B22 | 3.6, S. 28 (#200) | GLiNER: 10 000 Testbeispiele stehen in Tab. 6 des Anhangs; der Satz zur InstructUIE-Stichprobe betrifft die Trainingsdaten | Fundstelle „Anh., Tab. 6“ ergänzen; „dem Vorgehen von InstructUIE folgend“ |
| B23 | 3.6, S. 29 (#210, E3) | Keraghel et al.: auf WNUT2017 liegt GPT-4o gleichauf (44,73 gegenüber 44,81); die „Beispiele“ sind Beispielentitäten je Typ. 3.2.1 nennt die Ausnahme bereits | „in der Regel deutlich“ |
| B24 | 4.1, 4.3.1, 4.3.3 (#213, #221, #225) | „offizielle Splits“: Aufsatz und Kartentext nennen sie nicht ausdrücklich; das Repository enthält sie bei der zitierten Revision (geprüft). „mit Arbeiten vergleichbar, die dieselben Splits verwenden“ trägt keine der Quellen | „die im Repository veröffentlichten Splits“; den Vergleichbarkeitssatz als Folgerung formulieren |
| B25 | 4.2.1, S. 32 (#214) | Keraghel et al.: „the DeBERTa variants appear to be the most effective for NER“; „dienen häufig als starke Baselines“ steht dort nicht | zweiten Halbsatz streichen |
| B26 | 4.4.4, S. 41 (#232); 5.7.2, S. 67 (#273) | Semins angereicherte Variante nutzt Zeichenindizes; die Wortnummerierung stammt von Ramponi et al. (Tab. 9; Tab. 3 enthält nur die Ergebnisse) | in 4.4.4 unkritisch, der Folgesatz stellt es klar; in 5.7.2 „Tab. 9; Ergebnisse in Tab. 3“ |
| B27 | 4.4.5, S. 42 (#243) | Dettmers et al. 2023: 0,373 Bit in Abschn. 3; „ohne Leistungseinbuße“ in Abschn. 4 | Zusatz „Abschn. 3 und 4“ |
| B28 | 4.5.1, S. 44 (#244) | Sammelzitat, siehe 4.2 | optional Zitate den Satzteilen zuordnen |
| B29 | 5.7.1, S. 66 (#262, #264) | GLiNER wird ohne Zieldaten ausgewertet (Zero-Shot), NuNER mit wenigen. Der Gegenwert 92,77 bei Tan et al. ist deren retrieval-gestütztes System | „bei wenigen oder keinen Trainingsdaten der Zieldomäne“; bei Tan et al. unkritisch (3.2.1 erklärt es) |
| B30 | 1.2, 3.4, 3.6, 5.7.2 (#209, #279 u. a.) | Obeidat et al.: Die Faktoren 40 (JNLPBA) und knapp 220 (BioRED) stehen im Fließtext der Abschnitte III und IV; Tab. IV liefert die Zeiten | Zusatz „Abschn. III–IV, Tab. IV“ |
| B31 | 3.2.2, S. 20 (#129); 3.5, S. 26 (#174) | „LLMs im Bereich von 7 bis 13 Mrd. Parametern“: NuNER wird nur mit UniNER-7B verglichen (Bogdanov et al., Tab. 2), GLiNER mit UniNER-7B und -13B | „mit 7 Mrd. (NuNER) bzw. 7 bis 13 Mrd. Parametern (GLiNER)“ |

### Teil C: Bibliografische Korrekturen

| Nr. | Eintrag | Maßnahme |
|---|---|---|
| C1 | `hoffmann2022training` | Titel nach dem veröffentlichten PDF und Crossref: „Training Compute-Optimal Large Language Models“; Zusatz ersetzen durch „in den Proceedings online gelistet als ‚An empirical analysis of compute-optimal large language model training‘“. Alternativ den jetzigen Titel behalten und nur den Zusatz berichtigen („Titel des PDF: …“) |
| C2 | `manning1999foundations` | URL vor der Abgabe erneut prüfen. Ist sie weiter nicht erreichbar, das Feld `url` streichen (bei einem gedruckten Buch entbehrlich) |
| C3 | `qwen2026qwen35blog` | im Browser Titel und Datum prüfen; wenn ein Tagesdatum angegeben ist, eintragen. Für die Leserschaft wäre ein Archivlink hilfreich, weil die Seite ohne JavaScript leer bleibt |
| C4 | `weischedel2013ontonotes` | Herausgeber sichtbar machen (Eintragstyp `@dataset` mit `publisher`, oder `organization`/`howpublished = {Linguistic Data Consortium}`), danach das gesetzte Verzeichnis ansehen |

### Teil D: Formale Vereinheitlichungen (optional)

- `mayhew2024uner`: `{LJ}` schützen, damit „Miranda, LJ“ statt „Miranda, L.“ erscheint.
- `jurafsky2024speech`: an den drei Zitierstellen das Kapitel angeben (1.1 und 2.1: Kap. 1; Anwendungen von NER: Kap. 18).
- Abb. 2.2 und 2.3: wie bei Abb. 2.1 die Abbildungsnummer der Vorlage ergänzen („Abb. 1“).
- `address` ist nur bei sechs der 27 ACL-Einträge gesetzt; im APA-Stil wird das Feld nicht ausgegeben, eine Angleichung ist daher nicht nötig.
- CrossNER-Repository als eigenen Online-Eintrag aufnehmen (siehe 4.4).

### Teil E: Noch am Original zu prüfen (liegt bei dir)

| Nr. | Quelle | Betroffene Stellen | Was zu prüfen ist |
|---|---|---|---|
| E1 | Manning und Schütze (1999) | 1.1, S. 1 (#1); 2.1, S. 6 (#33, #34) | allgemeine Bestimmung von NLP; „von Hand entworfene Merkmale wie Wortformen, Wortarten oder Einträge in Namenslisten, die ein statistisches Modell gewichtete“. Das Buch lag mir nicht vor. Für den zweiten Satz trägt ersatzweise Ma und Hovy (2016), Einleitung: traditionelle Sequenzlabeling-Systeme stützten sich stark auf handgefertigte Merkmale und aufgabenspezifische Ressourcen wie Namenslisten (geprüft) |
| E2 | Efron und Tibshirani (1993) | 4.5.1, S. 45 (#247) | Verweis auf das allgemeine Bootstrap-Verfahren; das Buch lag mir nicht vor. Die Zuschreibung ist Standard. Koehn (2004) trägt das gepaarte Ziehen derselben Satzstichproben für beide Systeme (geprüft) |
| E3 | Qwen-Blog | 2.5, S. 13 (#81) | „im Februar 2026 erschienenes erstes Qwen3.5-Modell“; siehe C3 und A10 |
| E4 | Hripcsak und Rothschild (2005) | 2.8, S. 16 (#105) | nur das Abstract war einsehbar; es enthält die zitierte Aussage ausdrücklich |
| E5 | zehn OpenReview-Links, ACM-Link | Verzeichnis | einmal im Browser anklicken |
| E6 | Tagungs- und Zeitschriftenfassungen | siehe Abschnitt 6 | Abschnitts-, Tabellen- und Anhangsnummern bei He et al. 2023 (Abschn. 4.1), Hu et al. (Abschn. 4.1), Zhou et al. (Tab. 2, Abschn. 5.2) und Sainz et al. (Tab. 3; Anh. B, Tab. 6; Anh. D.2, Tab. 9; Anh. D.3, Tab. 10; Abschn. 5.2) habe ich nur an den arXiv-Fassungen bestätigt. Bei Sainz et al. und Zhou et al. sind die letzten arXiv-Versionen als ICLR-Fassung gekennzeichnet; eine Abweichung ist unwahrscheinlich, aber nicht ausgeschlossen |

Ein sachlicher Hinweis ohne Handlungsbedarf: Der Testumfang von 32 908 Beispielen, den GoLLIE für MultiNERD ausweist (Anh. D.2, Tab. 9), ist genau das Doppelte des englischen Testsplits mit 16 454 Sätzen. Der Text sagt „etwa das Doppelte“; „genau das Doppelte“ wäre richtig und stützt die Vermutung, dass dort nicht nur der englische Testsplit ausgewertet wurde.

## 6. Abdeckung

### Literatureinträge (84)

| | Anzahl |
|---|---|
| erfasst und bibliografisch geprüft | 84 |
| davon vollständig an einem Register oder der Publikation bestätigt | 78 |
| teilweise bestätigt (Efron und Tibshirani; Lafferty et al.; Seiten bei Ramshaw und Marcus) | 3 |
| Korrektur empfohlen (Hoffmann et al.; URL bei Manning und Schütze) | 2 |
| offen (Qwen-Blog) | 1 |

Verfügbarkeit der Volltexte für die inhaltliche Prüfung:

| Art | Einträge | Schlüsselverwendungen |
|---|---|---|
| Volltext in der zitierten Fassung | 58 | 242 |
| Volltext in anderer Fassung | 18 | 95 |
| nur Auszug (Jurafsky und Martin: Kap. 1 und 18; van Rijsbergen: Kap. 7; Manning et al. 2008: Abschn. 8.3 und 13.6 aus dem Archiv; OntoNotes: Katalogseite) | 4 | 13 |
| nur Abstract (Hripcsak und Rothschild) | 1 | 1 |
| nicht verfügbar (Manning und Schütze 1999; Efron und Tibshirani 1993; Qwen-Blog) | 3 | 5 |

„Andere Fassung“ heißt: 15-mal arXiv statt Tagungs- oder Zeitschriftenfassung (`he2020deberta`, `he2021debertav3`, `hu2021lora`, `han2024parameter`, `zhou2024universalner`, `sainz2024gollie`, `wei2022emergent`, `clark2020electra`, `yang2025gated`, `holtzman2020curious`, `liu2021crossner`, `li2018survey`, `bengio2013representation`, `zhao2023survey`, `lu2024llmstruggle`), dazu Lafferty et al. als Vorlesungskopie, Schuster und Nakajima als Kopie bei Google Research und Obeidat et al. als Autorenmanuskript der veröffentlichten Fassung in PMC. Für diese Einträge gelten Abschnitts- und Tabellennummern für die eingesehene Fassung.

### Zitierstellen (298 Befehle, 356 Schlüsselverwendungen, 508 Einzelaussagen)

| | Schlüsselverwendungen |
|---|---|
| erfasst | 356 |
| anhand des Originals in der zitierten Fassung inhaltlich geprüft | 242 |
| anhand einer anderen Fassung, eines Auszugs oder des Abstracts geprüft | 109 |
| offen (Quelle nicht zugänglich) | 5 |

### Wer hat was geprüft

- Die Einzelprüfung jeder Zitierstelle haben neun nur lesende Prüfagenten kapitelweise vorgenommen, jeweils mit dem Thesis-Quelltext und den Quelltexten. Die Agenten haben in ihren Abschnitten auch den ganzen Text auf unbelegte Aussagen und manuelle Quellenangaben gelesen.
- Selbst nachgeprüft habe ich: alle Befunde in Teil A und Teil C, alle Zeilen mit „Kein Beleg“ und „Widerspricht“, die in Teil B mit Werten oder Zitaten belegten Punkte und rund 40 der als tragfähig beurteilten Zahlen und Aussagen. Acht Agentenurteile habe ich dabei geändert, alle zugunsten der Arbeit (in der Matrix an der Begründung erkennbar), und bei drei weiteren das Urteil präzisiert. Abbildung 3 von Lu et al. habe ich am PDF angesehen.
- Zusätzlich habe ich maschinell geprüft, ob die von den Agenten angeführten wörtlichen Quellzitate im jeweiligen Quelltext stehen: 448 von 471 ließen sich lokalisieren; die übrigen 23 sind Kurzformeln, Zusammenfassungen oder Stellen aus zerfaserten Tabellen und betreffen keinen der Befunde.
- **Nicht einzeln wiederholt** habe ich die Urteile „Trägt“ der Agenten, soweit sie nicht in die Stichprobe fielen. Für diese Zeilen ist der Bericht eine dokumentierte Agentenprüfung mit Fundstelle, keine Zweitprüfung durch mich.

### Kapitel

| Teil | Zitierstellen | Suche nach unbelegten Aussagen |
|---|---|---|
| Abstract | keine Zitate | vollständig gelesen, kein Befund |
| Kapitel 1 | vollständig (32 Befehle) | vollständig |
| Kapitel 2 | vollständig (73) | vollständig, einschließlich der Bildunterschriften |
| Kapitel 3 | vollständig (105), jede Zelle von Tab. 3.1 und 3.2 | vollständig |
| Kapitel 4 | vollständig (45) | vollständig; eigene Messwerte und Laufdetails ausgenommen |
| Kapitel 5 | vollständig (32) | 5.1 und 5.7 vollständig gelesen, 5.5 und 5.6 in Teilen; der übrige Ergebnisteil (5.2 bis 5.4) nur auf Zitate und Quellensignale durchsucht |
| Kapitel 6 | vollständig (7) | vollständig |
| Anhang | vollständig (4) | vollständig; Laufdaten ausgenommen |
| Erklärung, Verzeichnisse, erzeugte Tabellen, Abbildungsdateien | keine Zitate | nur durchsucht |

### Grenzen

- Tabellenwerte stammen aus Textfassungen der PDFs; bei zerfaserten Tabellen ist die Zuordnung von Zeile und Spalte rekonstruiert und, wo möglich, über Summen oder Fließtext abgesichert.
- Negativaussagen über die Literatur („fehlt“, „keine der Arbeiten“) sind nur gegen die 84 zitierten Quellen geprüft, nicht gegen die Literatur insgesamt.
- Eigene Messwerte der Arbeit waren nicht Gegenstand dieser Prüfung.
- Hilfsdateien dieser Prüfung (Quelltexte, Metadatenabrufe, Skripte) liegen im Sitzungsverzeichnis `/private/tmp/claude-501/-Users-luajosman-bachelor-thesis-llm-vs-encoder-ner/a04e689c-f6de-45e9-b200-4bbf6327696c/scratchpad/` und können dort verloren gehen; die beiden Berichtsdateien sind davon unabhängig.

## Nachtrag: Umsetzung (24.11.2026)

Nach deiner Freigabe umgesetzt in `chapters/01` bis `06`, `chapters/anhang.tex` und `references.bib`. Die Abschnitte 1 bis 6 dieses Berichts und die Matrix beschreiben den Stand davor (`e82599f`).

**Stand danach:** 110 Seiten, Haupttext S. 1–76, Literatur ab S. 77, Anhang ab S. 86; Build ohne Fehler, ohne Warnungen, ohne Overfull- oder Underfull-Meldungen; weiterhin 84 Literatureinträge. Einzige Umbruchänderung: Abschnitt 3.6 beginnt oben auf S. 27 statt unten auf S. 26, Tabelle 3.2 steht auf S. 28. Kapitel 3 endet unverändert auf S. 29. Nicht committet.

| Punkt | Umsetzung |
|---|---|
| A1 | 3.4: Sammelaussage auf „einzelne Benchmarks“ eingegrenzt, mit den Datensätzen je Arbeit und dem Rückstand bei Zhan et al. auf OntoNotes 5.0, ACE2005 und GENIA. 3.6: „in der Tendenz übereinstimmen“; E1 mit „wie in der Literatur auf einzelnen Benchmarks“. Kapitel 6 brauchte keine Änderung |
| A2 | 2.4: Beleg auf die Konfigurationsdateien der Revisionen (Verweis auf Tabelle A.3) umgestellt; für BERT `\cite[Anh.~A.2]{devlin2019bert}` ergänzt |
| A3 | 3.3.1: „auf dem biomedizinischen Datensatz BC5CDR teils um mehr als 40 Punkte, auf OntoNotes 5.0 deutlich weniger“ |
| A4 | 5.7.2 und 4.4.4 wie vorgeschlagen |
| A5 | 5.7.3: Größenvergleich nur noch für GoLLIE und als eigene Rechnung gekennzeichnet |
| A6 | 2.5 (Position mit Semin et al. belegt), Einleitung zu Kapitel 4 (Beleg hinter „JSON-Array von Entitäten“), 4.4.4 („nur über eine API genutzt“), 4.5.3 (Zitate getrennt) |
| A7 | Tabelle 3.2: GLiNER „Trainingszeit“; Semin et al. „Prompting; Constrained Decoding“ und „Ausgabetokens (geschätzt)“; PaDeLLM und GoLLIE „berichtete Werte“; GPT-NER und CodeIE „teilweise“; Legende: „teilweise“ = nur ein Teil der Vergleichswerte selbst erzeugt, Ressourcen „erhoben oder geschätzt“ |
| A8 | 3.4: `\cite[Abschn.~3.2, Tab.~4]{zhan2026generativeNER}` am GENIA-Halbsatz |
| A9 | 3.1: „gehört zu den am häufigsten verwendeten NER-Benchmarks“; OntoNotes-Satz geteilt, Typzahl bei Li et al.; Li et al. in der Unterschrift von Tabelle 3.1 |
| A10 | 2.5: Chat-Template zusätzlich mit dem Qwen3-Bericht belegt (dort Tab. 9 mit Rollen und Steuertokens; die Rollenliste im Satz blieb); Reihenfolge („folgten“) gestrichen, MoE-Aussage auf den Schichtaufbau der Karten gestützt. 2.3: alle drei Karten zitiert. A.2.2: Kartenzitat hinter „hybrider Attention“, Grouped-Query-Attention und Ausgabe-Gating hängen damit an den Konfigurationsdateien. (f) nicht geändert |
| B1 bis B4 | umgesetzt (B1 ohne Kapitelangabe, weil das Zitat ein Sammelzitat ist; in B4 „häufig“, „verbreitet“, „die übliche Architektur“) |
| B6, B7, B8 | umgesetzt (B8 als „typischerweise Milliarden“) |
| B10, B11, B12 | umgesetzt |
| B15, B16, B17 | umgesetzt |
| B19 bis B27 | umgesetzt; bei B24 nur die Zuordnung des Zitats in 4.3.3 |
| B29, B31 | umgesetzt |
| aus 4.3 | Dropout-Satz, Adapter auf allen linearen Schichten (Dettmers et al. 2023), „seit CoNLL-2003 üblich“, InstructUIE-Zitat in 3.6, Größenklasse („passen … an“) in 1.2 und 3.6, enge Lesart der Größenstaffelung, Tedeschi-Beleg und „vermutlich“ zur Kontamination in 3.6, Obeidat-Zitat in 6.1, GoLLIE „genau das Doppelte“ |
| C1 | Titel „Training Compute-Optimal Large Language Models“, Zusatz nennt den Listentitel der Proceedings |
| C2 | URL entfernt (am 24.11. abends weiterhin nicht erreichbar) |
| C4 | Eintragstyp `@dataset`; im Verzeichnis jetzt „OntoNotes Release 5.0 (LDC2013T19) [Datensatz]. Linguistic Data Consortium.“ |
| D | Mayhew et al.: „Miranda, L. J.“; Abb. 2.2 und 2.3 mit „Abb. 1“ der Vorlage |

**Bewusst nicht umgesetzt:**

- B5, B9, B13, B14, B18, B28: im Bericht als unkritisch oder optional eingestuft.
- B30 (Fundstelle der Faktoren 40 und 220 bei Obeidat et al.): Im zweispaltigen Textauszug ließ sich nicht sicher entscheiden, in welchem nummerierten Abschnitt die beiden Sätze stehen. Der Zusatz „Tab. IV“ bleibt; die Tabelle enthält die Zeiten, aus denen sich die Faktoren ergeben.
- C3 und E1 bis E6: liegen bei dir.
- Teil D: Kapitelangaben bei Jurafsky und Martin (alle drei Stellen sind Sammelzitate) und ein eigener Eintrag für das CrossNER-Repository.
- Aus 4.3: zusätzliche Zitate in 4.2.2, 5.1.2, 5.5 und A.2.3, der Satz zur Precision ohne Vorhersagen in 2.8 und die Neuheitsbehauptung in 5.7.2 (jeweils unverändert).
