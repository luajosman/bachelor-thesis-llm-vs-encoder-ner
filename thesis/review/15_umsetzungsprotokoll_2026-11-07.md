# Umsetzungsprotokoll der 1,0-Überarbeitung (ab 7.11.2026)

Plan: `~/.claude/plans/guck-dir-bitte-kapitel-quiet-fairy.md` (freigegeben 7.11.2026). Zwei Phasen: **A** (Sa 7.11./So 8.11.) Korrekturen und Pflichtangaben → Betreuerentwurf (Tag `betreuerentwurf-2026-11-08`); **B** (Mo 9.11.–Mi 11.11.) Verdichtung und Gesamtabnahme → Änderungsübersicht und Bewertungsmatrix.

## Nutzerentscheidungen (7.11.)

- Nur Claude bearbeitet `thesis/chapters/*.tex`, `references.bib`, `main.tex`; Codex' unbestätigte Änderungen (STATUS.md, review/README.md, Bericht 09) bleiben unangetastet und werden erst in Block 9 mit committet.
- Kapitel 5/6: korrigieren, verdichten, Stil glätten; der Verfasser liest danach alle Kapitel gegen.
- Blockweise mit Zwischenbericht, ohne auf Freigabe zu warten.
- Titel bleibt wie auf dem Deckblatt (ausgeschrieben). KI-Angabe nennt Claude Code und OpenAI Codex (Codex-Modell/-Version: vom Verfasser einzutragen).
- Repo-URL `https://github.com/luajosman/bachelor-thesis-llm-vs-encoder-ner` wird eingetragen; Repo wird vor Abgabe öffentlich.
- Rechercheabsatz: Google Scholar, arXiv, ACL Anthology; 2019–2025; Stichwörter (Vorschlag, zu bestätigen): LLM named entity recognition, generative NER, instruction tuning NER, LoRA/QLoRA NER, DeBERTa NER, MultiNERD, CrossNER; Schneeballsuche ab Schlüsselarbeiten.
- Reihenfolge: Korrekturen zuerst, Verdichtung nach dem Betreuerentwurf.
- Abb. 2.4 bleibt; `eq:powerlaw`, Bahri-Satz und Nebenbefunde in 2.7 werden gestrichen.

## Feste Namen

- Neue Bib-Schlüssel: `ramponi2025fallacy`, `mayhew2024uner`, `ding2024gner`, `efron1993bootstrap`, `koehn2004bootstrap`, `he2021debertav3card`, `vonwerra2026trldocs`; vorhanden: `qwen2026card08b/4b/27b`, `goldberg2016primer`, `cawley2010selectionbias`.
- Neue Labels: `app:durchfuehrung`, `tab:app-qwen-architektur`, `tab:app-limit-seeds`, `tab:datennutzung`, `tab:synthese`.
- Nummern aus den Berichten → Labels: Tab. 4.1 = `tab:experiment-matrix`; Tab. 4.2 = `tab:qwen-architektur`; Tab. 4.4 = `tab:hardware`; 4.5.2 = `sec:effizienzmetriken`; 4.5.4 = `sec:einfluss-parameteranzahl`; Tab. 4.9 = `tab:abweichungen`; A.1 = `app:reproduzierbarkeit`; Tab. A.1 = `tab:app-laeufe`; Tab. A.2 = `tab:app-revisionen`; Tab. A.3 = `tab:app-einstellungen`; Tab. A.4 = `tab:app-seeds`; Tab. A.8 = `tab:app-beispiele`.

## Prüfwerkzeuge (`thesis/scripts/check/`)

`overlap.py` (8-Wort-Shingles gegen Quelltexte und Berichte), `numbers.py` (Zahlen in Diff/Datei gegen summary.json und generated/*.tex), `sentences.py`, `words.py`, `pages.sh`, `resolve_anchors.py`, `terms.sh`; gemeinsame Hilfsfunktionen in `texclean.py`. Interpreter `/Users/luajosman/miniforge3/bin/python3`.

## Baseline (HEAD 3c49499, Build vom 7.11. 07:07)

| Größe | Wert |
|---|---|
| PDF | 112 Seiten; Overfull 0, Underfull 0, undefined 0 |
| Haupttext | S. 1–90 (K1 1–6, K2 7–23, K3 24–37, K4 38–61, K5 62–84, K6 85–90); Literatur 91–98; Anhang 99–112 |
| Fließtextwörter (words.py) | K1 1 764, K2 4 947, K3 4 462, K4 6 407, K5 6 439, K6 1 776; Σ 25 795 |
| Sätze > 50 W (sentences.py) | K1 2, K2 7, K3 7, K4 9, K5 11, K6 1 |
| Terminologie-Altvarianten (terms.sh) | 47 Zeilen in K1–K6 + Anhang |
| Overlap-Nullmessung (n = 8, 93 Vergleichsdateien: 77 Quelltexte + 16 Berichte) | 0 Treffer gegen Quelltexte; 8 Treffer gegen Berichte, alle Zitate des bestehenden Thesis-Textes in den Berichten (02:447, 02:750, 03:733, 04:323, 05:244, 05:408, 05:871, 06:110) |
| numbers.py Kapitel 5 (ganze Datei) | 1 034 Zahlen, 1 004 belegt, 28 UNBELEGT (abgeleitete Werte wie Differenzen/Faktoren, Literaturzahlen, Splitstatistik; Liste als Prüfgrundlage für Block 5a) |

## Übertrag (Aufträge, die in einer anderen Datei landen als dem Block, der sie entdeckt)

| Quelle | Zieldatei | Anker | Auftrag | Status |
|---|---|---|---|---|
| K3-02 | 05 | „Obwohl die Ausgabelänge nur von der Zahl" | Ausgabelänge hängt auch von Spantext/Serialisierung ab | offen (5a) |
| K3-13 | 05 | 05:981 „geprompte" | „per Prompting genutzte LLMs" | offen (5a) |
| Keraghel „Few-Shot" | 01, 03, Tab. 3.1, 05 | 01:94–96; 03:106; 03:603; 05:978–981 | „mit Beispielentitäten im Prompt" | offen (3, 2a, 5a) |
| Block 1 Zhan-Note | 03 | 03:417–418 „zur Veröffentlichung in den Findings" | streichen, falls nicht belegbar | offen (2a) |
| Block 1 Goldberg | 02 | 02:58–60 | `goldberg2016primer` mit Abschnitt oder Seitenanspruch streichen | offen (4a) |
| K5 #13 | 06 | 06:176–187 | Trainingsspeicher nicht gemessen; allokierter Speicher begründet keine Eignung | offen (6) |
| K6-05 / K2-03 / K4-04 | 02, 04 | 02:770–775; 04:315–318 | „1 000 Sätze": Definition 2.8, Begründung 4.3.1, sonst Verweis | offen (4a, 7a) |
| Block 3 FF-Wortlaut | 06, abstract | — | finaler FF1–FF3-Wortlaut als Pflichteingabe | offen (nach 3) |
| 7a Daten-Agent | 06 | 06:107–120 | Val-Loss-Abgleich (x von 9) in 6.2 nennen | offen (6/7a) |

## Blockprotokoll

### Blöcke 2a, 3, 5a, 7a, 8 (Sa 7.11., Phase A) – Übersicht
Je Block: Schreib-Agent nach Vorlage 5.1 → eigener Diff-Review → Prüf-Agent nach Vorlage 5.2 → Korrekturen → gemeinsamer Build. Alle Prüf-Agenten bestätigten die Zahlen (summary.json) und die Quellenwiedergabe an den Quelltexten; die gemeldeten Restpunkte wurden behoben (siehe je Block). Build nach allen Blöcken: 117 Seiten, Haupttext 1–93, Overfull 0, Underfull 0, undefined 0. Haupttext wächst in Phase A um ≈3 Seiten (Korrekturen, Tabellen); die Verdichtung folgt in Phase B.

**Block 2a – Kapitel 3 (Commit „Kapitel 3: Quellenwiedergabe korrigiert …"):** F01 (GoLLIE Tab. 10: 44,5/79,5/94,6 h, CO₂; keine Inferenz), F02 (Anh. D.2 Standardsplits; Tab. 9: 32 908 = 2 × 16 454, Beziehung ungeklärt), F03 (Anh. G: LLaMA-3.1-8B und BERT-MRC auf L40 selbst trainiert), F04/F04-b (zwei getrennte NuNER-Vergleiche; nur GLiNER/UniNER teilen Pile-NER), F05 (GPT-NER: feinjustierter Tagger, Abschn. 3.2.2), F10 (Synthese: generativ nur Zhan/Obeidat; Dukić Klassifikationskopf; Inferenzkosten nur Obeidat; Formate wachsen mit der Eingabelänge), F11 (Ramponi Tab. 9 als Ursprung; Semin Systematisierung; JSON/Nachverarbeitung eigene Umsetzung), K3-01 (341/353 vs. 336/435 Mio.), K3-02, K3-04 (UNER ≠ UniversalNER, `mayhew2024uner`), K3-05 (Zhan Tab. 11: LoRA 0,14–0,48 vor Full-FT), K3-07, K3-08 (< 500 $), K3-09, K3-13 (Terminologie inkl. IOB2), Keraghel („Beispielentitäten im Prompt"; WNUT2017 44,73 vs. 44,81, GLiNER-L dort höher), L-19 (Postnotes: Keraghel Tab. 5, Zhan Tab. 3, Obeidat Tab. III/IV, GoLLIE Tab. 3, UniNER Tab. 2, Tan Tab. 4, GPT-NER Tab. 1), F06 (3.5: Lücke 1 = Datenbasis und Pipeline, Lücke 2 = Ressourcenerfassung und Modellgröße; UniversalNER/GLiNER/GoLLIE-MultiNERD-Werte abgegrenzt; OntoNotes-Genres mit `weischedel2013ontonotes`), E1–E3 (E1 „erreicht das Niveau … oder übertrifft es leicht"), K3-06 (GNER-Absatz + Tab.-3.1-Zeile `ding2024gner`, arXiv 2402.16602v2 geprüft), K3-03 (Rechercheabsatz: Google Scholar, arXiv, ACL Anthology, 2019–2025, Stichwörter, Schneeballsuche, keine Trefferzahlen). Wörter 4 462 → 5 104 (+642; Budget überschritten, Verdichtung in Block 2b). Nach Prüfbericht korrigiert: WNUT-Satz, GNER-Vergleichswerte (teils neu evaluiert), OntoNotes-Genres, Verweis `sec:datenaufteilung`, Formulierung F05 (Überschneidung mit Bericht 08 beseitigt), Bezugswort „Dieser Vorsprung", zwei Tab.-3.1-Zellen (Overfull).

**Block 3 – Kapitel 1 (Commit „Kapitel 1: Forschungslücke …"):** F06 (vier Aspekte als zwei Lücken, absolute Aussage entfernt), K1-01, F11 (Ramponi), K1-02 (FF2 „innerhalb der untersuchten Konfigurationen beider Modellfamilien", FF3 „unter den gemessenen Bedingungen"), K1-05, K1-06/07, Keraghel-Übertrag, Kürzungen K1 #1, #3–#7 (#2 als zwei Sätze), DFKI als ein Satz. Wörter 1 764 → 1 520. Nach Prüfbericht: „etablierte allgemeine Benchmarks", „innerhalb jeweils einer Modellfamilie", „in 4 Bit", Prompt-/Decoding-Bedingungen in 1.3. FF-Wortlaut (Pflichteingabe Block 6/8) im Bericht des Schreib-Agenten festgehalten.

**Block 5a – Kapitel 5 (Commit „Kapitel 5: Überdeutungen …"):** K5-Ü1–Ü11, K5-Z1 (18,6 GiB), K5-Z2 (Transfer: LLM 12,4 bei 1 024 mit Referenz 99,6; Encoder 14,5), K5-Z3–Z8 (0,5–0,6; 1,03–1,04; knapp das Dreißigfache; 5,6-/6,4-Fache; 3,8–5,2; Differenzen mit Intervall zweistellig: 1,34/0,18/0,14; 0,82/0,51; 2,13; 13,29), K5-Z9 (Encoder-Recall auf abgeschnittenen Sätzen 93,8–96,6), K5-N1–N8 (Ramponi/Semin präzisiert; Sampling-Bedingung; „nicht untersucht"; Hauptvergleiche vs. explorativ; Vorlauf-Satz jetzt mit Tab. `tab:app-vorlaeufe` ≤ 0,12; Obeidat Tab. III; Keraghel Tab. 5; GoLLIE Tab. 6; Modellkarte ohne Trainingsdaten; „ähnelt"/„nahelegen"), Synthesetabelle `tab:synthese`, alternative Erklärungen (5.7.4) und Vermutungssatz (5.7.1), Relevanzschwelle (SD 0,03–0,10; base/large 0,3), „illustrative Auswahl". Greedy-Absatz gestrichen (steht in 4.4.3). Wörter 6 439 → 6 734. Nach Prüfbericht: zwei Sätze neu formuliert (Überschneidung mit Bericht 08), Satz > 50 W geteilt, Rundung 5.3/5.6.

**Block 7a – Kapitel 4 (Commit „Kapitel 4: Fairnessaussage …"):** K4-01 („intern fair" ersetzt), K4-02, F06 (K4), F12a (Efron & Tibshirani, Koehn; Verfahren wie implementiert), F12b (DeBERTa-Modellkarten MIT), F12c (TRL 1.8.0, `assistant_only_loss` False), F11 (Ramponi; Dagdelen nur JSON), K4-03 (Qwen-Modellkarten statt URL-Fußnote), K4-04 (Gold-Stichprobe 1 000 Sätze, ≥ 20 je Typ, Tab. 5; Grund), K4-05 (Sampling: Modellkarte je Modus; Abweichungen top-p 1,0, keine Presence-Penalty), K4-06 (vier dekorative Zitate gebunden/gestrichen), K4-07 (Zhan Anh. D), K4-08, K4-09 (Doppel-Label, `\paragraph`, „1 Minute", Validierungsregeln als Liste, Tempus), 04:651 entlastend, Datenlizenzen (MultiNERD CC BY-NC-SA 4.0; CrossNER MIT), Tab. `tab:datennutzung` (8 Zeilen) + Verweis in `tab:abweichungen`, Diagnose Checkpoint-Auswahl (1/9, 0/6; 12/15 letzte Epoche; 200 Sätze ±3,1 Pp., ohne BIO/INST; Statistik in `thesis/generated/validation_subset_stats.md`, Skript `ba-ner/scripts/validation_subset_statistics.py`). Wörter 6 407 → 6 743. Nach Prüfbericht: OntoNotes ohne „numerische Ausdrücke", CrossNER-Lizenz über Repository-Commit, Zero-Shot-Paarung im Bootstrap-Satz.

**Block 6 – Kapitel 6 (Commit „Kapitel 6: Antworten je Forschungsfrage …"):** 6.1 als je ein Antwortabsatz zu FF1–FF3 mit Abgleich der Erwartungen E1–E3 (E1 weitgehend bestätigt bei ausreichendem Limit; E2 bestätigt; E3 bestätigt); K6-N1 (Punktwertunterschiede ohne Intervalle), K6-N2 (MultiNERD und vergleichbar annotierte Wikipedia-Texte; Latenz- und Speicherbudget), K6-N3 („blieb ungeprüft"), K6-N4 (Testnutzung: zwei Arten, Checkpoint-Diagnose, BF16-Durchsatzwahl, Sensitivitätsanalysen post hoc, Verzerrung nicht bezifferbar), K6-03 (FF-Zuordnung in allen sechs Limitationsabsätzen), K6 #7 (Abweichungen in zwei Sätzen + `tab:abweichungen`), K6-Eff (Trainingsspeicher, reservierter Speicher, Übertragbarkeit der Faktoren), K6-02c (Hauptvergleiche vs. explorativ, Dokumentabhängigkeiten, keine Korrektur für Testnutzung), F09 (LogitMatch als Constrained Decoding auf gültige Eingabespans, Semin Abschn. 4), K6-N5 (Tedeschi-Stichprobe mit ≥ 20 je Typ; Dequantisierungs-Halbsatz gestrichen, nicht belegbar), Ausblick mit drei priorisierten Anschlussfragen. `\paragraph` 15 → 9; Wörter 1 776 → 1 397 (nach Wiederaufnahme der vom Prüf-Agenten als fehlend gemeldeten Limitationsinhalte). Nach Prüfbericht: Gleichwertigkeitsaussage zu base zurückgenommen („keine Rangfolge nachweisbar"), Zero-Shot nicht als Obergrenze, Speicherbereich auf trainierte Konfigurationen eingeschränkt (Zero-Shot-27B 50,7 GiB), „eine gut gestützte Wahl". Build: 118 Seiten, Haupttext 1–94, 0/0/0.

**Block 4a – Kapitel 2 (Commit „Kapitel 2: technische Aussagen präzisiert …"):** A1 (Decoder kausal, Textausgabe in dieser Pipeline, rechter Kontext über Prompt, Klassifikationskopf nach Dukić), A2 (Prefill, Decoding-Schritt mit wachsendem Kontext), A3/F08 (Constrained Decoding verhindert Syntax-/Schemafehler, nicht inhaltliche Richtigkeit), A4 (LoRA-α, Hu Abschn. 4.1), A5 (QLoRA: 4-Bit-Speicher, 16-Bit-Rechnung; Paged Optimizers verringern Risiko; Leistung auf GLUE, Super-NaturalInstructions, MMLU), A6, A7 (Kaplan: mehr als sieben Größenordnungen; Abbildung schematisch), A8 (DeBERTa-v3: weitgehend gleiche Einstellungen, 160 GB wie DeBERTa-v2, Abschn. 4.1), K2-02, K2-03 (1 000 Sätze, ≥ 20 je Typ, Tab. 5; Typkürzel mit Langform), K2-04, K2-05, K2-07 (Dimensionen, Formalisierung nach Li, Greedy als Auswahlregel), L-17 (`goldberg2016primer`, Abschnittsangabe entfernt, da nur am arXiv-Rendering lokalisierbar). Wörter 4 947 → 5 066. Prüf-Agent am Rate-Limit abgebrochen → eigener Diff-Review; Nachprüfung in Block 9.

**Argumentationsgang (Commit „Argumentationsgang: …"):** Prüf-Agent (Abstract, 1.2–1.4, 3.5, 4.1, 4.3.3, 5.7, 6) meldete zehn Punkte, alle umgesetzt: Hauptvergleiche vs. explorativ in 4.5.1 verankert (Experimentmatrix bei 256 Tokens; Sensitivitätsanalysen explorativ); 1.4 „Systemnachricht des Zero-Shot-Prompts"; FF2-Antwort mit Verlauf je Familie und Zero-Shot-Größeneffekt (28,8/34,5 als Formtreue-Effekt); FF1-Antwort mit Zero-Shot-Formtreue (0.8B Syntax/Typen; 4B/27B 3 930/3 397 inkonsistente Elemente); E1 auf MultiNERD bezogen, CrossNER nur Einordnung; 3.5 „Formtreue der LLM-Ausgaben" und „nicht vollständig reproduziert"; 1.2 Aspekt 3 „nur unvollständig gemeinsam"; „Silberreferenz" statt „Silver Standard" in 3.5; „Inferenzlatenz" statt „Modelllatenz" in Abstract, 5.7.4, 6.1 (Begriff aus 4.5.2). Bestanden: vier Aspekte ↔ zwei Lücken, E1–E3-Abgleich durch 5.7 gedeckt, Abstract ↔ 6.1 zahlengleich, Testnutzung 4.3.3/5.7.5/6.2 deckungsgleich. Phase A abgeschlossen mit Tag `betreuerentwurf-2026-11-08` (Build 119 Seiten, Haupttext 95, 0/0/0). Offen aus Phase A: Zahlen-Audit aller Kapitel und Prüfung von Kapitel 2 (beide Agenten am Rate-Limit abgebrochen) → in Block 9 nachholen.

**Block 8 – Anhang, Frontmatter, main.tex (Commit „Anhang und Frontmatter …"):** AN-01 (Repo-URL, Tag, Zugangsregel), AN-02 (`\clearpage`), AN-03 (Kurztitel), AN-04 (Captions „256 Ausgabetokens", Spaltenköpfe), AN-05 (`tab:app-limit-seeds`), AN-06 (Tab. `tab:app-beispiele`: Spalte Nr. = Listenindex, Grenz-/Typfehler-Beispiel Seed 123, Arboretum-Zusatz, illustrativ, `\cite{tedeschi2022multinerd}`), AN-07 (Absatz „Ausführungszeiten und Cluster-Jobs"), Vorläufe (`tab:app-vorlaeufe`, aus dem Skript), AN-08 (bestätigt, keine Änderung). F13 (KI-Angabe: Claude Code und Codex, alle Kapitel; Codex-Modell rot als TODO), FM-02 (Komma), FM-03 (Deckblatt „Berlin, 25. November 2026", TODO-Kommentar), FM-05/06 (BIO gestrichen, ELECTRA und HBM3 aufgenommen), M-01 (römische Seitenzahlen sichtbar: `\pagestyle{plain}` ab Erklärung), Penalties gegen Hurenkinder/Schusterjungen, PDF-Metadaten (Titel, Autor), `terms.sh`-Fehlalarm „QLoRA-Regime" behoben.

### Block 0 (Sa 7.11.) – Vorbereitung
- origin/main = HEAD 3c49499 (0/0); Arbeitsbaum nur Codex' drei Dateien.
- Prüfskripte angelegt und Baseline gemessen (Tabelle oben).
- Arbeitsliste (Anhang A–C des Plans) unten angefügt. Commit 152896e.

### Block 1 (Sa 7.11.) – Literatur (`references.bib`)
| Änderung | Quelle/Prüfung | Ergebnis |
|---|---|---|
| Neue Einträge `ramponi2025fallacy`, `mayhew2024uner`, `ding2024gner`, `koehn2004bootstrap` | ACL-Anthology-Seiten (Abruf 7.11.2026): 2025.naacl-long.34 (S. 762–784, DOI), 2024.naacl-long.243 (S. 4322–4337), 2024.findings-acl.206 (S. 3461–3475), W04-3250 (S. 388–395) | eingetragen |
| `efron1993bootstrap` | Chapman & Hall 1993, Monographs on Statistics and Applied Probability 57, DOI 10.1201/9780429246593 (Verlagsseite liefert nur Titel/Autoren; übrige Angaben Standardmetadaten) | eingetragen |
| `he2021debertav3card`, `he2021debertav3cardlarge` (Abweichung vom Plan: zwei Schlüssel) | HF-Modellkarten microsoft/deberta-v3-base/-large: „License: mit", 86 Mio./304 Mio. Backbone-Parameter, 128k Vokabular | eingetragen |
| `vonwerra2026trldocs` | TRL-Doku v1.8.0, SFT Trainer: `assistant_only_loss` Standard `False`; `completion_only_loss` `None` → Verlust über die ganze Sequenz bei Konversationsdaten | eingetragen; stützt 04:669–673 |
| 20 arXiv-`@misc`: `doi = {10.48550/arXiv.<id>}`, `note = {arXiv-Preprint}` (bei Zhan/Semin vorangestellt); `type = {Preprint}` druckt biblatex-apa nicht | PDF-Text (pypdf im Container python:3.12-slim) | gedruckt als „Titel [arXiv-Preprint]. https://doi.org/10.48550/arXiv.…" (13 zitierte Einträge) |
| Zhan/Semin `eprint` mit v3/v2; Zhan-Note | arXiv-Abstractseite 2601.17898, Comments: „Accepted to EMNLP 2026 Findings" (Stand 7.11.2026) | Annahme belegt, Note präzisiert; 03:417–418 kann bleiben |
| ICLR-booktitle einheitlich (10 Einträge) | — | „International Conference on Learning Representations (ICLR)" |
| Jurafsky-Note deutsch; Lafferty-URL (ACM DL 10.5555/645530.655813); Manning 2008 DOI 10.1017/CBO9780511809071; van Rijsbergen Kapitel 7 (Seite geprüft: E-Maß mit β definiert) | WebFetch Chapter.7/Ch.7.html | erledigt |
| `commoncrawlblog2024inside` entfernt (nicht zitiert, schwache Quelle) | — | 108 Einträge |
| Offen aus Block 1 | L-16/17 Goldberg-Seitenangabe → Block 4a (`goldberg2016primer`); L-18 eingeschränkt geprüfte Quellen → Block 9 | |
Build: 112 Seiten, Overfull 0, Underfull 0, undefined 0. Werkzeughinweis: kein pdftoppm/pdftotext lokal; PDF-Text per `docker run --rm -v "$PWD":/work -w /work python:3.12-slim sh -c "pip install -q pypdf; python …"`.

### Block 5a, Teil Skript (Sa 7.11.) – `aggregate_results.py` (Commit f63c611)
Neue reine Funktionen `coarse_reference`, `transfer_losses`, `encoder_recall_on_truncated`, `tab_synthese`, `tab_app_limit_seeds`; drei neue Tests (8/8 grün); zwei Läufe byte-identisch. Neue Werte (summary.json):
| Größe | Wert |
|---|---|
| Transferverlust PER/ORG/LOC (MultiNERD-Referenz gepoolt − CrossNER gepoolt) | Encoder 99,54 − 84,99 = 14,55; LLM bei 256: 98,46 − 71,69 = 26,76; LLM bei 1 024: 99,57 − 87,12 = 12,45 (`crossner.transfer`) |
| Encoder-Recall auf den je LLM-Seed abgeschnittenen Sätzen (27B; n = 65/68/62) | DeBERTa-v3-large 93,81–96,62 (seedgleich 96,62/93,81/95,14); base 94,99–96,62; LLM dort 0 (256) bzw. 97,8–98,4 (1 024) (`output_limit_multinerd.encoder_subset`) → ersetzt die unbelegte Angabe „96,2 bis 96,5" (K5-Z9) |
| `tab-synthese.tex` (`tab:synthese`) | MultiNERD 256: 95,82 vs. 94,48, Δ +1,34 [+0,99; +1,68]; 1 024: 95,64, Δ +0,18 [+0,02; +0,33]; CrossNER 256: 84,99 vs. 71,69, Δ +13,29 [+11,28; +15,45]; 1 024: 87,12, Δ −2,13 [−2,80; −1,47] (Δ = Encoder − LLM) |
| `tab-app-limit-seeds.tex` (`tab:app-limit-seeds`) | je Seed F1 256/1 024, Δ, neu erzeugt, weiterhin abgeschnitten (alle 0), Präfix gleich (67/68, 67/68, 67/67; 70/72, 66/67, 66/67; 63/65, 67/68, 61/62) |

### Datenbefunde Daten-Agent 1 (Sa 7.11.) – Checkpoint-Auswahl (für Block 6 und 7a)
- Alle 15 Läufe protokollieren einen Validierungsverlust je Epoche (`training_history.json`, `eval_loss`; LLM: teacher-forced SFT-Verlust auf dem vollständigen Validierungssplit, 16 410 Sätze; Encoder: Token-Klassifikationsverlust, vollständiger Split).
- Gewählte Epoche = Epoche mit minimalem Validierungsverlust: LLM **1 von 9** (nur 4B Seed 42), Encoder **0 von 6**. Der generative Dev-F1 (200 Sätze) bzw. der seqeval-F1 wählt systematisch spätere Epochen als das Verlustminimum (LLM: Verlustminimum E1/E2, F1-Maximum meist E3; Encoder: Verlustminimum E2/E3, F1-Maximum E3–E5).
- Letzte Epoche gewählt: **12 von 15** (8 von 9 LLM; 4 von 6 Encoder); Ausnahmen 4B Seed 42 (E2/3), base Seed 42 (E4/5), large Seed 42 (E3/4).
- Erste 200 Validierungssätze (Reihenfolge des Loaders, kein Shuffle; `src/decoder/train.py` Z. 516–539, `gen_eval_max_samples: 200`): 283 Entitäten (passt zu `dev_recall` 262/283), 1,415 Entitäten/Satz (Split 1,587), Satzlänge 22,3 (Split 23,1); Typanteile weichen um höchstens 3,1 Pp. ab (PLANT −3,1, PER −2,7, ORG +2,6, FOOD −2,5, TIME +1,9, MEDIA +1,8, VEHI +1,8); **BIO und INST kommen nicht vor** (Split: 14 bzw. 34 Entitäten). Rohdaten: Scratchpad `valstats/valstats.out`.
- Verwendung: Tab. `tab:datennutzung` und ein Satz in 4.4 (Checkpoint-Auswahl: Abgleich mit Verlustminimum 1/9 bzw. 0/6; 200er-Stichprobe ohne BIO/INST, sonst ±3 Pp.) und 6.2 (Untertraining 12/15; Auswahlkriterium F1 statt Verlust).

### Datenbefunde Daten-Agent 2 (Sa 7.11.) – Anhang-Provenienz (für Block 7a und 8)
- **Tab. A.8 (`tab:app-beispiele`), 0-basierte Listenindizes in `test_predictions.json` (Reihenfolge in allen Läufen identisch), jeweils Seed 42 des genannten Laufs:** Filthy Friends (PER→ORG) 327, large und 27B-QLoRA; perch (ANIM→FOOD) 396, large und 27B-QLoRA; University of Georgia Campus Arboretum → „University of Georgia Campus" (ORG) 59, **nur large s42** (zusätzlich „Arboretum (LOC)" vorhergesagt, in der Tabelle nicht genannt); burro → „wild burro" 383, 27B-QLoRA (alle Seeds); copra ausgelassen 35, large s42 (Vorhersage leer; s123/s456 sagen „coconuts (FOOD)"); whiptails ausgelassen 4, 27B-QLoRA (alle Seeds; zusätzlich Zusatz „fence lizard"); desert cottontail (Zusatz) 5, large (alle Seeds) und 27B-QLoRA s42/s456; Propranolol (MED) 10805, 27B-Zero-Shot (`unknown_types: 1`); Easter Monday (TME) 2405, 27B-Zero-Shot. Beispiel für „Grenz- und Typfehler": Index 59, large **Seed 123**: Gold „University of Georgia Campus Arboretum" (ORG), Vorhersage „University of Georgia" (ORG) + „Campus Arboretum" (LOC). Skript/Ausgabe: Scratchpad `find_examples.py/.out`.
- **Zero-Shot-Läufe:** Dateien vom 1.8.2026 (mtime), Commit 137bffd (sauber), 27B auf H100 NVL (serv-3310), 4B/0.8B auf A100-SXM4-80GB (serv-3336); Slurm-Job-IDs nicht in den Laufdateien (Kandidaten in `configs/training_monitor.yaml`: 27B 3221500/3226656, 4B 3221499/3226655, 0.8B 3221498/3226654/3230044; Slurm-CSV beginnt erst am 7.8.) → im Anhang Datum und Commit nennen, Job-ID als „nicht protokolliert". Greedy-Kontrolllauf: 29.7.2026, Commit 4679bf1 (dirty), A100.
- **Nachinferenz 1 024:** Jobs 3673515–3673523, 28.10.2026 16:35–21:49 UTC, Commit ff91342 (sauber), H100 80GB HBM3 (serv-3340), mem 160/64 GB; `matches_source_gpu: false` für 27B (Quelle H100 NVL) und 4B (Quelle A100). **CrossNER 1 024:** Jobs 3673475/3673476/3673477, Start 28.10. 14:30/15:03/21:32 UTC, Ende 28.10. 23:27 / 29.10. 00:06 / 29.10. 06:33 UTC; Aggregation 3673478 am 29.10. 09:13 CET (CPU); Partition H100, 160 GB; Erstversuch 3672273–3672276 mit „Frozen repository mismatch" abgebrochen.
- **AN-08:** Validierungs-Batchgrößen 32/16/8/4/1 und Partitionen (H100/H100/H100/A100-80GB/H100-Trails) in Configs, `config_resolved.yaml` und Slurm-CSV bestätigt; keine Abweichung. Eval-Jobs liefen auf CPU-Partition `batch`.
- **Datenlizenzen:** MultiNERD: `license: cc-by-nc-sa-4.0`; „restricted to only non-commercial research purposes under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License" (HF-Datensatzkarte README, Abruf 7.11.2026). CrossNER: MIT License, Copyright (c) 2020 Zihan Liu (GitHub-Repo LICENSE).


### Codex-Neubewertung (Bericht 17, Sa 7.11. abends) – Übernahme R1–R8 und `references.bib`
Codex' zweite Session hat parallel `references.bib` bereinigt (16:31, unbestätigt) und Bericht 17 abgelegt. Befund: HEAD b4eddcf zitiert `geng2023grammar` (02, 04) und `qwen2026card397b` (01, 02); beide Schlüssel stammen aus Codex' Arbeitsbaum-Fassung der bib, nicht aus der committeten. Der Schreib-Agent von Block 4a hat sie dort vorgefunden; ein sauberer Checkout von Tag `betreuerentwurf-2026-11-08` hätte daher zwei undefinierte Zitate. Entscheidung: Codex' bib-Fassung wird übernommen und committet (Abweichung von der Regel „nur Claude bearbeitet die bib", dem Nutzer in der Änderungsübersicht ausgewiesen).
| Prüfung der bib-Änderung | Quelle (Abruf 7.11.2026) | Ergebnis |
|---|---|---|
| `geng2023grammar` (neu, @inproceedings) | ACL Anthology 2023.emnlp-main.674: Geng, Josifoski, Peyrard, West; EMNLP 2023, S. 10932–10952, DOI 10.18653/v1/2023.emnlp-main.674 | bestätigt |
| `zhao2023survey` @misc → @article | Crossref 10.1007/s11704-026-60308-3: „A Survey of Large Language Models", Front. Comput. Sci. 20(12), online 9.5.2026 | bestätigt; `number = {12}` ergänzt (eid 2012627 nicht in Crossref, belassen) |
| `wei2023chatie` v1 → v2 | arXiv 2302.10205: v2 vom 27.5.2024, Titel „ChatIE: …" | bestätigt |
| `nakayama2018seqeval`, `mangrulkar2022peft` als @online mit Version | PyPI seqeval 1.2.2 (24.10.2020); PEFT 0.19.1 = Version in 04:546 | bestätigt |
| `qwen2026card08b/4b/27b` URLs auf README der Revision aus Tab. A.2; `qwen2026card397b` neu | Revisions-Hashes identisch mit `tab:app-revisionen`-Zeilen (anhang.tex 163–165) | bestätigt |
| 16 unbenutzte Einträge entfernt | keiner in Kapiteln/Frontmatter zitiert (comm cited/bibkeys: 0 fehlend; 97 Einträge) | übernommen |
| `eprinttype` statt `archivePrefix` | biblatex-Feldname | übernommen |
Befunde R1–R8 aus Bericht 17 (Codex-Bewertung ≈ 1,3 / 8,5):
| ID | Befund | Umsetzung |
|---|---|---|
| R1 | 3.5 Lücke 2 „nicht gemeinsam erfasst" zu absolut | → Schreib-Agent Block 2b: „nur ausschnittsweise erfasst" |
| R2 | 3.4 „Generativ arbeiten dabei nur Zhan und Obeidat" (Lu ebenfalls generativ) | → Schreib-Agent Block 2b: Zhan, Obeidat, Lu generativ; Dukić Klassifikationskopf; Obeidat-Ausgabe „je Eingabetoken ein Tag" am Quelltext prüfen |
| R3 | 6.2 „Verzerrung begrenzt" unbelegt; „unter 2 GiB Inferenzspeicher" ohne Messbezug | 06: „Einfluss nicht auszuschließen, Ausmaß nicht bestimmbar"; „gemessener allokierter Inferenzspeicher unter 2 GiB" (selbst, 7.11.) |
| R4 | „vorab festgelegt" (Hauptvergleiche) | Formulierung in 4.5.1 lautet bereits „in der Experimentmatrix angelegt" (Argumentationsgang-Commit); kein Protokolldatum behauptet → keine Änderung; Nutzerpunkt: bestätigen, dass die Matrix vor den Läufen stand |
| R5 | Tab. A.8 Satz 59: unter strikter IOB2 (`metrics.py`: `mode="strict", scheme=IOB2`) sind I-LOC ohne B-LOC keine Spans; „Arboretum (LOC)" / „Campus Arboretum (LOC)" existieren als Vorhersage nicht | Zeile 59 (Seed 42): Vorhersage „University of Georgia Campus (ORG)", Klasse Grenzfehler; Seed-123-Zeile entfernt; neue Zeile 311 (Seed 42, large): Referenz „Robert Fulton Birthplace (LOC)", Vorhersage „Robert Fulton (PER)" (Tags B-PER I-PER I-LOC), Klasse Grenz- und Typfehler; geprüft an `test_predictions.json` beider Seeds (selbst, 7.11.) |
| R6 | 5.7.2 „die Indizes aber nicht" legt Indexfeld fest; Dequantisierungsanteil nicht gemessen; Latenz nachgenerierter Sätze nicht isoliert | → Schreib-Agent Block 5b (R6a/R6b) |
| R7 | Seitenziel ≈ 80 Haupttext (K2 14–15, K3 12–13, K4 21–22, K5 21–22) | Phase B unverändert (Ziel ≤ 76–80) |
| R8 | Begleitnotiz: „Tabelle 4.x", 40 GPU-h unbelegt, „unabhängige Evaluation" zu stark | Bericht 16: Tabellenname statt Nummer; Frage 1 ohne GPU-Stunden, „zusätzliche Evidenz", Hinweis, dass der Testsplit derselbe bliebe (selbst, 7.11.) |

### Prüfung Block 4a – Kapitel 2 (Prüf-Agent, abgeschlossen Sa 7.11. abends)
Sieben Befunde; Umsetzung: (1/2) `qwen2026card397b` (02:517, 01:55) und `geng2023grammar` (02:547, 04:849) stammen aus Codex' Arbeitsbaum-bib → bib committet (siehe oben); 397B-Aussage („397 Mrd. Parameter, 17 Mrd. aktiv") entspricht dem Modellnamen Qwen3.5-397B-A17B der Modellkarte. (3) Quellentausch Willard → Geng in A3: arXiv-Abstract 2305.13971 (Abruf 7.11.) belegt „guaranteeing that the output follows a given structure" → Paraphrase gedeckt; beibehalten, `willard2023outlines` bleibt entfernt. (4) Dettmers-Vergleichsmaßstab präzisiert: „volles Fine-Tuning bei GLUE und Super-NaturalInstructions, LoRA bei MMLU" (dettmers2023qlora.txt, Abschn. 4/App. C.1). (5) „304 Mio." nicht im DeBERTaV3-Paper (Tab. 9 nur base/small/xsmall) → zusätzlich `he2021debertav3cardlarge` zitiert (Modellkarte: 304M Backbone). (6) Wortlaut A1 belassen. (7) Satz je Zeile wiederhergestellt (02:545). Zahlen 32/32 belegt; Overlap 0 gegen Quelltexte; Labels/Refs/Umgebungen unverändert; terms.sh 0.

---

# Arbeitsliste (Stand 7.11., Zeilen gegen HEAD 3c49499; verbindlich sind die Anker)

## Anhang A: Konsolidierte Befundliste (Explore-Agent, 7.11., Zeilen gegen HEAD 3c49499 verifiziert)

Kürzel: 01–06 = `thesis/chapters/0x_*.tex`, A = `anhang.tex`, FM = `frontmatter/*.tex`, M = `main.tex`, BIB = `references.bib`. „bestätigt" = am Volltext geprüft (08a V / 12); „plausibel" = nur Abstract/Metadaten.

### Frontmatter
| ID | Stelle | Beginnt mit | Kat. | Zielzustand | Volltext? | Status |
|---|---|---|---|---|---|---|
| F13 | eigenstaendigkeitserklaerung.tex 18–27 | „% Entwurf der KI-Angabe…" / „Claude Code (Anthropic PBC…" | Formalia | Alle genutzten KI-Werkzeuge mit Produkt, Hersteller, Version, Nutzungsart: Claude Code, Codex (OpenAI), KI-Prüfläufe; Kommentar entfernen | nein | offen |
| FM-01 | deckblatt.tex 32–33 | „Fine-Tuning für Named Entity Recognition:…" | Formalia | Titel bleibt (Nutzerentscheidung); Umbrüche prüfen | nein | entschieden |
| FM-02 | deckblatt.tex 43–44 | „Fakultät IV," / „Fachgebiet Quality \& Usability" | Formalia | Komma streichen; Bezeichnung prüfen | nein | offen |
| FM-03 | deckblatt.tex (kein Datum); erkl 37 | „\noindent Berlin, den \rule…" | Formalia | Abgabedatum aufs Deckblatt; Datum/Unterschrift bei Abgabe | nein | offen (Nutzer) |
| FM-04 | deckblatt.tex 10–11 | „\includegraphics…dfki.jpg" | Formalia | Logo-Freigabe klären | nein | offen (Nutzer) |
| FM-05 | abkuerzungsverzeichnis.tex 23 | „\acro{BIO}{Beginning, Inside, Outside…" | Terminologie | Eintrag streichen (kollidiert mit MultiNERD-Typ BIO; IOB Z. 34 reicht) | nein | offen |
| FM-06 | abkuerzungsverzeichnis.tex 8–12 | Kommentarregel | Formalia | HBM3 (04:511–520, 1282, 05:565) und ELECTRA (02:398, 04:186) aufnehmen | nein | offen |
| FM-07 | abstract.tex 1 (+ M:187) | „\chapter*{Abstract}" | Formalia | „Abstract" bleibt (Annahme) | nein | Annahme |
| FM-08 | abstract.tex 4–42 | — | Struktur | Nach Kap. 5/6 abgleichen; Zahlen stimmen derzeit | nein | offen (niedrig) |

### Kapitel 1
| ID | Stelle | Beginnt mit | Kat. | Zielzustand | Volltext? | Status |
|---|---|---|---|---|---|---|
| F06 (K1) | 109–113 | „Erstens verwenden die Studien allgemeine Datensätze… nicht herangezogen." | Überdeutung | Absolute Aussage streichen (UniversalNER Tab. 2 auf MultiNERD, Tan/MultiCoNER II 33 Typen, Zhan/OntoNotes 18 Typen); Lücke als belegte Kombination; nur zusammen mit K3/K4 ändern | nein (bestätigt) | offen |
| K1-01 | 109–110 ↔ 03:651, 679 | „Für die vorliegende Arbeit sind dabei vier Aspekte relevant…" | Struktur | Zählung vereinheitlichen | nein | offen |
| F11 (K1) | 170–173 | „Für die LLMs wird ein indexbasiertes JSON-Ausgabeformat nach \textcite{semin…}" | fehlende Quelle | Ramponi et al. 2025 als Primärquelle; Semin als Systematisierung; Schema/Nachverarbeitung als eigene Umsetzung | nein (bestätigt) | offen |
| K1-02 | 158–160 | „…fragt die dritte Forschungsfrage (FF3)…" | Überdeutung | „unter den gemessenen Bedingungen" | nein | offen |
| K1-03 | 192–199 ↔ 216–218; 81–82 ↔ 135–137 ↔ 196–198; 175–182 ↔ 209–210; 131–134 ↔ 235–241; 138–143 ↔ 243–252 | Doppelungen | Redundanz | Jede Aussage einmal; ≈5 Seiten | nein | offen |
| K1-04 | 45–49 | „Die vorliegende Arbeit entstand am Deutschen Forschungszentrum…" | Struktur | Kürzen/als Nebensatz | nein | offen |
| K1-05 | 79–80 | „Parameter-Efficient Fine-Tuning (PEFT) umfasst Verfahren, die stattdessen nur eine kleine Zahl zusätzlicher Parameter…" | Sachfehler (gering) | „zusätzlicher oder ausgewählter vorhandener Parameter" (wie 02:555–557) | nein | offen |
| K1-06 | 129, 222 | „…im Zero-Shot-Einsatz…" | Terminologie | „Zero-Shot-Regime" | nein | offen |
| K1-07 | 229–233; 265–267; 270 | Doppelung „weitere/andere Modellfamilien"; Füllsatz; „parameter-effizientes" | Sprache | auflösen; streichen; „parametereffizient" | nein | offen |

### Kapitel 2 (A1–A8 aus Bericht 11)
| ID | Stelle | Beginnt mit | Kat. | Zielzustand | Volltext? | Status |
|---|---|---|---|---|---|---|
| A1 | 278–285 | „…ein Decoder sieht beim Erzeugen eines Tokens nur die Vergangenheit und muss die Entitäten als Text ausgeben." | Sachfehler/Überdeutung | Eingabesatz steht vollständig im Prompt; Unterschied = tokenweise bidirektionale Repräsentation; „muss" → „gibt in der hier verwendeten Pipeline … aus" (Dukić mit Klassifikationskopf) | nein | offen |
| A2 | 447–449 | „Die Inferenzzeit wächst damit vor allem mit der Länge der Ausgabe, nicht der Eingabe…" | Überdeutung | Prefill und kontextabhängige Decoding-Kosten nennen | nein | offen |
| A3 = F08 | 534–536 | „…Constrained Decoding binden die Erzeugung an eine formale Grammatik und schließen solche Fehler aus" | Überdeutung | „kann syntaktische und im Schema kodierte Fehler verhindern; semantische Richtigkeit nicht garantiert" | nein (bestätigt) | offen |
| A4 | 580–581 | „…sodass bei einer Änderung von r die Lernrate nicht neu eingestellt werden muss." | Überdeutung | „kann den Bedarf an erneuter Abstimmung verringern" (LoRA 4.1) | nein (bestätigt) | offen |
| A5 | 639–641; 628–630; 646–648 | „Paged Optimizers … verhindern so Abbrüche…" | Überdeutung | „fangen Speicherspitzen ab, verringern das Risiko"; Speicher- vs. Rechenpräzision trennen; Leistungsaussage auf QLoRA-Tests begrenzen | nein (bestätigt) | offen |
| A6 | 921–926 | „…deren Verhältnis den Effizienzgewinn parameter-effizienter Verfahren beschreibt" | Überdeutung | „quantifiziert die Reduktion trainierbarer Parameter; Speicher, Durchsatz, Zeit sind gesonderte Größen" | nein | offen |
| A7 | 679–680 | „Diese Beziehungen gelten über mehrere Größenordnungen, wie Abbildung … zeigt." | Überdeutung (gering) | „…wie die schematische Darstellung nach Kaplan et al. veranschaulicht" | nein | offen |
| A8 | 411–412 | „DeBERTa-v3 wurde mit denselben Einstellungen und Daten wie DeBERTa vortrainiert" | Sachfehler? | Am DeBERTaV3-Paper prüfen (V1- oder 160-GB-V2-Daten) | ja | offen |
| K2-01 | figures/ner_grundprinzip.tex 24–25 | „(t_1, t_2, PER) … Barack Obama" | Konsistenz | Indizes statt Tokens passend zu 02:83–86 | nein | offen |
| K2-02 | 263–264 (Caption Abb. 2.2) | „…der Decoder ergänzt eine maskierte Self-Attention…" | Sachfehler (gering) | „ersetzt die Self-Attention durch eine maskierte und ergänzt…" | nein | offen |
| K2-03 | 770–775 | „Zusätzlich stellen die Autoren eine kleine, manuell geprüfte englische Teststichprobe…" | Struktur | „1 000 Sätze" hier nennen; Typkürzel beim ersten Auftreten | nein | offen |
| K2-04 | 305–306; 917–918 | Kontrastmuster | Sprache | umformulieren | nein | offen |
| K2-05 | 22; 925 | „parameter-effizientes" | Sprache | „parametereffizient" | nein | offen |
| K2-06 | 40–62; 369–372; 374–380; 656–730; 855–864 | Kürzungskandidaten | Struktur | siehe Anhang B | nein | offen |
| K2-07 | 53–54; 83–86; 405–410; 451–452 | Dimension als Beispiel; Tripel als eigene Formalisierung; Residual-Embeddings; kein Hardware-Determinismus | Sachfehler (gering) | präzisieren | nein | optional |

### Kapitel 3
| ID | Stelle | Beginnt mit | Kat. | Zielzustand | Volltext? | Status |
|---|---|---|---|---|---|---|
| F01 | 365–368; Tab. 625–627; 679–683 | „…keine der Arbeiten misst Trainings- oder Inferenzaufwand" / GoLLIE-Zeile „-- & --" / „…ohne Ressourcenmessung" | Quellenfehler | GoLLIE Anh. D.3 Tab. 10: Training über drei Größen (7B 1×A100 44,5 h, 1,57 kg CO₂; 13B 79,5 h; 34B 94,6 h), keine Inferenz; UniversalNER misst nichts | bestätigt | offen |
| F02 | 351–353; 78–81; 657–659 | „Welche Fassung des Datensatzes und welcher Split…, nennt die Arbeit nicht" | Quellenfehler | Anh. D.2 „standard splits", Tab. 9: 32 908 MultiNERD-Testbeispiele (= 2 × 16 454, Beziehung ungeklärt) → eingeschränkte Vergleichbarkeit | bestätigt | offen |
| F03 | 402–404; Tab. 628–630 | „Die Werte von BERT-MRC entsprechen… ob die Vergleichswerte neu gemessen wurden, sagt die Arbeit nicht." | Quellenfehler | Anh. G: LLaMA3.1-8B und BERT-MRC für Kostenvergleich selbst trainiert (L40); Herkunft weiterer Baselines separat | bestätigt | teilweise |
| F04 | 150–154; 521–526 | „…GPT-4 erreicht, sobald mehr als ein Dutzend…" / „…auf denselben LLM-erzeugten Daten" | Quellenfehler | Zwei Vergleiche trennen (GPT auf 2 Datensätzen, verkleinerte Testsets, Token-Makro-F₁, „remains unclear"/„competes"; UniNER auf 4 Testsets, Entity-Mikro-F₁); „dieselben Daten" nur GLiNER/UniNER; NuNER = C4 mit GPT-3.5 | bestätigt | offen |
| F05 | 263–264 (+ Tab. 613–614; 267–269; 361–364) | „Die frühen LLM-Ansätze für NER verzichten auf jedes aufgabenspezifische Training." | Quellenfehler | „verzichten auf NER-spezifische Anpassung der Gewichte; GPT-NER nutzt in der stärksten Retrieval-Variante einen feinjustierten NER-Tagger" (Abschn. 3.2.2) | bestätigt | offen |
| F06 (K3) | 651–659; 165–168; 480–482; 78–81; ↔ 324–329 | „…ein allgemeiner, feingranularer Datensatz wie MultiNERD … bislang nicht herangezogen" | Überdeutung | Lücke als belegte Kombination; OntoNotes/MultiCoNER II inhaltlich abgrenzen; GLiNER-MultiNERD prüfen | bestätigt / GLiNER plausibel | offen |
| F10 | 473–477; 483–489 | „Drei der vier Arbeiten konvergieren…" / „Die Ausgabeformate reproduzieren den Eingabetext…" | Quellenfehler | Synthese trennen: generative Ausgabe, Klassifikationskopf (Dukić), gemessene Ressourcen (nur Obeidat misst Inferenz) | bestätigt | offen |
| F11 (K3) | 211–212; 240–245 | „Eine Indexing-Variante mit nummerierten Wörtern … nicht evaluiert." | fehlende Quelle | Ramponi 2025 als Ursprung | bestätigt | offen |
| K3-01 | 527–531 | „\textcite{obeidat2025llm} halten die Zahl trainierbarer Parameter konstant…" | Quellenfehler | „auf ähnliche Größenordnung abgestimmt" (341/353 Mio. vs. 336/435 Mio.) | bestätigt | offen |
| K3-02 | 249–250 (+ 05:1071) | „…Ausgabelänge hängt nur von der Zahl der Entitäten ab…" | Sachfehler | auch Spantext und Serialisierung | nein | offen |
| K3-03 | 15–23 | Auswahlkriterien | Methodentransparenz | Rechercheabsatz (Nutzerangaben) | nein | offen |
| K3-04 | 200–202; Tab. 610–612 | „…UniversalNER-Benchmark mit den Typen Person, Organisation und Ort." | Verwechslungsgefahr | Fußnote: UNER (Mayhew 2024) ≠ UniversalNER (Zhou) | ja | offen |
| K3-05 | 449–452 (+ 04:783–787) | „…vermuten, dass vollständiges Fine-Tuning den Abstand schließen könnte." | fehlender Gegenbeleg | Zhan Anh. D: LoRA 0,14–0,48 F₁ vor Full-FT | bestätigt | offen |
| K3-06 | fehlt (3.2) | — | fehlende Quelle | GNER (Ding et al. 2024) einordnen | ja | offen |
| K3-07 | 733–737 | „Diese Abweichungen entsprechen in Art und Umfang denen der direkten Vergleiche…" | Überdeutung | belegen oder abschwächen | ggf. | offen |
| K3-08 | Tab. 607–609 | NuNER-Zeile Ressourcen „--" | strittig | „Annotationskosten < 500 $; keine Latenz-/Speichermessung" | ja | strittig |
| K3-09 | 422 | „Für 5 biomedizinische…" | Formalia | „fünf" | nein | offen |
| K3-10 | 3.1 43–175, 3.2 176–378 | keine Unterabschnitte | Struktur | Unterabschnitte; Synthese nach Vergleichsfragen; ≈11 Seiten | nein | offen |
| K3-11 | 539–556 ↔ 04:201–213/270–281; 721–737 ↔ 04:126–152/Tab. `tab:abweichungen` | Doppelungen zu Kap. 4 | Redundanz | nur Verweise | nein | offen |
| K3-12 | 58–63; 564–565 | überlanger Satz; Kontrast | Sprache | teilen | nein | offen |
| K3-13 | „Zero-Shot-Setting" 313, 336, 447, 511; „-Einsatz" 146; „Silberannotation" 70; „Silber-Referenz" 668; „geprompte" 51, 158, 199, 257, 361 (+05:981) | Terminologie | einheitlich „Zero-Shot-Regime/-Szenario", „Silberreferenz", „per Prompting genutzte LLMs" | nein | offen |

### Kapitel 4
| ID | Stelle | Beginnt mit | Kat. | Zielzustand | Volltext? | Status |
|---|---|---|---|---|---|---|
| K4-01 | 323–326 | „Der Vergleich bleibt intern fair, da alle Modelle mit denselben Trainingssignalen…" | Überdeutung | gemeinsame Trainingssätze/Referenz; Zero-Shot ohne Training; Verlust, Anpassungsumfang, Auswahl verschieden; Silberreferenz nicht neutral | nein | offen |
| K4-02 | 1077–1079 | „…ihr Verhältnis quantifiziert den Effizienzvorteil des Adaptertrainings" | Überdeutung | „Reduktion aktualisierter Parameter"; kein Zeitvorteil (266 h vs. 1,2 h) | nein | offen |
| F06 (K4) | 300–304; 308–311 | „…bislang keine Vergleichswerte auf diesen Splits publiziert" | Überdeutung | nach F02 einschränken; OntoNotes-Abgrenzung anpassen | nein | offen |
| F12a | 969–990 | „…gepaarten Satz-Bootstrap…" | fehlende Quelle | Efron & Tibshirani 1993; Koehn 2004; Lakens nur für „keine Äquivalenz" | ja | offen |
| F12b | 195–196 | „Beide Modelle stehen unter der MIT-Lizenz… \cite{wolf2020transformers}" | Quellenfehler | Modellkarte/LICENSE; wolf2020 nur Bibliothek | ja | offen |
| F12c | 669–673 | „Entsprechend dem Standardverhalten der verwendeten TRL-Version…" | fehlende Quelle | TRL-1.8.0-Doku/Code | ja | offen |
| F11 (K4) | 372–374; 701–704; 715–718 | Index-Format nach Semin | fehlende Quelle | Ramponi als Primärquelle; eigene Umsetzung kennzeichnen; Dagdelen nur für JSON-Parsebarkeit | bestätigt | offen |
| K4-03 | 217–219; 204–206 | Fußnote mit Modellkarten-URLs; Apache-2.0 | Formalia | `qwen2026card*` zitieren | ja | offen |
| K4-04 | 315–318 | „…kleinen, manuell annotierten Testdatensatz…" | Begründung fehlt | Grund nennen | ggf. | offen |
| K4-05 | 641–643 | „…Sampling … (Temperatur 1,0, top-k = 20…" | fehlende Quelle | Herkunft (Modellkarte oder ad hoc) | ja | offen |
| K4-06 | 35; 281; 1179; 1112–1113 | dekorative Belege (keraghel; kaplan/hoffmann; tedeschi/tam) | fehlende Quelle | eigene Festlegungen kennzeichnen | nein | offen |
| K4-07 | 783–787 | „LoRA wird gewählt, weil…" | fehlende Quelle | Zhan Anh. D als Stütze | bestätigt | offen |
| K4-08 | 547–550 | „Tabelle … ordnet jedem Lauf Konfiguration, Commit, Revisionen…" | Formalia | Verweis auf Tab. `tab:app-revisionen`/A.3 | nein | offen |
| K4-09 | 105–106; 37/68/126/154; 1028; 897–910; 1277–1281; 363 | Doppel-Label; `\subsection*`; „1 Minuten"; 110-W-Satz; unverständlich; Tempus | Formalia/Sprache | beheben | nein | offen |
| K4-10 | 784; 946 | „token-level"; „Entity-Level" | Terminologie | „auf Tokenebene/Entitätsebene" | nein | offen |
| K4-11 | Tab. `tab:qwen-architektur` 248–268; 489–494; 536–539; 1029–1046; 1330–1335 | Anhangskandidaten | Struktur | ≈3–4 Seiten in den Anhang | nein | offen |
| K4-12 | 680ff. | Checkpoint-Auswahl auf 200 Sätzen | optional | Verteilungsdiagnose | nein | optional |

### Kapitel 5
| ID | Stelle | Beginnt mit | Kat. | Zielzustand | Status |
|---|---|---|---|---|---|
| K5-Ü1 | 390–393 | „Ein Greedy-Kontrolllauf des 0.8B-Modells … das Sampling hebt den Wert…" | Überdeutung | Greedy + alter Prompt 0,5; Sampling + Glossar-Prompt 6,2; Beiträge nicht getrennt | offen |
| K5-Ü2 | 378–379; 1052–1055 | „…das Modell hat also falsch gezählt" | Überdeutung | „Inkonsistenz zwischen Text- und Indexfeld" | offen |
| K5-Ü3 = F07a | 1029–1038 | „Das stützt die … Erwartung, dass Indizes … den Einbruch vermeiden…" | Überdeutung | „Format praktisch nutzbar; Vorteile nicht isoliert untersucht"; Zhan nur Einordnung | offen |
| K5-Ü4 = F07b | 1070–1077 | „Für die Effizienz hat das Format den erhofften Effekt nicht gebracht…" | Überdeutung + Sachfehler | Schluss streichen; „Inferenz bleibt erheblich langsamer"; Länge hängt auch von `text` ab | offen |
| K5-Ü5 | 638–641 | „Die Pareto-Front besteht … allein aus den beiden Encodern…" | Überdeutung | „bei 256 Ausgabetokens" | offen |
| K5-Ü6 | 831–834 | „…weil dort die vielen nicht abgebildeten Fachtypen als Zusätze zählen…" | Überdeutung | „spricht dafür, dass…" | offen |
| K5-Ü7 | 1063–1069 | „Ein Constrained Decoding … würde hier wenig helfen…" | Überdeutung | als Vermutung | offen |
| K5-Ü8 | 1108–1113 | Emergenz-Bezug | Überdeutung | vorsichtiger | optional |
| K5-Ü9 | 776–780; 1006–1010 | „…viele davon sind Spans wie…" / „Die häufigsten Zusätze…" | Überdeutung | quantifizieren oder abschwächen | offen |
| K5-Ü10 | 527–529 (+127–132, 241–247, 264–267, 589–593, 616–620, 630–633, 776–780, 841) | Interpretation in 5.1–5.6 | Struktur | nach 5.7 | offen |
| K5-Ü11 | 334/753 vs. 399; 243–244; 235–236; 409–410 | Mittel vs. Einzellauf; „vorn"; „Niveau" | gering | kennzeichnen | offen |
| K5-Z1 | 850–851 | „…1,7 bzw. 19,0 GiB" | Zahlenfehler | 18,6 GiB | offen |
| K5-Z2 | 870–872 | „…verliert das LLM beim Transfer dann 12,4 Punkte, der Encoder 14,5" | Zahlenfehler | Referenz 1 024 ins Skript; Wert aus summary.json; STATUS.md mitkorrigieren | offen |
| K5-Z3 | 406–408 | „0,5 bis 0,7" | Rundung | „0,5 bis 0,6" | offen |
| K5-Z4 | 609–612 | „1,04" | Rundung | „1,03 bis 1,04" | offen |
| K5-Z5 | 589–590 | „etwa das Dreißigfache" | Rundung | „knapp das Dreißigfache" (27,9) | offen |
| K5-Z6 | 1213–1215 | „vier bis fünf Punkten Breite" | Rundung | „3,8 bis 5,2" | offen |
| K5-Z7 | 1088–1091 | „Fünf- bis Sechsfache" | Rundung | „5,6- bzw. 6,4-Fache" | offen |
| K5-Z8 | 6–7; 53–55; 108, 170/1202, 925, 927/968/1210, 957, 1212, 1106 | Rundungskonvention | Konsistenz | „Differenzen mit Intervall zweistellig"; 925/957 angleichen | offen |
| K5-Z9 | 756–757 | „…Encoders auf denselben Sätzen je nach Seed bei 96,2 bis 96,5" | Zahl unbelegt | Teilmengenwert aus Vorhersagen berechnen (Skript) | offen |
| K5-Z10 | 129; 262–263; 412–413; 587 | 30,1 %; 4 190/5 354; fünf Kontrollsätze; 70,8 h | nicht aus summary.json | gegen Rohartefakte bestätigen | niedrig |
| K5-S1 | 514–537 ↔ 1082–1125 | 5.3 ↔ 5.7.3 | Redundanz | 5.7.3 ohne Zahlen | offen |
| K5-S2 | 573–577, 601–608, 626–628 ↔ 1130–1138, 1154–1156 | 5.4 ↔ 5.7.4 | Redundanz | 5.7.4 entzahlen | offen |
| K5-S3 | 5.7.1 ab 924 | — | Struktur | Synthesetabelle vier Bedingungen; drei Kernaussagen | offen |
| K5-S4 | 189–192, 333–339, 396–420, 752–758, 841–865, 949–953, 1042–1046, 1139–1148, 1209–1214 | Trunkierung | Redundanz | bündeln (Inventar b) | offen |
| K5-S5 | 941–949; 1025/1042; 558–559, 578–588 | Erstens…; „ausnahmslos/ausschließlich"; 27B-Chronik | Sprache | auflösen | offen |
| K5-T1 | 312–327 | Tab. 5.3 Caption | Formalia | kürzen, Verweis Tab. 4.7 | offen |
| K5-T2 | 553–570; tab-effizienz.tex | Tab. 5.6 | Formalia | Encoder-Minuten; „(Herstellerangabe)"; Caption kürzen | offen |
| K5-T3 | 647–719; 269–300 | Abb. 5.3; Abb. 5.1 | Formalia | Überlagerung lösen, GPU kodieren, Direktbeschriftung; Gruppentrenner | offen |
| K5-T4 | 880–908 | Tab. 5.9/5.10 | Formalia | „95-%-Intervall" einheitlich; Einheit | offen |
| K5-Sp | 941/945/949; 1042–1043, 1049; 1029, 1059, 1101, 1150; 1049/1060; > 50 W: 765, 842, 1092, 1006, 341, 578, 1063, 229, 248, 1029; 181, 227, 359, 745, 763; 524–526 | Sprache | glätten | offen |
| K5-N1 (Gutachter-Agent) | 1036–1038 | „…hatten die Wortindex-Variante nicht evaluiert; die vorliegenden Ergebnisse zeigen…" | Quellenfehler (F11) | Ramponi et al. evaluieren eine Indexvariante; „hier erstmals in Kombination mit QLoRA und strikter Entitätsmetrik auf MultiNERD" | offen |
| K5-N2 | 1059–1062 | „Das zeigt, dass Formatvorgabe und Typinventar ohne Training zum Flaschenhals werden" | Überdeutung | „unter den gewählten Sampling-Bedingungen"; Einzellauf; Verweis 6.2 | offen |
| K5-N3 | 1175–1176; (06:93–95) | „…erfolgreiche Änderung des Typinventars per Prompt wurde nicht nachgewiesen." | irreführend | „wurde nicht untersucht" (Prompt unverändert, Typen abgebildet) | offen |
| K5-N4 | 1241–1243 | „Die Ergebnisse der einzelnen Vergleiche sind explorativ" | zu vorsichtig | vorab festgelegte Hauptvergleiche (large–27B bei 256; Größenschritte) vs. explorative übrige | offen |
| K5-N5 | 1194–1196 | „Die ebenfalls geringen Abstände historischer Vorläufe…" | unbelegt | Vorlaufwerte in A.1 nennen oder Satz streichen | offen |
| K5-N6 | 931–934; 1084–1087; 984–989 (+03:349); 1101–1104; 1114–1120 | Obeidat „Precision-getrieben"; Keraghel „base/large tauschen Plätze"; GoLLIE „acht der 15 Typen"; „entspricht"; „zeigen" | Quellenaussagen ungeprüft/zu stark | am Quelltext prüfen (`obeidat2025llm.txt`, `keraghel2024recent.txt`, `sainz2024gollie.txt`); Kategorienfehler Keraghel („zwischen Datensätzen"); „ähnelt"; „nahelegen" | offen |
| K5-N7 | 1012–1014 | „Da Wikipedia- und Wikinews-Texte mit hoher Wahrscheinlichkeit Teil des Vortrainings sind" | unbelegt | Vortrainingsdaten nicht offengelegt; Webkorpora enthalten Wikipedia (`commoncrawl2024dataset`) | offen |
| K5-N8 | 5.2 (bei 753/951 ↔ 04:632); 5.5 (763–782); 5.7.4; 5.7.5 | 65–69 abgeschnittene Modellantworten vs. 55 Referenzantworten; Fehlerbeispiele; CrossNER-Umkehr; Relevanzschwelle | fehlende Einordnung | Halbsatz „Modellantworten, nicht Referenzantworten"; „illustrative, nicht zufällige Auswahl"; Absatz „Alternative Erklärungen" (Vortrainingswissen, Typabbildung, Gold- vs. Silberreferenz, nur LLM neu inferiert); SESOI-Satz | offen |

### Kapitel 6
| ID | Stelle | Beginnt mit | Kat. | Zielzustand | Status |
|---|---|---|---|---|---|
| F09 | 247–250 (↔ 03:208–210, 05:1066–1069) | „…Wortindizes nicht erzeugen, sondern aus den Logits ableiten, etwa LogitMatch…" | Quellenfehler | „spanbeschränktes Matching-Verfahren wie LogitMatch könnte die freie Generierung numerischer Wortindizes ersetzen" (Semin Abschn. 4) | offen |
| K6-01 | 205–275 | Ausblick, 6 Blöcke | Struktur | drei Prioritäten | offen |
| K6-02 | 26–38 ↔ 05:924–937…; 59–75 ↔ 05:1082–1100…; 77–98 ↔ 05:1165–1184; 137–144 ↔ 05:1000–1010; 188–200 ↔ 05:1238–1246 | Redundanz | Antwortabsatz je FF; ≈4 Seiten | teilweise |
| K6-03 | 107; 137; 188 | `\paragraph` ohne FF-Zuordnung | Struktur | „Betroffen ist FF…" ergänzen | teilweise |
| K6-04 | 15 `\paragraph`; 18–21 | Roadmap | Sprache | kürzen | offen |
| K6-05 | 222 | „…1 000 Sätze…" | Struktur | in 2.8 einführen | offen |
| K6-N1 (Gutachter-Agent) | 40–42 | „Die typbezogenen Ergebnisse zeigen unterschiedliche Stärken." | zu weitgehend | „beobachtete Punktwertunterschiede ohne eigene Intervalle" | offen |
| K6-N2 | 78–79 | „Für ein festes Typinventar und Daten ähnlich MultiNERD…" | unbestimmt | „für MultiNERD und vergleichbar annotierte Wikipedia-Texte" | offen |
| K6-N3 | 93–95 | „…keinen Nachweis, dass neue Entitätstypen allein durch Promptänderung…" | irreführend | „wurde nicht untersucht" | offen |
| K6-N4 | 107–120 | „Anschließend blieben die Hyperparameter…" | unvollständig | Neufassung Testnutzung (Block 6, Vorschlag Gutachter-Agent) + Verweis `tab:datennutzung` | offen |
| K6-N5 | 221–227; 260–264 | „Eine manuell geprüfte Teststichprobe, etwa im Umfang der 1 000 Sätze…" / „Letzteres vermeidet die Dequantisierung…" | zu weitgehend / unbelegt | „für die häufigen, verrauschten Typen ANIM, PLANT, FOOD"; `\cite{dettmers2023qlora}` oder streichen | offen |

### Anhang
| ID | Stelle | Kat. | Zielzustand | Status |
|---|---|---|---|---|
| AN-01 | 17–20; 88–90; 04:89–92 | Formalia | Repo-URL + Tag + Zugangsregel für Ergebnisdaten | offen |
| AN-02 | vor 188 | Formalia | `\clearpage` vor A.2 | offen |
| AN-03 | 209 | Formalia | Kurztitel Tab. `tab:app-seeds` | offen |
| AN-04 | 220–223; 232–236; 245–247 | Formalia | „256 Ausgabetokens"; Spaltenköpfe | offen |
| AN-05 | fehlt | Struktur | Seed-Werte der 1 024er-Läufe | offen |
| AN-06 | 253–303; 203–204, 256; 05:781–782 | Formalia | Lauf/Seed/Satz-ID; Klasse „Grenz- und Typfehler"; `\cite{tedeschi2022multinerd}` | offen |
| AN-07 | 55–95 | Formalia | Job-IDs/Daten 1 024er-Läufe; Zero-Shot-Laufdaten | offen |
| AN-08 | 146; 162 | Formalia | Validierungs-Batchgrößen; „H100-Trails" | offen |

### Literatur/bib
| ID | Zielzustand | Status |
|---|---|---|
| L-01 Ramponi 2025 | NAACL 2025, 2025.naacl-long.34 | offen |
| L-02 Mayhew 2024 | Universal NER, NAACL 2024 | offen |
| L-03 Bootstrap | Efron & Tibshirani 1993; Koehn 2004; (Dror 2018) | offen |
| L-04 GNER | Ding et al. 2024, 2024.findings-acl.206 | offen |
| L-05/06/07 | DeBERTa-v3-Modellkarte; TRL-1.8.0-Doku; `qwen2026card*` zitieren | offen |
| L-08 Preprints | 20 arXiv-`@misc` drucken ohne „arXiv/Preprint" trotz eprint-Feldern → Kennzeichnung erzwingen und im PDF prüfen | offen |
| L-09 ICLR | eine booktitle-Variante | offen |
| L-10–L-12 | Jurafsky-note deutsch; Lafferty-URL; Manning 2008 DOI | offen |
| L-13 | Zhan/Semin eprint-Versionen; EMNLP-Annahme prüfen | offen |
| L-14 | Keraghel-Fassung | niedrig |
| L-15 | `commoncrawlblog2024inside` entfernen | niedrig |
| L-16/17 | van Rijsbergen Kap. 7; Goldberg-Seite | offen |
| L-18 | eingeschränkt geprüft: lakens, hripcsak, goldberg, schuster, manning1999, manning2008, qwen2026qwen35blog → Paraphrasen abgleichen | offen |
| L-19 | Tabellen-/Abschnittsangabe bei tragenden Literaturzahlen | optional |

### main.tex
| ID | Stelle | Zielzustand | Status |
|---|---|---|---|
| M-01 | 172–173; 193 | römische Seitenzahlen sichtbar (`\pagestyle{plain}` nach Deckblatt) | offen |
| M-02 | 187 | an FM-07 anpassen (bleibt) | — |
| M-03 | Build | neu bauen, Sichtprüfung | offen |

### Bereits erledigt (nicht mehr anfassen)
Erklärungswortlaut; Abgrenzung zu Semin; α/r in Abb. 2.3; RoFormer/CoNLL-Belege; Abb. 2.3/2.4 neu; SD-Summenregel entfernt; Bootstrap Hauptinstrument; Testnutzung explorativ; 27B-Epochen 2→3; Abweichungen QLoRA dokumentiert; Code-Stichproben Kap. 4 stimmen; Bericht-03-Punkte A1–A7, B8–B12, C15, C17, C18, D22, D23, D26–D28; „obere Abschätzung"; Kontamination; CEL; GB→GiB (außer K5-Z1); Pareto-Bedingung 5.7.4; drei Empfehlungsszenarien; CrossNER-Bootstrap; Log-Achsen Abb. 5.3; Präfixe 594/604; Listing A.1; „Unbekannter Typ"; Verzeichnisse/Paginierung; ICLR-/CoNLL-DOI; Makro-Gleichstand; ≈240/250 Zahlen stimmen.

## Anhang B: Kürzungsliste (Explore-Agent, 7.11.; Fließtextwörter ohne Floats; S = streichen, Z = zusammenziehen, A = Anhang, V = Zahlenwiederholung durch Verweis)

### Kapitel 1 (1 772 W; P1 ≈385 W)
| # | Zeilen | Beginn | Art | ~W | Erhalten bleibt | Prio |
|---|---|---|---|---|---|---|
| 1 | 8–16 | „Dieses Kapitel erläutert Motivation…" | S | 50 | – | P1 |
| 2 | 109–122 | „…vier Aspekte relevant…" | Z | 55 | ein Aufzählungssatz + Verweis 3.5 | P1 |
| 3 | 131–134 | „Das Zero-Shot-Regime beschreibt…" | S | 30 | geht in 235–241 auf | P1 |
| 4 | 135–137 | „QLoRA trainiert Low-Rank-Adapter…" | S | 25 | geht in Titelabsatz auf | P1 |
| 5 | 209–210 | „CrossNER wird ausschließlich…" | S | 15 | 175–182 bleibt | P1 |
| 6 | 191–201 + 216–228 | Titelabweichung / DeBERTa | Z | 90 | Titelabweichung; linearer Kopf; drei Qwen-Größen; „Nicht untersucht…" 229–233 | P1 |
| 7 | 265–303 | Aufbau | Z | 120 | ein Satz je Kapitel; Anhanginhalt | P1 |
| 8 | 45–49 | DFKI | Z | 20 | Nebensatz | P2 |
| 9 | 76–82 | PEFT-Motivation | Z | 25 | ein Satz | P2 |
| 10 | 243–252 | Metrikdefinition | Z | 45 | „Nicht erfasst werden…" 253–260 | P2 |

### Kapitel 2 (5 063 W; P1 ≈1 325 W)
| # | Zeilen | Beginn | Art | ~W | Erhalten bleibt | Prio |
|---|---|---|---|---|---|---|
| 1 | 9–26 | Kapitelvorschau | Z | 75 | 11–13 | P1 |
| 2 | 40–62 | NLP-Geschichte | Z | 190 | 31–36, 63–75 | P1 |
| 3 | 163–183 | CRF/BiLSTM/CNN-Historie | Z | 100 | ein Satz CRF; 184–185 | P1 |
| 4 | 222–226 | Original-Hyperparameter | S | 35 | 218–221 | P1 |
| 5 | 305–310 | Quantisierung vs. Attention | S | 45 | 311–313 | P1 |
| 6 | 326–339 | WordPiece/RoBERTa/SentencePiece | Z | 70 | DeBERTa-v3 SentencePiece 128k; 340–346 | P1 |
| 7 | 369–372 | BERT auf CoNLL | S | 35 | – | P1 |
| 8 | 374–380 | RoBERTa | S | 50 | – | P1 |
| 9 | 405–410, 416–422 | GDES; Keraghel-Modellwahl | S | 75 | 382–404, 411–415 | P1 |
| 10 | 494–510 | OSI/Release-Historie | Z | 100 | „offene Gewichte, Apache 2.0"; Namen | P1 |
| 11 | 511–521 | Architektur (52-W-Satz) | V | 55 | Thinking-Voreinstellung → 4.4.3 | P1 |
| 12 | 544–562 | Full-FT/PEFT-Taxonomie | Z | 50 | Speichergründe; PEFT-Definition | P1 |
| 13 | 649–651 | LoRA-Varianten | S | 20 | – | P1 |
| 14 | 659–715 (+ Gl. powerlaw; Abb. 2.4 bleibt) | Skalierung | Z | 230 | je ein Satz Kaplan/Chinchilla; Grenzen 717–729 | P1 |
| 15 | 742–758 | CoNLL/OntoNotes | Z | 60 | IOB1/strikt (ein Satz); Tab. 2.2 | P1 |
| 16 | 778–780 | Kontamination | S | 25 | Verweis 3.5 | P1 |
| 17 | 850–867 + Gl. fbeta | F_β | Z | 60 | F₁ = harmonisches Mittel | P1 |
| 18 | 921–930 | Ressourcen | S | 50 | Halbsatz → `sec:effizienzmetriken` | P1 |
| 19–26 | 118–126/133–134; 246/252–253; 465–469; 582–584/594–602; 875–881; 883–898; 909–919; 794–796 | Reserve | Z/S/V | 270 | siehe Agent | P2 |

### Kapitel 3 (4 487 W; P1 ≈1 075 W)
| # | Zeilen | Beginn | Art | ~W | Erhalten bleibt | Prio |
|---|---|---|---|---|---|---|
| 1 | 12–41 | Kapitelvorschau | Z | 90 | 15–23 Auswahlkriterien | P1 |
| 2 | 46–53 | drei Leitfragen | S | 50 | – | P1 |
| 3 | 55–74 | MultiNERD-Beschreibung | Z | 90 | mBERT; 1 000 Sätze 0,85; 75–81 | P1 |
| 4 | 179–188 | Abschnittsvorschau | Z | 50 | Zweiteilung in einem Satz | P1 |
| 5 | 240–261 | Einordnung des eigenen Formats (69-W-Satz) | Z | 80 | drei Vorteile kurz; Einschränkung | P1 |
| 6 | 263–285 | GPT-NER/Xie Mechanik | Z | 95 | 90,91 vs. 92,8; Modellaufrufe | P1 |
| 7 | 300–306 | CodeIE | S | 50 | Tab. 3.1-Zeile; Zitat 364 | P1 |
| 8 | 308–329 | UniversalNER (2 Absätze) | Z | 75 | 93,73 vs. 91,25; Aufbereitung | P1 |
| 9 | 331–353 | GoLLIE (2 Absätze) | Z | 70 | QLoRA-Richtlinien; 76,0; CrossNER 55,5–67,8 | P1 |
| 10 | 480–496 | Lückenliste | V | 90 | geht in 3.5 auf; 473–479 bleibt | P1 |
| 11 | 501–507 | Skalierungsgesetze | S | 45 | – | P1 |
| 12 | 509–537 | Größeneffekte (55-W-Satz) | Z | 100 | Kernbefunde | P1 |
| 13 | 539–556 | Folgerung für die Arbeit | Z | 80 | Größenbereich | P1 |
| 14 | 721–743 | Vergleichsbedingungen | V | 110 | ein Satz + Verweis 4.7 | P1 |
| 15–27 | 83–99; 101–120; 122–142; 144–154; 169–174; 199–202/208–213; 227–238; 287–298; 383–391; 405–418; 438–440/453–457/466–471; 651–677; 697–719 | Reserve | Z/S | 660 | siehe Agent | P2 |

### Kapitel 4 (6 347 W; P1 ≈1 360 W, davon ≈570 W + Tab. `tab:qwen-architektur` in den Anhang)
| # | Zeilen | Beginn | Art | ~W | Erhalten bleibt | Prio |
|---|---|---|---|---|---|---|
| 1 | 12–24 | Kapitelvorschau | Z | 40 | Verweis A.1 | P1 |
| 2 | 78–92 | Vorläufe (53-W-Satz) | A | 55 | Vorläufe existierten; Thinking aus; keine Auswertung | P1 |
| 3 | 148–151 | Wiederholung 1.3 | S | 30 | 152 | P1 |
| 4 | 182–191 | DeBERTa-Architektur | S | 45 | Parameter, Lizenz, Begründung | P1 |
| 5 | 203–206, 214–222 | Qwen-Tradition/Checkpoints | Z | 45 | 207–213; „Basismodell" | P1 |
| 6 | 235–246 + Tab. `tab:qwen-architektur` | multimodale Details | A | 35 | nur Textmodell; Parameter; 3:1 | P1 |
| 7 | 270–281 | Größenklassen-Subsection | Z | 45 | in Caption Tab. 4.1 | P1 |
| 8 | 313–326 | Annotationsqualität | Z | 55 | Silber; „intern fair" präzisiert | P1 |
| 9 | 350–353 | Tokenizer | S | 30 | – | P1 |
| 10 | 454–464 | Benchmark-Detail | Z/A | 40 | Testnutzung; explorativ | P1 |
| 11 | 469–478 | Pipeline-Module | A | 50 | Halbsatz | P1 |
| 12 | 486–494 | Softwareversionen | A | 45 | „einheitliche Versionen" | P1 |
| 13 | 527–546 | Protokollfelder | A | 100 | Kohortenunterschied; keine Pins; Sampling bei Wiederaufnahme | P1 |
| 14 | 576–581 | Early-Stopping-Epochen | V | 45 | ein Satz | P1 |
| 15 | 654–659 | Stichprobe | Z | 30 | ein Satz | P1 |
| 16 | 687–694 | 200 Sätze | Z | 40 | ≈18 min vs. 25 h; nicht zufällig | P1 |
| 17 | 712–718 | Index-Adressierung | S | 45 | Verweis 3.2 | P1 |
| 18 | 783–790 | LoRA-Begründung | Z | 45 | r, α, Dropout, Heuristik | P1 |
| 19 | 820–823 | Adaptergrößen | A | 30 | FP32 in Tab. `tab:abweichungen` | P1 |
| 20 | 838–843 | Lernrate/Epochen | S | 40 | – | P1 |
| 21 | 897–910 | Validierung (86-W-Satz) | Z/A | 50 | vier Regeln | P1 |
| 22 | 951–956 | seqeval | V | 30 | ein Satz | P1 |
| 23 | 1029–1043 | 27B-Chronologie | A | 110 | Dauer = Zuteilungen; Knoten geteilt; 1044–1046 | P1 |
| 24 | 1105–1119 | Fehlerklassen (63-W-Satz) | Z | 50 | vier Klassen; Asymmetrie | P1 |
| 25 | 1176–1190 | Parameteranzahl | Z | 60 | deskriptiv | P1 |
| 26 | 1203–1218 | Auswahlregel | Z | 60 | Kernargument; 1222–1226 | P1 |
| 27 | 1292–1303 | CrossNER-Bootstrap | Z | 50 | Schichtung; dieselbe Ziehung | P1 |
| 28 | 1330–1335 | Vorstudie (61-W-Satz) | A | 60 | 1336–1344 | P1 |
| 29–42 | 53–59; 167–172; 308–311; 339–342; 391–397; 562–567; 619–621/633; 667–668; Listing 4.1; 991–998; 1061–1064; 1081–1085; 1088–1097; 1324–1329 | Reserve | — | 365 | siehe Agent | P2 |

### Kapitel 5 (6 424 W; P1 ≈1 215 W)
| # | Zeilen | Beginn | Art | ~W | Erhalten bleibt | Prio |
|---|---|---|---|---|---|---|
| 1 | 29–46 | Kapitelvorschau | Z | 75 | FF-Zuordnung; „5.1–5.6 berichten, 5.7 interpretiert" | P1 |
| 2 | 48–74 | Konventionen | V/A | 120 | Skala, Rundung, Mittel ± SD, Zero-Shot Einzellauf, Intervall-Lesart in einem Satz; Skriptsatz → A.1 | P1 |
| 3 | 189–192 | Vorgriff Ausgabelimit | S | 30 | – | P1 |
| 4 | 215–240 | Typwerte (52/59-W-Sätze) | V | 130 | Richtung je Typ; 241–247 | P1 |
| 5 | 341–357 | Nullbefunde (66-W-Satz) | Z | 110 | Einzelfälle; 45/12/4 | P1 |
| 6 | 361–380 | Formtreue-Zähler | V | 90 | Muster je Größe; erfundene Typen | P1 |
| 7 | 390–393 | Greedy-Kontrolllauf | S | 35 | – (steht in 4:643–646; K5-Ü1 dort korrigieren) | P1 |
| 8 | 578–588 | 27B-Chronik | V | 75 | ein Satz + Verweis `sec:effizienzmetriken` | P1 |
| 9 | 924–929, 954–971 | Hauptbefund-Wiederholung | V | 110 | Interpretation; 0,18 [0,02; 0,33] → 403–408 | P1 |
| 10 | 1024–1028, 1042–1046 | Parse-Fehler/Limit | S | 75 | 1029–1041 | P1 |
| 11 | 1084–1100 | Größe (88-W-Satz) | V | 120 | Stabilität; abnehmende Zugewinne | P1 |
| 12 | 1130–1138 | FF3-Zahlen | S | 65 | Verweis 5.4/Abb. 5.3 | P1 |
| 13 | 1154–1163 | Trainingszeiten | S | 70 | → 6.2 | P1 |
| 14 | 1198–1220 | Bootstrap-Zahlen | V | 110 | Zählaussagen | P1 |
| 15–43 | 79–81/86–89; 106–116; 118–132; 134–147; 149–166; 173–179; 248–253; 255–267; Abb. 5.1; 397–420; 514–537; 542–551; 596–633; 725–731; 753–759; 763–782; 784–791; 796–800; 821–839/847–851; 864–869/876–878; Tab. 5.10; 973–983; 1000–1018; 1051–1058; 1121–1125; 1150–1152; 1182–1184; 1189–1196; 1237–1247 | Reserve | — | 1 335 | bevorzugt #19, #27, #33 | P2 |

### Kapitel 6 (1 735 W; P1 ≈580 W)
| # | Zeilen | Beginn | Art | ~W | Erhalten bleibt | Prio |
|---|---|---|---|---|---|---|
| 1 | 9–21 | Aufbau-Wiederholung | Z | 55 | je ein Satz | P1 |
| 2 | 27–38 | FF1 | Z | 30 | 1,34 [0,99; 1,68] → 0,18 [0,02; 0,33]; base keine Rangfolge | P1 |
| 3 | 40–46 | Typen | Z | 30 | Makro-Intervall enthält 0 | P1 |
| 4 | 48–57 | Formtreue | Z | 40 | 0,4 % nur Limit; Zero-Shot 6,2–69,4; kein isolierter Effekt | P1 |
| 5 | 60–75 | FF2 | Z | 50 | Intervalle ohne 0, kein Skalierungsgesetz; ≈210; Speicher; Trainingszeit = Aufwand | P1 |
| 6 | 78–98 | FF3 | Z | 50 | Antwort mit Bedingungen; 87,1 vs. 85,0; keine allgemeine Überlegenheit | P1 |
| 7 | 157–166 | Abweichungen | V | 35 | Stichworte + Tab. `tab:abweichungen` | P1 |
| 8 | 205–275 | Ausblick | Z | 290 | drei Prioritäten (siehe Block 6) | P1 |
| 9–12 | 103–105; 116–120; 125–131; 189–200 | Reserve | — | 95 | — | P2 |

### Redundanz-Inventar (bleibende Stelle fett)
- (a) Hauptbefund 1,34/0,18: **05:168–172**, **05:403–408**, **06:27–38**; streichen/verweisen: 05:924–929, 954–971, 1142–1144, 1201–1203, 1209–1213, 06:80–82.
- (b) Ausgabelimit: **04:627–633**, **04:1195–1226**, **05:333–339**, **05:397–420**, **05:841–875**, **06:122–135**; kürzen: 05:189–192, 753–759, 924–929/949–958, 1024–1026/1042–1046, 1252–1254, 06:30–34/48–50/211–219; streichen 04:1324–1329.
- (c) Modellwahl/QLoRA: **01:191–201(+216–228)**, **04:180–196**, **04:207–213**; streichen 01:135–137, 02:416–422; kürzen 02:505–521, 03:118–120, 03:540–548, 04:53–59, 04:783–787.
- (d) CrossNER PER/ORG/LOC: **01:175–182**, **03:369–378**, **04:1244–1250**; streichen 01:209–210, 03:738–743; Verweise 02:794–796, 04:286–288, 05:796–800.
- (e) Empfehlungen: **05:1165–1176**, **06:78–98**; streichen 04:1187–1190; 05:1150–1152 → 1165; 05:1182–1184 → 6.2; 06:257–259 → Anschlussfrage 2.
- (f) „kein isolierter Effekt": bleibt 1.3 (235–241), 4.1 (145–152), Tab. `tab:abweichungen`; sonst Halbsatz (01:134, 03:730, 04:657/834/1309, 05:132/415/875/971/977, 06:38/55/65/166).
- (g) Skalierungsgesetze: bleibt 2.7 (717–727); kürzen 03:501–507/549–553, 04:1180–1186, 05:535–537/1121–1125, 06:63–65.
- (h) Silberreferenz: 2.8 Definition, 5.7.1 Interpretation, 6.2; kürzen 03:58–74, 04:313–326.
- (i) Zero-Shot-Abstand 25–87: 5.1.1 und 6.1; kürzen 05:973–977, 1104–1107.
- (j) Latenzfaktor 210: 5.4, 5.7.2, 6.1; kürzen 05:1136–1137.
- (k) 27B-Chronologie: Anhang; Konsequenz `sec:effizienzmetriken`; Tab. 5.6; 6.1; kürzen 05:578–588, 1097–1098, 1154–1159.
- (l) Kontamination: 3.5 und 6.2; kürzen 01:255–257, 02:778–780, 05:1011–1018.
- (m) Testnutzung: 4.3.3 und 6.2; kürzen 05:142–147.
- (n) Literaturzahlen (Keraghel 65,17/90,98; Obeidat 2–8 / 40–220): Kap. 3 + eine Nennung in 5.7.
- (o) Kleindoppelungen: Epochen 04:579–581 ↔ 05:121–123, 04:685–686 ↔ 05:158–160; Greedy-Kontrolllauf 04:643–646 ↔ 05:390–393; 27B-Epochen 04:444–447/841–843/06:109–111.
- (p) Constrained Decoding: bleibt 4.4.4 (770–778); kürzen 01:172–173, 02:534–536, 03:260–261, 04:170–172.
- (q) Index-Format: bleibt 3.2; kürzen 04:712–718.

### Sprachinventar
- Sätze > 50 W: K1 192–199; K2 294–301, 511–517; K3 132–137, 246–252, 422–428, 513–518; K4 78–84, 207–213, 628–633, 897–907, 1004–1010, 1105–1115, 1213–1218, 1330–1335; K5 215–220, 229–234, 248–253, 341–347, 361–366, 578–584, 765–769, 1029–1035, 1063–1069, 1092–1100; K6 222–227.
- „nicht X, sondern Y": 02:305; 03:66, 356, 469, 564–565; 04:350, 909; 05:1049; 06:248. Variante ohne „sondern": 01:181; 02:447, 616, 726, 917; 03:377, 743; 04:277, 1058, 671, 1223; 05:1043, 1064; 06:38, 75.
- Erstens/Zweitens/…: 01:111–120; 02:384/392, 718/722; 03:157/161, 356/361/365, 540/549; 04:294–305, 1158/1162/1165; 05:941/945/949.
- Ankündigungsformeln: 01:109; 02:717; 03:15, 156, 246, 355, 539, 697; 04:293, 1157.

### Summentabelle (kalibrierte Zielseiten)
| Kap. | Ist-S | P1-Ersparnis [W] | Ziel-S (kalibriert) | Richtwert |
|---|---|---|---|---|
| 1 | 6 | 385 | 4,7 | 5 |
| 2 | 17 | 1 325 | 12,6 | 13 |
| 3 | 14 | 1 075 | 10,9 | 11 |
| 4 | 24 | 1 360 (≈570 Anhang) | 19,5 | 20 |
| 5 | 23 | 1 215 | 19,6 | 20 |
| 6 | 6 | 580 | 4,0 | 4 |
| Σ | 90 | 5 940 | ≈71–76 | 73 |
Anhang wächst um ≈2–2,5 Seiten.

## Anhang C: Quellenlage (Explore-Agent, 7.11.)

- biblatex: `backend=biber, style=apa, natbib=true, doi=true, url=true, eprint=true` (main:20–27), biblatex 3.22a / biblatex-apa 9.20; `\let\cite\parencite`; 179× `\cite`, 118× `\textcite`, 360 Verweise auf 79 Schlüssel; **keine Postnotes (keine Seitenangaben)**; `\printbibliography[heading=bibintoc]`, nur zitierte Einträge gedruckt.
- Preprints: 20 `@misc` mit `eprint`+`archivePrefix`+`primaryClass`; im PDF erscheint nur „Autor (Jahr). Titel. URL" ohne „arXiv:" oder „Preprint" → Darstellungsproblem (L-08). Zitiert: keraghel (19×), zhan (19×, eprint ohne Version), semin (17×, ohne Version), liu2019roberta, zhao2023survey, kaplan2020scaling, qwen2024qwen25, wang2024parameter, ashok2023promptner, qwen2025qwen3, wei2023chatie, willard2023outlines, wang2023instructuie (ohne Version); nicht zitiert: lin2021survey, ansar2024survey, openai2023gpt4, guo2023evaluating, minaee2024large, chen2025survey, qwen2026qwen35omni.
- Weitere `@misc/@online` ohne eprint: Software (mangrulkar2022peft, vonwerra2020trl, nakayama2018seqeval), Datensätze (weischedel2013ontonotes, ldc2024penntreebank), Web (qwen2026qwen35blog, osi2024osaid, qwen2026card08b/4b/27b, commoncrawl2024dataset, commoncrawlblog2024inside, babelscape2023multinerdcard); hoffmann2022training als NeurIPS mit arXiv-note (korrekt).
- Text ↔ bib: 0 fehlende Schlüssel; 22 unbenutzte Einträge (nicht gedruckt).
- Wörtliche Zitate: keine (alle Anführungszeichen sind Beispiele/Daten/Kategorien); englische Passagen nur in Tab. `tab:app-beispiele` (MultiNERD-Sätze, **ohne `\cite{tedeschi2022multinerd}`**), Prompt-Listings, Abkürzungen; 8-Wort-Overlap mit 74 Quelltexten: keiner.
- Eigene Gegenprobe am Quelltext (Abweichungen zu Berichten): NuNER nennt „costing less than 500$" (stützt 08 gegen 12); GPT-NER-Tagger in Abschn. 3.2.2; Keraghel: „few-shot" steht nicht in der Quelle (nur „custom prompts … included multiple examples", dort „GPT-4"), und auf WNUT2017 liegt GPT-4o (44,73) nahezu gleichauf mit DeBERTa-v3-large (44,81) → 03:112–114 „auf allen Datensätzen deutlich hinter" abschwächen, 01:94–96/03:106/Tab. 03:603/05:978–981 „Few-Shot" → „mit Beispielentitäten im Prompt"; Semin: LogitMatch-Gewinne nur bei tokenisierter Eingabe; PaDeLLM-Faktoren 1,76–10,22 gelten für das eigene Parallel-Decoding; vaswani-Eintrag in 08a hat einen arXiv-Tippfehler (Matrix, nicht Thesis); Goldberg 02:58–60 braucht Seitenangabe; Dagdelen belegt JSON-Parsebarkeit, nicht die Nummerierung.
- Eingeschränkt geprüft (L-18): goldberg2017neural (02:60), manning1999foundations (01:28, 02:33, 02:44), manning2008introduction (01:246, 02:836, 02:904, 04:947, 04:958), qwen2026qwen35blog (01:71, 01:129, 01:225, 02:507, 04:204, 04:229), schuster2012japanese (02:328), lakens2017equivalence (04:980), hripcsak2005agreement (02:914).
- Fassungen: Zhan „Findings EMNLP 2026 angenommen" (03:417–418, bib-note) im v3-Text nicht belegt → prüfen/entfernen; keraghel v3 Preprint (kein Fehler); hu/he/he jetzt ICLR (erledigt).
