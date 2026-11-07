# Zweitgutachten Kapitel 3, Kapitel 4 und Literaturapparat (7.11.2026, Stand 5b40126)

Quellen per Volltext verifiziert: GoLLIE (Anhang D.2/D.3/E), Zhan (Anhang D/G, Tab. 2/13), NuNER (Abschn. 6), UniversalNER (Tab. 2), Semin (Abschn. 3.2/4), GPT-NER. Code-Stichproben Kap. 4 (Ausgabelimit, Sampling-Parameter am Tag, LoRA-Hyperparameter, 200 Validierungssätze, Double-Quant-Fehlbenennung, Parser, seqeval, Encoder-Einstellungen, Bootstrap, CrossNER-Mapping, Antwortlängen) stimmen alle.

## Noten
- **Kapitel 3 7/10** – fünf bestätigte Quellenfehler/Überzeichnungen (F01, F02, F04, F05, F10), Forschungslücke widerspricht 03:324–329 (F06), Primärquelle Wortindex fehlt (F11), keine Suchstrategie, keine Unterabschnitte in 3.1/3.2, Redundanz zu Kap. 4.
- **Kapitel 4 8,5/10** – Reproduzierbarkeit sehr hoch; früher offene Begründungen stehen; offen: 04:323–326 „intern fair“, 04:1077–1079 Parameteranteil = Effizienzvorteil, Bootstrap ohne Methodenbeleg, Lizenz/TRL-Belege (F12), 04:302–304 zu absolut, B8 (manuell geprüfter Teildatensatz), Formalia, 24 Seiten.
- **Literaturapparat 7,5/10** – fehlende Belege: Ramponi et al. 2025 (Wortindex), Mayhew et al. 2024 (Universal NER = UNER bei Semin, nicht Zhou et al.s UniversalNER), Bootstrap-Methodenquelle (Efron & Tibshirani 1993; Koehn 2004; Dror et al. 2018); Qwen-Modellkarten in der Bib, aber in 04:217–219 nur als URL; keine Quelle für DeBERTa-Lizenz; schwache Quelle `commoncrawlblog2024inside`.

## Kapitel 3 – Befunde (alle verifiziert)
1. F01 GoLLIE Ressourcen (03:365–368, Tab. 03:625–627, 03:679–683): Anhang D.3 Tab. 10 nennt 7B 1×A100, 44,5 h, 1,57 kg CO₂eq; 13B 79,5 h; 34B 94,6 h; keine Inferenzlatenz. UniversalNER: wirklich keine Messung.
2. F02 GoLLIE Split (03:351–353): Anhang D.2 „standard train, dev, test splits“; Tab. 9: 32 908 MultiNERD-Testbeispiele (= 2 × 16 454; prüfen).
3. F05 GPT-NER (03:263–264): Retrieval-Variante nutzt feinjustierten NER-Tagger (Abschn. 4.2.2).
4. F04 NuNER (03:150–154, 03:521–526): zwei getrennte Vergleiche (GPT-4 auf 2 Datensätzen, verkleinerte Testsets, Token-Makro-F₁, „remains unclear“; UniNER-Vergleich 4 Datensätze, Entity-Mikro-F₁); NuNER aus C4, UniversalNER aus Pile → „dieselben Daten“ falsch (gilt nur GLiNER/UniNER).
5. F10 Synthese 3.3 (03:473–477, 483–489): nur Obeidat misst Inferenz; Dukić nutzt Klassifikationskopf.
6. F03 teilweise: 03:412–416 korrekt; Tab. 03:628–630 „berichtete Werte“ unvollständig (BERT-MRC selbst trainiert; eigener Kostenvergleich).
7. F06 Forschungslücke (03:651–659, 03:165–168, 03:481–482; 01:111–113; 04:302–304): widerspricht 03:324–329; OntoNotes 18 Typen bei Zhan.
8. Suchstrategie fehlt (03:15–23) – nur Autor, nichts erfinden.
9. F11 Ramponi (03:211–212, 240–245; 04:372–374, 715–718).
10. UNER-Verwechslung möglich (03:200–202, Tab. 03:610–612) – Fußnote + Bib Mayhew et al. 2024.
11. Zhan Anhang D ungenutzt: LoRA 0,14–0,48 F₁ vor Full-FT → stützt 04:783–787, relativiert 03:452.
12. GNER (Ding et al. 2024) fehlt.
13. 03:733–737 „entsprechen in Art und Umfang denen der direkten Vergleiche“ unbelegt.
Formalia: 03:422 „5“ → „fünf“; Unterabschnitte 3.1 (03:43–175) und 3.2 (03:176–378); Redundanz 03:539–556 ↔ 04:201–213/270–281, 03:721–737 ↔ 04:126–152/Tab. 4.9; Satz 03:58–63; Tab. 3.1 Korrekturen; „geprompte“ (Jargon).

## Kapitel 4 – Befunde
1. 04:323–326 „intern fair, da alle Modelle mit denselben Trainingssignalen“ – Zero-Shot ohne Training; Vorschlag im Bericht.
2. 04:1077–1079 Parameteranteil ≠ Effizienzvorteil (266 h vs. 1,2 h).
3. 04:302–304 „keine Vergleichswerte auf diesen Splits“ → nach F02 einschränken.
4. 04:195–196 Lizenz mit `wolf2020transformers` → Modellkarte; 04:669–673 TRL → Versionsdoku.
5. F12 Bootstrap-Beleg (04:969–990).
6. B8 04:315–318 Grund für Nichtnutzung des manuell geprüften Teildatensatzes.
7. 04:641–642 Herkunft der Sampling-Parameter (Modellkarte oder ad hoc).
8. Dekorative Belege 04:35, 04:281, 04:1179, 04:1112–1113.
Formalia: Doppel-Label 04:105–106; `\subsection*` 04:37/68/126/154; 04:1028 „1 Minuten“; 04:897–910 110-Wort-Satz; 04:1278–1279 kryptisch; 04:363 „entfielen“; „token-level“/„Entity-Level“ vs. „auf Entitätsebene“; Anhangskandidaten (Tab. 4.2 Vision/MTP 04:248–268, Softwareversionen 04:489–494, Zero-Shot-Metadaten 04:536–539, 27B-Chronik 04:1029–1046, Vorstudie 04:1330–1335) ≈ 3–4 Seiten.

## Status F01–F13
F01 offen; F02 offen; F03 teilweise; F04 offen; F05 offen; F06 offen (01, 03, 04); F07 offen (Kap. 5); F08 offen (02:534–535); F09 offen (06:247–249); F10 offen; F11 offen; F12 offen; F13 offen. Abweichung zu Audit 08: NuNER nennt keine bezifferten Kosten, „--“ in Tab. 3.1 vertretbar.

## Aufwand
KI-Assistenz ≈ 11 h (F01/F02 0,75; F05 0,25; F04 0,75; F10 0,75; F03 0,25; F06 1,5; F11 0,5; UNER 0,25; Zhan Anh. D 0,5; F12/Belege 0,75; Fairness/Parameteranteil 0,25; Formalia 1; Unterabschnitte 0,75; Redundanz 1; Anhangsverschiebung 1,5; Build 0,5). Autor ≈ 8–9 h (Suchstrategie 1; GNER 1,5; B8 0,5; Sampling-Herkunft 0,25; F13 0,5; GoLLIE-Split 0,5–1; Kürzungsentscheidungen 1,5; Zhan-Fassung 0,25; eigene Sprache Kap. 3/4 2).
