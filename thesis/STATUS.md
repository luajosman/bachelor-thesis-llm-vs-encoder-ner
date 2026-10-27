# Stand der Bachelorarbeit (27.10.2026, abends)

Abgabe: 26.11.2026; Verlängerung um ein bis zwei Wochen mit Attest möglich (Entscheidung bis 6.11.). Abstract nur deutsch. Der angemeldete Titel ("Fine-Tuning für NER: Ein Vergleich von BERT-basierten Modellen und LLMs mit LoRA") ist vermutlich fix; die Abweichung zu DeBERTa/QLoRA/Zero-Shot wird im Text (1.3) erklärt, nicht durch Umbenennen.

## Ablage

- LaTeX: `thesis/` in diesem Repo. Bauen: `cd thesis && latexmk -pdf main.tex` (103 Seiten, keine Overfull-Boxen, keine offenen Verweise, keine `\PH`-Platzhalter mehr).
- Ergebnisse: `ba-ner/results/` (in Git ignoriert; liegt auf dem MacBook Air und auf dem Cluster unter `/netscratch/losman/ba-ner/results/`). Auf einem neuen Rechner per `bash sync_from_cluster.sh` holen.
- Aggregation: `thesis/scripts/aggregate_results.py` (benötigt PyYAML) liest `ba-ner/results/` und schreibt `thesis/generated/` (Tabellenkörper `tab-*.tex`, Abbildungsdaten `fig-*.dat`, Kennzahlen `summary.json`). Die Kapitel binden die Tabellenkörper per `\input` ein; die Zahlen im Fließtext von Kapitel 5 und 6 stammen aus `summary.json`. Das Skript prüft seine strikte Auswertung gegen die gespeicherten Werte und bricht bei Abweichung ab.
- Cluster: `losman@login1.pegasus.kl.dfki.de`; Repo dort `/home/losman/bachelor-thesis-llm-vs-encoder-ner`; Commit 137bffd (Zero-Shot-Läufe, Anhang A.1) liegt nur in der Beweiskopie `.../bachelor-thesis-llm-vs-encoder-ner-1` auf dem Cluster.

## Kapitelstatus

- Kapitel 1 bis 4: überarbeitete Fassungen. Am 27.10. ergänzt: Unterscheidbarkeitsregel und Fehlerklassen in 4.5.1, Greedy-Kontrolllauf (0,5 Punkte) in 4.4.3, Encoder-Latenz 8,9 ms in 4.5.2, Sensitivitätsanalyse zum Ausgabelimit in 4.6, Kapitelbeschreibung in 1.4.
- Kapitel 5: vollständig mit Messwerten gefüllt (KI-gestützt erstellt, in der Erklärung anzugeben). Struktur: 5.1 Erkennungsleistung (5.1.1 Gesamt, 5.1.2 je Typ, Abb. 5.1), 5.2 Formtreue, 5.3 Modellgröße (Abb. 5.2), 5.4 Effizienz (Abb. 5.3), 5.5 Fehleranalyse, 5.6 CrossNER mit Tab. 5.8 (Ausgabelimit), 5.7 Diskussion.
- Kapitel 6: vollständig gefüllt; 6.1 entlang FF1 bis FF3.
- Anhang: A.1 vollständig (DeBERTa-Revisionen eingetragen), A.2 mit Seed-Einzelwerten, P/R/F1 je Typ und Fehlerbeispielen.

## Zentrale Befunde (Kurzfassung)

- Micro-F1 (Mittel ± SD über Seeds 42/123/456): DeBERTa-v3-base 95,50 ± 0,04; DeBERTa-v3-large 95,82 ± 0,03; Qwen3.5-0.8B/4B/27B QLoRA 93,14 ± 0,10 / 93,97 ± 0,05 / 94,48 ± 0,08; Zero-Shot 6,17 / 34,93 / 69,41 (Einzelläufe).
- Encoder vorn (1,3 Punkte, unterscheidbar); Vorsprung ausschließlich im Recall; rund die Hälfte davon durch das 256-Token-Ausgabelimit (65 abgeschnittene Antworten mit 656 Referenzentitäten). Makro-F1 gleichauf (86,7 vs. 86,9).
- Effizienz: Latenz 8,9/16,5 ms (Encoder) gegenüber 1 876 bis 5 505 ms (QLoRA); Speicher 0,7/1,7 GB gegenüber 0,9 bis 18,7 GB; Training 0,4/1,1 h gegenüber 12,1/62,9/≥50,5 h.
- CrossNER (PER/ORG/LOC): DeBERTa-large 85,0, Qwen-27B 71,7; 1024-Token-Stichprobe hebt Recall des LLM auf ~90.

## Offene Punkte (Pflicht vor Abgabe)

1. Kapitel 5 und 6 kritisch lesen und in eigene Sprache bringen; Richtung der Aussagen ist festgelegt (Encoder vorn, Ausgabelimit als Obergrenzen-Argument).
2. Abstract (deutsch), Erklärung mit Datum und KI-Angabe (Produkt, Hersteller, Version, Art der Nutzung), Titelabsatz in 1.3.
3. Abb. 2.1 (doppelte Beschriftung, deutsches Beispiel gegen englische Tab. 2.1), Abb. 2.2 (Beschriftung berührt Linie), "Full-FT" in Tab. 3.1 definieren.
4. Terminologie (LoRA-/QLoRA-Regime; Formtreue/Strukturtreue/Robustheit; Micro/Mikro; BIO/IOB2), Leerseite 4, Abbildungs- und Tabellenverzeichnis, Tabellenstil Kapitel 4 und A.1 auf booktabs.
5. Anhang A.1: Hinweis ergänzen, dass Commit 137bffd nur in der Beweiskopie auf dem Cluster liegt (oder diese pushen); TODO zu 137bffd auflösen.
6. Literaturangaben stichprobenartig prüfen (GPT-4o-Wert bei Keraghel et al. unverifiziert).
7. Zeitplan: Entwurf an Betreuung bis 11.11.; Formalia und Kapitel 1 bis 4 bis 16.11.; Feedback und Gesamtkorrektur bis 23.11.; Abgabe 25.11.
