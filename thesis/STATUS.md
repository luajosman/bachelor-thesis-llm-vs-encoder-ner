# Stand der Bachelorarbeit (27.10.2026)

Abgabe: 26.11.2026; Verlängerung um ein bis zwei Wochen mit Attest möglich (Entscheidung bis 6.11.). Abstract nur deutsch. Der angemeldete Titel ("Fine-Tuning für NER: Ein Vergleich von BERT-basierten Modellen und LLMs mit LoRA") ist vermutlich fix; die Abweichung zu DeBERTa/QLoRA/Zero-Shot wird im Text (1.3) erklärt, nicht durch Umbenennen.

## Ablage

- LaTeX: `thesis/` in diesem Repo (seit 27.10.2026). Der frühere Desktop-Ordner "Bachelorarbeit - Luaj (1)" auf dem ersten Mac ist nur noch Sicherung.
- Code und Läufe: `ba-ner/`. Repo auf dem Cluster: `/home/losman/bachelor-thesis-llm-vs-encoder-ner`; daneben die Beweiskopie `.../bachelor-thesis-llm-vs-encoder-ner-1` mit Commit 137bffd.
- Ergebnisse: `/netscratch/losman/ba-ner/results/` auf dem Cluster, Laufordner wie `multinerd/deberta-v3-base-canonical-seed123`, `multinerd/qwen35-27b-qlora-3ep-seed456`, dazu Zero-Shot- und CrossNER-Läufe. Je Lauf: `evaluation_metrics.yaml`, `inference_metrics.yaml`, `results.yaml`, `config_resolved.yaml`, `test_predictions.json`.
- Die in Anhang A.1 zitierten Commits 137bffd und b620041 liegen nur im Cluster-Klon, noch nicht auf GitHub.

## Kapitelstatus

- Kapitel 1 bis 4: überarbeitete Fassungen (Kopfzeilen 16./21.10.2026).
- Kapitel 5 und 6: vollständige Entwürfe vom 27.10.2026 (KI-gestützt, in der Erklärung anzugeben). Struktur ist final: 5.1 Erkennungsleistung (5.1.1 Gesamt, 5.1.2 je Typ), 5.2 Formtreue, 5.3 Modellgröße, 5.4 Effizienz, 5.5 Fehleranalyse, 5.6 CrossNER, 5.7 Diskussion; 6.1 entlang FF1 bis FF3, 6.2 Limitationen, 6.3 Ausblick. Alle Messwerte und richtungsabhängigen Aussagen sind rote `\PH{...}`-Platzhalter (rund 775 in allen Kapiteln). In 5.7.1 stehen zwei Bedingungsabsätze ("Falls das LLM vorn liegt / Falls der Encoder vorn liegt"); der nicht zutreffende ist zu streichen.
- Anhang A.2: Skelette für Seed-Einzelwerte, P/R/F1 je Typ und Fehlerbeispiele.
- Build: sauber, 97 Seiten, keine Overfull-Boxen, keine offenen Verweise.

## Konventionen in Kapitel 5

- Werte in Punkten (0 bis 100); trainierte Konfigurationen als Mittelwert ± Stichprobenstandardabweichung über die Seeds 42, 123, 456; Zero-Shot als Einzellauf (Seed 42, Sampling).
- Ein Unterschied gilt nur als Unterschied, wenn er die Summe der beiden Standardabweichungen übersteigt (definiert in 4.5.1).
- Fehlerklassen (korrekt, Typfehler, Grenzfehler, Grenz- und Typfehler, Auslassung, Zusätze) sind in 4.5.1 definiert.

## Nächste Schritte

1. Ergebnisse holen (vom Mac mit Cluster-Zugang, Hostname einsetzen):
   `rsync -avm --include='*/' --include='*.yaml' --include='*.json' --exclude='*' losman@HOST:/netscratch/losman/ba-ner/results/ ba-ner/results/`
2. Cluster-Commits auf GitHub bringen: `git remote add cluster losman@HOST:bachelor-thesis-llm-vs-encoder-ner && git fetch cluster && git rebase cluster/main && git push origin main` (Rebase nur des Thesis-Commits; die zitierten Hashes bleiben erhalten).
3. Aggregation: vorhandene Skripte nutzen (`ba-ner/src/evaluate/aggregate_seeds.py`, `seed_summary.yaml/csv`), ergänzen um typbezogene Werte, Makro-F1, Fehlerklassen aus `test_predictions.json`, Formtreue-Zähler aus `inference_metrics.yaml`, CrossNER gepoolt und je Domäne; Tabellenfragmente nach `thesis/generated/` schreiben.
4. Platzhalter füllen, drei Abbildungen (pgfplots): `fig:typen`, `fig:groesse`, `fig:tradeoff`; Diskussion auf eine Linie festlegen; 6.1 auf sechs bis acht Sätze je Forschungsfrage kürzen.
5. Formale Pflichtpunkte: Abstract (deutsch), Erklärung mit Datum und KI-Angabe (Produkt, Hersteller, Version, Art der Nutzung), Titelabsatz in 1.3, Abb. 2.1 (doppelte Beschriftung, deutsches Beispiel gegen englische Tab. 2.1), Abb. 2.2 (Beschriftung berührt Linie), "Full-FT" in Tab. 3.1 definieren, Terminologie (LoRA-/QLoRA-Regime; Formtreue/Strukturtreue/Robustheit; Micro/Mikro; BIO/IOB2), Leerseite 4, Abbildungs- und Tabellenverzeichnis, Tabellenstil Kapitel 4 und A.1 auf booktabs, 4.4.3 "unter 0,01" zu "unter 1", Entscheidung zum 1024-Token-Lauf (sonst Satz in 4.6 streichen), Platzhalter für DeBERTa-Revisionen in A.1.
6. Zeitplan: Daten bis 2.11.; Diskussion, Fazit, Abbildungen, Abstract bis 9.11.; Entwurf an Betreuung bis 11.11.; Formalia und Kapitel 1 bis 4 bis 16.11.; Feedback und Gesamtkorrektur bis 23.11.; Abgabe 25.11.
