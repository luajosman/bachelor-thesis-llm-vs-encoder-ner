# Zahlen-Audit Abstract, Kap. 1, Kap. 5, Kap. 6 gegen generated/summary.json (7.11.2026, Stand 5b40126)

Prüfskript (nur lesend) erzeugte alle Mittel/SD/Seedwerte, Differenzen, Intervalle, Faktoren, Zählungen und Einheitenumrechnungen aus `summary.json`; rund 250 Aussagen verglichen. K5 = `chapters/05_ergebnisse_und_diskussion.tex`, K6 = `chapters/06_fazit_und_ausblick.tex`.

## Abweichungen

| Stelle | Textaussage | Korrekt laut summary.json | Schwere |
|---|---|---|---|
| K5:871–872 | „verliert das LLM beim Transfer dann 12,4 Punkte, der Encoder 14,5“ | Gepoolt: LLM 256-Referenz 98,46 − 87,12 = **11,3** (12,4 ist der Verlust des Domänenmittels, 98,46 − 86,05; Encoder-Pendant 15,5). Der Satz mischt zwei Aggregationen. Hinweis: Der am 30.10. eingesetzte Wert 12,4 entstand aus einer ad hoc berechneten 1 024-Token-Referenz (PER/ORG/LOC: 99,57), die nicht in summary.json steht. Entweder 11,3 (256-Referenz, wie Tab. 5.8) oder die 1 024-Referenz explizit nennen und in `aggregate_results.py` aufnehmen. Auch `STATUS.md` nennt 12,4. | Fehler |
| K5:850–851 | CrossNER-Spitzenspeicher „1,7 bzw. 19,0 GiB“ | 19 045,78 MiB ÷ 1024 = **18,6 GiB** (19,0 ist ÷ 1000). 1,7 ist korrekt. | Fehler (Einheit) |
| K5:406–408 | „Makro-F₁ steigt um 0,5 bis 0,7 Punkte“ | 0,63 / 0,50 / 0,65 → „0,5 bis 0,6“ | Rundung |
| K5:609–612 | Encoder P95/Median „1,04“ | base 1,040, large 1,033 → „1,03 bis 1,04“ | Rundung |
| K5:589–590 | „etwa das Dreißigfache“ | 43 452 s / 1 555 s = 27,9 → „knapp das Dreißigfache“ oder „rund das 28-Fache“ | Formulierung |
| K5:1213–1215 | Makro-Intervallbreiten „vier bis fünf Punkte“ | 3,8 bis 5,2 | Formulierung |
| K5:1088–1091 | „Fünf- bis Sechsfache“ je Größenschritt | 5,6 und 6,4 | sehr gering |
| mehrere | Nachkommastellen 1,34 / 1,3 und 0,18 / 0,2 gemischt | alle Werte korrekt gerundet; Konvention in K5:6–7 präzisieren („Differenzen mit Intervall zweistellig“) | Konsistenz |

256-Token-Hauptwerte und 1 024-Token-Sensitivitätswerte werden nirgends vermischt. Alle übrigen rund 240 Aussagen (Abstract, Kap. 1 Zahlen, 5.1–5.7, Kap. 6, alle Tabellen) stimmen.

## Nicht aus summary.json prüfbar
Validierungs-LOC-Anteil 30,1 % (Tab. 4.3, konsistent); „4 190 der 5 354 TIME-Zusätze mit Ziffer“ (eigene Auszählung); „fünf Kontrollsätze byte-gleich“ (Nachinferenz-Protokoll); 70,8 h des 2-Epochen-Vorlaufs (Kap. 4: 15,5 s/Schritt × 16 410 = 70,65 h); Encoder-Recall „auf denselben Sätzen 96,2–96,5“ (K5:756–757; die Zahlen entsprechen dem Gesamt-Recall der large-Seeds, nicht belegbar als Teilmengenwert); Literaturwerte.

Korrekturaufwand gesamt rund 30 Minuten plus Build.
