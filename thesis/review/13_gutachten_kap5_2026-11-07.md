# Zweitgutachten Kapitel 5 (7.11.2026, Stand 5b40126)

26 Zahlenstichproben: 24 korrekt, 2 Fehler (05:849 „19,0 GiB“ → 18,6; 05:862–864 „12,4“ mischt Domänenmittel und gepoolt → 11,3 gepoolt oder 1 024-Referenz explizit). Statistik nach 6.11. konsistent (Bootstrap als Hauptinstrument, keine SD-Regel, keine Äquivalenz aus Intervallen mit null, 256/1 024 sauber getrennt). PDF S. 62–84 gesichtet.

## Noten
**Kapitel 5 gesamt 8/10** – Darstellung 5.1–5.6: 8; Diskussion 5.7: 7,5 (5.7.1 und 5.7.5 auf 1,0-Niveau; F07 in 5.7.2 offen; 5.7.3/5.7.4 wiederholen 5.3/5.4; keine Synthese der vier Bedingungen).

## Überinterpretationen (offen seit 28.10./6.11.)
1. 05:390–393 „das Sampling hebt den Wert auf 6,2“ – Prompt und Decoding zugleich geändert (04:644–651). Vorschlag: „Kontrolllauf mit Greedy und ursprünglichem Prompt 0,5; Konfiguration mit Sampling und Glossar-Prompt 6,2; Einzelbeiträge nicht getrennt.“
2. 05:378–379, 05:1053–1055 „falsch gezählt“ – nur Inkonsistenz Text/Indizes beobachtet.
3. 05:1029–1037 Formatvorteil aus Studienvergleich (Zhan, Zeichenoffsets 54) – isoliert keinen Formateffekt (F07).
4. 05:1070–1077 „Format hat den erhofften Effizienzeffekt nicht gebracht“ + 05:1071 „Ausgabelänge nur von der Zahl der Entitäten“ (falsch: Feld `text`) – F07.
5. 05:638–641 Pareto-Satz in 5.4 ohne „bei 256 Ausgabetokens“.
6. 05:831–834 „weil … Fachtypen als Zusätze zählen“ – plausibel, domänenweise Zusätze nicht berichtet → „spricht dafür“.
7. 05:1063–1069 Constrained Decoding „würde wenig helfen“ – Vermutung kennzeichnen.
8. 05:1108–1113 Emergenz aus Einzelläufen – optional.
9. 05:776–780, 05:1006–1010 „viele davon“ – unquantifiziert; 50 Zusätze je Klasse sichten oder abschwächen.
10. 05:527–529 Interpretation in Darstellungsabschnitt (gedeckt durch Tab. 5.3).
Kleinere Unschärfen: 65–69 vs. 62–72 (Mittel vs. Einzellauf) ohne Hinweis; 05:242–245 „seltene Typen, bei denen das LLM vorn liegt oder gleichzieht“ (BIO: Encoder vorn) → „überwiegend“; 05:236–237 ORG 98,9 vs. 98,8 → „praktisch gleich“; 05:409–410 „auf dem Niveau der Encoder“ → „nahe bei“.

## Struktur und Redundanz
5.3 ↔ 5.7.3 (05:514–534 ↔ 05:1081–1118; 05:530–534 ≈ 05:1115–1118); 5.4 ↔ 5.7.4 (05:573–577, 601–608, 626–628 ↔ 05:1129–1135, 1152–1154); 5.7 ↔ 6.1 zahlengleich; Trunkierung an elf Stellen (05:189–192, 333–339, 396–420, 752–758, 841–865, 949–974, 1042–1046, 1139–1148, 1210–1214; 06); Darstellung/Interpretation (05:43–46 vs. 05:127–132, 241–247, 264–267, 390–393, 527–537, 589–593, 616–620, 630–633, 638–645, 776–780, 831–834, 841, 862–865); fehlende Synthese-Tabelle der vier Bedingungen (MultiNERD/CrossNER × 256/1 024: 94,48/95,64 bzw. 71,69/87,12 gegenüber 95,82 bzw. 84,99); Erstens/Zweitens/Drittens 05:941–949; 27B-Zeit dreifach (04:1030–1040, Tab. 5.6 Fn. a, 05:578–588); „ausnahmslos/ausschließlich“ zweimal (05:1024–1026, 1042–1046).

## Tabellen/Abbildungen
Tab. 5.3 Caption kürzen (Verweis Tab. 4.7); S. 70 Float-Seite ok; Tab. 5.6 `\scriptsize`, Encoder-Testinferenz 0,05/0,09 h → Minuten, „(Herstellerangabe)“ bei 80 GB, Caption wiederholt 4.5.2; Abb. 5.3 4B/27B überlagert, GPU nicht kodiert → Direktbeschriftung; Abb. 5.1 ohne Gruppentrenner; Tab. 5.9/5.10 Float-Seite S. 77 mit 24 % Leerraum, „95-%-Intervall“ vs. „95 %-KI“, Latenz s vs. ms benennen; S. 83 16 % Leerraum.

## Sprache (Umschreiben in eigener Sprache)
Erstens/Zweitens/Drittens 05:941/945/949; „nicht X, sondern Y“ 05:1049, 1042–1043; „Das/Dies + Verb“ 05:1029, 1038, 1059, 1101, 1150; „Flaschenhals“ 05:1049/1060; 37 Sätze > 40 Wörter, > 50: 05:765 (95), 842 (92), 1092 (88), 1006 (73), 341 (66), 578 (60), 1063 (60), 229 (59), 248 (56), 1029 (56), 215, 361, 1000, 50, 158, 241, 521, 1070; Topic-Satz-Schablonen 05:181, 227, 359, 745, 763; „damit“ 13×, „nahezu“ 7×, „also“ 7×; 05:524–526 „um ein bis zwei Größenordnungen“.

## Status
Erledigt seit 28.10.: A1–A7, B8–B12, C15, C17, C18, D22, D23, D26–D28, „obere Abschätzung“, 1,2/0,2-Zerlegung, Kontamination, CEL, Typinventar, Batching, 24 GB, „in jeder Hinsicht“, isolierter Anpassungseffekt, GB→GiB (außer 05:849), 61 %/51 %, Pareto-Bedingung in 5.7.4, drei Empfehlungsszenarien, CrossNER-Bootstrap. Offen: Sampling, Zählfehler, F07 (zwei Stellen), „viele davon“, Emergenz (optional), Redundanzen, Synthese-Tabelle, Abb. 5.1/5.3, Tab. 5.3/5.6-Captions, Float S. 77, Sprache.

## Aufwand
KI ≈ 6 h (Zahlen 0,5; Formulierungsvorschläge Überinterpretationen 1; Dopplungen markieren 0,5; Synthese-Tabelle 0,5; Trunkierung bündeln 0,3; Pareto/Versprechen 0,2; Tabellen/Abbildungen 1,5; Diagnoseliste Sprache 0,5; Build 0,3). Autor ≈ 11–13 h (eigene Fassung der Überinterpretationen 1; 5.7.3/5.7.4 entzahlen 2; Synthese 1; Trunkierung 1; 6.1 verdichten 1,5; Darstellung/Interpretation 0,8; Sichtkontrolle 0,5; Sprache 3; Zusätze-Stichprobe 1,5 oder abschwächen 0,2; Abgleich 0,5).
