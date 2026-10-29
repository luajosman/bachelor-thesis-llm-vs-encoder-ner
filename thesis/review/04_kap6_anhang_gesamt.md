# Zweitgutachten Kapitel 6, Anhang, Literatur, Gesamtkonsistenz (28.10.2026, Stand 99d9a56)

Geprüft: `chapters/06_fazit_und_ausblick.tex`, `chapters/anhang.tex` + `generated/tab-app-*.tex`, `references.bib`, `main.tex`, Kap. 1 und 5 (ganz), Frontmatter; PDF-Seiten 2–9 und 90–107. Build: 0 Overfull, 0 undefinierte Verweise. Alle 76 zitierten Schlüssel existieren.

## 1. Kapitel 6 – Note 7/10

**Gut:** 6.1 beantwortet FF1–FF3 in der Reihenfolge und mit den Begriffen aus 1.2; jede Zahl ist in Kap. 5 belegt (alle gegengeprüft). Empfehlungen aus 5.7.4 abgeleitet, Bedingungen genannt. Limitationen ungewöhnlich ehrlich. Ausblick konkret. Länge angemessen.

**Warum nicht höher:**

a) **FF1 ohne Bootstrap-Intervalle; 6.2 behauptet, auf Inferenzstatistik sei verzichtet worden** (Z. 180–182) – Widerspruch zu Kap. 1 (Beitrag 5) und 5.7.5/Tab. 5.9. (korrigiert 28.10.)

b) **Redundanz zu 5.7 stellenweise wortgleich:** Z. 77–95 (FF3) ↔ Kap. 5 Z. 1047–1075; Z. 72–74 ↔ Kap. 5 Z. 468–472 und 1006–1009; Z. 38–41 ↔ Kap. 5 Z. 863–881; Z. 106–116 ↔ Kap. 5 Z. 904–914; Z. 128–129 = Kap. 5 Z. 856–857 wörtlich; Z. 52–53 ↔ Kap. 5 Z. 941–942. 6.1 soll verdichten, wiederholt aber in gleicher Länge. **Offen.**

c) **Limitationen sind eine Liste, nicht nach Wirkung geordnet.** Ausgabelimit (wirksamster Störfaktor) steht als letzter Satz von „Vergleichsbedingungen“ (Z. 147–153). Vorschlag: Reihenfolge (1) Ausgabelimit, (2) Silberreferenz, (3) Anpassungsasymmetrie, (4) Hardware-Heterogenität, (5) Zero-Shot-Einzellauf und Testnutzung, (6) Statistik; je Absatz zuerst: welche FF betroffen, Richtung oder nur Betrag. **Offen.**

d) Einzelne Unschärfen (Abschnitt 4, Punkte 6–10; korrigiert).

e) **Sprache/Textbaustein-Charakter:** 14 `\paragraph{Label.}`-Blöcke (Z. 26, 55, 76, 105, 126, 155, 168, 180, 199, 209, 217, 231, 243, 252) – Protokollcharakter. Semikolon-/Doppelpunktketten: Z. 76–87 ein Satz, Z. 26–37, Z. 61–68. Kontrastformeln: Z. 127, Z. 235 (ebenso Kap. 5 Z. 874, 878, 952, 1030; Kap. 1 Z. 181). Schlussfloskeln: Z. 206–207 „deren Beantwortung nur Rechenzeit kostet“, Z. 236–237 „setzen genau an dem Fehler an“, Z. 200 „Die naheliegendste Ergänzung“. Triade Z. 114 = Kap. 5 Z. 911. Roadmap-Satz Z. 18–21 im Muster von Kap. 1 Z. 8–16 und Kap. 5 Z. 29–45. Einheitenwechsel Z. 63/66 („1,2 h“ … „266 Stunden“).

## 2. Anhang – Note 8/10

**Gut:** A.1 vorbildlich (Tab. A.1–A.3, Artefaktliste, Listing A.1, ehrliche Fußnoten); A.2 liefert, was Kap. 5 auslagert; alle Werte stimmen; jeder Anhangsteil wird aus dem Haupttext erreicht.

**Warum nicht höher:**
a) **Fundort des Repositorys fehlt in der gesamten Arbeit** (anhang.tex Z. 17–18, Kap. 4 Z. 88–89): keine URL, kein Archiv-DOI, keine Zugangsregel. **Offen (Nutzerentscheidung).**
b) **Listing A.1** (Z. 174–186) beginnt auf S. 103, läuft als Fragment auf S. 104 weiter; Caption erst nach Tab. A.2/A.3. **Offen.**
c) Tab. A.8 „Beispiele je Fehlerklasse (5.5)“ enthält zwei Zeilen „Unbekannter Typ“ (Formtreue-Kategorie aus 5.2). **Offen.**
d) Einzeltabellen A.1–A.7 nie per Label referenziert. **Offen.**
e) Tab. A.5/A.7: Gruppenköpfe breiter als Datenspalten; Tab. A.3 Z. 146–148 zwei Validierungs-Batchgrößen ohne Erklärung, Z. 162 „H100-Trails“ ohne GPU-Typ. **Offen.**

## 3. Gesamtkonsistenz

- Titel/FF/Beiträge/Aufbau/Fazit konsistent; Aufbau-Absatz (Kap. 1 Z. 282–302) nennt 4.5.5 und A.2 nicht. **Offen.**
- Begriffe einheitlich; Reste: „Silver Standard“ (Tab. 4.9, Kap. 4 Z. 1282) neben „Silberreferenz“ (Kap. 5) und „Silberannotation“ (Kap. 6 Z. 116); „BIO-Tags“ Kap. 5 Z. 963 (Zitat Lu et al.); Nomen „Fine-Tuning“ neben Verb „feinjustiert“.
- Zahlen: Abstract/1/5/6 gleich; Unschärfen Abstract Z. 30–31 (korrigiert), Kap. 6 Z. 36–37 (korrigiert), Kap. 5 Z. 182–184 „Hälfte“ (korrigiert).
- **Literaturverzeichnis:** 97 Einträge, 76 zitiert; biblatex-apa konsistent. hu2021lora, he2020deberta, he2021debertav3 nur arXiv statt ICLR (OpenReview nZeVKeeFYf9, XPZIaotutsD, sE7-XhLxHA); keraghel2024recent arXiv v3 (Zeitschriftenfassung prüfen); sang2003conll ohne DOI (10.3115/1119176.1119195); Preprints in der gedruckten Liste nur mit URL, ohne „arXiv:ID“ (`eprint`-Feld wird nicht gedruckt: `archivePrefix`/`primaryClass` statt `eprinttype`/`eprintclass` prüfen); jurafsky2024speech Z. 11 englische Note; lafferty2001crf Z. 947 dblp-URL; Bib-Header „78 Einträge“ (97); 21 ungenutzte Einträge. **Offen.**
- **Verzeichnisse:** Abstract, Abkürzungs-, Abbildungs-, Tabellenverzeichnis fehlen im Inhaltsverzeichnis; Frontmatter arabisch (unsichtbar) gezählt, erste sichtbare Seitenzahl „9“; dritte TOC-Seite fast leer. **Offen.**

## 4. Mängel nach Schwere

**Hoch**
1. `06:180–186` Inferenzstatistik-Widerspruch (korrigiert).
2. `anhang.tex:17–20`, `04:88–89` Repository-Fundort (offen, Nutzer: URL/DOI oder Zugangsregel).
3. `anhang.tex:174–186` Listing A.1 von Caption getrennt (offen; `float=htbp` oder `\clearpage`).
4. `06:77–95` ↔ `05:1047–1075` u. a. wortgleiche Redundanz (offen; 6.1 je FF auf 8–12 Zeilen).
5. `06:100–103, 147–153` Limitationen unsortiert (offen).

**Mittel**
6. `06:34–37` verschachtelt, zahlenmäßig unscharf (korrigiert).
7. `06:64–65` Latenz 0.8B/27B auf verschiedenen GPUs ohne Vorbehalt (korrigiert).
8. `06:69–71` „erst das 27B-Modell hält Format und Typinventar ein“ – 4B erzeugt überwiegend gültiges JSON (korrigiert).
9. `06:247–250` NF4-Merge ohne Dequantisierung nicht möglich (korrigiert).
10. `06:258–261` GoLLIE ist zero-shot, PromptNER few-shot (korrigiert).
11. `anhang.tex:256–260, 294–299` Tab. A.8 Caption/Kategorie (offen).
12. `references.bib` ICLR-Fassungen, Preprint-Kennzeichnung, CoNLL-DOI (offen).
13. `01:282–289, 300–302` Aufbau-Absatz ohne 4.5.5/A.2 (offen).
14. Pauschale Anhangsverweise statt Labels (offen).

**Niedrig**
15. „Hälfte“ vs. „größter Teil“ (korrigiert). 16. Abstract Trainingsdauer (korrigiert). 17. `06:81–82` 24 GB vs. 18,7 GB (korrigiert). 18. Tab. A.3 Validierungs-Batchgrößen, Partition→GPU (offen). 19. Spaltenausrichtung A.5/A.7 (offen). 20. Terminologie-Reste Silberreferenz/BIO-Tags (offen). 21. `main.tex:136–137` Paginierung/TOC (offen). 22. Bib-Header, ungenutzte Einträge, Note, dblp-URL (offen). 23. Kommentare anhang.tex Z. 191–193, 304; pgfplots-Stil in 05:13–24 (offen, kosmetisch). 24. Stil Kap. 6 (Abschnitt 1e, Nutzeraufgabe).

## 5. Was zur 10 fehlt

**Kapitel 6:** Sensitivitätsanalysen einbauen; Redundanz zu 5.7 auflösen; Limitationen nach Wirkung ordnen; eigene Sprache; Präzisierungen (erledigt).
**Anhang:** Repository-Fundort; Listing A.1; Labels aus Kap. 4/5; Tab. A.8/A.3/A.5/A.7; nach den Läufen Zeilen mit Commit ff91342 um Datum/Job-IDs ergänzen, Laufdatum der Zero-Shot-Läufe.
**Arbeit als Ganzes:** Sensitivitätsanalysen synchron an allen Stellen; Literatur (ICLR, Preprints, DOI, ungenutzte Einträge); Formalia (römische Paginierung, Verzeichnisse im TOC, Titelwortlaut, Abstract-Überschrift); Terminologie-Feinschliff; Aufbau-Absatz 1.4.
