# Zweitgutachten Frontmatter, Kapitel 1 und 2 (28.10.2026, Stand 99d9a56)

Basis: frontmatter/*.tex, main.tex, chapters/01_einleitung.tex, chapters/02_grundlagen.tex, figures/ner_grundprinzip.tex, figures/vaswani_transformer.tex; PDF-Seiten 1–5, 18, 21, 27, 29, 31. Alle 57 Zitierschlüssel aus Kap. 1/2 existieren; alle 43 `\ref`-Ziele existieren. Gegen Fremdquellen geprüft: Keraghel et al. (arXiv 2401.10825v3), Qwen3.5-27B-Modellkarte, seqeval-README. Abstract-Zahlen stimmen mit Kap. 5/6 und summary.json überein.

## 1. Frontmatter: 7/10
Deckblatt sauber; Abstract dicht, mit Zahlen, deckt alle drei FF; Abkürzungsverzeichnis vollständig. Abzüge: kein Abgabedatum/Institut; Erklärungstext weicht von der TU-Vorlage ab (prüfen); ein ungenauer Abstract-Satz (korrigiert); „BIO“ doppeldeutig im Abkürzungsverzeichnis; arabische Zählung des Vorspanns (Kap. 1 beginnt auf S. 9).

## 2. Kapitel 1: 8/10
Motivation und Problemstellung schlüssig; FF präzise, operationalisiert, in 3.5/5/6 adressiert; Beitragsliste korrekt gerechnet; Titeldiskrepanz in 1.3 erklärt; Aufbau-Absatz stimmt mit Gliederung überein. Abzüge: Redundanzen in 1.2/1.3, „einzige Größenklasse“ (korrigiert), DFKI-Absatz an falscher Stelle, rhetorische Muster.

## 3. Kapitel 2: 8/10
Fachlich sehr solide; alles Eingeführte wird später gebraucht (außer F_β und Bahri et al.). Abb. 2.1/2.2 (TikZ) sauber. Abzüge: BILOU-Buchstaben (korrigiert), DeBERTa-Terme (korrigiert), 512-Positionen-Schluss (korrigiert), zwei Grammatikfehler „begrenzt“ (korrigiert), RoPE-Zitat, Typkürzel unerklärt, Abb. 2.3 winzige Beschriftung.

## 4. Mängel nach Schwere

### Inhaltliche Fehler / Ungenauigkeiten
1. `02:136–137` BILOU: L/U statt E/S. (korrigiert)
2. `02:392–394` DeBERTa: Inhalt-zu-Inhalt, Inhalt-zu-Position **und** Position-zu-Inhalt; Position-zu-Position entfällt. (korrigiert)
3. `abstract.tex:30–31` „Stunden gegen Tage bis Wochen“ – 12 h / 62,9 h / 266 h ≈ 11 Tage. (korrigiert)
4. `01:120–122` „einzige Größenklasse 3–9 Mrd.“ gilt nur für die adaptierten LLMs der direkten Vergleiche (Keraghel nutzt GPT-4o). (korrigiert)
5. `02:371–374` 512-Positionen-Schluss von BERT auf DeBERTa-v3 ohne Beleg. (korrigiert: DeBERTa-v3 ebenfalls auf 512 vortrainiert)
6. `02:238–240` RoPE mit Modellkarte belegt; Primärquelle Su et al. (RoFormer) fehlt. **Offen.**
7. `02:452–454` „Inferenzzeit wächst vor allem mit der Länge der Ausgabe, nicht der Eingabe“ ohne Begründung (Prefill vs. sequentielle Dekodierung). **Offen.**

### Argumentationslücken / Konsistenz
8. `01:109–110` vs. 3.5: „vier Fragen“ vs. „erste/zweite Lücke“. **Offen.**
9. `01:171–173` Beitrag 3 grenzt sich nicht von Semin et al. ab. (korrigiert)
10. `01:192–202, 217–219` DeBERTa doppelt eingeführt; `01:183–187 vs. 210–211` CrossNER-Einschränkung doppelt; `01:134–136 vs. 196–199` QLoRA/27B doppelt. **Offen.**
11. `01:45–49` DFKI/SLT-Absatz unterbricht den Fluss; „zentrales Anwendungsfeld“ unbelegt. **Offen.**
12. `02:786–788` MultiNERD-Typkürzel erstmals verwendet, erst in Tab. 4.3 aufgelöst. **Offen.**
13. `02:765` „CoNLL-2003 ist bis heute der am häufigsten verwendete NER-Benchmark“ ohne Beleg. **Offen.**

### Formalia
14. `abkuerzungsverzeichnis.tex:23, 34` „BIO“ als Schema vs. MultiNERD-Typ; doppelt mit „IOB“. **Offen.**
15. `deckblatt.tex` kein Abgabedatum, kein Institut; „Fakultät IV,“ mit Komma. **Offen (Nutzer).**
16. `eigenstaendigkeitserklaerung.tex:4–6` Wortlaut mit aktueller Fak.-IV-Vorlage abgleichen. **Offen (Nutzer).**
17. `main.tex:136, 154–168` Vorspann arabisch; Abkürzungsverzeichnis vor Inhaltsverzeichnis; Sternkapitel nicht im TOC. **Offen.**
18. `02:265–271` Caption Abb. 2.2 „ergänzt eine maskierte Self-Attention“ missverständlich; Abb. 2.2 allein auf Gleitseite (PDF S. 21). **Offen.**
19. `02:612`/figures/lora_adapter.pdf (Abb. 2.3) Beschriftung ~5 pt; Stil weicht von TikZ-Abbildungen ab. **Offen.**
20. figures/kaplan_scaling.pdf (Abb. 2.4) „Compute“ englisch, Schrift klein. **Offen.**
21. `figures/ner_grundprinzip.tex:24–25` Tupel (t₁, t₂, PER) vs. Formalisierung mit Indizes. **Offen.**

### Sprache / Stil
22. `02:681–683, 728–730` „begrenzt“ ohne Objekt. (korrigiert)
23. Kontrastmuster „nicht X, sondern Y“: `01:67–68, 181, 186–187; 02:312, 741, 932–933` (01:181 korrigiert).
24. `02:15` „handentworfenen“; `01:271, 02:22, 02:940` „parameter-effizient“ → „parametereffizient“.
25. „feinjustiert“ vs. „Fine-Tuning“; „Entity-Level“ vs. „auf Entitätsebene“ – bewusst festlegen.
26. `01:266–267` Füllsatz; `01:21–25` Eröffnung ohne Beleg.
27. `01:26–34, 57–61` Definitionen nahezu wortgleich mit 2.1/2.2/2.5.
28. `02:664–745` (2.7) Skalierungsgesetze zu lang; `02:870–882` F_β nicht benötigt.

## 5. Was zur 10 fehlt
**Frontmatter:** Abgabedatum, Institut, Titelwortlaut laut Anmeldung; Erklärungswortlaut aus Vorlage; „BIO“-Doppeldeutigkeit; Paginierung/Reihenfolge des Vorspanns.
**Kapitel 1:** Punkt 8 angleichen; Redundanzen 1.2/1.3; DFKI-Absatz; Füllsatz/Kontrastmuster; Definitionen straffen; Beitrag 5 nach den Läufen prüfen.
**Kapitel 2:** RoPE-Primärquelle; Typkürzel glossieren; CoNLL-Beleg; Abb. 2.3/2.4 neu; Caption Abb. 2.2; Tupelnotation Abb. 2.1; 2.7 und F_β kürzen; Terminologie.
