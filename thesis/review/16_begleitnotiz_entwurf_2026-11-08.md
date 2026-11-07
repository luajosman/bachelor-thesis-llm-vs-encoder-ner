# Entwurf der Begleitnotiz an den Betreuer (Dr. Philippe Thomas) – Stand 7.11.2026

Vom Verfasser vor dem Versand anzupassen (Anrede, Tag/Commit des versandten Stands, Rückmeldefrist). Keine Selbsteinstufung der Note.

---

Sehr geehrter Herr Dr. Thomas,

anbei sende ich Ihnen den vollständigen Entwurf meiner Bachelorarbeit „Fine-Tuning für Named Entity Recognition: Ein Vergleich von BERT-basierten Modellen und LLMs mit LoRA" (Stand: Git-Tag `betreuerentwurf-2026-11-08`, 119 Seiten, davon 95 Seiten Haupttext). Abgabetermin ist der 26. November 2026.

**Was der Entwurf enthält.** Der Vergleich von DeBERTa-v3-base/-large (vollständiges Fine-Tuning) mit Qwen3.5-0.8B/4B/27B (Zero-Shot und QLoRA) auf den offiziellen englischen MultiNERD-Splits ist vollständig ausgewertet (drei Seeds je trainierter Konfiguration, gepaartes Satz-Bootstrap, CrossNER-Transfertest). Die drei Kernbefunde: (1) Bei 256 Ausgabetokens liegt DeBERTa-v3-large 1,34 Punkte Mikro-F1 vor dem 27B-Modell im QLoRA-Regime; nach der Nachinferenz der am Limit abgeschnittenen Antworten mit 1 024 Tokens schrumpft der Abstand auf 0,18 Punkte (Intervall 0,02 bis 0,33). (2) Auf CrossNER kehrt sich die Rangfolge mit dem höheren Limit um (LLM 87,1 gegenüber 85,0). (3) Die Inferenz der LLMs ist hardwaregleich rund 210-mal langsamer; bei festem Typinventar und engem Latenz- und Speicherbudget bleiben die Encoder die gut gestützte Wahl.

**Was ich offen ausweise.** Der MultiNERD-Testsplit wurde vor der Hauptstudie in Vorläufen ausgewertet; Prompt und Decoding des Zero-Shot-Regimes wurden nach Testbeobachtungen eines 0.8B-Vorlaufs gewählt (Tabelle „Datengrundlage der Konfigurationsentscheidungen" in Abschnitt 4.3.3 und Abschnitt 6.2). Die Testreferenz ist eine Silberreferenz. Die Analysen zum Ausgabelimit sind nachträgliche Sensitivitätsanalysen.

**Was noch folgt.** Die Verdichtung der Kapitel 2 bis 5 von derzeit 95 auf etwa 75 bis 80 Seiten Haupttext, die sprachliche Durchsicht und die letzten Formalia (Codex-Version in der KI-Angabe, Abgabedatum, Veröffentlichung des Repositorys).

Für Ihre Rückmeldung bis zum 15. November wären mir vor allem diese Punkte wichtig:

1. **Testnutzung und Umfang:** Reicht die Offenlegung in 4.3.3 und 6.2 aus Ihrer Sicht, oder erwarten Sie zusätzliche Evidenz (etwa einen Zero-Shot-Neulauf mit auf Validierungsdaten festgelegtem Prompt und Decoding; der bekannte Testsplit bliebe dabei derselbe, der Aufwand wäre gesondert zu kalkulieren)?
2. **Hauptauswertung:** Ist es richtig, die 256-Token-Läufe als Hauptauswertung und die 1 024-Token-Nachinferenz als Sensitivitätsanalyse zu führen, oder sollte ein vollständiger Neulauf mit 1 024 Tokens die Hauptauswertung werden?
3. **Forschungslücke:** Trägt die Formulierung der Lücke als Kombination (offizielle Splits, eigene Encoder-Referenz in derselben Pipeline, Größenstaffelung beider Familien, gemeinsame Erfassung von Erkennung, Formtreue und Ressourcen), oder sollte der Beitrag stärker auf den Ausgabelimit-Befund zugespitzt werden?
4. **Verdichtung:** Welche Teile der Grundlagen (Kapitel 2) und der Methodik (Kapitel 4) sollen aus Ihrer Sicht ausführlich bleiben, welche können in den Anhang?
5. **Formalia:** Gibt es ein Bewertungsraster des Fachgebiets? Ist der angemeldete Titel mit der Abgrenzung in Abschnitt 1.3 ausreichend behandelt? Wird neben dem deutschen Abstract eine englische Fassung erwartet? Ist das DFKI-Logo auf dem Deckblatt in Ordnung?

Das Repository (Code, Konfigurationen, Auswertungsskripte, generierte Tabellen) stelle ich Ihnen auf Anfrage bereit; es wird vor der Abgabe öffentlich geschaltet.

Mit freundlichen Grüßen
Luaj Osman
