# Entwurf der Begleitnotiz an den Betreuer (Dr. Philippe Thomas) – Stand 8.11.2026

Vom Verfasser vor dem Versand anzupassen: Anrede, Fristen, Anhang (`thesis/main.pdf`). Keine Selbsteinstufung der Note. Vorgängerfassungen in der Git-Historie; diese Fassung ersetzt die Frage nach dem vollständigen Neulauf durch den eingereichten Kontrolllauf (Cluster-Inventur 8.11.).

---

**Betreff:** Entwurf meiner Bachelorarbeit – Bitte um kurze Rückmeldung zu einem Neulauf

Sehr geehrter Herr Dr. Thomas,

anbei sende ich Ihnen den vollständigen Entwurf meiner Bachelorarbeit „Fine-Tuning für Named Entity Recognition: Ein Vergleich von BERT-basierten Modellen und LLMs mit LoRA" (Stand 8. November 2026, 82 Seiten Haupttext, 116 Seiten gesamt). Abgabe ist am 26. November.

Kurz zum Stand: DeBERTa-v3-base und -large mit vollständigem Fine-Tuning sowie Qwen3.5-0.8B, -4B und -27B im Zero-Shot- und QLoRA-Regime sind auf den offiziellen englischen MultiNERD-Splits ausgewertet, mit drei Seeds, gepaartem Bootstrap und einem CrossNER-Transfertest. Bei 256 Ausgabetokens liegt DeBERTa-v3-large 1,34 F1-Punkte vor dem 27B-Modell, nach Nachinferenz der abgeschnittenen Antworten nur noch 0,18. Auf CrossNER kehrt sich die Rangfolge mit höherem Limit um. Qwen3.5-0.8B ist auf gleicher Hardware rund 210-mal langsamer als DeBERTa-v3-base.

**Eine Frage mit kurzer Frist, wenn möglich bis Dienstag, 10. November:**

Das Ausgabelimit von 256 Tokens hatte ich ohne Längenanalyse gesetzt. Es schneidet rund 0,4 % der Antworten ab und kostet das LLM auf MultiNERD gut einen F1-Punkt, auf CrossNER 15. Korrigiert ist das bisher durch eine Nachinferenz nur der abgeschnittenen Sätze mit 1 024 Tokens. Die Längen der Referenzantworten in Training und Validierung zeigen, dass sich 1 024 Tokens auch ohne Testdaten begründen lassen: Das Limit deckt alle Validierungsantworten und bis auf eine alle Trainingsantworten ab.

Einen vollständigen Neulauf aller neun QLoRA-Testinferenzen (rund 165 GPU-Stunden) halte ich vor der Abgabe nicht für planbar. Der SLT-Knoten ist derzeit voll belegt, und auf den allgemeinen H100-Partitionen warten über 1 400 Jobs. Stattdessen habe ich einen Kontrolllauf mit rund 11 GPU-Stunden eingereicht:
- Je QLoRA-Lauf werden 1 000 zufällig gezogene, nicht abgeschnittene Testsätze mit 1 024 Tokens auf einer H100 neu erzeugt, mit protokolliertem Stoppgrund.
- Er prüft, ob die zusammengesetzte Auswertung einem vollständigen Lauf entspricht.
- Er liefert Latenzen aller Modellgrößen auf demselben GPU-Typ wie die Encoder.

Was bis 16. November nicht vorliegt, nehme ich nicht mehr auf.

Halten Sie dieses Vorgehen für ausreichend? Falls Sie den vollständigen Neulauf für nötig halten: Ließe sich dafür kurzfristig GPU-Zeit auf dem SLT-Knoten bekommen, etwa acht GPUs für ein bis zwei Tage?

**Weitere Fragen, gern bis 15. November:**

1. Wo sehen Sie die größten Schwächen der Arbeit, und was würde sie aus Ihrer Sicht am meisten stärken?
2. Reicht die Offenlegung der früheren Testnutzung (Tabelle 4.3 und Abschnitt 6.2)? Die manuell geprüfte MultiNERD-Stichprobe von Tedeschi und Navigli ist in den veröffentlichten Daten nicht enthalten. Stattdessen prüfe ich selbst blind rund 180 Abweichungen zwischen den besten Modellen und der Silberreferenz, um Modell- von Annotationsfehlern zu trennen.
3. Erwarten Sie für FF3 konkrete Budgetszenarien für Latenz und Speicher, oder genügt die Antwort über die Pareto-Front?
4. Sind 82 Seiten Haupttext für das Fachgebiet angemessen?
5. Gibt es ein Bewertungsraster des Fachgebiets? Wird neben dem deutschen Abstract eine englische Fassung erwartet? Ist das DFKI-Logo auf dem Deckblatt in Ordnung?
6. Ist die KI-Angabe in der Eigenständigkeitserklärung in Form und Umfang so in Ordnung?
7. Ist der angemeldete Titel mit der Abgrenzung in Abschnitt 1.3 ausreichend behandelt?

Offen sind noch meine eigene sprachliche Durchsicht, das Abgabedatum auf dem Deckblatt und die Veröffentlichung des Repositorys. Code, Konfigurationen und Auswertungsskripte stelle ich Ihnen gern vorab bereit.

Mit freundlichen Grüßen
Luaj Osman
