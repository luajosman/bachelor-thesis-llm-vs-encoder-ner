# Mail an den Betreuer (Dr. Philippe Thomas) – Stand 8.11.2026, abends

Vor dem Versand prüfen: Anrede, Fristen. Anhang: `thesis/main_betreuerentwurf_9b1e273.pdf` (frischer Build aus Commit 9b1e273, 116 Seiten). Keine Selbsteinstufung der Note. Frühere Fassungen in der Git-Historie.

---

**Betreff:** Entwurf meiner Bachelorarbeit – Bitte um Feedback

Sehr geehrter Herr Dr. Thomas,

anbei sende ich Ihnen den aktuellen Stand meiner Bachelorarbeit „Fine-Tuning für Named Entity Recognition: Ein Vergleich von BERT-basierten Modellen und LLMs mit LoRA“ (82 Seiten Haupttext, 116 Seiten gesamt). Abgabe ist am 26. November, und ich würde mich sehr über Ihr Feedback freuen.

Kurz zum Stand: DeBERTa-v3-base und -large mit vollständigem Fine-Tuning sowie Qwen3.5-0.8B, -4B und -27B im Zero-Shot- und QLoRA-Regime sind auf den offiziellen englischen MultiNERD-Splits ausgewertet, mit drei Seeds, gepaartem Bootstrap und einem CrossNER-Transfertest. Bei 256 Ausgabetokens liegt DeBERTa-v3-large 1,34 F1-Punkte vor dem 27B-Modell, nach Nachinferenz der abgeschnittenen Antworten nur noch 0,18. Auf CrossNER kehrt sich die Rangfolge mit höherem Limit um. Qwen3.5-0.8B ist auf gleicher Hardware rund 210-mal langsamer als DeBERTa-v3-base.

**Zu den Läufen:** Das Ausgabelimit von 256 Tokens hatte ich ohne Längenanalyse gesetzt. Es schneidet rund 0,4 % der Antworten ab und kostet das LLM auf MultiNERD gut einen F1-Punkt, auf CrossNER 15. Korrigiert ist das durch eine Nachinferenz der abgeschnittenen Sätze mit 1 024 Tokens. Die Längen der Referenzantworten in Training und Validierung zeigen, dass sich dieses Limit auch ohne Testdaten begründen lässt.

Einen vollständigen Neulauf aller neun QLoRA-Testinferenzen (rund 165 GPU-Stunden) halte ich vor der Abgabe nicht für planbar. Stattdessen habe ich einen Kontrolllauf mit rund 11 GPU-Stunden eingereicht:
- Je QLoRA-Lauf werden 1 000 zufällig gezogene, nicht abgeschnittene Testsätze mit 1 024 Tokens auf einer H100 neu erzeugt.
- Er prüft, ob die zusammengesetzte Auswertung einem vollständigen Lauf entspricht.
- Er liefert Latenzen aller Modellgrößen auf demselben GPU-Typ wie die Encoder.

Die Jobs warten derzeit auf dem SLT-Knoten, der voll ausgelastet ist; der Scheduler schätzt den Start auf den 14. November. Was bis 16. November nicht vorliegt, nehme ich nicht mehr auf.

**Zwei Fragen mit kurzer Frist, wenn möglich bis Dienstag, 10. November:**
1. Ließe sich für diese kurzen Jobs (je höchstens vier Stunden, zusammen rund 11 GPU-Stunden) früher Platz auf dem SLT-Knoten finden?
2. Halten Sie den Kontrolllauf statt eines vollständigen Neulaufs für ausreichend?

**Bitte um Feedback, gern bis Montag, 16. November:** Falls die Zeit knapp ist, wären mir Ihre Anmerkungen zu Kapitel 4 (Methodik) und Kapitel 5 (Ergebnisse und Diskussion) am wichtigsten. Besonders interessieren mich:
1. Wo sehen Sie die größten Schwächen der Arbeit, und was würde sie aus Ihrer Sicht am meisten stärken?
2. Reicht die Offenlegung der früheren Testnutzung (Tabelle 4.3 und Abschnitt 6.2) und der Silberreferenz als Einschränkung?
3. Erwarten Sie für FF3 konkrete Budgetszenarien für Latenz und Speicher, oder genügt die Antwort über die Pareto-Front?
4. Sind 82 Seiten Haupttext für das Fachgebiet angemessen?
5. Gibt es ein Bewertungsraster des Fachgebiets? Wird neben dem deutschen Abstract eine englische Fassung erwartet? Ist das DFKI-Logo auf dem Deckblatt in Ordnung?
6. Ist die KI-Angabe in der Eigenständigkeitserklärung in Form und Umfang so in Ordnung?
7. Ist der angemeldete Titel mit der Abgrenzung in Abschnitt 1.3 ausreichend behandelt?

Offen sind noch die Ergebnisse des Kontrolllaufs, meine eigene sprachliche Durchsicht und die Veröffentlichung des Repositorys. Code, Konfigurationen und Auswertungsskripte stelle ich Ihnen gern vorab bereit.

Mit freundlichen Grüßen
Luaj Osman
