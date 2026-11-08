# Mail an den Betreuer (Dr. Philippe Thomas) – Stand 8.11.2026, abends, Du-Fassung

Anhang: `thesis/main_betreuerentwurf_9b1e273.pdf` (frischer Build aus Commit 9b1e273, 116 Seiten). Keine Selbsteinstufung der Note. Frühere Fassungen (Sie-Form, nur Kontrolllauf) in der Git-Historie.

---

Betreff: Entwurf meiner Bachelorarbeit – Bitte um Feedback

Hallo Philippe,

anbei der aktuelle Stand meiner Bachelorarbeit „Fine-Tuning für Named Entity Recognition: Ein Vergleich von BERT-basierten Modellen und LLMs mit LoRA“ (82 Seiten Haupttext, 116 Seiten insgesamt). Abgabe ist am 26. November, und ich würde mich sehr über dein Feedback freuen.

Kurz zum Stand: DeBERTa-v3-base und -large (volles Fine-Tuning) sowie Qwen3.5-0.8B, -4B und -27B (Zero-Shot und QLoRA) sind auf den offiziellen englischen MultiNERD-Splits ausgewertet, mit drei Seeds, gepaartem Bootstrap und einem CrossNER-Transfertest. Mit 256 Ausgabetokens liegt DeBERTa-v3-large 1,34 F1-Punkte vor dem 27B-Modell, nach einer Nachinferenz der abgeschnittenen Antworten nur noch 0,18. Auf CrossNER dreht sich die Reihenfolge mit dem höheren Limit sogar um. Qwen3.5-0.8B ist auf gleicher Hardware rund 210-mal langsamer als DeBERTa-v3-base.

Zu den Läufen: Das Ausgabelimit von 256 Tokens hatte ich ohne vorherige Längenanalyse gesetzt. Es schneidet rund 0,4 % der Antworten ab und kostet das LLM auf MultiNERD gut einen F1-Punkt, auf CrossNER 15. Bisher ist das über eine Nachinferenz nur der abgeschnittenen Sätze mit 1 024 Tokens korrigiert. Die Antwortlängen in Training und Validierung zeigen, dass sich 1 024 Tokens auch ohne Testdaten begründen lassen.

Sauberer wäre ein vollständiger Neulauf aller neun QLoRA-Testinferenzen mit 1 024 Tokens auf einheitlicher Hardware. Das sind rund 150 GPU-Stunden. Der SLT-Knoten ist im Moment aber komplett ausgelastet, und auf den allgemeinen H100-Partitionen warten über 1 400 Jobs. Deshalb habe ich erst mal nur einen Kontrolllauf mit rund 11 GPU-Stunden eingereicht: Je QLoRA-Lauf werden 1 000 zufällig gezogene, nicht abgeschnittene Testsätze neu erzeugt. Das zeigt, ob die zusammengesetzte Auswertung einem vollständigen Lauf entspricht, und liefert Latenzen aller Modellgrößen auf demselben GPU-Typ wie die Encoder. Laut Scheduler startet er frühestens am 14. November.

Meine wichtigste Frage, wenn möglich bis Dienstag, 10. November: Könntest du GPU-Kapazität auf dem SLT-Knoten für mich organisieren oder reservieren?
- Ideal wären acht GPUs für ein bis zwei Tage. Damit könnte ich den vollständigen Neulauf noch vor der Abgabe machen; den Code dafür habe ich in ein bis zwei Tagen startklar.
- Wenn das nicht geht, würde schon ein früherer Slot für den Kontrolllauf helfen (kurze Jobs von höchstens vier Stunden, zusammen rund 11 GPU-Stunden).

Und hältst du den Kontrolllauf für ausreichend, falls der vollständige Neulauf nicht klappt?

Feedback gern bis Montag, 16. November. Wenn die Zeit knapp ist, wären mir Anmerkungen zu Kapitel 4 (Methodik) und Kapitel 5 (Ergebnisse und Diskussion) am wichtigsten. Besonders interessiert mich:
1. Wo siehst du die größten Schwächen der Arbeit, und was würde sie am meisten stärken?
2. Reicht die Offenlegung der früheren Testnutzung (Tabelle 4.3 und Abschnitt 6.2) und der Silberreferenz als Einschränkung?
3. Erwartest du für FF3 konkrete Budgetszenarien für Latenz und Speicher, oder genügt die Antwort über die Pareto-Front?
4. Sind 82 Seiten Haupttext okay?
5. Gibt es ein Bewertungsraster? Braucht es zusätzlich ein englisches Abstract? Ist das DFKI-Logo auf dem Deckblatt in Ordnung?
6. Passt die KI-Angabe in der Eigenständigkeitserklärung so?
7. Ist der angemeldete Titel mit der Abgrenzung in Abschnitt 1.3 ausreichend abgedeckt?

Offen sind noch die Ergebnisse der Läufe, meine eigene sprachliche Durchsicht und die Veröffentlichung des Repositorys. Code, Konfigurationen und Auswertungsskripte schicke ich dir gern vorab.

Viele Grüße
Luaj
