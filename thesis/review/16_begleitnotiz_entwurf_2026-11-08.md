# Entwurf der Begleitnotiz an den Betreuer (Dr. Philippe Thomas) – Stand 8.11.2026

Vom Verfasser vor dem Versand anzupassen: Anrede, Fristen, Anhang (`thesis/main.pdf`). Keine Selbsteinstufung der Note. Vorgängerfassung vom 7.11. in der Git-Historie.

---

**Betreff:** Entwurf meiner Bachelorarbeit – Bitte um kurze Rückmeldung zu einem Neulauf

Sehr geehrter Herr Dr. Thomas,

anbei sende ich Ihnen den vollständigen Entwurf meiner Bachelorarbeit „Fine-Tuning für Named Entity Recognition: Ein Vergleich von BERT-basierten Modellen und LLMs mit LoRA" (Stand 8. November 2026, 82 Seiten Haupttext, 116 Seiten gesamt). Abgabe ist am 26. November.

Kurz zum Stand: DeBERTa-v3-base und -large mit vollständigem Fine-Tuning sowie Qwen3.5-0.8B, -4B und -27B im Zero-Shot- und QLoRA-Regime sind auf den offiziellen englischen MultiNERD-Splits ausgewertet, mit drei Seeds, gepaartem Bootstrap und einem CrossNER-Transfertest. Bei 256 Ausgabetokens liegt DeBERTa-v3-large 1,34 F1-Punkte vor dem 27B-Modell, nach Nachinferenz der abgeschnittenen Antworten nur noch 0,18. Auf CrossNER kehrt sich die Rangfolge mit höherem Limit um. Die LLM-Inferenz ist auf gleicher Hardware rund 210-mal langsamer.

**Eine Frage mit kurzer Frist, wenn möglich bis Dienstag, 10. November:**

Das Ausgabelimit von 256 Tokens hatte ich ohne Längenanalyse gesetzt. Es schneidet rund 0,4 % der Antworten ab und kostet das LLM auf MultiNERD gut einen F1-Punkt, auf CrossNER 15. Die Korrektur liegt bisher nur als selektive Nachinferenz der abgeschnittenen Sätze auf einem anderen GPU-Typ vor.

Ich würde deshalb gern alle neun QLoRA-Testinferenzen vollständig neu laufen lassen:
- mit einem Limit, das ich vorab aus den Antwortlängen von Training und Validierung ableite;
- auf einheitlicher Hardware (H100) und mit protokolliertem Stoppgrund;
- ohne neues Training, mit den vorhandenen Adaptern.

Der Aufwand liegt bei rund 165 GPU-Stunden; die Läufe wären bis 14. November fertig. Die bisherigen Ergebnisse bei 256 Tokens blieben als dokumentierte Fehlkonfiguration und Sensitivitätsanalyse in der Arbeit. Wenn Zeit bleibt, würde ich zusätzlich das Zero-Shot-Regime mit auf Validierungsdaten gewähltem Prompt und Decoding wiederholen. Was bis 15. November nicht vollständig vorliegt, nehme ich nicht mehr auf.

Halten Sie den Neulauf für sinnvoll, oder soll ich die Zeit lieber in den Text stecken?

**Weitere Fragen, gern bis 15. November:**

1. Reicht die Offenlegung der früheren Testnutzung (Tabelle 4.3 und Abschnitt 6.2), oder wäre zusätzlich eine Auswertung auf der manuell geprüften MultiNERD-Stichprobe von Tedeschi und Navigli sinnvoll, sofern sie verfügbar ist?
2. Erwarten Sie für FF3 konkrete Budgetszenarien für Latenz und Speicher, oder genügt die Antwort über die Pareto-Front?
3. Sind 82 Seiten Haupttext für das Fachgebiet angemessen?
4. Gibt es ein Bewertungsraster des Fachgebiets? Wird neben dem deutschen Abstract eine englische Fassung erwartet? Ist das DFKI-Logo auf dem Deckblatt in Ordnung?
5. Ist die KI-Angabe in der Eigenständigkeitserklärung in Form und Umfang so in Ordnung?
6. Ist der angemeldete Titel mit der Abgrenzung in Abschnitt 1.3 ausreichend behandelt?

Offen sind noch meine eigene sprachliche Durchsicht, das Abgabedatum auf dem Deckblatt und die Veröffentlichung des Repositorys. Code, Konfigurationen und Auswertungsskripte stelle ich Ihnen gern vorab bereit.

Mit freundlichen Grüßen
Luaj Osman
