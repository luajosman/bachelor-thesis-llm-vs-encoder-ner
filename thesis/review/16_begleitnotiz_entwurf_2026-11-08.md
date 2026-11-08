# Mail an den Betreuer (Philippe Thomas) – Stand 8.11.2026, abends, im gewohnten lockeren Ton

Anhang: `thesis/main_betreuerentwurf_9b1e273.pdf` (frischer Build aus Commit 9b1e273, 116 Seiten). Frühere, formellere Fassungen in der Git-Historie.

---

Betreff: Entwurf meiner BA

Hi Philippe,

wie angekündigt, hier der aktuelle Stand meiner Bachelorarbeit (82 Seiten Haupttext, 116 insgesamt). Alle Kapitel, Abstract und Anhang sind drin. Kurz zu den Ergebnissen: Mit 256 Ausgabetokens liegt DeBERTa-v3-large 1,34 F1-Punkte vor Qwen3.5-27B, nach einer Nachinferenz der abgeschnittenen Antworten nur noch 0,18. Auf CrossNER dreht sich die Reihenfolge mit dem höheren Limit sogar um, und das kleinste Qwen-Modell ist auf gleicher Hardware rund 210-mal langsamer als DeBERTa-v3-base.

Es wäre super, wenn du mir wieder das Liebermann-Feedbackdokument ausfüllen und eine Einschätzung zur vorläufigen Note geben könntest, gerne auch, wo du die größten Schwächen siehst. Wenn die Zeit knapp ist, sind mir Kapitel 4 (Methodik) und 5 (Ergebnisse und Diskussion) am wichtigsten. Wenn du es bis etwa 16.11. schaffst, hätte ich noch genug Zeit zum Einarbeiten.

Eine Sache noch zu den Läufen: Das Ausgabelimit von 256 Tokens war für die LLMs zu knapp. Bisher habe ich nur die abgeschnittenen Sätze mit 1 024 Tokens nachgerechnet. Sauberer wäre ein kompletter Neulauf der neun QLoRA-Testinferenzen auf einheitlicher Hardware, das wären rund 150 GPU-Stunden. Der SLT-Knoten ist gerade aber komplett belegt. Deshalb läuft erst mal nur ein kleiner Kontrolllauf mit rund 11 GPU-Stunden, und selbst der startet laut Scheduler voraussichtlich erst am 14.11.

Hast du eine Idee, ob man da kurzfristig GPUs reservieren könnte, zum Beispiel acht GPUs für ein bis zwei Tage, oder wen ich dafür am besten frage? Damit könnte ich den kompletten Neulauf noch vor der Abgabe schaffen. Und falls das nicht klappt: Reicht dir der Kontrolllauf?

Ein paar kurze Fragen noch:
1. Erwartest du für die dritte Forschungsfrage (Ressourcenbudgets) konkrete Budgetszenarien, oder reicht die Pareto-Front?
2. Braucht es zusätzlich ein englisches Abstract, und ist das DFKI-Logo auf dem Deckblatt okay?
3. Passt die KI-Angabe in der Eigenständigkeitserklärung so?
4. Sind 82 Seiten Haupttext okay?

Danke dir schon mal!

Viele Grüße,
Luaj
