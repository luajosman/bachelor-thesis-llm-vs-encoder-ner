# Mail an den Betreuer (Philippe Thomas) – Stand 8.11.2026, spätabends, im gewohnten lockeren Ton

Anhang: `thesis/main_betreuerentwurf_0816513.pdf` (frischer Build aus Commit 0816513: Sprachprüfung plus bündiges Abkürzungsverzeichnis, 118 Seiten). Grundlage ist die vom Nutzer eingefügte Fassung; aktualisiert wurden Seitenzahlen, CrossNER-Einschränkung, Scheduler-Termin (squeue am 8.11. abends: Smoke-Job 3724607 voraussichtlich 12.11. 02:26), Frage 2 (abgeleitete Schwellen) und Frage 4 („ohne Hilfe Dritter“). Frühere Fassungen in der Git-Historie.

---

Betreff: Entwurf meiner BA

Hi Philippe,

wie angekündigt, hier der aktuelle Stand meiner Bachelorarbeit (85 Seiten Haupttext, 118 insgesamt). Alle Kapitel, Abstract und Anhang sind drin.

Kurz zu den Ergebnissen:

Mit 256 Ausgabetokens liegt DeBERTa-v3-large 1,34 F1-Punkte vor Qwen3.5-27B, nach einer Nachinferenz der abgeschnittenen Antworten nur noch 0,18. Auf CrossNER dreht sich die Reihenfolge mit dem höheren Limit sogar um, wobei rund drei Viertel dieses Vorsprungs an der Abbildung der CrossNER-Typen hängen. Außerdem ist das kleinste Qwen-Modell auf gleicher Hardware rund 210-mal langsamer als DeBERTa-v3-base.

Es wäre super, wenn du mir wieder das Liebermann-Feedbackdokument ausfüllen könntest, gerne auch mit den größten Schwächen, die du siehst. Wenn die Zeit knapp ist, sind mir Kapitel 4 (Methodik) und 5 (Ergebnisse und Diskussion) am wichtigsten. Wenn du es bis etwa 16.11. schaffst, hätte ich noch genug Zeit zum Einarbeiten.

Eine Sache noch zu den Läufen: Das Ausgabelimit von 256 Tokens war für die LLMs zu knapp. Bisher habe ich nur die abgeschnittenen Sätze mit 1 024 Tokens nachgerechnet. Sauberer wäre ein kompletter Neulauf der neun QLoRA-Testinferenzen auf einheitlicher Hardware, das wären rund 150 GPU-Stunden. Der SLT-Knoten ist gerade aber komplett belegt. Deshalb läuft erst mal nur ein kleiner Kontrolllauf mit rund 11 GPU-Stunden, und selbst der startet laut Scheduler voraussichtlich erst am 12.11.

Hast du eine Idee, ob man da kurzfristig GPUs reservieren könnte, zum Beispiel acht GPUs für zwei bis drei Tage, oder wen ich dafür am besten frage? Damit könnte ich den kompletten Neulauf noch vor der Abgabe schaffen. Und falls das nicht klappt: Reicht dir der Kontrolllauf?

Ein paar kurze Fragen noch:

1. Wo würdest du die Arbeit im aktuellen Stand notenmäßig ungefähr sehen, und was würde am meisten fehlen, um noch eine Stufe besser zu werden?
2. Für die dritte Forschungsfrage (Ressourcenbudgets) habe ich keine festen Budgets gesetzt, sondern Latenz- und Speicherschwellen aus den Messwerten abgeleitet. Passt das für dich, oder erwartest du konkrete Einsatzszenarien?
3. Braucht es zusätzlich ein englisches Abstract?
4. Passt die KI-Angabe in der Eigenständigkeitserklärung so? Im Mustertext steht ja auch „ohne Hilfe Dritter“, verträgt sich das mit der KI-Nutzung?
5. Sind 85 Seiten Haupttext okay?
6. Weißt du, worauf Möller und Küpper bei der Bewertung besonders achten? Gibt es dafür ein Raster?
7. Ist es okay, wenn ich das Repository mit Code und Auswertungen öffentlich mache, oder lieber nur mit Zugang für die Gutachter?

Danke dir schon mal und sorry für die lange Mail!

Viele Grüße,
Luaj
