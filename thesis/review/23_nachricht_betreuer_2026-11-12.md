# Folgenachricht an den Betreuer (Philippe Thomas) – Stand 12.11.2026

Anhang: `thesis/main_betreuerentwurf_f233071.pdf` (Build aus Commit f233071: gekürzte Fassung mit eingebautem Kontrolllauf; 111 Seiten, Haupttext 77). Die erste Mail (Notiz 16) ging mit der langen Fassung (118 Seiten, Haupttext 85) raus; diese Nachricht ergänzt sie.

---

Betreff: BA: kürzere Fassung und Ergebnis des Kontrolllaufs

Hi Philippe,

kurzes Update zu meiner letzten Mail, zwei Dinge haben sich geändert.

Erstens habe ich die Arbeit gekürzt: 77 statt 85 Seiten Haupttext, 111 insgesamt. An den Ergebnissen ändert das nichts, rausgeflogen sind Wiederholungen und Grundlagen, die später nicht gebraucht werden. Die neue Fassung hängt an. Falls du mit der alten schon angefangen hast, lies ruhig dort weiter: Kapitel 4 und 5 sind bis auf den Kontrolllauf inhaltlich gleich.

Zweitens ist der Kontrolllauf durch. Für alle neun QLoRA-Läufe habe ich je 1 000 zufällige Testsätze mit 1 024 statt 256 Ausgabetokens neu erzeugt. 8 988 von 9 000 Antworten sind zeichengleich mit den alten, keine erreicht das neue Limit. Das stützt die Nachrechnung der abgeschnittenen Sätze und steht jetzt als Sensitivitätsanalyse in der Arbeit (Abschnitt 5.2 und Tabelle A.8).

Damit zu meiner Frage von neulich: Reicht dir das so, oder soll ich den kompletten Neulauf (rund 150 GPU-Stunden) trotzdem noch versuchen? Dafür bräuchte ich weiterhin die GPUs.

Danke dir und viele Grüße
Luaj
