# Prüfbericht Verdichtung und Kapitelabgrenzung (Bericht 28, 21.11.2026)

## Context

Auftrag: die gesamte Arbeit auf sinnvolle Verdichtung prüfen, mit Schwerpunkt auf der Abgrenzung von Grundlagen (Kap. 2), verwandten Arbeiten und Korpora (Kap. 3) und Methodik (Kap. 4). Erst Bericht, keine Änderung an Quellen, Tabellen, Abbildungen oder Anhang. Anlass ist die Begleitmail des Betreuers vom 17.11. (Trennung der drei Kapitel; Methodik stellenweise zu ausführlich).

Diese Datei **ist** der Prüfbericht. Mit der Freigabe des Plans passiert nur Schritt 1 unter „Vorgehen nach Freigabe" (Bericht im Repo ablegen). Textänderungen folgen erst, wenn du einzelne IDs freigibst.

---

## 1 Geprüfte Version und Umfang

- **Quellen:** Branch `lektorat-2026-11-25`, Commit `5fb2070` (21.11.2026, 17:54), Arbeitsbaum sauber, 7 Commits vor `origin` (nicht gepusht).
- **PDF:** `thesis/main.pdf` vom 21.11., 17:51, 112 Seiten, aus genau diesen Quellen gebaut. Haupttext S. 1–77 (Kap. 1: 5, Kap. 2: 12, Kap. 3: 13, Kap. 4: 22, Kap. 5: 20, Kap. 6: 5 Seiten).
- **Nicht verwendet (ältere Stände):** `thesis/main_long12112026.pdf` (lange Fassung, 118 S.), `main`/`origin/main` (12.11., 111 S.), `fassung-lang-2026-11-12`, `korrekturen-2026-11-25` (in den aktuellen Branch übernommen).
- **Gelesen:** Abstract und Kapitel 1 bis 6 vollständig im Quelltext; Anhang A.1 und A.2 vollständig, A.3 nach Beschriftungen (nur zum Abgleich, keine Vorschläge zum Anhang). Abschnitts- und Seitenzahlen aus `build/main.toc` und `build/main.aux`.
- **Nicht gemacht:** kein Neubau, keine visuelle Seitenprüfung. Zeilenangaben („04 Z. 300") beziehen sich auf die Kapiteldatei im Stand `5fb2070` und sind maßgeblich. Seitenangaben sind aus den Abschnittsanfängen abgeleitet und können um eine Seite abweichen.
- **Umfang Fließtext** (ohne Tabellen, Abbildungen, Listings): rund 22 450 Wörter (Kap. 1: 1 620, Kap. 2: 3 380, Kap. 3: 4 230, Kap. 4: 5 780, Kap. 5: 5 730, Kap. 6: 1 720).
- **Abgleich mit Bericht 24:** Dessen Kürzungen K-09 bis K-18 sind bis auf K-12 und K-16 (beide „wahlweise") umgesetzt. Seine Liste „nicht kürzen" habe ich beachtet; wo ich davon abweiche, steht es beim Vorschlag.

## 2 Kurzfazit

**Die Trennung der drei Kapitel ist seit dem Umbau vom 21.11. im Kern klar.** Kapitel 2 erklärt Begriffe, ohne Studien zu referieren; Kapitel 3 wiederholt keine Grundlagen; Kapitel 4 erklärt keine Grundlagen neu, sondern verweist knapp zurück. Die allgemeine Korpusbeschreibung (3.1) und die konkrete Nutzung (4.3) sind sauber getrennt.

Unscharf ist die Grenze noch an fünf Stellen:

1. **Die Begründung des Ausgabeformats steht in Kapitel 3** (3.3.1, rund 190 Wörter mit „drei Gründen"), nicht in 4.4.4. Herkunft und Einordnung des Formats werden zudem in fünf Abschnitten erzählt (1.2, 3.3.1, 4.3.2, 4.4.4, 5.7.2).
2. **Abschnitt 2.7 beschreibt zur Hälfte das eigene Design** (zweite „Grenze"), das 4.2.3 fast wortgleich noch einmal festlegt.
3. **Abschnitt 2.8 enthält zwei Methodenentscheidungen** (Wahl des strikten `seqeval`-Modus; was neben der Leistung gemessen wird), die in 4.5.1 und 4.5.2 wiederholt werden.
4. **Abschnitt 3.6 führt zwei Einschränkungen der eigenen Studie** als bloße Weiterverweise; die Kontamination wird über eine Verweiskette 1.3 → 3.6 → 5.7.1 → 3.6 erreicht.
5. **Kapitel 4 beschreibt die Prompt-Struktur viermal** (Kapiteleinleitung, 4.3.2, 4.4.3, 4.4.4).

Der größere Teil der Wiederholungen betrifft nicht die drei Kapitel untereinander, sondern Vorbehalte und Fakten, die Kapitel 4 an der Entscheidung nennt und 6.2 fast wörtlich wiederholt.

**Größenordnung:** verlustfrei etwa 540 Wörter (Kategorie A) und netto etwa 135 Wörter durch Umordnung (B), zusammen rund 3 % des Fließtexts. Mehr ist ohne Verlust nicht zu holen; die Arbeit ist nach zwei Kürzungsrunden dicht. Ob dabei eine Seite wegfällt, ist offen und nicht das Ziel. Der Gewinn liegt in der klareren Zuständigkeit der Kapitel.

## 3 Kapitelzuordnung

### Gut platziert (so lassen)

| Inhalt | Ort | Warum richtig |
|---|---|---|
| Begriffe NER, IOB2, Transformer, Subwörter, Decoding, LoRA/QLoRA, Metriken | 2.2–2.8 | Jeder Baustein wird in Kap. 3 bis 5 gebraucht (z. B. NF4, Double Quantization, Paged Optimizer für 4.4.5; Sampling-Parameter für 4.4.3; GQA und Gated DeltaNet für 4.4.5) |
| Herkunft, Annotation, Qualitätsstichprobe, bisherige Nutzung von MultiNERD und CrossNER | 3.1 | allgemeine Korpusbeschreibung; 4.3 verweist zurück |
| Auswahlkriterien, Filter, Splits, Typverteilung, Testnutzung | 4.3.1–4.3.3 | konkrete Nutzung mit Begründung |
| Wahl von DeBERTa-v3 und Qwen3.5 mit Begründung, Lizenzen, Parameterzahlen | 4.2 | Entscheidung der eigenen Studie |
| Einordnung früherer Formate (Tagging, Indexing, Matching) | 3.3.1 | Forschungsstand |
| Erwartungen E1 bis E3 | 3.6 | folgen aus der Literatur |
| Constrained Decoding: Begriff in 2.5, Entscheidung in 4.4.4, Deutung in 5.7.2 | – | Musterfall der gewünschten Dreiteilung |
| Literaturvergleich mit Zahlen | 5.7.1–5.7.3 | Zahlen stehen dort, wo verglichen wird; kein Zurückblättern nötig |

### Im falschen Kapitel oder doppelt verankert

| Inhalt | Steht in | Gehört nach | Vorschlag |
|---|---|---|---|
| Beschreibung und drei Gründe für das Wortindex-Format | 3.3.1 (03 Z. 313–336) | 4.4.4 | VB-01 |
| „Diese Arbeit passt veröffentlichte Checkpoints an …" (zweite Grenze der Skalierungsgesetze) | 2.7 (02 Z. 508–513) | 4.2.3 | VB-02 |
| Begründung für den strikten `seqeval`-Modus | 2.8 (02 Z. 567–569) | 4.5.1 | VB-03 |
| „Neben der Erkennungsleistung erfasst diese Arbeit den Ressourcenbedarf …" | 2.8 (02 Z. 595–599) | 4.5.2 | VB-04 |
| Abfolge Qwen2.5, Qwen3, Qwen3.5 (allgemeine Modellbeschreibung) | 4.2.2 (04 Z. 165–166) | 2.5 | VB-05 |
| Einschränkung „Vergleichsbedingungen" der eigenen Studie | 3.6 (03 Z. 769–772) | 1.3, 4.1, 4.7 (steht dort schon) | VA-05 |
| Vorwegnahme der Forschungslücke | 3.2.2 (03 Z. 250–252) | 3.6 (steht dort schon) | VA-02 |

### Zu den Verwandten Arbeiten insgesamt

Kapitel 3 referiert nicht nur: Jeder Block endet mit einer Synthese (Ende 3.2.2, 3.3.4, Ende 3.4), 3.5 liest die Arbeiten quer nach Modellgröße, Tabelle 3.2 vergleicht alle nach denselben Kriterien. Am stärksten aufzählend ist 3.3.2 (GPT-NER, Xie et al., PromptNER nacheinander); die Einordnung folgt erst in 3.3.4. Das ist vertretbar und nach zwei Kürzungsrunden nicht weiter verlustfrei zu straffen.

## 4 Wiederholungsübersicht

Je Thema: Fundstellen mit ihrer Funktion, bester Ort der vollständigen Erklärung, Behandlung der übrigen Stellen, was bleiben muss.

**W-01 Ausgabeformat: Herkunft, Einordnung, Begründung**
- Fundstellen: 1.2 (01 Z. 154–159, Beitrag) · 3.3.1 (03 Z. 283–285 Forschungsstand; Z. 313–336 Beschreibung und Begründung) · 4.3.2 (04 Z. 306–309, Nebenbemerkung) · 4.4.4 (04 Z. 670–684, Definition; nur ein Verweissatz zur Einordnung) · 5.7.2 (05 Z. 959–964, Neuheitsaussage). „Semin et al. haben die Wortindex-Variante nicht evaluiert" steht dreimal (03 Z. 285, Z. 333; 05 Z. 962).
- Vollständig nach: **4.4.4** (Beschreibung und Begründung). 3.3.1 behält Forschungsstand und einen Satz zur Einordnung samt Vorbehalt; 4.3.2 verliert den Satz; 5.7.2 verweist. 1.2 bleibt (Beitrag).
- Muss bleiben: Zuschreibung Ramponi (Nummerierung), Semin (Einordnung), „eigene Umsetzung"; die drei Gründe; der Vorbehalt „nicht evaluiert, nur Prompting"; „soweit bekannt, erstmals".
- Vorschläge: VB-01, VA-07, VA-13.

**W-02 Prompt- und Chat-Struktur innerhalb von Kapitel 4**
- Fundstellen: Kapiteleinleitung (04 Z. 20–22, Überblick) · 4.3.2 (Z. 300–305, als Vorverarbeitung) · 4.4.3 (Z. 627–629, Definition Single-Turn-SFT) · 4.4.4 (Z. 670–677, Definition) · Anhang A.2.4.
- Vollständig nach: **4.4.4**. 4.3.2 verweist und behält nur die Ableitung der Zielantwort.
- Muss bleiben: „Zählung ab 0", deterministische Ableitung der Zielantwort, Prüfung der Hin- und Rückkonvertierung.
- Vorschlag: VA-07. Die Kapiteleinleitung bleibt (vom Betreuer gewünschter Überblick).

**W-03 Silberreferenz und manuell geprüfte Stichprobe**
- Fundstellen: 1.3 (Geltungsbereich) · 2.8 (02 Z. 588–593, Deutung des F₁) · 3.1 (03 Z. 67–87, vollständige Beschreibung) · Tab. 3.1 · 4.3.1 (04 Z. 240–264, Wiederholung der Entstehung plus Folgen) · Tab. 4.9 · 5.1.2, 5.7.1 (Vorbehalt an den Werten) · 6.2 (Limitation) · 6.3 (Ausblick). Die Stichprobe („1 000 Sätze, mindestens 20 Instanzen je Typ") wird dreimal beschrieben (03 Z. 75–77, 04 Z. 246–248, 06 Z. 241–243). Der Begriff „Silberreferenz" wird zweimal eingeführt (2.8 und 3.1).
- Vollständig nach: **3.1** (Entstehung, Qualität), **4.3.1** (Folgen für diese Studie). 2.8 nur der allgemeine Satz; 4.3.1 und 6.3 verweisen auf 3.1.
- Muss bleiben: jeder Vorbehalt neben einem Ergebnis (5.1.2, 5.7.1), der Absatz in 6.2, die Begründung, warum die Stichprobe nicht genutzt wird.
- Vorschläge: VA-06, VA-16, VA-17g.

**W-04 Datenkontamination**
- Fundstellen: 1.3 (01 Z. 236–238, verweist auf 3.6) · 3.1 (03 Z. 100–102, verweist auf 5.7.1) · 3.6 (03 Z. 759–767, verweist auf 5.7.1) · 4.3.1 (04 Z. 223–225, verweist auf 5.7.1) · 5.7.1 (05 Z. 920–942, vollständig; verweist zurück auf 3.6) · 5.7.4, 6.2 (je ein Halbsatz).
- Vollständig nach: **5.7.1**. Je ein Verweis in 1.3 und 4.3.1 genügt; in Kapitel 3 einer statt zwei.
- Muss bleiben: der Hinweis auf Zhan et al. und CleanCoNLL (nur in 3.6), der ganze Absatz 5.7.1.
- Vorschlag: VA-05.

**W-05 „Konkrete Konfigurationen, keine isolierten Effekte"**
- Fundstellen: Abstract · 1.3 (Z. 217–223, vollständig) · 3.6 (Z. 769–772) · 4.1 (Z. 121–124, 139–142) · 4.4.3 (Z. 619–622) · 4.4.5 (Z. 787–788) · 4.6 (Z. 1238–1242) · 5.1.1, 5.2, 5.6, 5.7.1 (zweimal), 5.7.3 · 6.1 (zweimal) · 6.2 (zweimal) · Tab. 4.9. Insgesamt 16 Stellen mit „isolier…".
- Einordnung: Das sind überwiegend verschiedene Vorbehalte (Architektur, Anpassung, Domäne, Antwortlänge), jeweils neben der Zahl, die sie einschränken. Sie schützen vor Überdeutung.
- Vollständig nach: **4.1 mit Tab. 4.9** (Design) und **6.2** (Folgen). Entbehrlich sind nur 3.6 (falsches Kapitel) und 4.4.5 (steht wortnah in 6.2).
- Vorschläge: VA-05, VA-10. Alle übrigen Stellen bleiben.

**W-06 Testnutzung und Wahl der Zero-Shot-Einstellungen**
- Fundstellen: 4.3.3 (04 Z. 377–393, Text) · Tab. 4.3 · 4.4.3 (Z. 606–622, Hergang) · Tab. 4.9 · 5.1.1 (Z. 111–115) · 6.2 (Z. 124–148) · Abstract.
- Einordnung: Text, Tabelle und 6.2 überschneiden sich stark, aber jede Stelle trägt etwas: 4.3.3 die Einstufung, 4.4.3 den Hergang mit Zahlen, 6.2 die Folgen je Forschungsfrage. Bericht 24 hat 4.3.3 als „notwendig ausführlich" eingestuft; dabei bleibe ich.
- Verlustfrei ist nur ein doppelter Satz in 4.3.3 (VA-17e).

**W-07 Checkpoint-Auswahl (200 Sätze, Verlustminimum, Untertraining)**
- Fundstellen: 4.4.3 (04 Z. 639–662, alle Fakten) · Tab. 4.9 · 5.1.1 (Verlauf über die Epochen) · 6.2 (06 Z. 140–144 und Z. 193–196, dieselben Fakten fast wörtlich).
- Vollständig nach: **4.4.3** (Fakten), **6.2** (Folgerung mit Rückverweis).
- Muss bleiben: in 6.2 die Folgerungen „bleibt offen" und „Untertraining nicht auszuschließen".
- Vorschlag: VA-14.

**W-08 Skalierungsgesetze und deskriptiver Größenvergleich**
- Fundstellen: 2.7 (02 Z. 499–515) · 3.5 (Z. 558–561, Aufhänger; Z. 590–594, eigenes Design) · 3.6 (Z. 751–757) · 4.2.2 (Z. 172–175) · 4.2.3 (Z. 200–209) · 5.3 (Z. 449–450) · 5.7.3 (Z. 1039–1040) · 6.1 (Z. 60–61).
- Vollständig nach: **2.7** (was die Gesetze beschreiben) und **4.2.3** (was daraus für diese Studie folgt). 3.5 kürzer; 5.3, 5.7.3 und 6.1 sind knappe Erinnerungen und bleiben.
- Vorschläge: VB-02, VA-04, VA-08.

**W-09 Strikte Entitätsmetrik, `seqeval`, Mikro und Makro**
- Fundstellen: „Grenzen und Typ" in 1.3, 2.2, 2.8, 4.5.1 (zweimal), Tab. 5.1 · strikter Modus in 2.8 (Z. 555–569) und 4.5.1 (Z. 888–891, gleicher Nebensatz) · Makro-Definition in 2.8, 4.5.1, Tab. 5.1, 5.1.2 · „seltene Typen unsicher" in 2.8, 4.5.1, 5.1.2, 5.7.4, 5.7.5.
- Vollständig nach: **2.8** (Definitionen), **4.5.1** (Wahl und Begründung). 1.3 und 2.2 bleiben (erste Erklärung von „strikt" bzw. Teil der Aufgabendefinition).
- Vorschläge: VB-03, VA-17c, VA-17d.

**W-10 Forschungslücke und Kurzbeschreibung des eigenen Designs**
- Fundstellen: Abstract · 1.2 (Z. 98–125, 147–161) · 1.3 (Z. 182–199) · 3.2.2 (Z. 250–252) · 3.6 (Z. 723–735, 751–757, Tab. 3.2 letzte Zeile) · Kapiteleinleitung 4 · 4.1 · Einleitung Kap. 6 · 6.1 „Beitrag".
- Einordnung: weitgehend strukturbedingt (Einleitung, Lücke, Methodenüberblick, Fazit). Vollständig in **4.1 mit Tab. 4.1** (Design) und **3.6** (Lücke).
- Verlustfrei entbehrlich: 3.2.2 (VA-02) und ein Satz in 1.3 (VA-17f). Wahlweise 3.6 straffen (VC-02).

**W-11 Aufbau des CrossNER-Transfertests**
- Fundstellen: 1.2 (Z. 163–170) · 3.1 (Z. 104–118) · 3.3.4 (Z. 446–455, sagt dasselbe in Satz 1 und 3) · 4.6 (Z. 1163–1172, Domänen erneut aufgezählt).
- Vollständig nach: **3.1** (Korpus), **4.6** (Versuch).
- Vorschläge: VA-01, VA-11.

**W-12 Ressourcenerfassung früherer Arbeiten**
- Fundstellen: je Arbeit im Text (3.2.2, 3.3.3, 3.4) · 3.3.4 (Z. 442–445) · Tab. 3.2, Spalte „Ressourcen gemessen" · 3.6 (Z. 737–750).
- Vollständig nach: **Tab. 3.2 und 3.6**. 3.3.4 ist entbehrlich (VA-03).

**W-13 Literaturzahlen (65,17/90,98; 93,85/93,04; Faktor 40 bis 220)**
- Fundstellen: 1.2, 3.2.1 bzw. 3.4, 3.6 (E2), 5.7.1, 5.7.2, 6.1.
- Einordnung: Motivation, Forschungsstand, Abgleich. Jede Stelle braucht die Zahl selbst. **Nicht ändern.**

**W-14 Ausgabelimit, „kein vollständiger Neulauf", Hardware, Batchgröße 1**
- Fundstellen: 4.4.3, 4.5.4, Tab. 4.3/4.4/4.9, 5.2, 5.4, 5.6, 5.7.1, 5.7.4, 6.1, 6.2 („GPU-Typ" 25-mal).
- Einordnung: Vorbehalte neben den Zahlen; von Bericht 24 ausdrücklich geschützt. **Nicht ändern**, bis auf den Eingangsabsatz von 5.7.4, der Ergebnisse aus 5.4 nacherzählt (VA-12).

**W-15 LogitMatch**
- Fundstellen: 5.7.2 (05 Z. 981–986, erklärt) · 6.3 (06 Z. 262–266, erneut erklärt). Vollständig in **5.7.2** (VA-15).

## 5 Konkrete Vorschläge

Felder je Vorschlag: Fundstelle · Original · Problem · Vorschlag · bleibt erhalten · Verweise/Quellen · Risiko · Einsparung. Alle A-Vorschläge erhöhen die Informationsdichte; bei B und C steht es dabei.

### A – verlustfreie Verdichtung

**VA-01 · 3.3.4, CrossNER-Einordnung (03 Z. 446–455, S. 25)**
- Original: „Für den CrossNER-Transfertest … liefern PromptNER und GoLLIE Orientierung, aber keine Vergleichswerte: … Der Versuch dieser Arbeit ist dagegen ein reiner Transfertest … Die von beiden Arbeiten berichteten CrossNER-Werte belegen deshalb nur, dass …; welches Niveau … zu erwarten ist, zeigen sie nicht."
- Problem: Satz 1 und Satz 3 sagen dasselbe („keine Vergleichswerte" / „zeigen sie nicht").
- Vorschlag: „Für den CrossNER-Transfertest dieser Arbeit (Abschnitt 4.6), einen reinen Transfertest ohne Zielbeispiele auf den Grobtypen PER, ORG und LOC, liefern PromptNER und GoLLIE keine Vergleichswerte: PromptNER sieht zwei Beispiele aus der Zieldomäne, GoLLIE trainiert auf den Richtlinien vieler Datensätze, und beide werten die domänenspezifischen Typinventare aus. Ihre Werte belegen nur, dass der Domänenwechsel auch für generative Modelle eine erhebliche Hürde bleibt."
- Erhalten: beide Unterschiede, die Folgerung. · Verweise: unverändert. · Risiko: keines. · rund −30 Wörter.

**VA-02 · 3.2.2, Schlusssatz (03 Z. 250–252, S. 22)**
- Original: „Keine der Arbeiten stellt beide Modellfamilien feinjustiert in derselben Pipeline auf den offiziellen MultiNERD-Splits mit gemeinsamer Ressourcenmessung gegenüber."
- Problem: nimmt die Forschungslücke aus 3.6 vorweg (dort 03 Z. 715–717 und Z. 748–750), in einem Abschnitt, der nur Encoder-Arbeiten behandelt.
- Vorschlag: streichen. · Erhalten: Aussage steht vollständig in 3.6 und 1.2. · Risiko: Der Abschnitt endet dann mit der Einschränkung der Gleichstände, was als Schluss trägt. · −22 Wörter.

**VA-03 · 3.3.4, Ressourcensatz (03 Z. 442–445, S. 25)**
- Original: „UniversalNER misst keinen Trainings- oder Inferenzaufwand; GoLLIE dokumentiert Trainingszeit und CO₂ je Größe, aber keine Inferenzlatenz und keinen Encoder-Vergleich [Anh. D.3, Tab. 10]."
- Problem: steht in Tab. 3.2 (Spalte Ressourcen) und in 3.6 (Z. 743–745).
- Vorschlag: streichen; die Fundstelle „Anh. D.3, Tab. 10" an das GoLLIE-Zitat in 3.6 hängen.
- Erhalten: Aussage und genaue Quellenstelle. · Risiko: keines. · −24 Wörter.

**VA-04 · 3.5, Schlusssatz (03 Z. 590–594, S. 28)**
- Original: „Die vorliegende Arbeit betrachtet die Modellgröße daher innerhalb jeder Familie getrennt und deskriptiv, mit zwei Encoder- und drei LLM-Größen (Abschnitt 4.2.3), die über den Größenbereich der direkten Vergleiche nach unten und oben hinausgehen (Tabelle 4.1)."
- Problem: Der zweite Teil steht in 3.6 (Z. 751–755) und 4.2.2 (Z. 172–175).
- Vorschlag: „Die vorliegende Arbeit betrachtet die Modellgröße daher innerhalb jeder Familie getrennt und deskriptiv (Abschnitt 4.2.3)."
- Erhalten: die Folgerung aus dem Abschnitt. · Risiko: keines. · −22 Wörter.

**VA-05 · Einschränkungsverweise in 3.1, 3.6 und 1.3**
- Fundstellen: 03 Z. 100–102 (S. 19); 03 Z. 769–772 (S. 30); 01 Z. 236–238 (S. 5).
- Problem: Kontamination wird in Kapitel 3 zweimal nur weiterverwiesen; 1.3 verweist auf 3.6, das selbst auf 5.7.1 verweist, und 5.7.1 verweist zurück. Der Absatz „Vergleichsbedingungen" in 3.6 ist eine Einschränkung der eigenen Studie, die in 1.3 (Z. 217–223), 4.1 und Tab. 4.9 vollständig steht.
- Vorschlag: (a) in 3.1 den Satz „Die mögliche Datenkontamination … diskutiert Abschnitt 5.7.1." streichen; (b) in 3.6 den Absatz „Die zweite Einschränkung betrifft die Vergleichsbedingungen …" streichen und den vorigen mit „Eine Einschränkung betrifft …" beginnen; (c) in 1.3 den Verweis von 3.6 auf 5.7.1 umstellen.
- Erhalten: Hinweis auf Zhan et al. und CleanCoNLL in 3.6; vollständige Diskussion in 5.7.1; Vorbehalt der Vergleichsbedingungen in 1.3, 4.1, 4.7.
- Verweise: `sec:research_gap` → `sec:disk-leistung` in 01 Z. 238. · Risiko: 3.6 ordnet die Lücke danach ohne den relativierenden Satz ein; der Vorbehalt steht aber in der Einleitung und im Methodenkapitel. · rund −55 Wörter.

**VA-06 · 4.3.1, Entstehung der Annotation und Stichprobe (04 Z. 240–248, S. 35)**
- Original: „MultiNERD ist automatisch annotiert: Verlinkte Textstellen … BabelNet … iterativ bereinigt (Abschnitt 3.1). Auch die Referenz des Testsplits ist somit eine Silberreferenz. Die von den Autoren manuell geprüfte Stichprobe von 1 000 zufällig gezogenen englischen Sätzen mit mindestens 20 Instanzen je Typ wird nicht verwendet."
- Problem: wiederholt die allgemeine Korpusbeschreibung aus 3.1 (03 Z. 67–77) im Methodenkapitel.
- Vorschlag: „Alle Splits von MultiNERD sind automatisch annotiert (Abschnitt 3.1); auch die Referenz des Testsplits ist somit eine Silberreferenz. Die manuell geprüfte Stichprobe der Autoren (Abschnitt 3.1) wird nicht verwendet."
- Erhalten: Entscheidung und alle drei Gründe (folgende Sätze unverändert), ebenso Z. 256–264, die Bericht 24 geschützt hat.
- Quellen: Das Zitat `tedeschi2022multinerd` bleibt über „[Tab. 5]" im Absatz. · Risiko: keines. · −35 Wörter.

**VA-07 · 4.3.2, Absatz „LLM-Pipeline" (04 Z. 300–309, S. 36)**
- Original: drei Sätze zur Chat-Struktur und zur nummerierten Tokenliste, dazu „Die Nummerierung der Eingabetokens nach Ramponi … erlaubt die Rückabbildung …".
- Problem: Die Prompt-Struktur ist in 4.4.4 definiert, die Herkunft der Nummerierung in 3.3.1 (bzw. nach VB-01 in 4.4.4).
- Vorschlag: „Für die LLM-Pipeline wird jedes Beispiel in das Prompt-Format aus Abschnitt 4.4.4 überführt (Listing 4.1); im Training kommt eine Assistentennachricht mit der Zielantwort hinzu." Danach unverändert ab „Die Zielantwort wird deterministisch …". In 4.4.4 hinter „im Format `Index: Token`" ergänzen: „(Zählung ab 0)".
- Erhalten: Ableitung der Zielantwort, Prüfung der Konvertierung, Zählung ab 0. · Quellen: `ramponi2025fallacy` und `semin2026spanLabeling` bleiben in 3.3.1/4.4.4 zitiert.
- Risiko: Vorwärtsverweis auf 4.4.4; den gibt es an dieser Stelle schon („den vier Feldern aus Abschnitt 4.4.4"). · rund −50 Wörter.

**VA-08 · 4.2.3, Schlusssatz (04 Z. 208–209, S. 34)**
- Original: „Leitfrage ist das Verhältnis von Leistungsgewinn und Ressourcenbedarf größerer LLMs gegenüber kompakten Encodern [Keraghel, Obeidat]."
- Problem: wiederholt FF2 und FF3 aus 1.2. · Vorschlag: streichen. · Quellen: beide anderswo zitiert. · Risiko: keines. · −17 Wörter.

**VA-09 · 4.4.5, Satz zur Adapter-Präzision (04 Z. 772–775, S. 44)**
- Original: „Die gespeicherten Adapter der 0.8B- und 4B-Läufe liegen in BF16 vor, die der 27B-Läufe nach der Wiederaufnahme aus Checkpoints … in FP32 (Anhang A.2.7)."
- Problem: steht wortgleich in Anhang A.2.7 und als Zeile in Tab. 4.9; für das Verständnis des Designs nachrangig.
- Vorschlag: Satz streichen; in Tab. 4.9, Zeile „Adapter-Präzision", die Spalte „Abschnitt" auf „Anhang A.2.7" setzen.
- Erhalten: Angabe in Tab. 4.9 und im Anhang. · Risiko: berührt eine Zelle von Tab. 4.9 (nur der Verweis); braucht deine Freigabe. · −28 Wörter.

**VA-10 · 4.4.5, Schlusssatz (04 Z. 787–788, S. 44)**
- Original: „Der Einfluss von Double Quantization, anderer Adapterplatzierung oder eines auf die Antwort beschränkten Verlusts wurde nicht isoliert gemessen."
- Problem: steht mit denselben Punkten in 6.2 (06 Z. 185–192) und in Tab. 4.9.
- Vorschlag: streichen oder durch „(Folgen in Abschnitt 6.2)" am vorangehenden Satz ersetzen.
- Erhalten: die Einschränkung in 6.2. · Risiko: gering; wer nur Kapitel 4 liest, findet die Abweichungen weiter in Tab. 4.9. · −18 Wörter.

**VA-11 · 4.6, Einleitung (04 Z. 1165–1167, S. 50)**
- Original: „Grundlage ist CrossNER mit den Domänen Künstliche Intelligenz, Literatur, Musik, Politik und Naturwissenschaften und jeweils domänenspezifischen Entitätstypen [Liu]."
- Problem: Aufzählung steht in 3.1 und in Tab. 5.6. · Vorschlag: „Grundlage ist CrossNER [Liu] (Abschnitt 3.1)." · Risiko: keines. · −17 Wörter.

**VA-12 · 5.7.4, Eingangsabsatz (05 Z. 1045–1056, S. 70)**
- Original: „Auf MultiNERD erreichen die Encoder die höchsten Mikro-F₁-Werte bei der geringsten Inferenzlatenz, und DeBERTa-v3-base benötigt zugleich den geringsten Inferenzspeicher … Der hardwaregleiche Vergleich … Latenzfaktor von etwa 210; Faktoren zwischen anderen Größen enthalten … Hardwareeffekte." Der nächste Absatz beginnt: „Bei 256 Ausgabetokens bilden die beiden Encoder die Pareto-Front (Abschnitt 5.4) …"
- Problem: erzählt 5.4 nach (dort Z. 503–515 und 545–549) und sagt die Pareto-Aussage zweimal hintereinander.
- Vorschlag: „FF3 fragt nach dem Verhältnis von Erkennungsleistung und Ressourcenbedarf unter Speicher- und Latenzbeschränkungen. Bei 256 Ausgabetokens bilden die beiden Encoder die Pareto-Front (Abschnitt 5.4, Abbildung 5.3); hardwaregleich benötigt Qwen3.5-0.8B etwa das 210-Fache der Latenz von DeBERTa-v3-base. Die Nachinferenz am Ausgabelimit verändert den Qualitätsvergleich, weil …" (weiter wie bisher).
- Erhalten: Pareto-Aussage, Faktor 210. Der Hardware-Vorbehalt zu den übrigen Faktoren steht in 5.4 neben diesen Faktoren. · Risiko: keines; hilft dem knappen Seitenstand von Kapitel 5. · rund −45 Wörter.

**VA-13 · 5.7.2, Herkunft der Wortindex-Variante (05 Z. 959–964, S. 68)**
- Vorschlag: „Nummerierte Eingabetokens setzen Ramponi et al. [Tab. 3, Tab. 9] für die Spanerkennung ein; Semin et al. beschreiben die Wortindex-Variante, ohne sie zu evaluieren (Abschnitt 3.3.1). Hier wird sie, soweit bekannt, erstmals mit QLoRA und strikter Entitätsmetrik auf MultiNERD geprüft."
- Erhalten: Neuheitsaussage mit Einschränkung, beide Quellenstellen. · Risiko: keines. · −14 Wörter.

**VA-14 · 6.2, wiederholte Fakten zur Checkpoint-Auswahl (06 Z. 140–144 und Z. 193–196, S. 75–76)**
- Original: „… bei den LLMs 200 Sätze, deren Typanteile um höchstens 3,1 Prozentpunkte vom Split abweichen und die keine BIO- und INST-Entitäten enthalten; ob …" und „Die nach F₁ gewählte Epoche fällt nur in einem von neun LLM- und in keinem von sechs Encoder-Läufen mit dem Verlustminimum zusammen; 12 von 15 Läufen wählten die letzte Epoche, ein Untertraining ist nicht auszuschließen."
- Problem: dieselben Zahlen stehen in 4.4.3 (04 Z. 647–656).
- Vorschlag: „… bei den LLMs nur die ersten 200 Sätze (Abschnitt 4.4.3); ob eine größere oder zufällig gezogene Stichprobe dieselben Epochen gewählt hätte, bleibt offen." und „Da 12 von 15 Läufen die letzte Epoche wählten (Abschnitt 4.4.3), ist ein Untertraining nicht auszuschließen."
- Erhalten: beide Folgerungen in 6.2, alle Zahlen in 4.4.3. · Risiko: 6.2 ist weniger selbsttragend; der Rückverweis gleicht das aus. · −34 Wörter.

**VA-15 · 6.3, LogitMatch (06 Z. 262–266, S. 77)**
- Vorschlag: „Beim Decoding könnte LogitMatch [Semin, Abschn. 4] die freie Erzeugung numerischer Wortindizes ersetzen (Abschnitt 5.7.2)."
- Erhalten: Verfahren und Begründung stehen in 5.7.2 (05 Z. 981–986). · Risiko: keines. · −22 Wörter.

**VA-16 · 2.8, Silberreferenz-Satz (02 Z. 588–593, S. 17)**
- Vorschlag: „Ein F₁-Score misst damit die Übereinstimmung mit der Referenz und nur mittelbar die Richtigkeit; bei einer automatisch erzeugten Referenz wie der von MultiNERD (Abschnitt 3.1) ist diese Unterscheidung wesentlich."
- Erhalten: der allgemeine Befund nach Hripcsak; die Anwendung steht in 4.3.1 (Z. 263–264) und 6.2. Der Begriff „Silberreferenz" wird dann nur einmal eingeführt (3.1). · Risiko: keines. · −20 Wörter.

**VA-17 · Kleinstellen (Sammelposten, zusammen rund −90 Wörter)**
- a) 2.5 (02 Z. 322–324): „; die Keys und Values der bisherigen Tokens werden dabei zwischengespeichert" streichen, Verweis auf 2.3 behalten (steht in 2.3, Z. 211–212).
- b) 2.2 (02 Z. 146–147): Satz „Die Encoder-Pipeline dieser Arbeit folgt der Sequenzlabeling-Formulierung mit IOB2-Tags" mit Z. 113–115 zusammenziehen (zweimal „diese Arbeit verwendet IOB2" im selben Abschnitt).
- c) 4.5.1 (04 Z. 943–945): Satz zu seltenen Typen an Z. 923–924 anhängen: „…; bei seltenen Typen sind sie wegen des geringen Supports besonders unsicher (Abschnitt 2.8, Tabelle 4.2)."
- d) 4.5.1 (04 Z. 947–950): Makro-Definition auf „Ergänzend wird der Makro-F₁ über alle 15 Typen berichtet (Abschnitt 2.8); er reagiert stark auf die seltenen Typen." kürzen.
- e) 4.3.3 (04 Z. 390–391): „Ihre Konfiguration wurde nach Testbeobachtungen gewählt." streichen; der Satz davor sagt dasselbe.
- f) 1.3 (01 Z. 192–193): „Auf der LLM-Seite werden drei Größen der Qwen3.5-Familie untersucht, die einen Vergleich innerhalb einer Modellfamilie erlauben." streichen (steht in 1.2); Folgesatz mit „Die Qwen3.5-Modelle werden …" beginnen.
- g) 6.3 (06 Z. 241–243): „die 1 000 Sätze mit mindestens 20 Instanzen je Typ, die … nutzen" → „die manuell geprüfte Stichprobe von Tedeschi et al. (Abschnitt 3.1)".
- h) 5.7.4 (05 Z. 1112–1115): „Die Zero-Shot-Konfigurationen erreichen … wesentlich niedrigere Werte; das 27B-Modell erzielt unter ihnen den höchsten F₁, doch" → „Ob ein Zero-Shot-Einsatz des 27B-Modells für Anwendungen ohne Trainingsdaten taugt und welche Prompt- und Decoding-Wahl optimal wäre, wurde nicht geprüft."

### B – sinnvolle Umordnung

**VB-01 · Begründung des Ausgabeformats von 3.3.1 nach 4.4.4** (höchste Priorität)
- Fundstelle: 03 Z. 313–336 (S. 23) → 04 Z. 682–684 (S. 42).
- Problem: Eine Methodenentscheidung wird im Forschungsstand beschrieben und begründet; 4.4.4 verweist dafür zurück.
- Neuer Schluss von 3.3.1: „Die Ausgabestrategie dieser Arbeit gehört zur Indexing-Familie in der angereicherten Variante auf Wortebene (Abschnitt 4.4.4). Semin et al. haben diese Variante nicht evaluiert, und ihre Befunde beruhen auf per Prompting genutzten, nicht feinjustierten Modellen; ob die Vorteile der angereicherten Indizierung auch im QLoRA-Regime und bei 15 Typen tragen, ist eine empirische Frage dieser Arbeit."
- Neu in 4.4.4 (statt des Verweissatzes): Einordnung nach Semin et al. [Abschn. 3.2] mit Rückverweis auf 3.3.1, Zuschreibung an Ramponi et al. [Tab. 9], „JSON-Schema und Nachverarbeitung sind eigene Umsetzungen", dann die drei Gründe im bisherigen Wortlaut (03 Z. 322–331).
- Erhalten: alle drei Gründe, alle Zuschreibungen, der Vorbehalt. Es entfällt nur die doppelte Formatbeschreibung (03 Z. 314–316).
- Verweise: 05 Z. 952–953 („die Erwartung aus Abschnitt 3.3") auf 4.4.4 umstellen; der Verweis auf 4.4.6 wandert mit.
- Risiko: Kapitel 4 wächst um etwa sieben Zeilen, Kapitel 3 schrumpft um etwa zehn. Die Zeilen in Kapitel 4 werden durch VA-06 bis VA-11 mehr als ausgeglichen; trotzdem Probesatz nötig (Listing 4.1 und Tab. 4.6 liegen in der Nähe).
- Bilanz: Kap. 3 rund −150, Kap. 4 rund +95, netto −55 Wörter. Dichte steigt, weil Beschreibung und Begründung an einer Stelle stehen.

**VB-02 · Zweite „Grenze" aus 2.7 nach 4.2.3**
- Fundstelle: 02 Z. 503–515 (S. 15) → 04 Z. 203–207 (S. 34).
- Neu 2.7: „Die Gesetze beschreiben allerdings den Verlust des Vortrainings von Modellen, die eigens für die Untersuchung trainiert wurden; über Kennzahlen einer nachgelagerten Aufgabe wie den F₁-Score bei NER sagen sie nichts unmittelbar aus. Empirische Befunde zum Größeneinfluss bei NER fasst Abschnitt 3.5 zusammen; was daraus für den Größenvergleich dieser Arbeit folgt, legt Abschnitt 4.2.3 fest."
- Neu 4.2.3: „Die Arbeit passt veröffentlichte Checkpoints fester Größe an, deren Vortraining nicht variiert werden kann, und mit der Größe ändern sich zugleich Hyperparameter und Hardware. Mit zwei Encoder- und drei LLM-Größen lassen sich Größenunterschiede deshalb zwar gegen die Variabilität zwischen den Seeds abgrenzen, Skalierungsgesetze im Sinne angepasster Potenzgesetze [Kaplan, Hoffmann] (Abschnitt 2.7) aber nicht bestimmen; beobachtete Unterschiede sind als Unterschiede zwischen Konfigurationen zu lesen."
- Erhalten: beide Grenzen, die Lesart. · Verweise: 05 Z. 1039–1040 auf „Abschnitte 2.7 und 4.2.3" erweitern.
- Risiko: 2.7 hat danach nur noch rund 80 Wörter (siehe VC-05). · netto rund −25 Wörter. Dichte steigt.

**VB-03 · Begründung des strikten Modus von 2.8 nach 4.5.1**
- Fundstelle: 02 Z. 555–569 (S. 16–17) → 04 Z. 888–891 (S. 45).
- Neu 2.8: Absatz bis „… sodass unzulässige Folgen keine Entität bilden." Es entfallen „Unterstützt werden IOB1, IOB2, IOE1 und IOE2 sowie im strikten Modus IOBES und BILOU" (für die Arbeit ohne Belang) und der Schlusssatz „Für den Vergleich generativer Modelle …" (wandert).
- Neu 4.5.1: „Berechnet werden die Werte mit `seqeval` im strikten Modus mit IOB2-Schema (Abschnitt 2.8); für den Vergleich mit generativen Modellen, deren Ausgaben erst in Tagfolgen überführt werden, ist das die eindeutigere Wahl."
- Erhalten: Definition beider Modi in 2.8, Wahl und Begründung in 4.5.1. Verloren geht nur die Liste der unterstützten Schemata.
- Risiko: **4.5.1 ist umbruchkritisch** (eine Zusatzzeile hat am 21.11. eine Seite gekostet). Nur zusammen mit VA-17c und VA-17d umsetzen, die dort mehr sparen, als hier dazukommt. · netto rund −28 Wörter.

**VB-04 · Schlussabsatz von 2.8 nach 4.5.2**
- Fundstelle: 02 Z. 595–599 (S. 17) → 04 Z. 1015–1019 (S. 48).
- Vorschlag: Absatz in 2.8 streichen. In 4.5.2 („Parameterzahlen"): „Ihr Verhältnis zeigt nur, wie stark das Adaptertraining die Zahl der aktualisierten Parameter senkt [Hu, Dettmers]; über Inferenzspeicher, Latenz und Trainingsdauer sagt es nichts aus, weshalb diese gesondert gemessen werden."
- Erhalten: der Gedanke vollständig, an der Stelle der Messung. · Risiko: keines. · netto rund −22 Wörter.

**VB-05 · Qwen-Linie von 4.2.2 nach 2.5** (niedrige Priorität)
- Fundstelle: 04 Z. 165–166 → 02 Z. 372–378.
- Vorschlag: „Qwen3.5 folgt … auf Qwen2.5 und Qwen3" samt den drei Zitaten in den Qwen-Satz von 2.5 ziehen; 4.2.2 beginnt mit Auswahl, Verweis auf 2.5 und Lizenz.
- Verweise/Quellen: `qwen2024qwen25` ist nur hier zitiert und muss mitwandern. · netto rund −5 Wörter; reine Ordnung.

### C – möglicherweise kürzbar, fachlich zu prüfen

- **VC-01 · 2.2, BIOES/BILOU (02 Z. 109–110):** Der Satz wird nirgends wieder gebraucht (einzige weitere Nennung ist die Schemaliste, die mit VB-03 entfällt). −22 Wörter. Entscheidung: Soll das Grundlagenkapitel die Varianten der Vollständigkeit halber nennen? Meine Empfehlung: streichen, wenn VB-03 umgesetzt wird.
- **VC-02 · 3.6, zwei Blöcke „Die vorliegende Arbeit …" (03 Z. 723–735, 751–757):** wiederholen 1.2 und die letzte Zeile von Tab. 3.2. Kürzbar um rund 50 Wörter, wenn man auf die Tabellenzeile verweist. Dagegen spricht die Scharnierfunktion des Abschnitts (Lücke, dann eigene Antwort). Empfehlung: nur kürzen, wenn der Betreuer Kapitel 3 noch als zu lang ansieht.
- **VC-03 · Lehrbuchnahe Stellen in Kapitel 2 ohne spätere Verwendung:** Representation Learning und Wortvektoren (2.1, Z. 27–37), WordPiece und Byte-Paar-Kodierung (2.4, Z. 235–236), Next Sentence Prediction (Z. 251–252), Enhanced Mask Decoder (Z. 281–284), 160 GB Vortrainingsdaten (Z. 294–296). Zusammen etwa 150 Wörter, kürzbar vielleicht die Hälfte. Empfehlung: belassen. Sie beschreiben die verwendeten Modelle, zwei Quellen (Bengio, Goldberg) wären sonst unzitiert, und 2.3 sowie 2.7 sind bereits gekürzt.
- **VC-04 · Titel von Kapitel 3:** Das Kapitel enthält seit dem 21.11. die Korpora, heißt aber „Verwandte Arbeiten". „Verwandte Arbeiten und Korpora" wäre genauer (berührt Kolumnentitel und Inhaltsverzeichnis, 1.4 passt schon). Deine Entscheidung.
- **VC-05 · 2.7 als eigener Abschnitt:** Nach VB-02 bleiben rund 80 Wörter. Möglich wäre, sie als Schlussabsatz in 2.5 aufzunehmen (2.8 würde 2.7). Empfehlung so kurz vor der Abgabe: Abschnitt behalten.
- **VC-06 · 4.4.3, Diagnose der Checkpoint-Wahl (04 Z. 644–656):** enthält Laufergebnisse (gewählte Epochen, Verlustminimum, auch der Encoder-Läufe) und wäre streng genommen Ergebnis (5.1.1). Empfehlung: nicht verschieben. Bericht 24 stuft den Absatz als notwendig ein, Kapitel 5 endet am Seitenende, und die Angaben dokumentieren die Durchführung.
- **VC-07 · 1.3 weiter straffen:** 1.3 wiederholt Teile von 1.2 (Modelle, Datenbasis, Metrik). Über VA-17f hinaus ist wenig verlustfrei: Die Definition von „strikt" in 1.3 ist die erste in der Arbeit, und die QLoRA-Erklärung trägt die Abweichung vom Titel.

### Randbefunde (keine Kürzung)

- **P-1:** Beschriftung von Tab. 3.1 („Vergleich der verwendeten NER-Datensätze"): CoNLL-2003 und OntoNotes werden laut 1.3 nicht verwendet. Genauer: „der für diese Arbeit relevanten NER-Datensätze".
- **P-2:** 4.4.3 nennt min-p und Presence-Penalty; 2.5 erklärt nur Temperatur, Top-k und Top-p.
- **P-3:** CodeIE wird erst im Zwischenfazit 3.3.4 eingeführt (und in Tab. 3.2), nicht in 3.3.2.
- **P-4:** Aus Bericht 24 sind K-12 (Spannen aus Tab. 5.4 im Text von 5.4) und K-16 (Parameterzahlen in der Beschriftung von Abb. 5.2) offen; Einschätzung unverändert „wahlweise".

## 6 Prioritäten

1. **VB-01** (Formatbegründung nach 4.4.4) mit **VA-07** und **VA-13**: größter Gewinn für die Trennung von Kapitel 3 und 4; beseitigt die fünffache Formatherleitung.
2. **VB-02, VB-03 (mit VA-17c/d und VC-01), VB-04, VA-16:** nimmt die eigene Studie aus den Grundlagen 2.7 und 2.8.
3. **VA-05, VA-02, VA-03, VA-04, VA-01:** Kapitel 3 endet ohne vorweggenommene Lücke, ohne Einschränkungen der eigenen Studie und ohne Verweiskette.
4. **VA-06, VA-08, VA-09, VA-10, VA-11:** antwortet auf „Methodik stellenweise zu ausführlich", ohne eine Angabe zur Durchführung zu verlieren.
5. **VA-12, VA-14, VA-15, VA-17:** Wiederholungen zwischen Kapitel 4, 5.7 und 6.
6. Wahlweise: VB-05, VC-02, VC-04, P-1 bis P-3.

Bei fünf Tagen bis zur Abgabe empfehle ich die Pakete 1 bis 4 in einem Durchgang mit Probesatz; Paket 5 kostet wenig und kann mitlaufen.

## 7 Nicht kürzen

- **Anhang:** unverändert, nichts dorthin verschoben.
- **Kapiteleinleitung 4 und 4.1:** Überblick und Detail überschneiden sich, der Überblick ist vom Betreuer gewünscht.
- **4.3.3 (Testnutzung) mit Tab. 4.3, 4.4.3 (Längengrenzen, Zero-Shot-Hergang, Checkpoint-Auswahl), 4.5.1 (Bootstrap), 4.5.4 (Nachinferenz):** tragen Einstufung und Nachrechenbarkeit; jede Angabe wird in Kapitel 5 oder 6 gebraucht.
- **4.4.5, Platzierung der Adapter und Abweichungen vom QLoRA-Rezept; 4.4.6, Validierungskette; Tab. 4.7:** Regeln und Zähler spiegeln sich, aber die Tabelle ordnet die Zähler den Fehlerklassen zu und wird zum Lesen von Tab. 5.2 gebraucht.
- **4.5.2, Trainingsdauer der 27B-Läufe:** steht ausführlicher in Anhang A.2.8, der Kern (vier Zuteilungen, geteilter Knoten, mehr als doppelte Schrittzeit) wird zum Lesen der 266 h gebraucht.
- **4.3.1, Z. 256–264 (gleiche Trainingsdaten, gleiche Referenz, aber …):** von Bericht 24 geschützt; bleibt.
- **Tabellen 4.3, 4.4, 4.9 und 3.2; alle Abbildungen.**
- **3.1:** Der Text zu CoNLL-2003 und OntoNotes deckt sich mit Tab. 3.1, umfasst aber nur zwei Sätze. Die Beschreibung von MultiNERD und CrossNER ist die einzige vollständige und bleibt.
- **3.2 bis 3.4:** zweimal gekürzt; die Zahlen werden in 5.7 wieder aufgenommen.
- **Vorbehalte neben Ergebnissen** (Silberreferenz, Hardware, Batchgröße 1, „kein isolierter Effekt", „kein vollständiger Neulauf") in 5.1 bis 5.7 und 6.1/6.2.
- **Literaturzahlen in 1.2, 3.x und 5.7** (W-13) und die Asymmetrie-Aussage in 5.7.1 und 6.1.
- **5.7.4, Schwellen (05 Z. 1069–1081):** nennt Tabellenwerte erneut, ist aber die Antwort auf FF3.
- **6.1:** Zusammenfassung; Wiederholung ist ihre Funktion.
- **Abstract, 1.1, 1.2, 1.4.**

---

## Vorgehen nach Freigabe

1. **Mit Freigabe dieses Plans:** Bericht unverändert als `thesis/review/28_verdichtungspruefung_2026-11-21.md` ablegen und in `thesis/review/README.md` eintragen. Keine Änderung an Kapiteln, Tabellen, Abbildungen, Anhang. Kein Commit ohne deine Ansage.
2. **Erst wenn du IDs freigibst** (z. B. „Pakete 1 bis 4" oder einzelne VA-/VB-Nummern): nur diese Stellen ändern, in `thesis/chapters/01…06_*.tex`; `anhang.tex`, `generated/`, `main.tex`, `references.bib` bleiben unberührt. Vorher `git status` und Änderungszeiten prüfen (parallele Sitzungen).
3. **Umsetzung wie am 21.11. bewährt:** Ersetzungen als Liste (ID, alt, neu) mit genau einem Treffer je Stelle; Ausgangsstand ist `5fb2070`, damit ist alles über `git diff` prüfbar und über `git checkout 5fb2070 -- <Datei>` rücknehmbar.

## Verifikation (nach einer Umsetzung)

- Probesatz in einer Kopie im Scratchpad: `docker run --rm -v "$PWD":/work -w /work texlive/texlive:latest latexmk -pdf main.tex`; erwartet 0 Fehler, 0 offene Verweise und Zitate, 0 Overfull-Meldungen.
- Seiten der Labels vor und nach der Änderung aus `main.aux` vergleichen (Kapitelanfänge, Tabellen, Abbildungen); besonders 4.4.4/4.5.1 und die Kapitelenden 3 und 5.
- Quellen: jede in den geänderten Stellen berührte `\cite`-Angabe ist weiter mindestens einmal zitiert (`qwen2024qwen25`, Fundstelle „Anh. D.3, Tab. 10").
- Anhang: `git diff 5fb2070 -- thesis/chapters/anhang.tex` muss leer sein.
- Zahlen: `scripts/check/numbers.py --diff` auf den geänderten Kapiteln; keine Zahl darf entfallen sein, die nicht an anderer Stelle im Haupttext steht.
- Gegenlesen der Übergänge an den verschobenen Stellen (Ende 3.3.1, 4.4.4, 2.7, 2.8, 4.2.3, 4.5.1).

---

## Nachtrag: Umsetzung am 22.11.2026

**Freigabe des Verfassers:** alle A- und B-Punkte sowie VC-01; Layout (`main.tex`) unverändert. Offen bleiben VC-02 bis VC-07 und P-1 bis P-4.

**Ergebnis:** 112 Seiten wie zuvor, Haupttext S. 1–77, alle Kapitelanfänge unverändert; Build ohne Fehler, Warnungen, offene Verweise oder Overfull-/Underfull-Meldungen. Fließtext 21 893 statt 22 456 Wörter (−563): Kap. 1 −12, Kap. 2 −168, Kap. 3 −249, Kap. 4 −26, Kap. 5 −49, Kap. 6 −59. Kapitel 4 schrumpft kaum, weil die Inhalte aus 3.3.1, 2.7 und 2.8 dort jetzt stehen.

**Warum keine Seite wegfällt:** Die Überschriftenregel (keine nummerierte Überschrift im unteren Seitendrittel) nimmt den Gewinn auf. Kapitel 2 endet mit 9 Zeilen auf S. 17, Kapitel 3 mit 18 Zeilen auf S. 30; davor stehen Leerräume, weil 2.5, 2.8 und 3.3.3 auf die Folgeseite rücken. Die Überschrift 2.8 verfehlt S. 15 um etwa zwei Zeilen.

| Punkt | Datei | Stand |
|---|---|---|
| VA-01 | 03 | umgesetzt; „CrossNER-Versuch" statt doppeltem „Transfertest" |
| VA-02, VA-03, VA-04 | 03 | umgesetzt; „Anh. D.3, Tab. 10" steht jetzt in 3.6 |
| VA-05 | 01, 03 | umgesetzt (a, b, c) |
| VA-06 | 04 | umgesetzt; ein Rückverweis auf 3.1 statt zwei |
| VA-07 | 04 | umgesetzt; „(Zählung ab 0)" in 4.4.4 |
| VA-08 | 04 | umgesetzt |
| VA-09 | 04 | umgesetzt; Tab. 4.9 verweist auf Anhang A.2.7 |
| VA-10 | 04 | umgesetzt (Satz gestrichen) |
| VA-11 | 04 | umgesetzt |
| VA-12, VA-13 | 05 | umgesetzt |
| VA-14, VA-15 | 06 | umgesetzt |
| VA-16 | 02 | umgesetzt |
| VA-17a, b | 02 | umgesetzt |
| VA-17c | 04 | abgewandelt: nur Verweis auf Tabelle 4.2 (Umbruch) |
| VA-17d, e | 04 | umgesetzt |
| VA-17f | 01 | umgesetzt; „Auf der LLM-Seite" bleibt als Satzanfang |
| VA-17g | 06 | umgesetzt |
| VA-17h | 05 | abgewandelt: „das im Zero-Shot-Regime den höchsten F₁ erzielt" bleibt als Nebensatz |
| VB-01 | 03, 04, 05 | umgesetzt; Verweis in 5.7.2 zeigt auf 4.4.4 |
| VB-02 | 02, 04, 05 | umgesetzt; Schlusssatz von 2.7 kürzer („die Folgen … nennt Abschnitt 4.2.3") |
| VB-03 | 02, 04 | abgewandelt: „im strikten Modus, der für generative Modelle … die eindeutigere Wahl ist" (Umbruch) |
| VB-04 | 02, 04 | abgewandelt: „Inferenzspeicher, Latenz und Trainingsdauer werden deshalb gesondert gemessen" (Umbruch) |
| VB-05 | 02, 04 | umgesetzt; „mit den Generationen Qwen2.5, Qwen3 und Qwen3.5" |
| VC-01 | 02 | umgesetzt |

**Umbruch:** Der erste Probesatz hatte 113 Seiten, weil 4.5.3 durch zwei Zusatzzeilen in 4.5.1 und 4.5.2 auf die Folgeseite rückte. Die drei kompakteren Fassungen (VA-17c, VB-03, VB-04) beheben das.

**Prüfungen:** Literaturverzeichnis mit denselben Einträgen wie zuvor; `anhang.tex`, `main.tex`, `references.bib`, `frontmatter/`, `generated/`, `figures/` ohne Diff; keine Zahl aus dem Haupttext entfallen (`scripts/check/numbers.py --diff 5fb2070`: alle belegt bis auf „210-Fache" in 5.7.4, das als Faktor schon vorher im Text stand). Ein unabhängiger Lauf des Agenten `thesis-pruefer` auf dem Diff bestätigt alle Rückverweise und verschobenen Passagen und fand keinen Grammatikfehler; seine fünf sprachlichen Hinweise sind eingearbeitet (VA-01, VA-17h, VB-03, VB-05, 3.3.1 „Ihre übrigen Befunde").

**Bewusst entfallen (vom Prüfer als Verlust gemeldet, im Bericht so vorgesehen und freigegeben):**
- BIOES/BILOU und die Liste der von `seqeval` unterstützten Schemata (VC-01, VB-03); beides wird in der Arbeit nicht verwendet.
- Der Satz zur Adapter-Präzision in 4.4.5 (VA-09); die Angabe steht in Tab. 4.9 und Anhang A.2.7, die BF16-Präzision der 0.8B- und 4B-Adapter nur noch im Anhang.
- „nicht isoliert gemessen" in 4.4.5 (VA-10); steht in 6.2.
- Der Leitfrage-Satz in 4.2.3 (VA-08); entspricht FF2/FF3.
- Der Ressourcensatz in 3.3.4 (VA-03); „keine Messung" bei UniversalNER und „CO₂" bei GoLLIE stehen nur noch in Tab. 3.2.

Rücknahme einzelner Stellen: `git diff 5fb2070 -- thesis/chapters/<Datei>` zeigt jede Änderung; `git checkout 5fb2070 -- <Datei>` stellt eine Datei wieder her.

### Umbruch (22.11.2026, zweiter Schritt)

Auf Wunsch des Verfassers laufen Unterkapitel jetzt im Seitenfluss weiter; nur Kapitel beginnen auf einer neuen Seite. Dazu ist in `main.tex` der Rand der Überschriftenregel von 30 % auf 15 % der Satzspiegelhöhe gesenkt (`\ueberschriftenrand`); eine nummerierte Überschrift rückt nur noch dann auf die Folgeseite, wenn unter ihr weniger als etwa vier Textzeilen Platz hätten.

- **Stand:** 110 Seiten (vorher 112). Haupttext S. 1–76 (Kap. 1: 5, Kap. 2: 11, Kap. 3: 13, Kap. 4: 22, Kap. 5: 20, Kap. 6: 5 Seiten), Literatur S. 77–85, Anhang S. 86–101. Build ohne Fehler, Warnungen und Overfull-/Underfull-Meldungen.
- **Woher die zwei Seiten kommen:** Kapitel 2 verliert seine letzte Seite (Textkürzung plus frei laufende Überschriften 2.5 und 2.8), der Anhang eine Seite allein durch den Umbruch. Der Inhalt des Anhangs ist unverändert.
- **Getestet:** 20 % ergibt 111 Seiten, 15 % ergibt 110, 10 % und 5 % ebenfalls 110, lassen aber Überschriften mit nur zwei Zeilen Text am Seitenende stehen (4.4.2, 4.4.6, 4.5.2, 4.7, 6.3). Bei 15 % stehen unter jeder Überschrift mindestens vier Zeilen; am tiefsten stehen 2.5 (S. 11), 3.6 (S. 26), 4.4.4 (S. 40), 5.3 (S. 57) und A.2.3 (S. 91).
- **Nicht erreichbar ohne weitere Kürzung:** Kapitel 3 endet mit 11 Zeilen auf S. 29; für eine weitere Seite müssten dort rund 130 Wörter entfallen.
