# Bericht 22: Sprachliche Prüfung der Gesamtarbeit

Stand 8. November 2026. Abschnitte 1 bis 6 sind der Prüfbericht vor der Umsetzung. Nach der Freigabe aller Befunde („setze das alles um“) wurde am selben Tag umgesetzt; Abschnitt 7 protokolliert Entscheidungen, Abweichungen und Prüfergebnisse.

## 1. Geprüfte Version und Umfang

- **Quellen:** Commit `ef7f24e` (die Thesis-Quellen sind identisch mit `ed693be`). Gelesen wurden vollständig: `frontmatter/` (Deckblatt, Eigenständigkeitserklärung, Abstract, Abkürzungsverzeichnis), `chapters/01` bis `06`, `chapters/anhang.tex` (A.1 bis A.3 einschließlich Listings und Captions), die Beschriftungen in `figures/*.tex` und in den Diagrammen von Kapitel 5, die Spaltenköpfe und Tabellennotizen in `generated/tab-*.tex` sowie das Literaturverzeichnis in der gerenderten Fassung.
- **PDF:** Build von `ed693be` (118 Seiten) zum Abgleich von Überschriften, Inhaltsverzeichnis, Captions und Literaturverzeichnis. Seitenangaben sind die gedruckten Seitenzahlen: Vorspann römisch, Haupttext S. 1 bis 85, Literatur S. 86 bis 94, Anhang ab S. 95.
- **Fußnoten:** Die Arbeit enthält keine (`\footnote` kommt nicht vor). Geprüft wurden stattdessen die Tabellennotizen.
- **Nicht sprachlich geprüft:** Zahlen in Tabellenzellen, die englischen Prompt-Listings (Code), der angemeldete Titel auf dem Deckblatt und der Mustertext der Eigenständigkeitserklärung.
- **Englische Originalquellen:** Sie liegen in diesem Checkout nicht vor (`thesis/literatur/` ist leer). **Keine Übersetzung wurde gegen das Original geprüft.** Stellen, deren richtige Formulierung von der Quelle abhängt, sind mit „Quelle prüfen“ markiert. Drei Befunde ließen sich ohne Quelle belegen:
  - K3-19 über die frühere Fassung im Git-Verlauf;
  - K4-13 über den Code (`ba-ner/src/decoder/inference.py`);
  - K5-08 über `generated/summary.json`.
- **Automatische Trennungen** im PDF wurden nicht als Fehler gewertet; maßgeblich ist der Quelltext.
- **Eigene Fehler:** Vier Befunde gehen auf meine Überarbeitungen vom 7. und 8. November zurück:
  - K4-14 (Commit `e624d7e`);
  - K5-22 (`c482382`);
  - K5-23 (`9b1e273`);
  - K6-03 (`f63af31`).

  K3-19 entstand durch den späteren Einschub zu Zhan et al. (Commit `8a00510`).

### Legende

- **Kategorien:** EF = Eindeutig falsch, SA = Sprachlich auffällig, UP = Unpräzise oder schwammig, RD = Redundant, UE = Uneinheitlich, ST = Stilistische Empfehlung, FP = Fachlich zu prüfen.
- **Sicherheit:**
  - *sicher* = rein sprachlich, die Aussage bleibt unverändert;
  - *Quelle prüfen* = die Bedeutung hängt vom englischen Original ab;
  - *Rückfrage* = die gemeinte Bedeutung ist nicht sicher erkennbar, es werden zwei Lesarten angeboten.
- **Kernaussage:** Sie wird bei Hoch- und Mittel-Befunden ausdrücklich genannt. Bei Niedrig-Befunden ist sie klar, und die Änderung betrifft nur die Form.

### Die fünf Hoch-Befunde auf einen Blick

| Nr. | Stelle | Problem |
|---|---|---|
| K3-19 | 3.2.3, S. 29 | „Als Gründe nennen sie …“ bezieht sich jetzt auf Zhan et al., gemeint sind Lu et al. |
| K4-13 | 4.5.2, S. 51 | „Dekodierung“ in der Liste der nicht gemessenen Schritte ist als Generierung lesbar, gemeint ist Detokenisierung |
| K4-14 | 4.5.2, S. 52 | Der Satz „Effizienzvergleiche werden daher …“ steht im falschen Absatz (eigener Fehler) |
| K5-08 | 5.2, S. 63 | „die acht Testsätze ohne Referenzentität lässt nur das 27B-Modell leer“ trifft nur auf 3 bis 4 der 8 Sätze zu |
| K5-23 | 5.7.1, S. 73 | „trifft nur DeBERTa-v3-large 319 Referenzspans“ ist als Gesamtzahl missverständlich (eigener Fehler) |

## 2. Befunde nach Kapitel

### Vorspann

**V-01 · Abstract, S. ii, 3. Absatz · Niedrig · SA**
- Original: „Die Encoder benötigen auf MultiNERD 9 bis 17~ms Inferenzlatenz je Satz und 0,7 bis 1,7~GiB Inferenzspeicher.“
- Problem: Eine Latenz wird gemessen, nicht „benötigt“; die Kollokation ist schief.
- Vorschlag: „Die Inferenzlatenz der Encoder beträgt auf MultiNERD 9 bis 17~ms je Satz bei 0,7 bis 1,7~GiB Inferenzspeicher.“ Der Folgesatz („Die QLoRA-Läufe … benötigen 1,9 bis 5,5~s“) kann bleiben.
- Erhalten: alle Zahlen und „je Satz“. Sicherheit: sicher.

**V-02 · Abstract, S. ii, 3. Absatz · Mittel · UP**
- Original: „Ihre Prompt- und Decoding-Wahl wurde durch Testbeobachtungen beeinflusst; zusätzliche Unterschiede in Prompt, Decoding und Präzision verhindern die Bestimmung eines isolierten Fine-Tuning-Effekts.“
- Problem:
  - „Präzision“ ist im Abstract ohne Kontext als Metrik (Precision) lesbar; gemeint ist die numerische Präzision.
  - „zusätzliche Unterschiede“: Der Bezug fehlt, gemeint ist „zusätzlich zum Training“.
  - Die Nominalkette „verhindern die Bestimmung eines isolierten …“ ist sperrig.
- Vorschlag: „Ihre Prompt- und Decoding-Wahl wurde durch Testbeobachtungen beeinflusst. Da sich Zero-Shot- und QLoRA-Konfigurationen neben dem Training auch in Prompt, Decoding und numerischer Präzision unterscheiden, lässt sich kein isolierter Fine-Tuning-Effekt bestimmen.“
- Erhalten: beide Einschränkungen. Sicherheit: sicher; deckt sich mit 5.1.1 („er umfasst neben dem Training auch Unterschiede in Prompt, Decoding und Präzision“). Hinweis: Das Abstract muss auf einer Seite bleiben; nach der Umsetzung prüfen.
- Kernaussage: Der Abstand zwischen Zero-Shot und QLoRA ist kein reiner Fine-Tuning-Effekt.

**V-03 · Abstract, S. ii, 4. Absatz · Mittel · UP**
- Original: „Auf MultiNERD ist damit unter jedem gemessenen Latenz- und Speicherbudget ein Encoder die beste Wahl“
- Problem: Budgets wurden nicht gemessen. Laut 4.5.2 wurden sie nicht vorab festgelegt; die Schwellen sind aus Messwerten abgeleitet. „gemessenen“ bezieht sich grammatisch auf das Budget.
- Vorschlag: „Auf MultiNERD ist damit unter jeder aus den Messwerten abgeleiteten Latenz- und Speichergrenze ein Encoder die beste gemessene Wahl“. Der Wortlaut lehnt sich an 5.7.4 an.
- Erhalten: die Beschränkung auf die gemessenen Konfigurationen. Sicherheit: sicher.
- Kernaussage: Auf MultiNERD ist bei jeder Ressourcengrenze ein Encoder vorn.

**V-04 · Abstract, S. ii · Niedrig · UP**
- Original: „Der Qualitätsvergleich hängt jedoch wesentlich von Ausgabelimit, Datensatz und Typabbildung ab.“
- Problem: „Typabbildung“ wird im Abstract nicht eingeführt; vorher steht nur „mit den gemeinsamen Typen PER, ORG und LOC“.
- Vorschlag: „… von Ausgabelimit, Datensatz und der Typabbildung auf PER, ORG und LOC ab.“ Das ist nur sinnvoll, wenn das Abstract auf einer Seite bleibt.
- Sicherheit: sicher.

**V-05 · Abstract, S. ii · Niedrig · UE**
- Original: „Die Arbeit zeigt damit sowohl den Ressourcenunterschied … als auch die Bedeutung ausreichender Ausgabelängen für die Bewertung generativer NER.“
- Problem: „damit“ steht zwei Sätze zuvor schon einmal. Zudem bezeichnet „Ausgabelänge“ sonst die Länge einer Antwort (K2, K3, 5.7.2); gemeint ist das Limit.
- Vorschlag: „Die Arbeit zeigt so sowohl … als auch die Bedeutung ausreichender Ausgabelimits für …“
- Sicherheit: sicher.

**V-06 · Abstract, S. ii, 2. Absatz · Niedrig · UP**
- Original: „erhöht den LLM-Wert auf 95,6“
- Vorschlag: „erhöht den Mikro-F$_1$ des LLM auf 95,6“. Sicherheit: sicher.

**V-07 · Eigenständigkeitserklärung, S. i, „Verwendete KI-Werkzeuge“ · Mittel · RD**
- Original: „Art der Nutzung: Unterstützung bei der Implementierung … sowie bei der Unterstützung der Literaturrecherche.“
- Problem: „Unterstützung … bei der Unterstützung“ ist doppelt.
- Vorschlag: „… sowie bei der Literaturrecherche.“
- Erhalten: die Literaturrecherche als Nutzungsart. Sicherheit: sicher.
- Kernaussage: Die KI half auch bei der Literaturrecherche.

**V-08 · Eigenständigkeitserklärung, S. i · Niedrig · UE/FP**
- Original: Der Formeltext sagt „KI-Tools“, der Absatztitel „Verwendete KI-Werkzeuge“; der Mustertext enthält zudem „ohne Hilfe Dritter“.
- Problem: Die Bezeichnungen wechseln. „ohne Hilfe Dritter“ steht neben der ausgewiesenen KI-Nutzung; das ist eine formale, keine sprachliche Frage.
- Vorschlag: Den Formeltext nicht ändern, falls er wörtlich dem TU-Muster entspricht; allenfalls den Absatztitel an „KI-Tools“ angleichen. Ob „ohne Hilfe Dritter“ zur KI-Angabe passt, beim Prüfungsamt oder Betreuer klären (steht bereits als Frage in der Begleitnotiz).
- Sicherheit: Rückfrage (Mustertext).

### Kapitel 1 Einleitung

**K1-01 · 1.1, S. 1 · Niedrig · UP**
- Original: „ein darauf trainierter Klassifikator weist jedem Token ein Label zu, aus dem sich die Entitäten ergeben (Sequenzlabeling)“
- Problem: Die Entitäten ergeben sich aus der Labelfolge, nicht aus einem einzelnen Label.
- Vorschlag: „… weist jedem Token ein Label zu; aus der Labelfolge ergeben sich die Entitäten (Sequenzlabeling).“
- Sicherheit: sicher.

**K1-02 · 1.1, S. 1 · Niedrig · SA**
- Original: „Large Language Models (LLMs) sind deutlich größere, generative Sprachmodelle, die ein breites Spektrum von Aufgaben mit demselben Modell bearbeiten“
- Problem: Die Modelle bearbeiten Aufgaben „mit demselben Modell“, also mit sich selbst; das ist schief.
- Vorschlag: „… sind deutlich größere generative Sprachmodelle, die als einzelnes Modell ein breites Spektrum von Aufgaben bearbeiten“.
- Sicherheit: sicher.

**K1-03 · 1.1, S. 2 · Niedrig · ST**
- Original: „erfordert zusätzlich Speicher für Gradienten und Optimiererzustände“
- Vorschlag: „erfordert neben den Gewichten Speicher für Gradienten und Optimiererzustände“, damit klar ist, wozu der Speicher zusätzlich ist. Sicherheit: sicher.

**K1-04 · 1.1, S. 2 · Niedrig · ST**
- Original: „Für die praktische Modellwahl … und wann ein kompakter Encoder die effizientere Wahl darstellt.“
- Problem: „Wahl“ steht doppelt, dazu das Funktionsverb „darstellt“.
- Vorschlag: „… und wann ein kompakter Encoder effizienter ist.“ Sicherheit: sicher.

**K1-05 · 1.2, S. 2 · Mittel · SA**
- Original: „Die direkten Vergleiche erfassen Erkennungsleistung und Ressourcenbedarf nur unvollständig gemeinsam (Inferenzzeit ohne Speicher oder Trainingskosten ohne Inferenz)“
- Problem: „nur unvollständig gemeinsam“ ist unidiomatisch und schwer zu erfassen.
- Vorschlag: „Die direkten Vergleiche erfassen neben der Erkennungsleistung nur Teile des Ressourcenbedarfs (Inferenzzeit ohne Speicher oder Trainingskosten ohne Inferenz)“.
- Erhalten: die Beispiele in Klammern und die Aussage „keine vollständige gemeinsame Erfassung“. Sicherheit: sicher.
- Kernaussage: Keine Vergleichsarbeit misst Qualität und alle Ressourcen gemeinsam.

**K1-06 · 1.2, S. 3, FF2 · Niedrig · ST**
- Original: „Untersucht wird, wie sich … innerhalb der untersuchten Konfigurationen …“
- Problem: „untersucht“ steht doppelt.
- Vorschlag: „… innerhalb der betrachteten Konfigurationen …“
- Erhalten: der Wortlaut der FF, falls er an anderer Stelle wörtlich wiederholt wird; dann nur gemeinsam ändern. Sicherheit: sicher.

**K1-07 · 1.2, S. 3, FF3 · Niedrig · ST**
- Original: „… welche Konfigurationen unter den gemessenen Bedingungen … Daraus soll hervorgehen, unter welchen Bedingungen der Einsatz eines LLM …“
- Problem: „Bedingungen“ steht in zwei aufeinanderfolgenden Sätzen.
- Vorschlag: im zweiten Satz „unter welchen Umständen“. Sicherheit: sicher.

**K1-08 · 1.2, S. 4 · Niedrig · ST/UE**
- Original: „… ergänzen die Auswertung und ermöglichen eine Einordnung der Unsicherheit sowie des Einflusses begrenzter Ausgabelängen.“
- Problem: Nominalstil; „Ausgabelängen“ meint das Limit (vgl. V-05).
- Vorschlag: „… ergänzen die Auswertung; sie zeigen, wie unsicher die Unterschiede sind und wie stark das begrenzte Ausgabelimit die Ergebnisse beeinflusst.“
- Sicherheit: sicher.

**K1-09 · 1.3, S. 4 · Niedrig · RD**
- Original: „im Folgenden werden die Verfahren mit präziseren Bezeichnungen benannt“
- Vorschlag: „im Folgenden werden die Verfahren präziser bezeichnet“. Sicherheit: sicher.

**K1-10 · 1.3, S. 4 · Mittel · SA**
- Original: „Eine Übertragung der Schlussfolgerungen auf andere Sprachen oder Anwendungsdomänen wird nicht abgeleitet.“
- Problem: Eine Übertragung wird nicht „abgeleitet“; die Kollokation ist schief.
- Vorschlag: „Die Schlussfolgerungen werden nicht auf andere Sprachen oder Anwendungsdomänen übertragen.“
- Sicherheit: sicher.
- Kernaussage: Die Ergebnisse gelten nur für den untersuchten Rahmen.

### Kapitel 2 Grundlagen

**K2-01 · 2.1, S. 6 · Niedrig · SA**
- Original: „Über lange Zeit beruhten NLP-Systeme …“
- Vorschlag: „Lange Zeit beruhten NLP-Systeme …“ Sicherheit: sicher.

**K2-02 · 2.1, S. 6 · Mittel · EF (Rechtschreibung)**
- Original: „dass Wörter mit ähnlichen Kontexten im Vektorraum nahe beieinander liegen“
- Problem: Nach Duden wird „beieinanderliegen“ zusammengeschrieben; Kapitel 5 schreibt es bereits so.
- Vorschlag: „nahe beieinanderliegen“. Sicherheit: sicher.

**K2-03 · Abb. 2.1, Caption, S. 7 · Niedrig · SA**
- Original: „…; Endindizes einschließend.“
- Problem: Die Ellipse ist unklar.
- Vorschlag: „…; der Endindex gehört zur Erwähnung.“ Sicherheit: sicher.

**K2-04 · 2.2, S. 7 · Niedrig · ST**
- Original: „Das verbreitetste Schema“
- Vorschlag: „Das am weitesten verbreitete Schema“. Sicherheit: sicher.

**K2-05 · Tab. 2.1, Caption, S. 7 · Niedrig · SA**
- Original: „in IOB1 erhält nur eine Entität ein \texttt{B-}-Tag, die unmittelbar auf eine Entität desselben Typs folgt, was in diesem Satz nicht vorkommt.“
- Problem: Der Relativsatz steht weit von seinem Bezugswort entfernt.
- Vorschlag: „in IOB1 erhält eine Entität nur dann ein \texttt{B-}-Tag, wenn sie unmittelbar auf eine Entität desselben Typs folgt; das kommt in diesem Satz nicht vor.“
- Sicherheit: sicher.

**K2-06 · 2.2, S. 8 · Mittel · SA (Lehnübersetzung „given the input“)**
- Original: „die bedingte Wahrscheinlichkeit der gesamten Labelfolge gegeben die Eingabe“
- Vorschlag: „… der gesamten Labelfolge bei gegebener Eingabe“. Sicherheit: sicher.
- Kernaussage: CRFs modellieren die Labelfolge als Ganzes.

**K2-07 · 2.3, S. 9 · Niedrig · ST (Zeichensetzung)**
- Original: „Für alle Tokens einer Sequenz zusammengefasst in den Matrizen $Q$, $K$ und $V$ ergibt sich“
- Vorschlag: „Für alle Tokens einer Sequenz, zusammengefasst in den Matrizen $Q$, $K$ und $V$, ergibt sich“. Sicherheit: sicher.

**K2-08 · 2.3, S. 9 · Mittel · FP**
- Original: „Jedes Token sammelt so Information aus genau den Positionen, die für es relevant sind.“
- Problem: Softmax-Attention gewichtet alle Positionen; der Satz davor spricht selbst von der „gewichteten Summe“. „genau den Positionen“ suggeriert eine harte Auswahl. „für es“ klingt zudem unschön.
- Vorschlag, Rückfrage mit zwei Varianten:
  - (a) „Jedes Token sammelt so Information vor allem aus den Positionen, die für dieses Token relevant sind.“
  - (b) „Jedes Token sammelt so Information aus allen Positionen, gewichtet nach ihrer Relevanz.“ Das ist näher an der Mechanik.
- Erhalten: die intuitive Erklärung. Sicherheit: Rückfrage.
- Kernaussage: Attention gewichtet den Kontext nach Relevanz.

**K2-09 · 2.3, S. 9 · Mittel · SA**
- Original: „mit einer weiteren Matrix $W^O$ auf die Modelldimension zurückgebildet“
- Problem: „zurückbilden“ bedeutet sich zurückentwickeln; gemeint ist eine Projektion.
- Vorschlag: „… auf die Modelldimension zurückprojiziert“. Sicherheit: sicher.
- Kernaussage: Die Kopfausgaben werden verkettet und auf $d_{\text{model}}$ projiziert.

**K2-10 · 2.3, S. 9 · Niedrig · SA**
- Original: „das Training tiefer Stapel“
- Vorschlag: „das Training tiefer Schichtstapel“. Sicherheit: sicher.

**K2-11 · 2.3, S. 10 bis 11 · Niedrig · ST (Kernaussage spät)**
- Original: Der Absatz „Für NER folgt daraus der zentrale Unterschied …“ nennt erst nach dem Einschub über Decoder mit Klassifikationskopf, warum der rechte Kontext wichtig ist („Der rechte Kontext ist für NER unmittelbar relevant …“).
- Vorschlag: Den Satz zum rechten Kontext unverändert direkt hinter den Encoder-Satz stellen, danach die Decoder-Sätze. Es wird nichts gestrichen.
- Sicherheit: sicher.
- Kernaussage: Encoder sehen beidseitigen Kontext; der Decoder nutzt ihn hier nur, weil der ganze Satz im Prompt steht.

**K2-12 · 2.4, S. 12 · Niedrig · UE**
- Original: „die \emph{disentangled Attention}“
- Vorschlag: „die \emph{Disentangled Attention}“. So steht der Name im Abkürzungsverzeichnis und in den Quellen. Sicherheit: sicher.

**K2-13 · 2.4, S. 12 · Mittel · SA (Lehnübersetzung „sample-efficient“)**
- Original: „was das Vortraining stichprobeneffizienter macht“
- Problem: „Stichprobe“ trifft hier nicht zu; gemeint ist, dass bei gleicher Datenmenge mehr Lernsignal entsteht.
- Vorschlag: „was das Vortraining dateneffizienter macht“.
- Sicherheit: Quelle prüfen (Clark et al. 2020, Wortlaut zur Effizienz).

**K2-14 · 2.4, S. 12 · Niedrig · UP**
- Original: „DeBERTa-v3 wurde mit weitgehend denselben Einstellungen wie DeBERTa und mit 160~GB Daten wie DeBERTa-v2 vortrainiert“
- Problem: „mit 160 GB Daten wie DeBERTa-v2“ ist elliptisch: dieselben Daten oder nur dieselbe Menge?
- Vorschlag, Rückfrage mit zwei Lesarten:
  - (a) „… und auf denselben 160~GB Daten wie DeBERTa-v2 vortrainiert“;
  - (b) „… und wie DeBERTa-v2 mit 160~GB Daten vortrainiert“.
- Sicherheit: Quelle prüfen (He et al. 2023, Abschn. 4.1).

**K2-15 · 2.5, S. 13 · Niedrig · SA**
- Original: „wodurch der unzuverlässige Rand der Verteilung abgeschnitten wird“
- Problem: „Rand“ steht hier für „tail“.
- Vorschlag: „wodurch die unzuverlässigen Ausläufer der Verteilung abgeschnitten werden“. Sicherheit: sicher.

**K2-16 · 2.5, S. 14 · Mittel · EF (Bezug des indem-Satzes)**
- Original: „Nachtrainierte LLMs lösen Aufgaben, indem die Aufgabe im Eingabetext beschrieben wird.“
- Problem: Der indem-Satz muss das Subjekt des Hauptsatzes aufnehmen; hier wechselt es. „Aufgabe“ steht zudem doppelt.
- Vorschlag: „Nachtrainierte LLMs lösen Aufgaben, die im Eingabetext beschrieben werden.“
- Sicherheit: sicher.
- Kernaussage: Die Aufgabe steckt im Prompt, nicht in den Gewichten.

**K2-17 · 2.5, S. 14 · Niedrig · SA**
- Original: „welche Komponenten geladen werden und dass der Thinking-Modus deaktiviert ist, beschreiben die Abschnitte~\ref{sec:llm-auswahl} und~\ref{sec:llm-pipeline}.“
- Problem: Das Subjekt besteht aus einer indirekten Frage und einem dass-Satz.
- Vorschlag: „Welche Komponenten geladen werden, beschreibt Abschnitt~\ref{sec:llm-auswahl}, die Deaktivierung des Thinking-Modus Abschnitt~\ref{sec:llm-pipeline}.“
- Sicherheit: sicher. Die Zuordnung ist geprüft: 4.2.2 nennt die geladenen Komponenten, 4.4.3 den Thinking-Modus.

**K2-18 · 2.5, S. 14 (ebenso 4.3.2, S. 39) · Mittel · SA**
- Original: „Diese Ausgabe muss anschließend auf die Tokenfolge zurückgeführt werden“; S. 39: „auf die Wortebene zurückgeführt“.
- Problem: „zurückführen auf“ bedeutet im Deutschen „als Ursache benennen“; gemeint ist eine Abbildung.
- Vorschlag: „auf die Tokenfolge abgebildet werden“; S. 39: „auf die Wortebene übertragen“. Sicherheit: sicher.
- Kernaussage: Die generierte Ausgabe wird wieder an die Tokens gebunden.

**K2-19 · 2.6, S. 15 · Mittel · SA**
- Original: „Adam führt je Parameter zwei zusätzliche Größen.“
- Problem: „führen“ ist hier eine Lehnübersetzung von „keep track of“.
- Vorschlag: „Adam speichert je Parameter zwei zusätzliche Größen.“ Sicherheit: sicher.

**K2-20 · 2.6, S. 15 · Niedrig · RD**
- Original: „beziffern … den Speicherbedarf mit Adam auf 1{,}2~TB GPU-Speicher“
- Vorschlag: „beziffern … den GPU-Speicherbedarf mit Adam auf 1{,}2~TB“. Sicherheit: sicher.

**K2-21 · 2.6, S. 15 · Mittel · SA (Lehnübersetzung „reduce the need to retune“)**
- Original: „die Konstante $\alpha$ skaliert den Beitrag des Adapters, was den Bedarf verringern kann, die Lernrate bei einer Änderung von $r$ neu abzustimmen“
- Vorschlag: „…; dadurch muss die Lernrate bei einer Änderung von $r$ seltener neu abgestimmt werden.“
- Erhalten: die vorsichtige Formulierung; „seltener“ statt „nicht“. Sicherheit: sicher.

**K2-22 · 2.6, S. 15 · Mittel · SA (Lehnübersetzung „reduce by X times“)**
- Original: „reduziert LoRA die Zahl der trainierbaren Parameter … um das 10\,000-Fache und den GPU-Speicher um das Dreifache“
- Problem: „um das X-Fache reduzieren“ ist logisch schief; „um das Dreifache“ kann sogar als „um 300 %“ gelesen werden.
- Vorschlag: „… auf ein Zehntausendstel und den GPU-Speicherbedarf auf ein Drittel“.
- Sicherheit: sicher. Die Werte entsprechen der bekannten Angabe von Hu et al. („10,000 times“, „3 times“).

**K2-23 · 2.6, S. 15 · Mittel · EF (Vergleichskonstruktion)**
- Original: „erreicht LoRA dabei eine gleichwertige oder bessere Qualität als vollständiges Fine-Tuning“
- Problem: „gleichwertig“ verlangt „wie“ oder „mit“, „besser“ verlangt „als“; das Zeugma ist grammatisch unsauber.
- Vorschlag: „erreicht LoRA dabei eine Qualität, die der des vollständigen Fine-Tunings entspricht oder sie übertrifft“.
- Sicherheit: sicher.

**K2-24 · Abb. 2.3, S. 17 · Niedrig · UE**
- Original: Die Caption und die Achse sagen „Test-Loss“, der Fließtext sagt „Verlust“.
- Vorschlag: Die Achse bleibt als Wiedergabe der Originalgrafik; in der Caption „Testverlust“. Optional. Sicherheit: sicher.

**K2-25 · 2.7, S. 17 · Mittel · UP**
- Original: „Zweitens werden in dieser Arbeit veröffentlichte Checkpoints fester Größe angepasst“
- Problem: Lesbar als „in dieser Arbeit veröffentlichte Checkpoints“.
- Vorschlag: „Zweitens passt diese Arbeit bereits veröffentlichte Checkpoints fester Größe an, deren Vortraining nicht variiert werden kann“.
- Sicherheit: sicher.
- Kernaussage: Die Modellgröße lässt sich hier nicht kontrolliert variieren.

**K2-26 · 2.8, S. 18 · Mittel · SA/UE**
- Original: „Es umfasst fünf Zieldomänen, Politik, Naturwissenschaft, Musik, Literatur und Künstliche Intelligenz, mit Sätzen aus Wikipedia, die manuell annotiert wurden. Jede Domäne besitzt ein eigenes Typinventar mit neun bis 17 Kategorien“
- Problem:
  - Nach dem Entfernen der Gedankenstriche ist die Aufzählung eine Komma-Apposition; man liest „fünf Zieldomänen, Politik“ als Reihe.
  - „Naturwissenschaft“ steht sonst überall im Plural (K4, K5, alle Tabellen).
  - Der Zahlbereich mischt Wort und Ziffer.
- Vorschlag: „Es umfasst fünf Zieldomänen (Politik, Naturwissenschaften, Musik, Literatur und Künstliche Intelligenz) mit manuell annotierten Sätzen aus Wikipedia. Jede Domäne besitzt ein eigenes Typinventar mit 9 bis 17 Kategorien“.
- Sicherheit: sicher.

**K2-27 · Tab. 2.2, S. 18 · Niedrig · UE**
- Original: „automatisch (Silver)“, „+ man.\ Teststichprobe“
- Vorschlag: „automatisch (Silberreferenz)“, „+ manuelle Teststichprobe“, sofern der Platz reicht. Sicherheit: sicher.

**K2-28 · 2.9, S. 20 (ebenso 4.5.3, S. 50) · Mittel · SA (Lehnübersetzung „lenient“)**
- Original: „nachsichtigere Modi“, „ist nachsichtig gegenüber formal unzulässigen Tagfolgen“, S. 50: „der nachsichtigen Evaluationsmodi“.
- Problem: „nachsichtig“ beschreibt Personen.
- Vorschlag: „weniger strenge Modi“, „toleriert formal unzulässige Tagfolgen“, „der weniger strengen Evaluationsmodi“. Sicherheit: sicher.

**K2-29 · 2.9, S. 20 · Mittel · EF (Kongruenz)**
- Original: „Für seltene Typen mit wenigen Referenzentitäten, dem sogenannten Support, sind …“
- Problem: Die Apposition im Singular hängt an einem Plural; Support ist die Zahl der Referenzentitäten, nicht die Entitäten selbst.
- Vorschlag: „Für seltene Typen mit geringem Support, also wenigen Referenzentitäten, sind …“
- Sicherheit: sicher.

### Kapitel 3 Verwandte Arbeiten

**K3-01 · Kapiteleinleitung, S. 21 · Niedrig · RD**
- Original: „Ergänzend wurde entlang der Zitationen der Schlüsselarbeiten per Schneeballsuche gesucht.“
- Vorschlag: „Ergänzend wurden die Zitationen der Schlüsselarbeiten per Schneeballsuche verfolgt.“ Sicherheit: sicher.

**K3-02 · 3.1.1, S. 22 · Niedrig · SA**
- Original: „Ergänzend prompten die Autoren ChatGPT (\texttt{gpt-3.5-turbo}) in drei Varianten“
- Vorschlag: „Ergänzend testen die Autoren ChatGPT (\texttt{gpt-3.5-turbo}) mit drei Prompt-Varianten“. Sicherheit: sicher.

**K3-03 · 3.1.2, S. 22 · Niedrig · SA**
- Original: „erweisen sich die DeBERTa-Varianten als am effektivsten“
- Vorschlag: „erweisen sich die DeBERTa-Varianten als die effektivsten“. Sicherheit: sicher.

**K3-04 · 3.1.3, S. 23 · Niedrig · ST**
- Original: „erreicht die mittlere Variante … im Mittel einen F$_1$ von 55{,}4“
- Vorschlag: „erreicht die mittelgroße Variante … durchschnittlich einen F$_1$ von 55{,}4“. Sicherheit: sicher.

**K3-05 · 3.2.1, S. 24 · Mittel · EF (fehlender Artikel beim Superlativ)**
- Original: „Tagging erweist sich als robusteste, aber tokenintensivste Strategie“
- Vorschlag: „Tagging erweist sich als die robusteste, aber auch tokenintensivste Strategie“. Sicherheit: sicher.

**K3-06 · 3.2.1, S. 24 · Mittel · UP (Bezug)**
- Original: „Eine auf Ramponi et al. zurückgehende Indexing-Variante … wird beschrieben, aber nicht evaluiert. Die Arbeit liegt bislang nur als Preprint vor.“
- Problem: „Die Arbeit“ kann nach dem Ramponi-Satz auf Ramponi et al. bezogen werden; das ist aber ein NAACL-Paper. Gemeint ist Semin et al. (arXiv 2026).
- Vorschlag: „Die Arbeit von Semin et al. liegt bislang nur als Preprint vor.“ Umsetzung mit `\textcite{semin2026spanLabeling}`.
- Sicherheit: sicher (laut Literaturverzeichnis ist Semin ein Preprint, Ramponi publiziert).
- Kernaussage: Die Belastbarkeit der Semin-Ergebnisse ist eingeschränkt.

**K3-07 · 3.2.1, S. 24 · Mittel · SA (Lehnübersetzung „first-order factor“)**
- Original: „Dass das Format auch bei feinjustierten Modellen ein Faktor erster Ordnung ist“
- Vorschlag: „Dass das Format auch bei feinjustierten Modellen die Leistung maßgeblich beeinflusst“.
- Sicherheit: Quelle prüfen; falls Zhan et al. „first-order“ wörtlich verwenden, kann man es als Zitat kennzeichnen.

**K3-08 · 3.2.1, S. 24 und 25 · Niedrig · SA**
- Original: „Sie trainieren acht offene LLMs mit LoRA auf fünf Ausgabeformaten“; „trainieren Modelle … per QLoRA auf sechs Ausgabeformaten“
- Vorschlag: „mit fünf Ausgabeformaten“, „mit sechs Ausgabeformaten“. Sicherheit: sicher.

**K3-09 · 3.2.1, S. 24 und 25 · Niedrig · UE/FP**
- Original: „auf den flachen Datensätzen rund 92 F$_1$ … dort im Mittel auf rund 54 fällt“; „Allein das Format verschiebt F$_1$ teils um mehr als 40\,\%“
- Problem: „F$_1$“ wird als Einheit verwendet, sonst heißt es „Punkte“; „verschiebt F$_1$“ hat keinen Artikel.
- Vorschlag: „rund 92 F$_1$-Punkte … auf rund 54 Punkte fällt“; „verschiebt den F$_1$-Wert teils um mehr als 40\,\%“.
- Sicherheit: Für „40\,\%“ die Quelle prüfen (Ravikumar et al. 2026: relative Prozent oder Prozentpunkte?).

**K3-10 · 3.2.1, S. 25 · Mittel · SA (Lehnübersetzung „speedup of X times“)**
- Original: „Ihr paralleles Decoding … verkürzt die Inferenz um das 1{,}76- bis 10{,}22-Fache“
- Vorschlag: „… beschleunigt die Inferenz um den Faktor 1{,}76 bis 10{,}22“. Sicherheit: sicher.

**K3-11 · 3.2.1, S. 25 · Niedrig · UE**
- Original: „JSON-Schema und Nachverarbeitung sind eigene Umsetzung.“
- Problem: In der Einleitung (S. 3) steht derselbe Satz als „eigene Umsetzungen dieser Arbeit“.
- Vorschlag: „sind eigene Umsetzungen.“ Sicherheit: sicher.

**K3-12 · 3.2.2, S. 26 · Mittel · SA (Lehnübersetzung „over-confidently“)**
- Original: „Weil GPT-3 auch Eingaben ohne Entität des abgefragten Typs überkonfident annotiert“
- Vorschlag: „Weil GPT-3 auch in Eingaben ohne Entität des abgefragten Typs mit übermäßiger Sicherheit Entitäten markiert“.
- Sicherheit: Quelle prüfen (Wang et al. 2025, Abschn. zur Selbstverifikation).

**K3-13 · 3.2.2, S. 26 · Mittel · UP**
- Original: „die mitgeführten überwachten Bestwerte liegen weiterhin weit darüber“
- Problem: „mitgeführt“ ist unklar.
- Vorschlag: „die zum Vergleich berichteten Bestwerte überwachter Verfahren liegen weiterhin weit darüber“.
- Sicherheit: sicher.
- Kernaussage: Prompting allein bleibt deutlich hinter überwachten Verfahren.

**K3-14 · Überschrift 3.2.3, S. 26 und Inhaltsverzeichnis S. iii · Mittel · SA**
- Original: „Instruktionsgetunte und angepasste LLMs“
- Problem: Das Wort ist eine deutsch-englische Mischbildung.
- Vorschlag, Rückfrage: (a) „LLMs mit Instruction-Tuning und Anpassung“; (b) „Instruktionsfeinabgestimmte und angepasste LLMs“.
- Erhalten: das Label `sec:k3-angepasst`. Sicherheit: sicher.

**K3-15 · 3.2.3, S. 26 und 3.3, S. 30 · Mittel · SA (Lehnübersetzung „student/teacher“)**
- Original: „übertreffen die Studentenmodelle … ihren Lehrer deutlich“; „vom 7B- zum 13B-Studenten“
- Problem: Bei der Destillation sind im Deutschen „Schülermodell“ und „Lehrermodell“ üblich.
- Vorschlag: „übertreffen die Schülermodelle … ihr Lehrermodell deutlich“; „vom 7B- zum 13B-Schülermodell“.
- Sicherheit: sicher.

**K3-16 · 3.2.3, S. 28 · Niedrig · UP**
- Original: „Die berichteten Werte belegen deshalb nur, dass …“
- Vorschlag: „Die von beiden Arbeiten berichteten CrossNER-Werte belegen deshalb nur, dass …“ Sicherheit: sicher.

**K3-17 · 3.2.3, S. 29 · Mittel · SA**
- Original: „Inferenzzeiten …, die je Testbeispiel um das 40- bis 220-Fache über denen des besten Encoders liegen“
- Problem: „um das X-Fache über“ ist schief; die Einleitung schreibt korrekt „das 40- bis 220-Fache“.
- Vorschlag: „Inferenzzeiten, die je Testbeispiel das 40- bis 220-Fache derjenigen des besten Encoders betragen“. Sicherheit: sicher.

**K3-18 · 3.2.3, S. 29 · Niedrig · ST**
- Original: „Nach einem LoRA-Fine-Tuning nähert sich … nach Darstellung der Autoren …“
- Vorschlag: „Mit LoRA-Fine-Tuning nähert sich … laut den Autoren …“ Sicherheit: sicher.

**K3-19 · 3.2.3, S. 29 · Hoch · UP (Bezugsfehler, fachlich missverständlich)**
- Original: „Diese Vermutung relativieren \textcite[Tab.~11]{zhan2026generativeNER}: In ihren Prüffällen … belegt. Als Gründe nennen sie die Abhängigkeit tokenbasierter NER von Einzeltoken-Repräsentationen, das darauf nicht ausgerichtete Pre-Training-Ziel der Decoder und die für die Modelle ungesehenen IOB2-Symbole. Die Modellgröße variiert die Arbeit nicht.“
- Problem: „sie“ und „die Arbeit“ beziehen sich grammatisch auf Zhan et al. Die Gründe und die fehlende Größenvariation stammen aber von Lu et al. Belegt ist das durch die Fassung vor Commit `8a00510`, in der „Als Gründe für die Schwierigkeiten nennen sie“ direkt auf Lu et al. folgte.
- Vorschlag: Die beiden Zhan-Sätze unverändert ans Absatzende stellen und den Anschluss anpassen: „… IOB2-Symbole. Die Modellgröße variiert die Arbeit nicht. Die Vermutung zum vollständigen Fine-Tuning relativieren \textcite[Tab.~11]{zhan2026generativeNER}: In ihren Vergleichen auf CoNLL-2003 und GENIA liegt LoRA um 0{,}14 bis 0{,}48 F$_1$-Punkte über vollständigem Fine-Tuning, was für Qwen3.5 und MultiNERD jedoch keine Gleichwertigkeit belegt.“
- Erhalten: alle Zahlen, Zitate und die Einschränkung zu Qwen3.5 und MultiNERD.
- Sicherheit: Die Zuordnung ist über den Git-Verlauf belegt, nicht erneut an Lu et al. geprüft.
- Kernaussage: Lu et al. finden LLMs bei tokenbasierter NER schwach und nennen drei Gründe; Zhan et al. finden LoRA etwa gleichauf mit vollständigem Fine-Tuning.

**K3-20 · 3.2.3, S. 29 (ebenso 4.4.5, S. 47) · Niedrig · SA**
- Original: „In ihren Prüffällen auf CoNLL-2003 und GENIA“; S. 47: „in den Prüffällen von Zhan et al.“
- Problem: „Prüffälle“ (test cases) ist für Versuchsbedingungen ungewöhnlich.
- Vorschlag: „In ihren Vergleichen auf …“, „in den Vergleichen von …“. Sicherheit: sicher.

**K3-21 · 3.2.3, S. 30 · Niedrig · SA**
- Original: „Drei der vier Arbeiten konvergieren in einem Punkt“
- Vorschlag: „Drei der vier Arbeiten stimmen in einem Punkt überein“. Sicherheit: sicher.

**K3-22 · 3.2.4, S. 30 · Mittel · SA (Komma-Apposition nach Entfernen der Gedankenstriche)**
- Original: „Die Grenzen dieser Vergleiche, Datenbasis, Ausgabeformate, Ressourcenerfassung, Größenbereich und Herkunft der Encoder-Referenzwerte, bündelt Abschnitt~\ref{sec:research_gap} zur Forschungslücke.“
- Problem: Subjekt und Objekt sind schwer zu erkennen; man liest zunächst eine Reihe von sechs Subjekten.
- Vorschlag: „Die Grenzen dieser Vergleiche (Datenbasis, Ausgabeformate, Ressourcenerfassung, Größenbereich und Herkunft der Encoder-Referenzwerte) bündelt Abschnitt~\ref{sec:research_gap} zur Forschungslücke.“
- Sicherheit: sicher.

**K3-23 · 3.3, S. 31 · Niedrig · ST**
- Original: „Und \textcite{dukic2024lookingright} beobachten, …“
- Vorschlag: „Zudem beobachten \textcite{dukic2024lookingright}, …“ Sicherheit: sicher.

**K3-24 · 3.4, S. 31 · Niedrig · UP**
- Original: „Die generativen Ausgabeformate der direkten Vergleiche wachsen überdies mit der Eingabelänge“
- Problem: Nicht die Formate wachsen, sondern die Ausgaben.
- Vorschlag: „Die Ausgaben der generativen Formate in den direkten Vergleichen wachsen überdies mit der Eingabelänge“. Sicherheit: sicher.

**K3-25 · 3.5, S. 33 · Niedrig · ST**
- Original: „…; im Zero-Shot-Regime könnte Vorwissen über die Artikel die Ergebnisse dagegen begünstigen; diese Frage kann die Arbeit nicht klären.“
- Vorschlag: Das zweite Semikolon durch einen Punkt ersetzen: „… begünstigen. Diese Frage kann die Arbeit nicht klären.“ Sicherheit: sicher.

### Kapitel 4 Methodik

**K4-01 · Kapiteleinleitung, S. 35 · Niedrig · ST**
- Original: „Zu den hier beschriebenen Läufen ordnet Anhang~\ref{app:reproduzierbarkeit} jedem Lauf … zu“
- Vorschlag: „Anhang~\ref{app:reproduzierbarkeit} ordnet jedem hier beschriebenen Lauf Konfiguration, Codestand, Revisionen und Ausgabeverzeichnis zu“. Sicherheit: sicher.

**K4-02 · 4.1, S. 35, Aufzählung · Mittel · EF (Durchkopplung)**
- Original: „\textbf{Encoder Fine-Tuning:}“, „\textbf{LLM Zero-Shot:}“, „\textbf{LLM Fine-Tuning mit LoRA (QLoRA-Regime):}“
- Problem: Zusammensetzungen mit Abkürzungen und englischen Bestandteilen werden durchgekoppelt (vgl. „LLM-Pipeline“, „Zero-Shot-Regime“ im selben Kapitel).
- Vorschlag: „Encoder-Fine-Tuning“, „LLM-Zero-Shot“, „LLM-Fine-Tuning mit LoRA (QLoRA-Regime)“. Sicherheit: sicher.

**K4-03 · 4.1, S. 36 · Niedrig · RD**
- Original: „Der Thinking-Modus wurde nach der generativen Validierung eines abgebrochenen 4B-Vorlaufs auf Validierungsdaten für alle Läufe deaktiviert“
- Problem: „Validierung … auf Validierungsdaten“ ist doppelt, und „auf Validierungsdaten für alle Läufe“ liest sich als Einheit.
- Vorschlag: „Nach der generativen Validierung eines abgebrochenen 4B-Vorlaufs wurde der Thinking-Modus für alle Läufe deaktiviert; die Entscheidung beruht damit nur auf Validierungsdaten“.
- Erhalten: der Hinweis, dass keine Testdaten verwendet wurden. Sicherheit: sicher (so steht es auch in Anhang A.2.1).

**K4-04 · 4.1, S. 37 · Niedrig · UE**
- Original: „Das Gesamt-F$_1$ bewertet damit …“
- Problem: Sonst ist F$_1$ maskulin („einen F$_1$ von“).
- Vorschlag: „Der Gesamt-F$_1$ bewertet damit …“ Sicherheit: sicher.

**K4-05 · 4.2.2, S. 37 und 38 · Mittel · UP (Doppelbedeutung)**
- Original: S. 37 „die dichte Größenleiter mit gemeinsamem Tokenizer“; S. 38 „alle drei Größen [sind] dichte Decoder mit hybrider Attention“.
- Problem: „dicht“ heißt auf S. 37 „fein abgestuft“, auf S. 38 „dense“ (ohne Mixture-of-Experts).
- Vorschlag: S. 37 „die fein abgestufte Größenleiter“; S. 38 unverändert. Sicherheit: sicher.

**K4-06 · 4.2.2, S. 38 (ebenso Anhang A.2.2, S. 100) · Mittel · UP**
- Original: „Von den multimodal angelegten Checkpoints laden beide Pipelines … nur das Textmodell“
- Problem: „beide Pipelines“ bezeichnet in 4.1 und A.2.3 Encoder- und LLM-Pipeline; hier sind die beiden LLM-Regime gemeint.
- Vorschlag: „laden beide LLM-Regime (Zero-Shot und QLoRA) …“.
- Sicherheit: Rückfrage; gemeint sind vermutlich Zero-Shot- und QLoRA-Inferenz einschließlich QLoRA-Training.

**K4-07 · 4.3.1, S. 38 · Niedrig · SA**
- Original: „Werte auf den offiziellen Splits sind nicht nachweisbar“
- Vorschlag: „Werte auf den offiziellen Splits sind nicht auffindbar“. Die vorsichtige Lesart bleibt. Sicherheit: sicher.

**K4-08 · 4.3.3, S. 40 · Mittel · SA (Projektjargon „canonical“)**
- Original: „Die kanonischen Läufe wurden anschließend mit drei Seeds neu trainiert.“
- Problem: Der Begriff wird nicht eingeführt; er stammt aus den Dateinamen (`*_canonical`).
- Vorschlag: „Die Läufe der Hauptstudie wurden anschließend mit drei Seeds neu trainiert.“ Sicherheit: sicher.

**K4-09 · 4.4.1, S. 41 und Tab. 4.9, S. 57 · Mittel · SA (Jargon)**
- Original: „ohne Revisions-Pins und mit reduzierten Metadaten“ (zweimal); „Eine bitgenaue Reproduzierbarkeit wird für keine Kohorte beansprucht“; S. 55: „in einer fest gepinnten … Revision“.
- Problem: „Pins“, „gepinnt“ und „Kohorte“ (cohort) sind Jargon.
- Vorschlag: „ohne explizite Revisionsangabe“ (so formuliert schon Anhang A.2.6); „wird für keinen der Läufe beansprucht“; „in einer festgelegten … Revision“. Sicherheit: sicher.

**K4-10 · 4.4.3, S. 45 und 46, 4.6, S. 55 · Niedrig · SA**
- Original: „erzeugt das Modell … greedy Vorhersagen“, „greedy decodiert“, „wird greedy, ohne Thinking … decodiert“
- Problem: Das englische Adjektiv bleibt unflektiert.
- Vorschlag: „erzeugt das Modell … Vorhersagen per Greedy-Decoding“, „mit Greedy-Decoding erzeugt“. Sicherheit: sicher.

**K4-11 · 4.4.6, S. 48 · Mittel · UP**
- Original: „Das Element muss ein Objekt mit allen vier Feldern sein, wobei zusätzliche Felder toleriert und verworfen werden“
- Problem: „verworfen“ steht im übergeordneten Satz für ganze Elemente; „toleriert und verworfen“ wirkt widersprüchlich.
- Vorschlag: „…, wobei zusätzliche Felder zulässig sind und ignoriert werden“. Sicherheit: sicher.
- Kernaussage: Zusatzfelder machen ein Element nicht ungültig.

**K4-12 · 4.5.1, S. 49 · Niedrig · UE**
- Original: „der mikro-gemittelte F$_1$-Wert“, „makro-gemittelt“
- Vorschlag: „mikrogemittelt“ und „makrogemittelt“; „Mikro-F$_1$“ bleibt unverändert. Sicherheit: sicher.

**K4-13 · 4.5.2 „Inferenzlatenz und Durchsatz“, S. 51 · Hoch · UP (fachlich missverständlich)**
- Original: „Nicht enthalten sind Laden, Tokenisierung, Datentransfer, Dekodierung, Validierung, Metrikberechnung und Speichern.“
- Problem: „Decoding“ bezeichnet in der Arbeit 41-mal die Generierung (in 3.2.1 heißt es auch „Dekodierung“). Leser können daher verstehen, die Generierung sei nicht in der Latenz enthalten; das widerspricht dem Satz davor. Gemeint ist die Detokenisierung: `tokenizer.batch_decode` läuft in `ba-ner/src/decoder/inference.py` nach dem Ende der Zeitmessung.
- Vorschlag: „Nicht enthalten sind Laden, Tokenisierung, Datentransfer, Detokenisierung, Validierung, Metrikberechnung und Speichern.“
- Sicherheit: sicher (über den Code geprüft).
- Kernaussage: Die Latenz misst den reinen Modellaufruf einschließlich der Generierung.

**K4-14 · 4.5.2, S. 52 · Hoch · UP (falsche Stelle, eigener Fehler aus `e624d7e`)**
- Original: Der Absatz „Ressourcenschwellen für FF3“ endet mit „Die Schwellen beschreiben die gemessenen Bedingungen (Batchgröße~1, jeweiliger GPU-Typ), keine Anforderungen einer konkreten Anwendung. Effizienzvergleiche werden daher vorrangig innerhalb gleicher Hardware- und Präzisionsbedingungen interpretiert.“
- Problem: Der letzte Satz gehörte zum Absatz „Vergleichbarkeit“ (Hardware und Präzision). Durch den eingefügten Absatz bezieht sich „daher“ jetzt auf die Schwellen.
- Vorschlag: Den Satz unverändert zurück an das Ende von „Vergleichbarkeit“ setzen, nach „… unterscheiden sich zudem in Präzision und Decoding.“
- Sicherheit: sicher (Herkunft im Diff von `e624d7e` belegt).
- Kernaussage: Effizienzwerte sind nur bei gleicher Hardware und Präzision direkt vergleichbar.

**K4-15 · 4.5.3, S. 52 · Mittel · FP**
- Original: „wobei Typen außerhalb des Inventars als halluzinierte Kategorien gelten \cite{tedeschi2022multinerd}“
- Problem: Es ist unklar, ob die Datensatzarbeit zu MultiNERD den Begriff „halluzinierte Kategorien“ belegt oder nur das Typinventar.
- Vorschlag: Je nach Quelle entweder den Beleg an „Inventar“ setzen („Typen außerhalb des Inventars \cite{tedeschi2022multinerd} gelten als halluzinierte Kategorien“) oder unverändert lassen.
- Sicherheit: Quelle prüfen.

**K4-16 · 4.5.4, S. 54 · Niedrig · UE**
- Original: „Der Einfluss der Parameteranzahl auf Performanz, Effizienz und Formtreue“
- Problem: Sonst heißt es 26-mal „Erkennungsleistung“.
- Vorschlag: „auf Erkennungsleistung, Effizienz und Formtreue“. Sicherheit: sicher.

**K4-17 · 4.5.4, S. 54 · Mittel · SA**
- Original: „Skalierungsgesetze im Sinne von Potenzgesetzfits“
- Problem: Das Kompositum ist aus „power-law fits“ übernommen.
- Vorschlag: „Skalierungsgesetze im Sinne angepasster Potenzgesetze“. Sicherheit: sicher.

**K4-18 · 4.6, S. 55 · Niedrig · UE**
- Original: „mit den Domänen künstliche Intelligenz, Literatur, …“
- Problem: Als Domänenname steht es sonst groß (K1, K2, K5, alle Tabellen).
- Vorschlag: „Künstliche Intelligenz“. Sicherheit: sicher.

**K4-19 · 4.6, S. 55 · Mittel · EF (Rechtschreibung)**
- Original: „laut der Lizenzdatei des Repositories“
- Problem: Der Duden führt den Genitiv „des Repositorys“ (Plural „Repositorys“).
- Vorschlag: „des Repositorys“. Sicherheit: sicher.

**K4-20 · Tab. 4.6, S. 48 · Niedrig · SA**
- Original: „Warm-up-Anteil (cosine)“
- Vorschlag: „Warm-up-Anteil (Kosinus-Schedule)“. Sicherheit: sicher.

**K4-21 · Tab. 4.9, S. 57 · Niedrig · SA**
- Original: „Provenienz“ (Zeilenkopf)
- Vorschlag: „Herkunft der Modelle und Daten“ oder kurz „Herkunft“. Sicherheit: sicher.

**K4-22 · Tab. 4.1, S. 36 · Niedrig · UE**
- Original: Der Spaltenkopf „trainierbar“ ist klein geschrieben, die übrigen Köpfe sind groß.
- Vorschlag: „Trainierbar“. Sicherheit: sicher.

**K4-23 · 4.4.5, S. 47 · Niedrig · FP**
- Original: „in den Prüffällen von \textcite[Anh.~D]{zhan2026generativeNER} … 0,14 bis 0,48 F$_1$-Punkte“
- Problem: Kapitel 3 belegt dieselben Zahlen mit „Tab.~11“.
- Vorschlag: Prüfen, ob Tab. 11 in Anhang D liegt; dann die Fundstelle einheitlich angeben, z. B. „Anh.~D, Tab.~11“.
- Sicherheit: Quelle prüfen.

**K4-24 · 4.5.2, S. 51 (ebenso Anhang A.2.8, S. 103) · Niedrig · UP**
- Original: „der explorative Vorlauf … benötigte … allein auf dem Knoten 15{,}5~s je Schritt“
- Problem: „allein“ ist mehrdeutig: „nur auf dem Knoten“ oder „als einziger Job auf dem Knoten“.
- Vorschlag: „… lief allein auf dem Knoten und benötigte 15{,}5~s je Schritt“. Sicherheit: sicher.

### Kapitel 5 Ergebnisse und Diskussion

**K5-01 · 5.1.1, S. 60 und 5.7.1, S. 72 (zweimal), 5.7.5, S. 78 · Mittel · UP**
- Original: „Unterschiede in Prompt, Decoding und Präzision“; „da sich auch Vortraining, Anpassungsverfahren und Präzision unterscheiden“; „Decoding und Präzision das Design nicht erlaubt“.
- Problem: Im Ergebniskapitel steht „Präzision“ neben der Metrik Precision (Tabellenspalte „P“, Text „Precision“). Abschnitt 4.1 führt „numerische Präzision“ ein, das Kapitel 5 verkürzt es.
- Vorschlag: An diesen vier Stellen „numerische Präzision“ schreiben.
- Sicherheit: sicher.
- Kernaussage: Die Vergleiche unterscheiden sich auch im Zahlenformat (BF16 gegenüber 4-Bit).

**K5-02 · 5.1.1, S. 60 · Niedrig · ST**
- Original: „Die Zero-Shot-Sampling-Streuung wurde nicht gemessen.“
- Vorschlag: „Die Streuung durch das Sampling im Zero-Shot-Regime wurde nicht gemessen.“ Sicherheit: sicher.

**K5-03 · 5.1.2, S. 60 · Mittel · UP**
- Original: „Die Typen ANIM, BIO, FOOD und PLANT, deren Silberreferenz nach \textcite{tedeschi2022multinerd} häufig verwechselt oder unvollständig verlinkt ist“
- Problem: Nicht die Referenz wird verwechselt, sondern die Typen werden in der Referenz verwechselt. Die Gruppenüberschrift in Tab. A.8 sagt richtig „Häufig verwechselte oder unvollständig verlinkte Typen“.
- Vorschlag: „Die Typen ANIM, BIO, FOOD und PLANT, die in der Silberreferenz nach \textcite{tedeschi2022multinerd} häufig verwechselt oder unvollständig verlinkt sind“.
- Sicherheit: sicher.
- Kernaussage: Die schwachen Typen sind die mit unzuverlässiger Annotation.

**K5-04 · 5.1.2, S. 60 · Mittel · UP**
- Original: „Diese Verteilung erklärt, warum Mikro- und Makro-F$_1$ die Klassen unterschiedlich ordnen.“
- Problem: „Klassen“ kann als Entitätsklassen gelesen werden; gemeint ist die Reihenfolge von Encoder und LLM.
- Vorschlag: „… warum Mikro- und Makro-F$_1$ Encoder und LLM unterschiedlich ordnen.“
- Sicherheit: sicher (der Folgesatz spricht von Encoder und LLM).
- Kernaussage: Im Mikro-F$_1$ führt der Encoder, im Makro-F$_1$ liegen beide nah beieinander.

**K5-05 · 5.1.2, S. 61 · Niedrig · SA**
- Original: „der Makro-Wert gewichtet die seltenen Typen gleich, …, und ergibt nahe beieinanderliegende Werte“
- Problem: „Der Makro-Wert ergibt Werte“ ist schief.
- Vorschlag: „…; im Makro-F$_1$ liegen beide deshalb nahe beieinander.“ Sicherheit: sicher.

**K5-06 · 5.1.2, S. 61 · Niedrig · UP**
- Original: „erkennt das 27B-Modell … MYTH, CEL, DIS, MEDIA, EVE und ANIM nur zur Hälfte“
- Problem: „nur zur Hälfte“ ist vage.
- Vorschlag: „… nur mäßig (F$_1$ zwischen 45 und 56 Punkten)“. Die Werte stammen aus Tab. A.8 (Zero-Shot 27B: ANIM 45,0 bis MYTH 56,3).
- Sicherheit: sicher (Tabellenwerte).

**K5-07 · 5.1.2, S. 61 · Niedrig · ST**
- Original: „Der niedrige Wert zeigt vor allem eine abweichende Typdefinition; der Recall von 18,7 … zeigt zugleich, dass …“
- Vorschlag: Das zweite „zeigt“ durch „macht zugleich deutlich“ ersetzen. Sicherheit: sicher.

**K5-08 · 5.2, S. 63 · Hoch · FP (Zahlenaussage)**
- Original: „…, und die acht Testsätze ohne Referenzentität lässt nur das 27B-Modell leer.“
- Problem: Laut `generated/summary.json` (`groups/qwen27-qlora/agg/form/empty_correct`) lässt das 27B-Modell nur 3, 4 bzw. 3 dieser acht Sätze leer (Seeds 42, 123, 456); das 0.8B- und das 4B-Modell lassen keinen leer. Der Satz liest sich, als blieben alle acht leer.
- Vorschlag, mit zwei Lesarten:
  - (a) „…; von den acht Testsätzen ohne Referenzentität lässt nur das 27B-Modell einige leer (3 bis 4 je Seed).“
  - (b) „…; nur das 27B-Modell lässt überhaupt Testsätze ohne Referenzentität leer (3 bis 4 der 8 je Seed).“
  - Beide Lesarten sind mit den Daten vereinbar; (a) ist näher am Original.
- Sicherheit: über die Daten geprüft.
- Kernaussage: Nur das größte Modell gibt bei Sätzen ohne Entität gelegentlich korrekt leere Antworten.

**K5-09 · 5.2, S. 63 · Mittel · UP (Agens)**
- Original: „Das 4B-Modell erzeugt überwiegend gültiges JSON, verwirft aber Tausende Elemente wegen Text-Abweichungen und gibt häufig leere Arrays aus“
- Problem: Das Modell verwirft nichts; die Validierung verwirft Elemente.
- Vorschlag: „Das 4B-Modell erzeugt überwiegend gültiges JSON, doch wegen Textabweichungen werden Tausende Elemente verworfen, und es gibt häufig leere Arrays aus“.
- Sicherheit: sicher.
- Kernaussage: Beim 4B-Modell scheitert die Ausgabe am Textabgleich, nicht am JSON.

**K5-10 · 5.2 „Nachinferenz mit 1\,024 Tokens“, S. 63 · Niedrig · UP**
- Original: „Werden nur die 62 bis 72 nicht parsebaren Sätze je Lauf … neu generiert“; „Auf den neu generierten Sätzen“
- Problem: Neu generiert werden die Antworten, nicht die Sätze.
- Vorschlag: „Werden die Antworten nur für die 62 bis 72 nicht parsebaren Sätze je Lauf … neu generiert“; „Auf diesen Sätzen“. Sicherheit: sicher.

**K5-11 · 5.2, S. 64 und 5.6, S. 70 · Mittel · SA (Agens)**
- Original: „Teilmengenauswahl und GPU-Wechsel isolieren jedoch keinen reinen Effekt der Antwortlänge.“; S. 70: „der verbleibende Unterschied zu MultiNERD umfasst außerdem Unterschiede in Texten und Annotation und isoliert keinen reinen Domäneneffekt.“
- Problem: Störfaktoren „isolieren“ nichts; gemeint ist, dass sich der Effekt wegen dieser Faktoren nicht isolieren lässt.
- Vorschlag: „Wegen der Teilmengenauswahl und des GPU-Wechsels lässt sich daraus jedoch kein reiner Effekt der Antwortlänge isolieren.“; S. 70: „…; ein reiner Domäneneffekt lässt sich daraus nicht isolieren.“
- Sicherheit: sicher.

**K5-12 · 5.4, S. 65 · Niedrig · UP**
- Original: „12,1~h für das 0.8B-Modell auf derselben H100“
- Problem: „derselben“ heißt dieselbe Karte; gemeint ist derselbe GPU-Typ (so in 5.6).
- Vorschlag: „ebenfalls auf einer H100“. Sicherheit: sicher.

**K5-13 · 5.4, S. 65 · Niedrig · SA**
- Original: „Selbst das kleinste Adaptertraining dauert rund das 28-Fache des vollständigen Fine-Tunings von DeBERTa-v3-base“
- Vorschlag: „… dauert rund 28-mal so lange wie das vollständige Fine-Tuning von DeBERTa-v3-base“. Sicherheit: sicher.

**K5-14 · 5.4, S. 67 · Niedrig · ST**
- Original: Absatztitel „Trade-off.“
- Vorschlag: optional „Abwägung von Qualität und Ressourcen.“; „Trade-off“ ist im Fach geläufig. Sicherheit: sicher.

**K5-15 · 5.5, S. 68 · Niedrig · UP**
- Original: „erreicht DeBERTa-v3-large je LLM-Seed einen Recall von 93,8 bis 96,6 (seedgleich 96,6/93,8/95,1)“
- Problem: „seedgleich“ ist knapp und nicht erklärt; die Reihenfolge der Seeds bleibt offen.
- Vorschlag: „(bei jeweils gleichem Seed beider Modelle: 96,6, 93,8 und 95,1)“, falls die Reihenfolge 42, 123, 456 ist, mit dieser Angabe.
- Sicherheit: Rückfrage (Reihenfolge prüfen).

**K5-16 · 5.5, S. 68 · Niedrig · ST**
- Original: „nähert sich die Auslassungsrate der QLoRA-Modelle der der Encoder an“
- Vorschlag: „… derjenigen der Encoder an“. Sicherheit: sicher.

**K5-17 · 5.6, S. 70 · Niedrig · SA**
- Original: „Die Profile unterscheiden sich im Ursprung des Verlusts:“
- Vorschlag: „Die Profile unterscheiden sich darin, worauf der Verlust entfällt:“ Sicherheit: sicher.

**K5-18 · 5.6, S. 70 · Niedrig · ST (Satz mit 44 Wörtern und Zahlenkette)**
- Original: „Auf CrossNER betrugen Latenz je Satz und GPU-Spitzenspeicher 25,7~ms und 1,7~GiB für DeBERTa-v3-large (das 1,6-Fache …) sowie 14\,691~ms (Median 13\,763~ms) und 18,6~GiB für Qwen3.5-27B (das 2,7-Fache; dessen MultiNERD-Wert stammt von der H100~NVL).“
- Vorschlag: Zwei Sätze: „Auf CrossNER betrugen Latenz je Satz und GPU-Spitzenspeicher für DeBERTa-v3-large 25,7~ms und 1,7~GiB, das 1,6-Fache seiner MultiNERD-Latenz auf demselben GPU-Typ. Für Qwen3.5-27B waren es 14\,691~ms (Median 13\,763~ms) und 18,6~GiB, das 2,7-Fache seiner MultiNERD-Latenz; dieser MultiNERD-Wert stammt von der H100~NVL.“
- Sicherheit: sicher.

**K5-19 · 5.7.1, S. 71 · Mittel · UP (Bezug)**
- Original: „Beide Intervalle schließen null aus; ihre praktische Größenordnung unterscheidet sich jedoch deutlich.“
- Problem: „ihre“ bezieht sich auf die Intervalle; gemeint sind die Abstände (1,34 gegenüber 0,18 Punkte).
- Vorschlag: „…; die beiden Abstände unterscheiden sich in ihrer praktischen Größenordnung jedoch deutlich.“
- Sicherheit: sicher.
- Kernaussage: Beide Unterschiede sind statistisch belastbar, aber nur der erste ist praktisch groß.

**K5-20 · 5.7.1, S. 71 · Niedrig · EF (Präposition)**
- Original: „Es liegt nahe am Befund von … und bei der Beobachtung von …“
- Vorschlag: „Es liegt nahe am Befund von … und an der Beobachtung von …“ Sicherheit: sicher.

**K5-21 · 5.7.1, S. 71 · Mittel · UP**
- Original: „Für die Einordnung sind drei Eigenschaften des Setups relevant.“
- Problem: Der dritte Punkt („Schließlich geht der Vorsprung des Encoders fast vollständig auf den Recall zurück …“) ist ein Befund, keine Eigenschaft des Versuchsaufbaus.
- Vorschlag: „Für die Einordnung sind drei Punkte relevant.“ Sicherheit: sicher.

**K5-22 · 5.7.1, S. 72 · Mittel · RD/UP (eigener Fehler aus `c482382`)**
- Original: „Mit deren Nachinferenz sinkt der Anteil der Auslassungen des 27B-Modells von 5,1 auf 2,5\,\% gegenüber 2,4\,\% bei DeBERTa-v3-large (Tabellen …). Die Nachinferenz dieser Sätze (Tabelle~\ref{tab:limit-multinerd}) macht daraus einen Messwert: Mit vollständigen Antworten liegt das 27B-Modell …“
- Problem: Die Nachinferenz wird zweimal eingeführt. „daraus“ bezog sich ursprünglich auf die Zuschreibung des Rückstands zu den abgeschnittenen Antworten; nach dem eingefügten Satz fehlt der Bezug.
- Vorschlag: „… abgeschnittenen Antworten. Die Nachinferenz dieser Sätze (Tabelle~\ref{tab:limit-multinerd}) macht daraus einen Messwert: Der Anteil der Auslassungen des 27B-Modells sinkt von 5,1 auf 2,5\,\% gegenüber 2,4\,\% bei DeBERTa-v3-large (Tabelle~\ref{tab:fehlertypen}). Mit vollständigen Antworten liegt das 27B-Modell knapp hinter DeBERTa-v3-large und 0,14 Punkte vor DeBERTa-v3-base; …“ (Rest unverändert).
- Erhalten: alle Zahlen und Verweise. Sicherheit: sicher.
- Kernaussage: Die Nachinferenz belegt messbar, dass der Rückstand des LLM überwiegend aus abgeschnittenen Antworten stammt.

**K5-23 · 5.7.1, S. 73 · Hoch · UP (missverständliche Zahl, eigener Fehler aus `9b1e273`)**
- Original: „Auf Span-Ebene (Seed~42) trifft nur DeBERTa-v3-large 319 Referenzspans exakt in Grenzen und Typ, nur das 27B-Modell mit 1\,024 Tokens 331; in 699 Fällen …“
- Problem: Lesbar als „DeBERTa-v3-large trifft insgesamt nur 319 Referenzspans“. Gemeint sind 319 Referenzspans, die ausschließlich DeBERTa-v3-large trifft, und 331, die ausschließlich das 27B-Modell trifft.
- Vorschlag: „Auf Span-Ebene (Seed~42) gibt es 319 Referenzspans, die nur DeBERTa-v3-large exakt in Grenzen und Typ trifft, und 331, die nur das 27B-Modell mit 1\,024 Tokens trifft; in 699 Fällen …“
- Sicherheit: sicher (Strata S1 und S2 aus `ba-ner/scripts/check_sample.py`).
- Kernaussage: Beide Modelle treffen jeweils einige Hundert Spans, die das andere verfehlt.

**K5-24 · 5.7.2, S. 74 · Mittel · SA (schiefer Vergleich)**
- Original: „…, dass die gemessene LLM-Inferenz um Größenordnungen hinter den Encodern zurückbleibt“
- Vorschlag: „…, dass die gemessene LLM-Inferenz um Größenordnungen langsamer ist als die der Encoder“. Sicherheit: sicher.

**K5-25 · 5.7.2, S. 74 · Mittel · UP**
- Original: „Die Faktoren von 40 bis 220 für BIO-Ausgaben \cite[Tab.~IV]{obeidat2025llm}“
- Problem: BIO ist in dieser Arbeit auch ein MultiNERD-Typ. Kapitel 3 (S. 29) beschreibt dieselbe Quelle mit „IOB2-Format“; das Abkürzungsverzeichnis führt BIO unter IOB.
- Vorschlag: „für IOB2-Ausgaben“. Sicherheit: sicher.

**K5-26 · 5.7.2, S. 75 · Niedrig · SA**
- Original: „…, die beide bei jeder Entität anfallen, gleich wie kompakt sie kodiert ist“
- Vorschlag: „…, unabhängig davon, wie kompakt sie kodiert ist“. Sicherheit: sicher.

**K5-27 · 5.7.3, S. 75 · Mittel · FP (schiefer Vergleich)**
- Original: „Bei den Encodern liegt die große Variante bei allen drei Seeds vor der kleinen; diese Stabilität über Seeds unterscheidet sich von \textcite{keraghel2024recent}, wo die Rangfolge … zwischen den Datensätzen wechselt“
- Problem: Verglichen werden zwei Dimensionen: Seeds hier, Datensätze dort.
- Vorschlag, Rückfrage: (a) „…; bei \textcite{keraghel2024recent} wechselt die Rangfolge … dagegen zwischen den Datensätzen“; (b) unverändert lassen, falls der Kontrast bewusst so gemeint ist.
- Sicherheit: Rückfrage.

**K5-28 · 5.7.3, S. 75 · Niedrig · EF (Numerus)**
- Original: „als der zwischen den kleinsten und größten QLoRA-Konfigurationen (1,3 Punkte)“
- Vorschlag: „als der zwischen der kleinsten und der größten QLoRA-Konfiguration (1,3 Punkte)“. Sicherheit: sicher.

**K5-29 · 5.7.4, S. 76 · Niedrig · ST**
- Original: „mit den Ressourcenwerten des 256er-Laufs“
- Vorschlag: „des Laufs mit 256 Ausgabetokens“. Sicherheit: sicher.

**K5-30 · 5.7.4, S. 76 · Niedrig · ST**
- Original: „Wo das 27B-Modell zulässig ist, ist es auch DeBERTa-v3-large, das auch mit 1\,024 Tokens vor ihm liegt.“
- Vorschlag: „…, das selbst mit 1\,024 Tokens vor ihm liegt.“ Sicherheit: sicher.

**K5-31 · 5.7.4, S. 76 · Mittel · UP**
- Original: „Eine solche Empfehlung bei kleinen Qualitätsunterschieden diskutieren auch \textcite{obeidat2025llm}.“
- Problem: Davor steht keine ausdrückliche Empfehlung, sondern die Zulässigkeitsaussage. Der Bezug von „Eine solche Empfehlung“ ist unklar.
- Vorschlag: „Die Empfehlung eines Encoders trotz kleiner Qualitätsunterschiede diskutieren auch \textcite{obeidat2025llm}.“
- Sicherheit: Rückfrage und Quelle prüfen (diskutieren Obeidat et al. genau das?).

**K5-32 · 5.7.5, S. 78 · Niedrig · RD (keine Änderung empfohlen)**
- Original: „Als Hauptvergleiche gelten die aus FF1 und FF2 abgeleiteten Vergleiche … die übrigen Vergleiche …“ Der Satz steht fast gleich in 4.5.1 (S. 50).
- Bewertung: Die Wiederholung ist im Abschnitt „Belastbarkeit“ fachlich sinnvoll. Die dreifache Verwendung von „Vergleiche“ in einem Satz lässt sich mildern: „… die übrigen (1\,024 Tokens, CrossNER) sind explorativ“.
- Sicherheit: sicher.

**K5-33 · 5.7.5, S. 79 · Niedrig · UP**
- Original: „Ihre Auswirkungen auf die Rangfolge sind nicht generell bekannt; beim Ausgabelimit ist eine Änderung der Rangfolge auf CrossNER bereits nachgewiesen.“
- Problem: „nicht generell bekannt“ ist vage.
- Vorschlag, mit zwei Lesarten: (a) „Wie sie sich auf die Rangfolge auswirken, lässt sich nicht allgemein angeben; …“; (b) „Wie sie sich auf die Rangfolge auswirken, ist überwiegend nicht bekannt; …“.
- Sicherheit: Rückfrage.

### Kapitel 6 Fazit und Ausblick

**K6-01 · 6.1, S. 80 · Niedrig · ST**
- Original: „Nach der Nachinferenz am Ausgabelimit mit 1\,024 Tokens, die die am Limit abgeschnittenen Antworten ersetzt, erreicht das LLM 95,6 Punkte“
- Problem: „Nach der Nachinferenz“, „die die“ und zweimal „Limit“.
- Vorschlag: „Ersetzt eine Nachinferenz mit 1\,024 Tokens die am Limit abgeschnittenen Antworten, erreicht das LLM 95,6 Punkte“. Sicherheit: sicher.

**K6-02 · 6.1, S. 81 · Niedrig · UE**
- Original: „Innerhalb der QLoRA-Familie steigt die Latenz …“
- Problem: „Familie“ bezeichnet sonst die Modellfamilie; QLoRA ist ein Regime.
- Vorschlag: „Innerhalb der QLoRA-Konfigurationen steigt …“ Sicherheit: sicher.

**K6-03 · 6.1, S. 81 · Mittel · UP (eigener Fehler aus `f63af31`)**
- Original: „Rund drei Viertel dieses Vorsprungs gehen allerdings auf Vorhersagen für Entitäten nicht abgebildeter CrossNER-Typen zurück, die als Fehler zählen (Abschnitt~\ref{sec:disk-effizienz}).“
- Problem: Offen bleibt, wessen Vorhersagen gemeint sind, und der Relativsatz bezieht sich grammatisch auf „Typen“. Laut 5.7.4 entsteht der Vorsprung, weil der Encoder solche falsch positiven Spans häufiger erzeugt: 61 gegenüber 53 % seiner falsch positiven Spans bei insgesamt geringerer Precision.
- Vorschlag: „Rund drei Viertel dieses Vorsprungs entfallen allerdings auf falsch positive Spans auf Entitäten nicht abgebildeter CrossNER-Typen, die der Encoder häufiger erzeugt als das LLM (Abschnitt~\ref{sec:disk-effizienz}).“
- Erhalten: „rund drei Viertel“ (2,1 auf 0,5 Punkte) und der Verweis.
- Fachlich zu prüfen: 5.7.4 schränkt ein: „sofern man auch Teilspans solcher Entitäten nicht als Fehler wertet“. Diese Einschränkung fehlt in 6.1. Ob sie dort ergänzt werden soll, ist eine inhaltliche Entscheidung; sie wird nicht ohne Freigabe eingefügt.
- Sicherheit: sicher (Zahlen aus 5.7.4).
- Kernaussage: Der CrossNER-Vorsprung des LLM ist großteils ein Artefakt der Typabbildung.

**K6-04 · 6.1 „Beitrag“, S. 82 · Niedrig · UE**
- Original: „Mit 1\,024 Ausgabetokens bleibt keine Antwort unparsebar.“
- Problem: Sonst steht 13-mal „nicht parsebar“.
- Vorschlag: „Mit 1\,024 Ausgabetokens sind alle Antworten parsebar.“ Sicherheit: sicher.

**K6-05 · 6.3, S. 84 · Mittel · SA (Zeugma und Bezug)**
- Original: „Er sollte das Zero-Shot-Regime einschließen, dessen nicht parsebare Antworten 3,8 bis 16,2\,\% der Referenzentitäten enthalten (…), sowie Batching und einheitliche Hardware, was die Empfehlungen zu FF3 auf die Offline-Verarbeitung ausweiten würde.“
- Problem: „einschließen“ passt zu „Zero-Shot-Regime“, aber kaum zu „Batching und einheitliche Hardware“. „was“ kann sich auf alles oder nur auf das Batching beziehen.
- Vorschlag: „Er sollte das Zero-Shot-Regime einschließen, dessen nicht parsebare Antworten 3,8 bis 16,2\,\% der Referenzentitäten enthalten (…). Zudem sollte er Batching und einheitliche Hardware verwenden; das würde die Empfehlungen zu FF3 auf die Offline-Verarbeitung ausweiten.“
- Sicherheit: Rückfrage (bezieht sich die Ausweitung auf die Offline-Verarbeitung nur auf das Batching?).

### Anhang (nur Vorschläge; Umsetzung nur nach ausdrücklicher Freigabe, keine Kürzung)

**A-01 · A.1 „Läufe und Artefakte“, S. 95 · Niedrig · SA**
- Original: „wissenschaftlicher Vertrag mit Code- und Revisions-Hashes (\texttt{scientific\_contract.yaml})“
- Problem: Der Dateiname ist wörtlich übersetzt; „Vertrag“ ist im Deutschen unklar.
- Vorschlag: „Festlegung der wissenschaftlichen Laufparameter (\texttt{scientific\_contract.yaml}) mit Code- und Revisions-Hashes“. Alternativ den Ausdruck in Anführungszeichen setzen.
- Sicherheit: Rückfrage (was legt die Datei fest?).

**A-02 · A.1 „Explorative Vorläufe“, S. 97 · Niedrig · ST**
- Original: „dokumentiert die explorativen Vorläufe (Abschnitt …), je ein Lauf mit Seed~42, die auf dem Testsplit ausgewertet wurden und nicht in die Hauptauswertung eingehen; ihre Abweichung …“
- Vorschlag: „dokumentiert die explorativen Vorläufe (Abschnitt …; je ein Lauf mit Seed~42). Sie wurden auf dem Testsplit ausgewertet und gehen nicht in die Hauptauswertung ein; ihre Abweichung …“ Inhalt unverändert. Sicherheit: sicher.

**A-03 · Tab. A.3, S. 98 · Niedrig · UE**
- Original: „den CrossNER-Versuch und die Sensitivitätsanalysen“
- Vorschlag: „den CrossNER-Transfertest und die Sensitivitätsanalysen“. Sicherheit: sicher.

**A-04 · A.2, S. 99 · Niedrig · SA**
- Original: „für die Nachvollziehung der Läufe aber relevant sind“
- Vorschlag: „für die Nachvollziehbarkeit der Läufe aber relevant sind“. Sicherheit: sicher.

**A-05 · A.2, S. 99 und Überschrift A.2.10, S. 104 (Inhaltsverzeichnis S. v) · Niedrig · UE**
- Original: Im Text steht „die Rückfallstrategien des Parsers“, die Überschrift lautet „Parsing-Fallbacks“; Kapitel 4 schreibt „zwei Rückfallstrategien“.
- Vorschlag: Überschrift „Rückfallstrategien des Parsers“. Sicherheit: sicher.

**A-06 · A.2.1, S. 99 · Niedrig · RD**
- Original: „nachdem die generative Validierung auf Validierungsdaten nach der ersten und zweiten Epoche …“
- Vorschlag: „auf Validierungsdaten“ streichen; der übernächste Satz sagt bereits, die Entscheidung beruhe „ausschließlich auf Validierungsdaten“. Es geht keine Information verloren.
- Sicherheit: sicher.

**A-07 · A.2.2, S. 100 · Mittel · UP**
- Original: „Beide Pipelines laden die Modelle jedoch über \texttt{AutoModelForCausalLM}“
- Problem: Wie K4-06; in A.2.3 bezeichnet „Beide Pipelines“ zwei Absätze später Encoder- und LLM-Pipeline.
- Vorschlag: „Beide LLM-Regime laden die Modelle jedoch über …“. Sicherheit: Rückfrage wie bei K4-06.

**A-08 · A.2.3, S. 101 · Niedrig · SA**
- Original: „gemeinsame Module für Datensatzladung, Metrikberechnung und Metadatenprotokollierung“
- Vorschlag: „gemeinsame Module für das Laden der Datensätze, die Metrikberechnung und die Protokollierung der Metadaten“. Sicherheit: sicher.

**A-09 · A.2.6, S. 103 · Mittel · SA (Jargon „cohort“)**
- Original: „Der Umfang der Reproduzierbarkeitsangaben unterscheidet sich zwischen den Kohorten.“; „Eine bitgenaue Reproduzierbarkeit wird für keine Kohorte beansprucht“
- Vorschlag: „… unterscheidet sich zwischen Trainings- und Zero-Shot-Läufen.“; „… wird für keinen der Läufe beansprucht“.
- Sicherheit: sicher (der Absatz unterscheidet genau diese zwei Gruppen).

**A-10 · A.2.6, S. 103 · Niedrig · ST**
- Original: „Die Zero-Shot-Ausgaben entstehen zudem durch Sampling; ihre exakte Wiederholung setzt daher auch eine ununterbrochene Ausführung voraus, da der Zustand … nicht wiederhergestellt wird.“
- Problem: „daher … da“ ist doppelt kausal.
- Vorschlag: „Die Zero-Shot-Ausgaben entstehen zudem durch Sampling. Weil der Zustand des Zufallszahlengenerators bei einer Wiederaufnahme nicht wiederhergestellt wird, setzt ihre exakte Wiederholung auch eine ununterbrochene Ausführung voraus.“
- Sicherheit: sicher.

**A-11 · A.2.8, S. 103 · Mittel · UP**
- Original: „Der vom Trainer gespeicherte Wert erfasst nur die letzte Zuteilung ab Schritt 20\,000 …“
- Problem: „Wert“ hat keinen Bezug; gemeint ist die Trainingsdauer.
- Vorschlag: „Die vom Trainer gespeicherte Trainingsdauer erfasst nur …“ Sicherheit: sicher.

**A-12 · A.2.8, S. 103 · Niedrig · UP**
- Original: „der explorative Vorlauf mit zwei Epochen benötigte allein auf dem Knoten 15,5~s je Schritt“
- Vorschlag: wie K4-24: „… lief allein auf dem Knoten und benötigte 15,5~s je Schritt“. Sicherheit: sicher.

**A-13 · A.2.10, S. 104 · Niedrig · ST**
- Original: „in Tabelle~\ref{tab:formtreue} als Kennzahl Parse-Strategien geführt“
- Vorschlag: „als Kennzahl \enquote{Parse-Strategien} geführt“. Sicherheit: sicher.

**A-14 · Tab. A.8, Caption, S. 106 · Niedrig · ST**
- Original: „Precision und Recall je Typ in Tabelle~\ref{tab:app-typen-prf}.“
- Vorschlag: „Precision und Recall je Typ nennt Tabelle~\ref{tab:app-typen-prf}.“ Sicherheit: sicher.

### Generierte Tabellen und Abbildungen

**T-01 · Tab. 5.8 (S. 72), Tab. 5.9 (S. 78), Tab. A.13 (S. 108) · Mittel · UE**
- Original: Tab. 5.8 hat die Spalte „$\Delta$ (Enc.\ $-$ LLM)“ und für CrossNER mit 1\,024 Tokens den Wert $-$2,13 [$-$2,80; $-$1,47]. Tab. A.13 rechnet „Qwen3.5-27B minus DeBERTa-v3-large“ und kommt auf $+$2,13 [$+$1,47; $+$2,80]. In Tab. 5.9 wechselt die Richtung je Zeile.
- Problem: Alles ist beschriftet und damit nicht falsch. Leser müssen aber zwischen Tabellen umdenken; Abstract und Fazit nennen „1,47 bis 2,80“ positiv.
- Vorschlag: In Tab. 5.8 und A.13 dieselbe Richtung verwenden, z. B. LLM minus Encoder, passend zu Abstract und Fazit. Das verlangt eine Änderung in `thesis/scripts/aggregate_results.py` und neu erzeugte Tabellen, ist also keine reine Textänderung. Optional.
- Sicherheit: sicher, aber Tabellen neu erzeugen.

**T-02 · Tab. 5.5, S. 68 · Niedrig · UE**
- Original: Die Spaltenköpfe lauten „Typfehler“, „Grenzfehler“, „Grenz- und Typfehler“, „Auslassung“, „Zusätze“.
- Problem: Der Numerus ist gemischt; Tab. 5.3 schreibt „Auslassungen [\%]“.
- Vorschlag: „Auslassungen“ (Skript). Sicherheit: sicher.

**T-03 · Abb. 5.1 (S. 61) und Abb. 5.3 (S. 67) · Niedrig · UE**
- Original: Die Legende lautet in Abb. 5.1 „Qwen3.5-27B (QLoRA)“, in Abb. 5.3 „Qwen3.5-27B, QLoRA“.
- Vorschlag: In Abb. 5.3 die Klammerform verwenden, wie in Text und Tabellen. Sicherheit: sicher.

### Literaturverzeichnis

**L-01 · S. 86 bis 88 und die zugehörigen Zitate im Text · Mittel · UE**
- Original: „Babelscape. (n.d.)“, „Microsoft. (n.d. a)“, „(n.d. b)“, „Hugging Face. (n.d.)“
- Problem: Die englische Abkürzung steht in einem sonst deutsch lokalisierten Verzeichnis („Zuletzt geprüft am“, „Aufl.“, „Artikel“). Im Deutschen ist „o.\,J.“ üblich.
- Vorschlag: In der Präambel `\DefineBibliographyStrings{ngerman}{nodate = {o.\,J\adddot}}` ergänzen. Das ist ein Zitierstring, keine Layout-Einstellung. Danach Text und Verzeichnis prüfen.
- Sicherheit: sicher.

**L-02 · S. 90 und 93 · Niedrig · UE**
- Original: „PyPI-Release vom 24.10.2020“, „revidierte Fassung vom 27.05.2024“; sonst „Fassung vom 19.~August 2026“.
- Vorschlag: „vom 24.~Oktober 2020“, „vom 27.~Mai 2024“. Sicherheit: sicher.

**L-03 · S. 88 · Niedrig · RD**
- Original: „PEFT Version 0.19.1: Software Release [Offizieller, versionsgebundener Software-Release]“
- Vorschlag: Notiz auf „[versionsgebunden]“ verkürzen. Optional. Sicherheit: sicher.

**L-04 und L-05 · Hinweise ohne Änderungsvorschlag**
- Die Titel wechseln zwischen Großschreibung im englischen Stil und Satzschreibung, z. B. „Structured information extraction from scientific text …“ neben „Representation Learning: A Review …“. Das sind Originaltitel; sie bleiben unverändert.
- „SemEval-2013 Task 9 : Extraction …“ hat ein Leerzeichen vor dem Doppelpunkt. Das ist vermutlich die Originalschreibweise der ACL Anthology; nur nach Abgleich ändern.

## 3. Fachbegriffe, Abkürzungen und Schreibweisen

Häufigkeiten aus den Quellen (Vorspann und Kapitel 1 bis 6, Anhang); Eigennamen, Modell- und Datensatznamen bleiben unverändert.

| Begriff | Varianten in der Arbeit | Empfehlung | Grund |
|---|---|---|---|
| Ebene der Bewertung | „Entity-Level-…“ (7: Abstract, 2.9, 4.1, Kap. 5), „auf Entitätsebene“ (5), „Token-Level-Evaluation“ | im Fließtext „auf Entitätsebene“ bzw. „auf Tokenebene“, z. B. „strikte Metriken auf Entitätsebene“ | deutsche Form ist eingeführt; Mischformen wirken uneinheitlich |
| Vortraining | „Vortraining“ (14), „Pre-Training“ (7, vor allem Kap. 3 und Tab. 3.1) | „Vortraining“; Originaltitel bleiben | Kapitel 2 führt „Vortraining“ ein |
| Slurm | „Slurm“ (8), „SLURM“ (6, u. a. Tab. A.4, A.2.3, 4.4.1) | „Slurm“ | heutige Schreibweise des Herstellers; vor der Umsetzung Präferenz klären |
| Top-k und Top-p | „Top-$p$“ als Verfahren (Kap. 2), „top-$k = 20$“ als Parameter (Kap. 4, Anhang) | keine Änderung | die Regel „Verfahren groß, Parameterwert klein“ wird eingehalten |
| Decoding | „Decoding“ (41), „Dekodierung“ (2: 3.2.1 Generierung, 4.5.2 Detokenisierung) | 3.2.1 „das sequenzielle Decoding“; 4.5.2 siehe K4-13 | „Dekodierung“ ist doppeldeutig |
| Constrained Decoding | „Constrained Decoding“ (4), „\emph{constrained decoding}“ (1, 4.4.4) | „Constrained Decoding“ | einheitlich |
| parsebar | „nicht parsebar“ (13), „unparsebar“ (1, 6.1) | „nicht parsebar“ | siehe K6-04 |
| Verschmelzen des Adapters | „ohne Merge“ (4.4.3), „nicht verschmolzenen Adapters“ (5.4) | „ohne Verschmelzung (Merge)“ in 4.4.3, sonst „nicht verschmolzen“ | einmal einführen, dann deutsch |
| F$_1$ | „der F$_1$“ (überwiegend), „Das Gesamt-F$_1$“ (1) | „der F$_1$“ | siehe K4-04 |
| Silberreferenz | „Silberreferenz“ (15), „(Silver)“ (Tab. 2.2) | „Silberreferenz“ | siehe K2-27 |
| Erkennungsleistung | „Erkennungsleistung“ (26), „Performanz“ (2: 4.5.4 und Überschrift 4.5.1 „Performanzmetriken“), „NER-Leistung“ (Überschrift 5.3) | „Erkennungsleistung“; Überschrift 4.5.1 „Metriken der Erkennungsleistung“, 5.3 „Einfluss der Modellgröße auf die Erkennungsleistung“ | Labels bleiben |
| Ausgabelimit oder Ausgabelänge | „Ausgabelimit“ (30); „Ausgabelänge“ (9), davon zweimal im Sinn von Limit (Abstract, 1.2) | „Ausgabelänge“ nur für die Länge einer Antwort | siehe V-05, K1-08 |
| CrossNER-Versuch | „CrossNER-Ergänzungsversuch“ (Überschrift 4.6), „CrossNER-Transfertest“ (Captions, Kap. 3), „Transfer auf CrossNER“ (Überschrift 5.6), „CrossNER-Versuch“ (Tab. A.3), „CrossNER-Evaluationen“ (Anhang) | im Fließtext und in Captions „CrossNER-Transfertest“; Überschriften können bleiben | einheitliche Bezeichnung |
| Nachverarbeitung | „Nachverarbeitung“ (Kap. 1, 2, 3, 4, 5), „Post-Processing“ (Überschrift 4.4.6) | Überschrift „Nachverarbeitung und Validierung der Ausgaben“ | einheitlich; Label bleibt |
| Rückfallstrategien | „Rückfallstrategien“ (4.4.6, A.2), „Parsing-Fallbacks“ (Überschrift A.2.10) | siehe A-05 | |
| Künstliche Intelligenz (Domäne) | groß (K1, K2, K5, Tabellen), klein (4.6) | groß | Domänenname |
| Naturwissenschaften (Domäne) | Plural (überall), Singular (2.8) | Plural | siehe K2-26 |
| Präzision | „numerische Präzision“ (4.1), sonst verkürzt „Präzision“ | in Abstract und Kap. 5 „numerische Präzision“ | Verwechslung mit Precision; siehe V-02, K5-01 |
| Familie | Modellfamilie (DeBERTa-v3, Qwen3.5), „QLoRA-Familie“ (6.1) | nur für Modellfamilien | siehe K6-02 |
| Durchkopplung mit Abkürzungen | „LLM-Pipeline“, „Zero-Shot-Regime“, aber „Encoder Fine-Tuning“, „LLM Zero-Shot“ (4.1) | durchkoppeln | siehe K4-02 |
| mikro und makro gemittelt | „Mikro-F$_1$“, „mikro-gemittelt“ (2) | „mikrogemittelt“ | siehe K4-12 |
| Kausale Maske | „kausale Maske“ (Kap. 2), „kausale Aufmerksamkeitsmaske“ (3.2.3) | „kausale Maske“ | Attention bleibt sonst englisch |
| Encoder-Komposita | „Encoder-Referenz“ (5), „Encodervergleich“ (1.2), „Encoderwerte“ (Tab. 5.8) | einheitlich mit Bindestrich | geringe Priorität |
| Feinjustierung | „feinjustiert“ (33, als Partizip), „Fine-Tuning“ (Substantiv) | keine Änderung | sinnvolle Arbeitsteilung, kein Mischfehler |
| F$_1$-Skala | Quellenwerte auf der Skala 0 bis 1 (3.2.3: 0,149; 0,744; 0,908), sonst Punkte von 0 bis 100 | optional einmal „(Skala 0 bis 1 wie in der Quelle)“ | Werte sind korrekt wiedergegeben |
| Multi-Token-Prediction | Abkürzungsverzeichnis, 4.2.2, A.2.2 | keine Änderung | einheitlich verwendet |
| Jargon aus dem Code | „kanonisch“, „Kohorte“, „Revisions-Pins“, „gepinnt“, „wissenschaftlicher Vertrag“ | siehe K4-08, K4-09, A-01, A-09 | im Text nicht eingeführt |

## 4. Wiederkehrende sprachliche Muster

1. **Lehnübersetzungen aus englischen Quellen**, gehäuft in Kapitel 2 und 3:
   - nachsichtig (lenient);
   - stichprobeneffizient (sample-efficient);
   - „gegeben die Eingabe“ (given);
   - zurückgebildet und zurückgeführt (projected, mapped back);
   - „um das X-Fache reduziert oder verkürzt“ (reduced by X times);
   - überkonfident (overconfident);
   - Studenten und Lehrer (student/teacher);
   - „Faktor erster Ordnung“ (first-order);
   - „konvergieren in einem Punkt“, „prompten“, „Prüffälle“, „Setup“, „Potenzgesetzfits“, „Instruktionsgetunte“.
2. **Jargon aus Code und Projekt**, vor allem in Kapitel 4 und im Anhang: kanonisch, Kohorte, Pins und gepinnt, Merge, wissenschaftlicher Vertrag.
3. **Unbelebte Subjekte mit Handlungsverben:**
   - „Teilmengenauswahl … isolieren keinen Effekt“;
   - „das 4B-Modell verwirft Elemente“;
   - „der Makro-Wert ergibt Werte“;
   - „die LLM-Inferenz bleibt hinter den Encodern zurück“;
   - „LLMs lösen Aufgaben, indem die Aufgabe beschrieben wird“.
4. **Gekippte Bezüge nach späteren Einschüben.** Diese Gruppe enthält die wichtigsten Befunde: K3-19, K3-06, K4-14, K5-22, K5-31, K6-03. Bei jedem Einschub wurden die Pronomen und Anschlüsse („sie“, „die Arbeit“, „daher“, „daraus“, „eine solche“) im Folgesatz nicht erneut geprüft.
5. **Komma-Appositionen nach dem Entfernen der Gedankenstriche** (K2-26, K3-22). Aufzählungen zwischen Kommas sind schwer vom Satzgerüst zu unterscheiden; Klammern helfen.
6. **Elliptische Vervielfachungen:** „das 28-Fache des Fine-Tunings“, „das 2,7-Fache;“, „um das 40- bis 220-Fache über denen“.
7. **Wiederholte Absicherungen.** Der Vorbehalt „Prompt, Decoding und Präzision“ steht im Abstract, in 5.1.1, zweimal in 5.7.1 und in 5.7.5; die Definition der Hauptvergleiche steht in 4.5.1 und 5.7.5. Die Wiederholungen stehen jeweils dort, wo die Aussage gebraucht wird, und sind fachlich nötig. Eine Kürzung empfehle ich nicht.
8. **Kernaussagen:** Ergebnisabsätze in Kapitel 5 beginnen fast immer mit dem Befund, und Abstract und Fazit nennen die zentralen Zahlen direkt. Spät oder indirekt kommt die Kernaussage in 2.3 (rechter Kontext, K2-11) und in 5.7.1 (die Schlussfolgerung zum Ausgabelimit steckt im dritten der „drei Eigenschaften des Setups“, K5-21).

## 5. Zusammenfassende Bewertung

- **Geprüfter Umfang:** die gesamte Arbeit (Abschnitt 1). Nicht gegen englische Originale geprüft; die betroffenen Stellen sind markiert.
- **Sprachliche und grammatische Korrektheit:** Überwiegend korrekt. Eindeutige Fehler sind selten:
  - Rechtschreibung: K2-02, K4-19;
  - Kongruenz und Bezug: K2-16, K2-29, K5-28;
  - Vergleichskonstruktion: K2-23;
  - fehlender Artikel: K3-05;
  - Durchkopplung: K4-02;
  - Präposition: K5-20.

  Schwerer wiegen fünf inhaltlich missverständliche Stellen (Hoch); vier davon sind durch spätere Einschübe entstanden.
- **Natürliches, wissenschaftliches Deutsch:** Überwiegend ja. Der Stil ist sachlich und dicht. Auffällig sind Lehnübersetzungen in der Wiedergabe englischer Quellen (Kapitel 2 und 3) und Projektjargon (Kapitel 4, Anhang).
- **Hinweise auf zu wörtliche Übersetzungen:** Ja, wiederkehrend, aber punktuell: etwa 15 Stellen (Muster 1). Ob die Übertragung den Sinn der Quelle trifft, lässt sich ohne Originale nicht bestätigen (K2-13, K2-14, K3-07, K3-09, K3-12, K4-15, K4-23, K5-31).
- **Schwammige oder unnötig ausführliche Formulierungen:** Wenige. Die Arbeit neigt eher zur Verdichtung als zur Weitschweifigkeit: lange Sätze mit Zahlenketten und Klammern (K5-18), elliptische Konstruktionen (Muster 6) und knappe Fachkürzel („seedgleich“, „256er-Lauf“). Echte Füllsätze habe ich nicht gefunden.
- **Konzision:** Insgesamt konzise. Die verbleibenden Wiederholungen sind gewollte Absicherungen an den Stellen, an denen sie gebraucht werden (Muster 7).
- **Zentrale Erkenntnisse:** Abstract, Ergebnisabschnitte und Fazit nennen sie klar und mit Zahlen. Unscharf sind vor allem 5.7.1 (K5-21, K5-22, K5-23) und der CrossNER-Satz im Fazit (K6-03).
- **Größter Überarbeitungsbedarf:**
  - Kapitel 3 (Lehnübersetzungen, Bezüge);
  - Abschnitt 5.7.1 (eigene Einschübe);
  - Kapitel 4 (Jargon, 4.5.2);
  - danach Kapitel 2.

  Kapitel 1 und 6 sowie der Anhang brauchen nur wenige Korrekturen.
- **Häufigste Uneinheitlichkeiten:** Entity-Level und Entitätsebene, Pre-Training und Vortraining, SLURM und Slurm, Präzision und Precision, die Bezeichnung des CrossNER-Versuchs, die Durchkopplung (Abschnitt 3).
- **Zwingend:**
  - die fünf Hoch-Befunde;
  - die eindeutigen Fehler (siehe oben);
  - K5-22, K6-03 und V-03, weil sie Aussagen verzerren.

  Alles Übrige sind Empfehlungen.
- **Ohne Originalquelle nicht sicher beurteilbar:** K2-13, K2-14, K3-07, K3-09 (40 %), K3-12, K4-15, K4-23, K5-31. Bei K3-19 ist die Zuordnung über den Git-Verlauf belegt, aber nicht erneut an Lu et al. geprüft.

## 6. Freigabe und weiteres Vorgehen

**Zahl der Befunde:**
- Vorspann und Kapitel 1 bis 6: 134 (5 Hoch, 53 Mittel, 76 Niedrig);
- Anhang: 14 Vorschläge (A-01 bis A-14; 3 Mittel, 11 Niedrig);
- generierte Tabellen und Abbildungen: 3 (T-01 bis T-03);
- Literaturverzeichnis: 3 (L-01 bis L-03) und zwei Hinweise ohne Änderung.

**Freigabe nach Nummern**, z. B. „alle Hoch und Mittel, Niedrig außer K2-04 und K5-14; Anhang: A-06, A-09, A-11“.

**Offene Rückfragen**, bei denen ich ohne Antwort nicht umsetze:
- K2-08 (Variante a oder b)
- K2-14 (gleiche Daten oder gleiche Menge)
- K3-14 (Überschrift a oder b)
- K4-06 und A-07 (Bedeutung von „beide Pipelines“)
- K5-08 (Lesart a oder b)
- K5-15 (Seed-Reihenfolge)
- K5-27 (Kontrast zu Keraghel)
- K5-31 (Bezug „Eine solche Empfehlung“)
- K5-33 (Lesart a oder b)
- K6-05 (Bezug von „was“)
- A-01 (Inhalt der Datei)
- V-08 (Mustertext der Erklärung)

**Nach der Freigabe:**
- nur die freigegebenen Stellen ändern; den Anhang nur mit ausdrücklicher Freigabe;
- Zahlen, Zitate, Verweise und Fachnamen bleiben unverändert;
- jede geänderte Passage im Zusammenhang ihres Absatzes erneut lesen;
- Build;
- Prüfskripte (`sentences.py`, `terms.sh`, Verweise, Labels);
- prüfen, ob das Abstract auf einer Seite bleibt und sich Seitenumbrüche verschieben;
- Commit und Push.

## 7. Umsetzung (8. November 2026)

**Freigabe:** alle Befunde einschließlich Anhang (A-01 bis A-14), generierter Tabellen und Abbildungen (T-01 bis T-03) und Literaturverzeichnis (L-01 bis L-03). L-04 und L-05 waren Hinweise ohne Änderung.

### 7.1 Offene Rückfragen: Entscheidung und Beleg

Für die Quellenfragen wurden die frei zugänglichen Originale abgerufen (arXiv, ACL Anthology) und der Wortlaut gesucht. Nicht abgerufen wurden Hu et al. 2021, Lu et al. 2024 und Keraghel et al. 2024; die zugehörigen Stellen sind daher weiterhin nicht gegen das Original geprüft.

| Nr. | Entscheidung | Beleg |
|---|---|---|
| K2-08 | Variante (b): „aus allen Positionen, gewichtet nach ihrer Relevanz“ | entspricht dem Satz davor („gewichtete Summe“) |
| K2-13 | „dateneffizienter“ | Clark et al. 2020, Abstract: „a more sample-efficient pre-training task“ |
| K2-14 | Lesart (a): „auf denselben 160 GB Daten wie DeBERTa-v2“ | He et al. 2023, Abschn. 4.1: „We train those models with 160GB data, which is the same as DeBERTaV2 and RoBERTa“ |
| K3-07 | umformuliert, kein Zitat | Zhan et al. 2026 (v3) enthalten „first-order“ nicht |
| K3-09 | nur Grammatik, Prozentangabe unverändert | Ravikumar et al. 2026: „format shifts F1 scores by over 40%“ |
| K3-12 | Vorschlag umgesetzt | Wang et al. 2025: „over-confidently label NULL inputs as entities“ |
| K3-14 | Variante (a): „LLMs mit Instruction-Tuning und Anpassung“ | Label `sec:k3-angepasst` unverändert |
| K4-06, A-07 | „beide LLM-Regime (Zero-Shot und QLoRA)“ bzw. „Beide LLM-Regime“ | Code: Zero-Shot- und QLoRA-Inferenz laden über `AutoModelForCausalLM` (`ba-ner/src/decoder/inference.py`), das QLoRA-Training ebenso (`train.py`) |
| K4-15 | Beleg an „Inventar“ verschoben | Tedeschi und Navigli 2022 enthalten „halluc…“ nicht |
| K4-23 | einheitlich „Anh.~D, Tab.~11“ in 3.3 und 4.4.5 | Zhan et al.: Vergleich LoRA/vollständiges Fine-Tuning „in Appendix D“, Tabelle 11 |
| K5-08 | Lesart (a): „von den acht Testsätzen ohne Referenzentität lässt nur das 27B-Modell einige leer (3 bis 4 je Seed)“ | `summary.json`, `empty_correct`: 27B 3/4/3, 0.8B und 4B 0 |
| K5-15 | „Seeds 42, 123 und 456: 96,6, 93,8 und 95,1“ | `summary.json`, `output_limit_multinerd/encoder_subset`, `seed_matched` in Seed-Reihenfolge |
| K5-27 | Variante (a) | Kontrast bleibt erhalten, nur sprachlich geglättet |
| K5-31 | Vorschlag umgesetzt | Obeidat et al. 2025 (arXiv 2504.00664v1, Abschn. III): „for domains where small performance differences are tolerable, encoders are still the better choice“ |
| K5-33 | weder (a) noch (b), sondern die wörtlichste Fassung: „Wie sie sich auf die Rangfolge auswirken, ist im Allgemeinen nicht bekannt“ | keine neue Bedeutung |
| K6-03 | Vorschlag plus Einschränkung aus 5.7.4 („sofern man auch Teilspans solcher Entitäten nicht als Fehler wertet“) | Einschränkung steht wörtlich in 5.7.4, wird nicht neu eingeführt |
| K6-05 | Vorschlag; „das“ bezieht sich wie im Original auf Batching und einheitliche Hardware gemeinsam | Bezug nicht verengt |
| A-01 | „Festlegung der wissenschaftlichen Laufparameter mit Code- und Revisions-Hashes“ | `ba-ner/src/seed_provenance.py`, `scientific_contract()`: Revisionen, Code- und Prompt-Hashes, Evaluationsvorgaben |
| V-08 | nur Absatztitel „Verwendete KI-Tools“; Mustertext unverändert | Frage zu „ohne Hilfe Dritter“ bleibt beim Betreuer bzw. Prüfungsamt |

### 7.2 Abweichungen vom Vorschlag

- **V-02, V-03:** Mit dem vorgeschlagenen Wortlaut lief das Abstract zwei Zeilen auf eine zweite Seite. Kürzere Fassungen mit gleicher Aussage:
  - V-02: „da sich die Regime über das Training hinaus in Prompt, Decoding und numerischer Präzision unterscheiden, ist kein isolierter Fine-Tuning-Effekt bestimmbar“;
  - V-03: „unter jeder Latenz- und Speichergrenze ein Encoder die beste gemessene Wahl“.

  Danach passte auch V-04 auf die Seite.
- **K2-27:** „automatisch (Silberreferenz) + manuelle Teststichprobe“ war 15 pt zu breit; die Zelle steht jetzt auf drei Zeilen, der Wortlaut ist vollständig.
- **K3-19:** Der Anschluss lautet wieder „Als Gründe für die Schwierigkeiten nennen sie“ (Fassung vor `8a00510`), damit sich die Gründe nicht auf die Vermutung zum vollständigen Fine-Tuning beziehen.
- **K1-02:** Derselbe Wortlaut („mit demselben Modell bearbeiten“) stand auch in 2.5 und wurde dort gleich korrigiert.
- **K5-23:** Der umformulierte Satz hatte 54 Wörter; er ist am Semikolon in zwei Sätze geteilt.
- **T-01:** Statt LLM minus Encoder zeigt jetzt Tab. A.13 Encoder minus LLM, wie Tab. 5.8 und die erste Zeile von Tab. 5.9. So ändert sich nur eine Tabelle; Abstract und Fazit nennen den Betrag des Intervalls und bleiben richtig. Umsetzung in `scripts/aggregate_results.py` (Vorzeichen und Intervallgrenzen getauscht), Caption in `anhang.tex` angepasst.
- **T-03:** Klammerform für alle sechs Qwen-Einträge der Legende von Abb. 5.3. Abb. 5.2 nennt nur Familien („Qwen3.5, QLoRA“) und blieb unverändert.

### 7.3 Begriffe (Abschnitt 3) umgesetzt

- „auf Entitätsebene“ bzw. „auf Tokenebene“ statt Entity-/Token-Level (Abstract, 2.9, 4.1, 4.5.1, 5);
- „Vortraining“ statt Pre-Training (Kap. 3, Tab. 3.1);
- „Slurm“ (4.4.1, A.2);
- „das sequenzielle Decoding“ (3.2.1);
- „Constrained Decoding“ (4.4.4);
- „ohne Verschmelzung (Merge)“ (4.4.3);
- „CrossNER-Transfertest“ (3.2.3, Einleitung Kap. 5, 5.6, Tab. A.3);
- Überschriften „Nachverarbeitung und Validierung der Ausgaben“ (4.4.6), „Metriken der Erkennungsleistung“ (4.5.1), „Einfluss der Modellgröße auf die Erkennungsleistung“ (5.3);
- „kausale Maske“ (3.3);
- „Encoder-Vergleich“, „Encoder-Werte“;
- Hinweis „Skala 0 bis 1 wie in der Quelle“ (3.3, einmal).

Labels blieben unverändert. Zusätzlich: „spaltenweise BIO-Tags“ → „IOB2-Tags“ in 3.2.1. Diese Stelle meldete `terms.sh` bereits vor der Umsetzung; sie gehört zum Muster von K5-25.

### 7.4 Prüfungen nach der Umsetzung

- Build (`latexmk` im Docker-Image `texlive/texlive:latest`): Exit 0, 118 Seiten wie vorher. Overfull, Underfull und undefinierte Verweise: 0.
- Kapitelanfänge unverändert: 1, 6, 21, 35, 58, 80, Literatur 86, Anhang 95.
- Abstract auf einer Seite (S. ii).
- `terms.sh`: keine Altvarianten.
- `sentences.py`: kein durch die Umsetzung neu entstandener Satz über 50 Wörter. Weiterhin über 50 Wörter wie vorher: 1.2 (64), 3.2.1 (51), 6.1 (68) sowie die Artefaktliste in A.1.
- `numbers.py --diff HEAD`: neu im Text nur die belegten Angaben aus K5-06, K5-08 und K5-15; die übrigen gemeldeten Zahlen stammen aus Literaturquellen und standen schon vorher im Text.
- `aggregate_results.py`: Ein zweiter Lauf ist byte-identisch. Geändert sind nur `tab-crossner-ci.tex`, `tab-fehlertypen.tex` und `tab-synthese.tex`.
- Unit-Tests: 10 bestanden.
- Jede geänderte Passage wurde im Wort-Diff mit ihrem Absatz erneut gelesen. Dabei korrigiert: „Textabweichungen“ (K5-09) → „Text-Abweichungen“, wie sonst in Kapitel 5.
- Im Anhang wurde außer dem freigegebenen doppelten „auf Validierungsdaten“ (A-06) nichts gestrichen.

