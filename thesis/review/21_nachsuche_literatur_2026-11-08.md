# Gezielte Literatur-Nachsuche (8.11.2026)

Anlass: Die Aussage in 5.7.2, die Wortindex-Variante werde „soweit bekannt, erstmals mit QLoRA und strikter Entitätsmetrik auf MultiNERD geprüft“, hing an einer Recherche mit Schwerpunkt 2019–2025. Durchgeführt mit KI-Unterstützung (Claude), Ergebnisse von mir gegen die Originalquellen geprüft.

## Abfragen

| Quelle | Abfrage | Ergebnis |
|---|---|---|
| arXiv-API (Metadaten: Titel, Abstract, Kommentare) | `abs:MultiNERD`, `all:MultiNERD`, `all:"MultiNERD"` | jeweils 0 Treffer (Gegenprobe `all:GLiNER`: 36 Treffer, API funktioniert) |
| Semantic-Scholar-API | `MultiNERD` | nicht auswertbar (Rate-Limit) |
| Websuche | `MultiNERD QLoRA LLM named entity recognition fine-tuning 2026` | keine Arbeit mit MultiNERD + QLoRA; LOCUS (arXiv 2512.06239) nutzt MultiNERD nur few-shot, ohne QLoRA, Ausgabeformat nicht angegeben |
| Websuche | `LLM NER output format token indices word index span fine-tuning LoRA 2026 arXiv` | Zhan et al. 2026 (bereits zitiert); **Ravikumar et al. 2026 (EACL)**; GPT-NER u. a. (älter, Prompting) |
| Websuche | `"MultiNERD" large language model fine-tuned NER 2025 OR 2026 aclanthology` | keine 2025/26-Arbeit, die ein LLM auf MultiNERD feinjustiert |
| Websuche | `generative NER "word indices" OR "token indices" output LLM fine-tuning MultiNERD OR CoNLL 2025 2026` | nur Index-Generierung mit BART (Yan et al. 2021) und Zhan et al. 2026 |

Geprüfte Kandidaten: arXiv 2606.08051 (Händlerdaten, LoRA, kein NER-Benchmark), arXiv 2510.08902 (biomedizinisch, Tagging), arXiv 2512.06239 (LOCUS, s. o.), ACL 2026.eacl-long.256.

## Ergebnis

- Keine Arbeit gefunden, die ein Wortindex-Format mit QLoRA und strikter Entitätsmetrik auf MultiNERD prüft. Die Aussage in 5.7.2 bleibt mit „soweit bekannt“.
- Aufgenommen: Ravikumar, Ibrahim und Batista-Navarro (2026), „Lost in Formatting: How Output Formats Skew LLM Performance on Information Extraction“, EACL 2026 (Vol. 1, S. 5498–5513, DOI 10.18653/v1/2026.eacl-long.256). Geprüft über die ACL-Anthology-Seite und das PDF: QLoRA (4 Bit), Modelle 3B–32B, sechs Formate (Freetext, Standoff-JSON, Standoff-Tupel, Inline-XML, Inline-Linear/BIO, Spalten), NER auf OntoNotes 5.0 und BC5CDR, strikte Mikro-F1; kein Indexformat, kein MultiNERD.
- Grenzen: keine Google-Scholar-Volltextsuche, Semantic Scholar nicht verfügbar, arXiv nur Metadaten. Die Nachsuche ist gezielt, nicht systematisch.
