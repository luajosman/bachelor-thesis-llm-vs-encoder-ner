# Typverteilung der ersten 200 Validierungssätze (MultiNERD, en)

Berechnet am 7.11.2026 auf dem DFKI-Cluster mit `ba-ner/scripts/validation_subset_statistics.py`
(Loader `src/data/dataset_loader.py`, Datensatzrevision `2814b78e7af4b5a1f1886fe7ad49632de4d9dd25`, Offline-Cache;
Teilmenge `validation.select(range(200))` in Loader-Reihenfolge, wie bei der Checkpoint-Auswahl der QLoRA-Läufe,
`gen_eval_max_samples: 200`). Entitäten = Zahl der B-Tags; Satzlänge = Zahl der Tokens.

| Kennzahl | Validierungssplit | erste 200 Sätze |
|---|---:|---:|
| Sätze | 16410 | 200 |
| Token | 379011 | 4462 |
| mittlere Satzlänge | 23.10 | 22.31 |
| Entitäten | 26041 | 283 |
| Entitäten je Satz | 1.587 | 1.415 |
| Sätze mit ≥ 1 Entität | 16405 | 200 |

| Typ | Anzahl (Split) | Anteil (Split) | Anzahl (200) | Anteil (200) | Differenz (Pp.) |
|---|---:|---:|---:|---:|---:|
| PER | 7507 | 28.8 % | 74 | 26.1 % | -2.7 |
| ORG | 2737 | 10.5 % | 37 | 13.1 % | +2.6 |
| LOC | 7850 | 30.1 % | 90 | 31.8 % | +1.7 |
| ANIM | 1134 | 4.4 % | 7 | 2.5 % | -1.9 |
| BIO | 14 | 0.1 % | 0 | 0.0 % | -0.1 |
| CEL | 94 | 0.4 % | 2 | 0.7 % | +0.3 |
| DIS | 1695 | 6.5 % | 18 | 6.4 % | -0.1 |
| EVE | 299 | 1.1 % | 4 | 1.4 % | +0.3 |
| FOOD | 2120 | 8.1 % | 16 | 5.7 % | -2.5 |
| INST | 34 | 0.1 % | 0 | 0.0 % | -0.1 |
| MEDIA | 919 | 3.5 % | 15 | 5.3 % | +1.8 |
| MYTH | 56 | 0.2 % | 1 | 0.4 % | +0.1 |
| PLANT | 1180 | 4.5 % | 4 | 1.4 % | -3.1 |
| TIME | 322 | 1.2 % | 9 | 3.2 % | +1.9 |
| VEHI | 80 | 0.3 % | 6 | 2.1 % | +1.8 |

```json
{
 "full_validation": {
  "n_sentences": 16410,
  "n_tokens": 379011,
  "mean_sentence_length": 23.0963436928702,
  "n_entities": 26041,
  "entities_per_sentence": 1.5868982327848873,
  "sentences_with_entity": 16405,
  "type_counts": {
   "PER": 7507,
   "ORG": 2737,
   "LOC": 7850,
   "ANIM": 1134,
   "BIO": 14,
   "CEL": 94,
   "DIS": 1695,
   "EVE": 299,
   "FOOD": 2120,
   "INST": 34,
   "MEDIA": 919,
   "MYTH": 56,
   "PLANT": 1180,
   "TIME": 322,
   "VEHI": 80
  },
  "type_share": {
   "PER": 0.28827617987020465,
   "ORG": 0.10510349064936063,
   "LOC": 0.30144771706155676,
   "ANIM": 0.04354671479589878,
   "BIO": 0.0005376137629123306,
   "CEL": 0.003609692408125648,
   "DIS": 0.06508966629545716,
   "EVE": 0.011481893936484773,
   "FOOD": 0.08141008409815291,
   "INST": 0.0013056334242156598,
   "MEDIA": 0.035290503436887985,
   "MYTH": 0.0021504550516493223,
   "PLANT": 0.045313160016896434,
   "TIME": 0.012365116546983603,
   "VEHI": 0.0030720786452133175
  }
 },
 "first_200": {
  "n_sentences": 200,
  "n_tokens": 4462,
  "mean_sentence_length": 22.31,
  "n_entities": 283,
  "entities_per_sentence": 1.415,
  "sentences_with_entity": 200,
  "type_counts": {
   "PER": 74,
   "ORG": 37,
   "LOC": 90,
   "ANIM": 7,
   "BIO": 0,
   "CEL": 2,
   "DIS": 18,
   "EVE": 4,
   "FOOD": 16,
   "INST": 0,
   "MEDIA": 15,
   "MYTH": 1,
   "PLANT": 4,
   "TIME": 9,
   "VEHI": 6
  },
  "type_share": {
   "PER": 0.26148409893992935,
   "ORG": 0.13074204946996468,
   "LOC": 0.31802120141342755,
   "ANIM": 0.024734982332155476,
   "BIO": 0.0,
   "CEL": 0.007067137809187279,
   "DIS": 0.0636042402826855,
   "EVE": 0.014134275618374558,
   "FOOD": 0.05653710247349823,
   "INST": 0.0,
   "MEDIA": 0.053003533568904596,
   "MYTH": 0.0035335689045936395,
   "PLANT": 0.014134275618374558,
   "TIME": 0.03180212014134275,
   "VEHI": 0.02120141342756184
  }
 },
 "n_validation": 16410,
 "first_200_first_tokens": [
  [
   "Between",
   "the",
   "end",
   "of",
   "World",
   "War"
  ],
  [
   "These",
   "ideas",
   "were",
   "prevalent",
   "among",
   "many"
  ],
  [
   "Most",
   "are",
   "from",
   "Indonesia",
   ",",
   "Vietnam"
  ]
 ]
}
```
