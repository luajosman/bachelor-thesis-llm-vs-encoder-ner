import json, sys
from collections import Counter
from src.data.dataset_loader import load_ner_dataset
ds, info = load_ner_dataset()
val = ds["validation"]
id2label = info.id2label
types = info.entity_types
def stats(rows):
    n_sent = 0; n_tok = 0; ent = Counter(); n_sent_with_ent = 0
    for r in rows:
        n_sent += 1
        toks = r["tokens"]; tags = r["ner_tags"]
        n_tok += len(toks)
        c = 0
        for t in tags:
            lab = id2label[t]
            if lab.startswith("B-"):
                ent[lab[2:]] += 1; c += 1
        if c: n_sent_with_ent += 1
    total = sum(ent.values())
    return {"n_sentences": n_sent, "n_tokens": n_tok, "mean_sentence_length": n_tok/n_sent,
            "n_entities": total, "entities_per_sentence": total/n_sent,
            "sentences_with_entity": n_sent_with_ent,
            "type_counts": {t: ent.get(t,0) for t in types},
            "type_share": {t: (ent.get(t,0)/total if total else 0.0) for t in types}}
out = {"full_validation": stats(val), "first_200": stats(val.select(range(200))),
       "n_validation": len(val), "first_200_first_tokens": [val[i]["tokens"][:6] for i in range(3)]}
print("RESULT_JSON_BEGIN"); print(json.dumps(out, indent=1)); print("RESULT_JSON_END")
