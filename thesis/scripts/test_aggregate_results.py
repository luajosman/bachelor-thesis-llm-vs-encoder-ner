"""Checks for pairing, pooling and reproducibility of the CrossNER intervals.

Run: python -m unittest discover -s thesis/scripts -p 'test_*.py'
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import aggregate_results as agg


class CrossnerBootstrapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = {"deberta-large": ["encoder-42", "encoder-123", "encoder-456"],
                     "qwen27-qlora": ["llm-42", "llm-123", "llm-456"]}
        self.records = {}
        predictions = {
            "a": [["B-PER", "O"], ["O", "O"], ["B-PER", "B-PER"]],
            "b": [["B-PER"] * 3, ["B-PER"] * 3, ["B-PER", "B-ORG", "B-ORG"]],
        }
        for dom, groups in predictions.items():
            for j, tags in enumerate(groups):
                root = "crossner_output_limit_1024" if j == 2 else "crossner"
                runs = self.runs["deberta-large" if j == 0 else "qwen27-qlora"]
                for run in runs:
                    records = [dict(sample_id=f"{dom}:{i}", tokens=[str(i)],
                                    gold_bio=["B-PER"], pred_bio=[tag])
                               for i, tag in enumerate(tags)]
                    path = self.root / root / run / dom / "test_predictions.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(records))
                    self.records[(j, run, dom)] = path
        # Hand-calculated F1: pooled encoder TP=4, FP=0, FN=1;
        # pooled 1024 LLM TP=3, FP=2, FN=2. Pooling differs from a domain mean.
        points = {"a": [2 / 3, 0, 1], "b": [1, 1, 1 / 3],
                  "pooled_micro": [8 / 9, 3 / 4, 3 / 5]}
        self.C = {
            "aggregate": {key: {dom: {"f1_mean": values[j]} for dom, values in points.items()}
                          for j, key in enumerate(self.runs)},
            "output_limit_full": {
                "complete": True,
                "full": {dom: {"1024": {"f1_mean": values[2]}} for dom, values in points.items()},
            },
        }
        patches = patch.multiple(agg, RES=self.root, CROSS_RUNS=self.runs,
                                 DOMAINS=[("a", "A"), ("b", "B")])
        patches.start()
        self.addCleanup(patches.stop)

    def test_pooled_counts_and_chunk_independent_reproducibility(self):
        first = agg.crossner_bootstrap(self.C, B=100, chunk=17)
        second = agg.crossner_bootstrap(self.C, B=100, chunk=1)
        self.assertEqual(first["domains"], second["domains"])
        point = first["domains"]["pooled_micro"]["configs"]["qwen27-qlora-1024"]
        self.assertAlmostEqual(point, 60)
        self.assertNotAlmostEqual(point, (100 + 100 / 3) / 2)
        self.assertEqual(first["settings"]["n_by_domain"], {"a": 2, "b": 3})

    def test_identical_predictions_have_zero_paired_difference(self):
        C = copy.deepcopy(self.C)
        for dom in ("a", "b"):
            for encoder, llm in zip(self.runs["deberta-large"], self.runs["qwen27-qlora"]):
                self.records[(1, llm, dom)].write_text(self.records[(0, encoder, dom)].read_text())
        C["aggregate"]["qwen27-qlora"] = copy.deepcopy(C["aggregate"]["deberta-large"])
        result = agg.crossner_bootstrap(C, B=100, chunk=17)
        for domain in result["domains"].values():
            self.assertEqual(domain["pairs"]["qwen27-qlora"]["ci"], [0, 0])
            self.assertFalse(domain["pairs"]["qwen27-qlora"]["excludes_zero"])

    def test_averages_seed_f1_instead_of_pooling_seeds(self):
        C = copy.deepcopy(self.C)
        for dom in ("a", "b"):
            path = self.records[(2, "llm-123", dom)]
            records = json.loads(path.read_text())
            for record in records:
                record["pred_bio"] = ["O"]
            path.write_text(json.dumps(records))
        for dom in ("a", "b", "pooled_micro"):
            C["output_limit_full"]["full"][dom]["1024"]["f1_mean"] *= 2 / 3
        result = agg.crossner_bootstrap(C, B=100)
        point = result["domains"]["pooled_micro"]["configs"]["qwen27-qlora-1024"]
        self.assertAlmostEqual(point, 40)
        self.assertNotAlmostEqual(point, 48)  # F1 from counts pooled over seeds

    def test_rejects_unpaired_or_malformed_records(self):
        path = self.records[(2, "llm-123", "a")]
        original = json.loads(path.read_text())
        for fault in ("gold", "order", "duplicate", "length"):
            with self.subTest(fault=fault):
                records = copy.deepcopy(original)
                if fault == "gold":
                    records[0]["gold_bio"] = ["B-ORG"]
                elif fault == "order":
                    records.reverse()
                elif fault == "duplicate":
                    records[1]["sample_id"] = records[0]["sample_id"]
                else:
                    records[0]["pred_bio"] = []
                path.write_text(json.dumps(records))
                with self.assertRaises(ValueError):
                    agg.crossner_bootstrap(self.C, B=10)
                path.write_text(json.dumps(original))

    def test_counts_only_common_types_and_strict_bio_entities(self):
        records = [dict(gold=["B-PER", "O", "B-DIS"],
                        pred=["B-PER", "B-DIS", "O"]),
                   dict(gold=["B-ORG"], pred=["I-ORG"])]
        micro, _ = agg.sentence_counts(records, "gold", "pred", agg.COARSE)
        np.testing.assert_array_equal(micro, [[1, 0, 0], [0, 0, 1]])


class TransferAndSubsetTests(unittest.TestCase):
    """coarse_reference, transfer_losses, encoder_recall_on_truncated auf synthetischen Satzzaehlern."""

    def setUp(self):
        nt = len(agg.TYPES)
        n = 4

        def run(name, micro, coarse_counts, regen=None):
            types = np.zeros((n, nt, 3), dtype=np.int32)
            for j, t in enumerate(agg.COARSE):
                types[:, agg.TYPES.index(t), :] = coarse_counts[j]
            types[:, agg.TYPES.index("DIS"), :] = [[5, 5, 5]] * n      # Nicht-Grobtyp darf nicht zaehlen
            r = dict(name=name, sent_micro=np.array(micro, dtype=np.int32), sent_types=types)
            if regen is not None:
                r["regen"] = regen
            return r

        # Encoder: je Satz tp/fp/fn; Satz 1 und 3 sind die "abgeschnittenen" Saetze des LLM-Seeds 42
        enc = run("enc", [[2, 0, 0], [1, 0, 1], [3, 1, 0], [0, 0, 2]],
                  coarse_counts=[[[1, 0, 0]] * n, [[1, 0, 1]] * n, [[0, 0, 0]] * n])
        llm = run("llm", [[2, 0, 0]] * n, coarse_counts=[[[1, 1, 0]] * n, [[1, 0, 0]] * n, [[0, 0, 1]] * n],
                  regen=dict(indices=[1, 3], subset_metrics=dict(baseline_256=dict(recall=0.25),
                                                                 output_limit=dict(recall=0.75))))
        self.G = {"deberta-large": dict(runs=[enc]), "qwen27-qlora": dict(runs=[llm]),
                  "qwen27-qlora-1024": dict(runs=[llm])}

    def test_coarse_reference_sums_only_coarse_types(self):
        ref = agg.coarse_reference(self.G, "deberta-large")
        s = ref["per_seed"][0]
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (8, 0, 4))      # PER 4/0/0 + ORG 4/0/4 + LOC 0
        self.assertAlmostEqual(s["f1"], 2 * 8 / (2 * 8 + 0 + 4))
        self.assertAlmostEqual(ref["f1"]["mean"], s["f1"])

    def test_transfer_losses_use_matching_limit(self):
        C = dict(aggregate={"deberta-large": {"pooled_micro": {"f1_mean": 0.5}},
                            "qwen27-qlora": {"pooled_micro": {"f1_mean": 0.4}}},
                 output_limit_full=dict(complete=True, full={"pooled_micro": {"1024": {"f1_mean": 0.6}}}))
        t = agg.transfer_losses(self.G, C)
        self.assertAlmostEqual(t["deberta-large"]["loss"], 2 * 8 / 20 - 0.5)
        llm_f1 = 2 * 8 / (2 * 8 + 4 + 4)                                # PER 4/4/0 + ORG 4/0/0 + LOC 0/0/4
        self.assertAlmostEqual(t["qwen27-qlora"]["loss_256"], llm_f1 - 0.4)
        self.assertAlmostEqual(t["qwen27-qlora"]["loss_1024"], llm_f1 - 0.6)

    def test_encoder_recall_on_truncated_subset(self):
        L = {"qwen27-qlora-1024": {}}
        out = agg.encoder_recall_on_truncated(self.G, L)
        e = out["qwen27-qlora-1024"]["per_llm_seed"][0]
        self.assertEqual(e["n"], 2)
        self.assertAlmostEqual(e["encoder"]["deberta-large"]["values"][0], 1 / (1 + 3))   # Saetze 1 und 3: tp 1, fn 3
        self.assertAlmostEqual(e["llm_recall_1024"], 0.75)
        self.assertAlmostEqual(out["qwen27-qlora-1024"]["deberta-large"]["min"], 0.25)


if __name__ == "__main__":
    unittest.main()
