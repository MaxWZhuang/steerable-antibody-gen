"""Known-answer learning controls, using synthetic context rather than biology.

The masked antibody is identical for two conflicting answers. Only the antigen
distinguishes them. This tests the information path; it is not transfer evidence.
"""
import copy
from dataclasses import replace
import math

import torch

from smallAntibodyGen.experiments.trastuzumab_poc.data import qualify_manifest
from smallAntibodyGen.experiments.trastuzumab_poc.factory import build_models
from smallAntibodyGen.experiments.trastuzumab_poc.runner import StageTrainer, TrainingConfig
from smallAntibodyGen.models.antibody_antigen.fusion import FusionConfig
from smallAntibodyGen.models.antibody_antigen.views import mask_complete
from smallAntibodyGen.tests.test_trastuzumab_data import _rec, _decl


def test_context_only_answer_can_be_learned_but_abonly_cannot_resolve_it():
    rows = [replace(_rec(record_id="a", H="ADEFGHIKL", L="MN", Ag="AAAA", tgt_family="a"),
                    h_eligible=(0,), l_eligible=None),
            replace(_rec(record_id="c", H="CDEFGHIKL", L="MN", Ag="WWWW", tgt_family="c"),
                    h_eligible=(0,), l_eligible=None)]
    population = qualify_manifest(_decl(), rows)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        results = {}
        for arm in ("paired", "abonly"):
            policy, encoder, origin = build_models({"kind": "tiny"}, FusionConfig(
                context_dim=8, bottleneck=12, lora_rank=4, lora_alpha=8,
                activation_checkpointing=True, update_mode="full"), seed=4)
            frozen = copy.deepcopy(encoder.state_dict())
            run = StageTrainer(policy, encoder, TrainingConfig(
                arm=arm, seed=4, total_updates=80, examples_per_update=2,
                learning_rate=0.01, weight_decay=0), population=population,
                data_identity={}, source_identity={}, initialization_identity=origin)
            def score(swap=False):
                policy.eval()
                with torch.no_grad():
                    losses = []
                    for index, row in enumerate(rows):
                        sequences = {"H": row.H, "L": row.L}
                        if arm == "paired":
                            sequences["Ag"] = rows[1 - index].Ag if swap else row.Ag
                        view, targets = mask_complete(sequences, (("H", 0),))
                        losses.append(float(run._reconstruct(view, targets)))
                    return sum(losses) / len(losses)
            before = score()
            run.run()
            results[arm] = before, score(), score(True)
            assert all(torch.equal(value, encoder.state_dict()[key]) for key, value in frozen.items())
            assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
        assert results["paired"][1] < 0.05
        assert results["paired"][2] > 2.0  # Wrong partner breaks the learned synthetic rule.
        assert results["abonly"][1] >= math.log(2) - 1e-6
        assert results["abonly"][1] == results["abonly"][2]
        assert results["abonly"][1] < results["abonly"][0]  # Its optimizer still learns.
    finally:
        torch.set_num_threads(previous)
