"""The pathway control removes only the information carried by the antigen."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from probe_antigen_pathway import ARMS, IGNORE, make_batch
from smallAntibodyGen.antigen_tokenization import build_antigen_tokenizer
from smallAntibodyGen.tokenizer import AminoAcidTokenizer


def test_pathway_control_changes_only_second_antigen_and_preserves_conflicting_labels():
    a, c = ARMS["A antigen-determined"], ARMS["C pathway control"]
    assert len(a) == len(c) == 2
    assert a[0] == c[0]
    assert a[1][:2] == c[1][:2]
    assert a[0][2] != a[1][2]
    assert c[0][2] == c[1][2] == a[0][2]

    tokenizer = AminoAcidTokenizer()
    antigen_tokenizer = build_antigen_tokenizer("scratch", tokenizer, "")
    batch_a, batch_c = [make_batch(tokenizer, antigen_tokenizer, spec, 48, 48,
                                   torch.device("cpu")) for spec in (a, c)]
    # Check the actual model inputs, including masks, rather than only the table.
    for index in (0, 1, 4):
        torch.testing.assert_close(batch_a[index], batch_c[index], rtol=0, atol=0)
    for index in (0, 1, 2, 3):
        torch.testing.assert_close(batch_c[index][0], batch_c[index][1], rtol=0, atol=0)
    assert not torch.equal(batch_a[2][0], batch_a[2][1])
    labels = batch_c[4]
    supervised = labels != IGNORE
    assert supervised[0].any()
    assert torch.equal(supervised[0], supervised[1])
    assert torch.all(labels[0][supervised[0]] != labels[1][supervised[1]])
    assert torch.all(batch_c[0][supervised] == tokenizer.mask_id)
