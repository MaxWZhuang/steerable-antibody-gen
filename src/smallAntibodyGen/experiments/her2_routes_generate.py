"""Batched raw generation, measured fill orders, remasking, and a scalable index.

Four things here that the earlier decoding stage did not have, each of which made
its numbers describe something other than the model:

* **the confidence was ``rng.random()``.** ``easy_first`` and ``hard_first`` were
  therefore two random orders with different labels, and the comparison between
  them was a comparison of two seeds. :func:`fill_batch` reads the model's own
  conditional entropy at every step, which is what "easy" has to mean.
* **generation was one draw per forward.** 10,000 draws of a ten-site window is
  100,000 single-row forwards, and the plan asks for 100,000 draws as well.
  Everything here is batched over draws; the AR route additionally expands each
  draw over its remaining coordinates in one forward, because an autoregressive
  decoder cannot know which site is easiest without scoring the candidates.
* **a CPU generator drove a CUDA multinomial.** That raises, so the sampling path
  had never run on the device it was profiled for. The generator is built on the
  model's device here.
* **clustering and nearest-training distance were Python loops.** Greedy
  assignment at radius one over 100,000 windows is 100,000 x medoids Hamming
  computations, and the nearest-training query against the 367,042-row library is
  another 37 billion. :func:`greedy_medoids` uses deletion-key hashing -- exact for
  the declared radius -- and the library query reuses the packed deletion-mask
  search from ``her2_routes_split``.

Nothing here labels an off-catalogue sample anything but an in-silico candidate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import her2_routes_data as data_lib
from . import her2_routes_energy_contract as energy_contract
from . import her2_routes_split as split_lib
from .her2_data import CANONICAL
from .her2_routes_decoding import (
    ORDER_DECLARED,
    ORDER_EASY_FIRST,
    ORDER_HARD_FIRST,
    ORDER_RANDOM,
    draw_accounting,
)
from .her2_runtime import require

#: The remasking strategy name. Not one of the fill ORDERS: a revision trajectory
#: has no closed-form density and is compared by samples only.
ORDER_REMASK = "remask"
#: The native DPLM decoder: iterative reparameterized diffusion decoding, all design sites
#: refined together over ``decode_steps`` denoiser forwards. Defined only for a model with an
#: ``iterative_decode`` method; the other routes have no such decoder and it is refused there.
ORDER_DIFFUSION = "diffusion_reparam"
STRATEGIES = (ORDER_DECLARED, ORDER_RANDOM, ORDER_EASY_FIRST, ORDER_HARD_FIRST, ORDER_REMASK,
              ORDER_DIFFUSION)


def device_generator(device, seed):
    """A generator on the device the tensors live on.

    ``torch.multinomial(cuda_tensor, generator=cpu_generator)`` raises. The failure
    is loud, which is why the earlier CPU generator proves the sampling path had
    never run on CUDA rather than that it ran and was wrong.
    """
    return torch.Generator(device=str(device)).manual_seed(int(seed))


@dataclass
class DrawResult:
    """Raw draws plus the three numbers every comparison is reported beside."""

    windows: tuple
    log_probabilities: object
    forward_calls: int
    seconds: float
    strategy: str
    orders: tuple = ()
    diagnostics: dict = field(default_factory=dict)

    def accounting(self, *, mode="full_mask"):
        return draw_accounting(draws=len(self.windows), forward_calls=int(self.forward_calls),
                               seconds=float(self.seconds), mode=mode)

    def document(self):
        return {"strategy": self.strategy, "draws": len(self.windows),
                "forward_calls": int(self.forward_calls), "seconds": float(self.seconds),
                "mean_log_probability": (None if self.log_probabilities is None
                                         or not len(self.log_probabilities) else
                                         float(np.mean(self.log_probabilities))),
                "diagnostics": dict(self.diagnostics)}


def _masked_step(services, route, model, constructs, revealed, candidates, unrevealed,
                 condition):
    """Conditionals at each row's candidate coordinates. One forward.

    ``unrevealed`` is masked in the input and ``candidates`` are the positions whose
    logits are read. Both arguments are needed and the distinction is the whole
    correctness of the fill: masking only the candidate would leave the not-yet-filled
    coordinates showing the SEED's residues, so the model would be conditioning on
    the wild-type window it is supposed to be replacing and the reported density
    would not be the one the draw came from.
    """
    from .her2_routes_serial import ablang_serialization
    views, per_row = [], []
    for construct, revealed_map, wanted, open_sites in zip(constructs, revealed, candidates,
                                                           unrevealed):
        current = construct.with_residues(tuple(revealed_map),
                                          "".join(revealed_map.values())) if revealed_map \
            else construct
        current = data_lib.with_sites(current, tuple(open_sites))
        views.append((current, ablang_serialization(current, masked_sites=tuple(open_sites),
                                                    order=tuple(wanted)), condition))
        per_row.append(tuple(wanted))
    batch = services.collate(route, views)
    log_probs = services.canonical_log_probs(route, model, batch)
    return per_row, log_probs, 1


def _causal_step(services, route, model, constructs, revealed, candidates, unrevealed,
                 condition, *, order_prefix, full_order):
    """Conditionals at each row's candidate coordinates, by expanding the batch.

    An autoregressive decoder's conditional depends on the pointer it is given, so
    "which site is easiest" cannot be read off one forward. Each row is expanded over
    its candidates and all of them are scored in one forward, which costs the same
    scored rows as sequential probing and one call instead of many.

    The emission order handed to the serialiser is the **whole** edit map for that row
    -- prefix, then the candidate, then the rest in a fixed order -- because
    ``causal_serialization`` masks every site of the map in the context and requires
    the order to be a permutation of it. Causal masking is what makes the conditional
    at step ``len(prefix)`` independent of whatever follows the candidate.

    ``full_order`` is **per row**: the revision path re-opens a different subset of
    coordinates in every row, so one shared order would mask the wrong positions in
    every row but the first.
    """
    from .her2_routes_serial import causal_serialization
    views = []
    for row, (construct, revealed_map, wanted) in enumerate(
            zip(constructs, revealed, candidates)):
        prefix = tuple(order_prefix[row])
        row_order = tuple(full_order[row])
        for site in wanted:
            tail = tuple(member for member in row_order
                         if member not in prefix and member != site)
            order = prefix + (site,) + tail
            current = data_lib.with_sites(construct, tuple(row_order))
            residues = ("".join(revealed_map[member] for member in prefix)
                        + current.residue_at(site)
                        + "".join(current.residue_at(member) for member in tail))
            views.append((current, causal_serialization(
                current, order=order, residues=residues,
                max_positions=services._positions(route)), condition))
    batch = services.collate(route, views)
    log_probs = services.canonical_log_probs(route, model, batch)
    per_row, gathered = [], []
    cursor = 0
    for row, wanted in enumerate(candidates):
        block = []
        for _site in wanted:
            block.append(log_probs[cursor, len(order_prefix[row])])
            cursor += 1
        per_row.append(tuple(wanted))
        gathered.append(torch.stack(block) if block else torch.empty(0))
    del unrevealed
    return per_row, torch.stack(gathered) if gathered else torch.empty(0), 1


@torch.no_grad()
def fill_batch(services, route, model, *, draws, sites, condition, strategy, seed,
               batch_size=32, temperature=1.0, remask_rounds=0, remask_fraction=0.3,
               seed_window=None):
    """Generate ``draws`` windows with the declared fill order. Batched, measured.

    ``easy_first`` and ``hard_first`` read the model's own conditional: the next site
    is the one whose current distribution has the lowest (or highest) entropy. That
    is a property of the model at that step, so the two strategies are genuinely
    different procedures rather than two random orders.

    The recorded log probability is always the **temperature-1** density of the
    realised residues along the order that produced them, so a draw can be rescored
    and must agree.
    """
    require(strategy in STRATEGIES, f"Unknown fill strategy {strategy!r}")
    if strategy == ORDER_DIFFUSION:
        return diffusion_fill(services, route, model, draws=draws, sites=sites,
                              condition=condition, seed=seed, batch_size=batch_size,
                              seed_window=seed_window, temperature=temperature)
    from .her2_routes_serial import her2_construct
    scaffold = services.scaffold()
    sites = tuple(sites)
    base = seed_window or services.config["her2"]["wild_type_core"]
    generator = device_generator(services.device, seed)
    numpy_rng = np.random.default_rng([int(seed), 17])
    windows, log_probabilities, orders = [], [], []
    forwards, started = 0, time.perf_counter()
    entropies = []
    # The confidence-ordered strategies score every remaining coordinate of every
    # draw in one forward, so the AR route's scored-row count is the draw batch TIMES
    # the window width. The batch is divided by that width rather than left to spill:
    # this card has about 2.3 GiB usable and a 160-row by 240-token FP32 forward does
    # not fit inside it.
    expansion = (len(sites) if (route == "piggen_causal_infiller"
                                and strategy in (ORDER_EASY_FIRST, ORDER_HARD_FIRST)) else 1)
    batch_size = max(1, int(batch_size) // int(expansion))
    for start in range(0, int(draws), int(batch_size)):
        count = min(int(batch_size), int(draws) - start)
        constructs = [her2_construct(scaffold, base, record_id=f"draw{start + index}")
                      for index in range(count)]
        constructs = [data_lib.with_sites(construct, sites) for construct in constructs]
        revealed = [{} for _ in range(count)]
        remaining = [list(sites) for _ in range(count)]
        prefix = [[] for _ in range(count)]
        totals = np.zeros(count, dtype=np.float64)
        row_orders = [[] for _ in range(count)]
        fixed = None
        if strategy in (ORDER_DECLARED, ORDER_REMASK):
            # The remasking arm FILLS in the declared order and then revises. Its
            # difference from ``declared`` is the revision rounds, not the first pass,
            # so leaving ``fixed`` unset here would have made the first pass an
            # attribute lookup on None.
            fixed = [list(sites) for _ in range(count)]
        elif strategy == ORDER_RANDOM:
            fixed = [[sites[int(index)] for index in numpy_rng.permutation(len(sites))]
                     for _ in range(count)]
        for _step in range(len(sites)):
            if strategy in (ORDER_EASY_FIRST, ORDER_HARD_FIRST):
                candidates = [list(row) for row in remaining]
            else:
                candidates = [[order[len(prefix[row])]] for row, order in enumerate(fixed)]
            if route == "piggen_causal_infiller":
                per_row, conditionals, calls = _causal_step(
                    services, route, model, constructs, revealed, candidates, remaining,
                    condition, order_prefix=prefix,
                    full_order=[sites for _ in range(count)])
            else:
                per_row, conditionals, calls = _masked_step(
                    services, route, model, constructs, revealed, candidates, remaining,
                    condition)
            forwards += calls
            with torch.no_grad():
                probabilities = conditionals.exp()
                entropy = -(probabilities * conditionals).sum(dim=-1)
            for row in range(count):
                available = per_row[row]
                if strategy == ORDER_EASY_FIRST:
                    position = int(torch.argmin(entropy[row, :len(available)]))
                elif strategy == ORDER_HARD_FIRST:
                    position = int(torch.argmax(entropy[row, :len(available)]))
                else:
                    position = 0
                site = available[position]
                distribution = conditionals[row, position]
                entropies.append(float(entropy[row, position]))
                scaled = torch.softmax(distribution / float(temperature), dim=-1)
                choice = int(torch.multinomial(scaled, 1, generator=generator).item())
                totals[row] += float(distribution[choice])
                revealed[row][site] = CANONICAL[choice]
                remaining[row] = [member for member in remaining[row] if member != site]
                prefix[row].append(site)
                row_orders[row].append(site)
        for row in range(count):
            filled = her2_construct(scaffold, base, record_id="draw")
            order = tuple(row_orders[row])
            filled = filled.with_residues(order, "".join(revealed[row][s] for s in order))
            windows.append(filled.residues(sites))
            log_probabilities.append(float(totals[row]))
            orders.append(order)
    result = DrawResult(
        windows=tuple(windows), log_probabilities=np.asarray(log_probabilities),
        forward_calls=forwards, seconds=time.perf_counter() - started, strategy=strategy,
        orders=tuple(orders),
        diagnostics={"mean_step_entropy": float(np.mean(entropies)) if entropies else None,
                     "temperature": float(temperature),
                     "draw_batch": int(batch_size), "row_expansion": int(expansion),
                     "confidence_source": ("the model's own conditional entropy at each step. An "
                                           "earlier revision used rng.random(), which makes "
                                           "easy_first and hard_first two random orders with "
                                           "different labels.")})
    if int(remask_rounds) > 0:
        return remask(services, route, model, result, sites=sites, condition=condition,
                      rounds=int(remask_rounds), fraction=float(remask_fraction),
                      seed=int(seed) + 1, batch_size=int(batch_size),
                      temperature=float(temperature))
    return result


@torch.no_grad()
def diffusion_fill(services, route, model, *, draws, sites, condition, seed, batch_size=32,
                   seed_window=None, temperature=1.0, rescore=True):
    """Generate windows with the model's native iterative reparameterized decoder.

    Every design site is ``<mask>`` at the start and every other position of the paired
    construct is fixed scaffold; ``model.iterative_decode`` refines all sites together over
    ``decode_steps`` denoiser forwards and asserts, per batch, that the scaffold is unchanged
    and that only canonical residues were produced.

    **No trajectory log probability is reported.** The decoder's token scores are the argmax of
    Gumbel-perturbed log-softmax, not a density. When ``rescore`` is true each final window is
    rescored in the declared fixed order at temperature one -- a normalised functional of the
    model's conditionals, labelled as the fixed-order density of the FINAL window, exactly as
    ``remask`` does -- and those forwards are counted. ``temperature`` is not a parameter of
    this decoder and any value other than 1.0 is refused rather than silently ignored.
    """
    require(hasattr(model, "iterative_decode"),
            f"{route} has no iterative diffusion decoder; {ORDER_DIFFUSION!r} is defined for the "
            "native DPLM routes")
    require(float(temperature) == 1.0,
            "The diffusion decoder samples by Gumbel-argmax and has no temperature; a value other "
            "than 1.0 would be recorded and ignored")
    from .her2_routes_serial import ablang_serialization, her2_construct
    scaffold = services.scaffold()
    sites = tuple(sites)
    base = seed_window or services.config["her2"]["wild_type_core"]
    generator = device_generator(services.device, seed)
    windows, forwards, started = [], 0, time.perf_counter()
    steps = None
    for start in range(0, int(draws), int(batch_size)):
        count = min(int(batch_size), int(draws) - start)
        views = []
        for index in range(count):
            construct = data_lib.with_sites(
                her2_construct(scaffold, base, record_id=f"draw{start + index}"), sites)
            views.append((construct, ablang_serialization(
                construct, masked_sites=sites, order=sites), condition))
        batch = services.collate(route, views)
        decoded = model.iterative_decode(batch.tokens, batch.query_slots,
                                         condition_indices=batch.condition_indices,
                                         generator=generator)
        forwards += int(decoded.forward_calls)
        steps = int(decoded.steps)
        indices = decoded.canonical.cpu().numpy()
        for row in range(count):
            windows.append("".join(CANONICAL[int(value)] for value in indices[row, :len(sites)]))
    scores, scoring_calls = None, 0
    if rescore and windows:
        scores = services.window_scores(route, model, windows, condition=condition,
                                        batch_size=batch_size, sites=sites,
                                        base=base).detach().cpu().numpy()
        scoring_calls = ((len(windows) + int(batch_size) - 1) // int(batch_size)) * len(sites)
        forwards += scoring_calls
    diagnostics = {
        **(decoded.document() if windows else {}),
        "decode_steps": steps, "draw_batch": int(batch_size),
        "denoiser_forward_calls_decoding": forwards - scoring_calls,
        "final_rescore_forward_calls": scoring_calls,
        "terminal_distribution_log_probability": None,
        "log_probability_note": (
            "the recorded log probability, when present, is the declared fixed-order density of "
            "the FINAL window at temperature one; the diffusion trajectory itself has none."),
        "scaffold_immutable": True, "sampling_support": "the twenty canonical residues"}
    return DrawResult(windows=tuple(windows), log_probabilities=scores, forward_calls=forwards,
                      seconds=time.perf_counter() - started, strategy=ORDER_DIFFUSION,
                      diagnostics=diagnostics)


@torch.no_grad()
def remask(services, route, model, result, *, sites, condition, rounds, fraction, seed,
           batch_size=32, temperature=1.0):
    """Stochastic revision: re-open a fraction of each window and refill it.

    The terminal distribution of a revision process has no closed form here, so the
    comparison against the fixed orders is by samples at shared draw counts. The
    returned log probability scores each final window in the declared fixed order
    at temperature one. It is not the terminal density of the revision process.
    """
    from .her2_routes_serial import her2_construct
    scaffold = services.scaffold()
    sites = tuple(sites)
    generator = device_generator(services.device, seed)
    numpy_rng = np.random.default_rng([int(seed), 23])
    windows = list(result.windows)
    forwards, started = int(result.forward_calls), time.perf_counter()
    for _round in range(int(rounds)):
        count = max(1, int(round(float(fraction) * len(sites))))
        for start in range(0, len(windows), int(batch_size)):
            block = windows[start:start + int(batch_size)]
            constructs, reopened = [], []
            for index, window in enumerate(block):
                construct = data_lib.with_sites(
                    her2_construct(scaffold, window, record_id=f"remask{start + index}"), sites)
                chosen = tuple(sites[int(position)] for position in
                               sorted(numpy_rng.choice(len(sites), size=count, replace=False)))
                constructs.append(construct)
                reopened.append(chosen)
            revealed = [{} for _ in constructs]
            prefix = [[] for _ in constructs]
            for depth in range(count):
                candidates = [[chosen[depth]] for chosen in reopened]
                # Only the re-opened coordinates are masked. The rest of the window is
                # the current draw's own residues, which is what makes this a REVISION
                # of a sample rather than a fresh fill of the whole window.
                open_sites = [tuple(chosen[depth:]) for chosen in reopened]
                if route == "piggen_causal_infiller":
                    per_row, conditionals, calls = _causal_step(
                        services, route, model, constructs, revealed, candidates, open_sites,
                        condition, order_prefix=prefix, full_order=list(reopened))
                else:
                    per_row, conditionals, calls = _masked_step(
                        services, route, model, constructs, revealed, candidates, open_sites,
                        condition)
                forwards += calls
                for row in range(len(constructs)):
                    scaled = torch.softmax(conditionals[row, 0] / float(temperature), dim=-1)
                    choice = int(torch.multinomial(scaled, 1, generator=generator).item())
                    revealed[row][per_row[row][0]] = CANONICAL[choice]
                    prefix[row].append(per_row[row][0])
            for row, construct in enumerate(constructs):
                order = tuple(reopened[row])
                updated = construct.with_residues(
                    order, "".join(revealed[row][site] for site in order))
                windows[start + row] = updated.residues(sites)
    scores = services.window_scores(route, model, windows, condition=condition,
                                    batch_size=batch_size).detach().cpu().numpy()
    scoring_calls = (len(windows) + int(batch_size) - 1) // int(batch_size)
    if route != "piggen_causal_infiller":
        scoring_calls *= len(sites)
    forwards += scoring_calls
    return DrawResult(
        windows=tuple(windows), log_probabilities=scores,
        forward_calls=forwards,
        seconds=float(result.seconds) + (time.perf_counter() - started),
        strategy=ORDER_REMASK, orders=result.orders,
        diagnostics={**result.diagnostics, "remask_rounds": int(rounds),
                     "remask_fraction": float(fraction),
                     "terminal_distribution_log_probability": None,
                     "final_rescore_forward_calls": scoring_calls,
                     "log_probability_note": ("final windows rescored in the declared fixed order "
                                              "at temperature one; the terminal density of the "
                                              "remasking process is unavailable.")})


def energy_generate(services, model, *, draws, context_rows, seed, batch_size=32,
                    sampling_options=None, requested_bin=None, mode=None):
    """Walk and jump. Raw continuous samples decoded to residues, not an imitation.

    The energy route has no masked-language interface, and giving it one would make
    its generation a different procedure from the one it was trained for. This is
    the pinned sampler: noise a random discrete seed, run the Langevin chain against
    the score, then decode through the denoiser.

    The conditioning mode is the model's VALIDATED runtime contract (read from its checkpoint's
    energy contract by ``load_for_scoring``), never its architecture: the null arm owns condition
    tables and context towers it was never given a context for. A model with no runtime contract
    is refused, and a ``mode`` that contradicts it is refused. In the conditioned mode the context
    is built from ``context_rows`` at ``requested_bin`` (default ``high``). In the null mode NO
    context is built and neither tower runs, ``context_rows`` are not read, and the requested
    quality bin is not applicable: asking for one is refused rather than silently ignored.
    """
    runtime = energy_contract.require_runtime(model, where="energy_generate", mode=mode)
    conditioned = runtime.conditioned
    if conditioned:
        require(context_rows, "Energy generation needs measured rows to build its context from")
        requested = "high" if requested_bin is None else requested_bin
    else:
        require(requested_bin is None,
                f"energy_generate: the null arm is given no context, so a requested quality bin "
                f"({requested_bin!r}) is not applicable and would be silently ignored")
        requested = None
    generator = services.energy_generator(seed)
    windows, started, forwards = [], time.perf_counter(), 0
    for start in range(0, int(draws), int(batch_size)):
        count = min(int(batch_size), int(draws) - start)
        context = None
        if conditioned:
            rows = [context_rows[(start + index) % len(context_rows)] for index in range(count)]
            context = services.energy_context(rows, requested_bin=requested)
        with torch.no_grad():
            indices = model.sample(count, context=context, generator=generator,
                                   sampling_options=sampling_options,
                                   slots=model.window)
        forwards += int((sampling_options or {}).get("steps", 100)) + 1
        for row in indices:
            windows.append("".join(CANONICAL[int(value)] for value in row))
    return DrawResult(
        windows=tuple(windows), log_probabilities=None,
        forward_calls=forwards, seconds=time.perf_counter() - started, strategy="walk_jump",
        diagnostics={"likelihood": ("none. A smoothed energy and a denoised sample do not supply a "
                                    "normalised discrete sequence likelihood."),
                     "requested_bin": requested,
                     "requested_bin_applicable": bool(conditioned),
                     "mode": runtime.mode, "arm": runtime.arm,
                     "context_built": bool(conditioned),
                     "initialization": runtime.initialization,
                     "sampling_options": dict(sampling_options or {})})


#: The scalable index lives in ``her2_routes_split`` because it is the same
#: deletion-mask argument the split's own distance work uses and because that module
#: imports no torch. Re-exported here so the generation call sites read in one place.
deletion_keys = split_lib.deletion_keys
greedy_medoids = split_lib.greedy_medoids
assign_to_medoids = split_lib.assign_to_medoids
HammingIndex = split_lib.HammingIndex
