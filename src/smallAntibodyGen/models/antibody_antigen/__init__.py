"""Antibody-antigen residue models behind one narrow, hidden-truth-free interface.

``views`` holds the immutable :class:`VisibleView` a backend is allowed to see and the
separate :class:`MaskedTargets` it is never shown. ``denoising`` holds the backend
contract plus the scorers (masked NLL, sequential log-probability, pair gap) and the
iterative masked completion. ``mint`` adapts the frozen native MINT release to that
contract. Nothing here imports ``experiments``, ``scripts`` or ``tests``.
"""
