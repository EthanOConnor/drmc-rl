"""Read-only loss-gradient geometry; no Adam step or causal strength inference."""
from __future__ import annotations

import math

import torch


def loss_gradient_geometry(named_parameters, terms):
    """Measure coefficient-weighted gradients without writing parameter .grad.

    Negative dot products describe conflicting Euclidean descent directions.
    They do not model Adam's moments, predict a finite PPO update or establish
    that an auxiliary loss caused a tournament regression.
    """
    parameters = [(n, p) for n, p in named_parameters if p.requires_grad]
    names = list(terms)
    if not parameters or not names or not all(torch.isfinite(t).item() for t in terms.values()):
        raise ValueError("finite scalar loss terms and trainable parameters are required")
    gradients = []
    for i, term in enumerate(terms.values()):
        gradients.append(torch.autograd.grad(term, [p for _, p in parameters],
            allow_unused=True, retain_graph=i + 1 < len(terms)))
    groups = {"all": list(range(len(parameters)))}
    if "policy_loss" in names and "value_loss" in names:
        actor, value = names.index("policy_loss"), names.index("value_loss")
        groups["shared_actor_value"] = [i for i in range(len(parameters))
            if gradients[actor][i] is not None and gradients[value][i] is not None]
    report = dict(terms=names, groups={})
    for group, indices in groups.items():
        gram = [[0.0 for _ in names] for _ in names]
        for index in indices:
            vectors = [g[index].detach().reshape(-1).double() if g[index] is not None else None
                       for g in gradients]
            for i, left in enumerate(vectors):
                for j in range(i, len(vectors)):
                    if left is not None and vectors[j] is not None:
                        gram[i][j] += float(torch.dot(left, vectors[j]))
        for i in range(len(names)):
            for j in range(i):
                gram[i][j] = gram[j][i]
        if not all(math.isfinite(v) for row in gram for v in row):
            raise RuntimeError("non-finite loss gradient geometry")
        norms = [math.sqrt(max(0.0, gram[i][i])) for i in range(len(names))]
        cosine = [[gram[i][j] / (norms[i] * norms[j]) if norms[i] and norms[j] else None
                   for j in range(len(names))] for i in range(len(names))]
        report["groups"][group] = dict(
            parameters=sum(parameters[i][1].numel() for i in indices),
            norms=dict(zip(names, norms)), gram=gram, cosine=cosine,
            combined_norm=math.sqrt(max(0.0, sum(map(sum, gram)))))
    return report
