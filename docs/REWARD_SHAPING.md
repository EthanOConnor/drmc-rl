# Competitive objective and auxiliary learning signals

## Authority

The final VS objective is match outcome:

```text
win  +1
Draw   0
loss  -1
```

A cumulative garbage, virus, speed, clear-type, or survival signal may not pay
enough to make a lower-win-probability policy optimal.

## Permitted uses of tactical information

- auxiliary prediction heads;
- start-state and replay prioritization;
- curriculum stage selection;
- counterfactual candidate labels;
- potential-based shaping tied to a frozen/slow target;
- early bootstrap shaping annealed to zero;
- lexicographic style preference inside an epsilon competitive-value envelope.

All shaping terms are logged separately from environment W/D/L. Promotion and
release evaluation ignore accumulated shaping return.

## Clear and attack styles

Clear-race, pressure, safety, and attack preferences belong in the trainer's
style decoder or bounded secondary objectives. Separate specialist reward
checkpoints may be used as exploiters/teachers, not shipped as independently
calibrated trainer styles.

## Single-player

Single-player speed/clear research may retain time-to-clear, risk/CVaR, and
potential shaping, but those values are not imported directly as the VS match
objective.
