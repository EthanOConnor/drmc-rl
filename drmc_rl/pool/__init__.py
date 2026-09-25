"""Continuously running rating pool for drmc-rl and Professor Pills players.

See docs/RATING_POOL.md. Everything in this package except the worker runtime
is torch-free so the coordinator can run on a host that never loads a model.
"""
