"""Sample Factory callback to log shaping stats & evaluator metrics per rollout.
Integrate per your SF version (APIs change; treat this as a sketch).
"""
import numpy as np


class ShapingLoggingCallback:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_env = 0.0
        self.sum_shape = 0.0
        self.sum_total = 0.0
        self.count = 0

    def on_step(self, infos):
        # infos: list[dict], each may contain r_env, r_shape, r_total
        for info in infos:
            re = info.get('r_env'); rs = info.get('r_shape'); rt = info.get('r_total')
            if re is not None and rs is not None and rt is not None:
                self.sum_env += float(re); self.sum_shape += float(rs); self.sum_total += float(rt)
                self.count += 1

    def summarize(self):
        if self.count == 0:
            return {}
        return {
            'shaping/mean_r_env': self.sum_env / self.count,
            'shaping/mean_r_shape': self.sum_shape / self.count,
            'shaping/mean_r_total': self.sum_total / self.count,
            'shaping/ratio_shape_env': (self.sum_shape / max(1e-6, abs(self.sum_env))) if self.sum_env != 0 else 0.0,
        }

