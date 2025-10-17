"""Evaluator-guided play harness.

Runs episodes using one-step planning guided by a TorchScript evaluator.
"""
import argparse
from envs.retro.register_env import register_env_id
from gymnasium import make
from models.evaluator.runtime import ScriptEvaluator
from models.policy.planning import plan_one_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluator', type=str, required=True, help='Path to TorchScript evaluator')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--obs-mode', choices=['state', 'pixel'], default='state')
    ap.add_argument('--mode', choices=['mean', 'quantile', 'cvar'], default='quantile')
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--alpha', type=float, default=0.25)
    args = ap.parse_args()

    register_env_id()
    env = make('DrMarioRetroEnv-v0', obs_mode=args.obs_mode)
    evaluator = ScriptEvaluator(args.evaluator)

    class DummyPolicy:
        def legal_actions(self, obs):
            # All actions legal by default
            return list(range(env.action_space.n))

    policy = DummyPolicy()
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_r = 0.0
        while not done:
            a = plan_one_step(env, policy, evaluator, risk_tau=args.tau, mode=args.mode, alpha=args.alpha)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            total_r += r
        print(f"episode {ep+1}: return={total_r:.1f}, cleared={info.get('viruses_remaining',1)==0}")
    env.close()


if __name__ == '__main__':
    main()

