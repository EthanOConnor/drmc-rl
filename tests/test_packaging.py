from __future__ import annotations

from importlib.resources import files

from setuptools import find_namespace_packages


def test_runtime_packages_are_discovered() -> None:
    packages = set(
        find_namespace_packages(
            include=[
                "drmc_rl*",
                "tools*",
            ],
            exclude=["data*", "docs*", "dr-mario-disassembly*", "legal_ROMs*"],
        )
    )
    assert {
        "drmc_rl",
        "drmc_rl.game",
        "drmc_rl.planning",
        "drmc_rl.planning.cuda",
        "drmc_rl.envs.libretro",
        "drmc_rl.training",
        "drmc_rl.seedlab",
    } <= packages


def test_runtime_resources_are_packaged() -> None:
    root = files("drmc_rl")
    assert (root / "game/specs/reward_config.json").is_file()
    assert (root / "envs/libretro/seeds/registry.json").is_file()
    assert (root / "planning/cuda/drm_reach.cu").is_file()
    assert (root / "training/configs/smdp_ppo.yaml").is_file()
