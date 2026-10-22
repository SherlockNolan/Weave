"""Training scale reported by Weave's author, without importing Isaac or PyTorch."""

import json
import os
from pathlib import Path

ISSUE_URL = "https://github.com/xiaohu-art/Weave/issues/2"
PAPER_WORLD_SIZE = 8
PAPER_ENVS_PER_RANK = 16384
PAPER_TOTAL_ENVS = PAPER_WORLD_SIZE * PAPER_ENVS_PER_RANK
PAPER_ITERATIONS = 100000
TRAIN_FILES = (
    "clothesstand_861clip.npz", "floorlamp_805clip.npz", "largebox_791clip.npz",
    "largetable_767clip.npz", "smallbox_637clip.npz", "smalltable_824clip.npz",
    "tripod_1131clip.npz", "whitechair_1066clip.npz", "woodchair_987clip.npz",
)


def make_training_recipe(world_size, envs_per_rank, steps=32, mini_batches=4, epochs=5, motion_files=()):
    if any(value <= 0 for value in (world_size, envs_per_rank, steps, mini_batches, epochs)):
        raise ValueError("Training scale, rollout length, epochs and mini-batches must be positive.")
    if envs_per_rank * steps % mini_batches:
        raise ValueError("Per-rank rollout must divide evenly into mini-batches.")
    total_envs = world_size * envs_per_rank
    rollout = total_envs * steps
    return {
        "source": ISSUE_URL,
        "world_size": world_size,
        "envs_per_rank": envs_per_rank,
        "total_envs": total_envs,
        "num_steps_per_env": steps,
        "global_rollout_samples": rollout,
        "num_mini_batches": mini_batches,
        "global_samples_per_mini_batch": rollout // mini_batches,
        "num_learning_epochs": epochs,
        "paper_total_envs": PAPER_TOTAL_ENVS,
        "fraction_of_paper_parallelism": total_envs / PAPER_TOTAL_ENVS,
        "matches_author_parallelism": total_envs == PAPER_TOTAL_ENVS,
        "motion_files": [str(Path(p).resolve()) for p in motion_files],
    }


def validate_paper_recipe(recipe):
    expected = {
        "world_size": PAPER_WORLD_SIZE, "envs_per_rank": PAPER_ENVS_PER_RANK,
        "num_steps_per_env": 32, "num_mini_batches": 4, "num_learning_epochs": 5,
    }
    differences = {key: (recipe.get(key), value) for key, value in expected.items() if recipe.get(key) != value}
    files = [Path(p) for p in recipe.get("motion_files", ())]
    if {p.name for p in files} != set(TRAIN_FILES) or len(files) != len(TRAIN_FILES):
        differences["motion_files"] = ("custom references", "the nine complete D_train files")
    if any(p.parent.name != "train" for p in files):
        differences["split"] = ("custom", "D_train")
    if differences:
        raise ValueError(f"Paper recipe mismatch (actual, required): {differences}")


def validate_paper_resume(checkpoint, recipe):
    """Only resume a run which recorded the same full-scale recipe."""
    record = Path(checkpoint).parent / "training_recipe.json"
    if not record.is_file():
        raise ValueError(f"Paper resume requires {record}; old 4096-env checkpoints are not this recipe.")
    saved = json.loads(record.read_text())
    validate_paper_recipe(saved)
    for key in ("world_size", "envs_per_rank", "num_steps_per_env", "num_mini_batches", "num_learning_epochs"):
        if saved[key] != recipe[key]:
            raise ValueError(f"Resumed recipe differs in {key}: {saved[key]} != {recipe[key]}")


if __name__ == "__main__":
    indices = os.environ.get("WEAVE_GPU_INDICES", "0,1").split(",")
    config = os.environ.get("WEAVE_TRAIN_CONFIG", "train_reference_dual")
    references = [Path("data/train") / name for name in TRAIN_FILES]
    if config == "train_pickups_dual":
        references = [Path("data/pickups") / f"{name}_pickups.npz" for name in ("tripod", "clothesstand", "whitechair")]
    recipe = make_training_recipe(
        len(indices), int(os.environ.get("WEAVE_TRAIN_ENVS_PER_GPU", "16384")),
        motion_files=references,
    )
    recipe.update({
        "plan_only": True,
        "config": config,
        "physical_gpu_indices": [int(index) for index in indices],
        "activation": os.environ.get("WEAVE_TRAIN_ACTIVATION", "scripts/reproduce/activate.sh"),
        "max_iterations": int(os.environ.get("WEAVE_TRAIN_ITERATIONS", "100000")),
        "run_name": os.environ.get("WEAVE_TRAIN_RUN_NAME", "reference-scaled-dual"),
        "note": (
            "Declared launch settings, before optional CLI/Hydra overrides. "
            "The training entry records resolved settings."
        ),
    })
    if recipe["config"] == "train_paper":
        validate_paper_recipe(recipe)
        if recipe["max_iterations"] != PAPER_ITERATIONS:
            raise ValueError("The paper recipe targets 100000 iterations.")
    print(json.dumps(recipe, indent=2))
