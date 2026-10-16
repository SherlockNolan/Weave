# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep Weave's existing YAML composition order on Isaac Lab 2 and 3.

Lab 3 removed the old decorator and consumes scalar overrides before YAML
composition. Weave needs its saved training YAMLs composed before CLI overrides.
"""

try:
    from isaaclab_tasks.utils.hydra import hydra_task_config
except ImportError:
    import functools
    from pathlib import Path

    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf

    from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
    from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices, to_dict, update_from_dict

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    def hydra_task_config(task_name, agent_entry_point):
        def decorate(function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                task = task_name.split(":")[-1]
                env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
                agent_cfg = load_cfg_from_registry(task, agent_entry_point)
                # RSL-RL 3 still consumes the original combined policy config.
                agent_cfg.actor = None
                agent_cfg.critic = None
                agent_cfg.empirical_normalization = False
                env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
                config = replace_slices_with_strings({"env": to_dict(env_cfg), "agent": to_dict(agent_cfg)})
                ConfigStore.instance().store(name=task, node=config)

                @hydra.main(config_path=str(Path(__file__).resolve().parent), config_name=task, version_base="1.3")
                def run(config):
                    config = replace_strings_with_slices(OmegaConf.to_container(config, resolve=True))
                    update_from_dict(env_cfg, config["env"])
                    update_from_dict(agent_cfg, config["agent"])
                    resolved_env = replace_strings_with_env_cfg_spaces(env_cfg)
                    return function(resolved_env, agent_cfg, *args, **kwargs)

                return run()

            return wrapper

        return decorate
