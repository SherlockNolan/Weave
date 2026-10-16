"""Searchable live policy evaluation with a browser connection that survives task changes."""

import copy
import gc
import json
import threading
import time
from pathlib import Path

import gymnasium as gym
import torch
from g1_hoi_learning.tasks.hoi.mdp.commands import MotionLoader
from isaaclab_visualizers.viser.viser_visualizer import ViserVisualizer
from rsl_rl.runners import OnPolicyRunner
from viser_catalog import OBJECT_LABELS, REPO, TaskCatalog
from viser_options import TerminalScene, configure_viser

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class TaskPanel:
    """GUI callbacks queue requests; only the simulation thread touches Isaac state."""

    PLACEHOLDER = "— 请选择任务 / Select a task —"

    def __init__(self, catalog, checkpoint):
        self.catalog = catalog
        self.checkpoint = Path(checkpoint).name
        self.lock = threading.RLock()
        self.pending = None
        self.query = ""
        self.split = "all"
        self.object = "all"
        self.current = catalog.tasks[0]
        self.selected = self.current
        self.server = None
        self.paused = False
        self.running = False

    def pop_request(self):
        with self.lock:
            request, self.pending = self.pending, None
            return request

    def attach(self, server, current):
        if self.server is not None:
            self.activate(current)
            return
        self.server = server
        self.current = current
        self.selected = current
        self.paused = self.running = False
        with server.gui.add_folder("任务选择 / Tasks", order=-1000):
            server.gui.add_markdown(f"统一策略：`{self.checkpoint}`")
            search = server.gui.add_text(
                "搜索全部任务", initial_value=self.query, hint="支持序列 ID、物体中英文名；多个关键词以空格分隔。"
            )
            scopes = {"全部": "all", "训练集 train": "train", "测试集 test": "test", "关注拾取": "pickups"}
            scope = server.gui.add_dropdown(
                "数据范围", options=tuple(scopes), initial_value=next(k for k, v in scopes.items() if v == self.split)
            )
            objects = {"全部物体": "all"}
            objects.update(
                {
                    f"{obj} / {OBJECT_LABELS.get(obj, obj)}": obj
                    for obj in sorted({t.object for t in self.catalog.tasks})
                }
            )
            obj = server.gui.add_dropdown(
                "物体", options=tuple(objects), initial_value=next(k for k, v in objects.items() if v == self.object)
            )
            self.matches = server.gui.add_markdown("")
            self.dropdown = server.gui.add_dropdown(
                "任务 / Task",
                options=(self.PLACEHOLDER,),
                hint="上方搜索覆盖全部任务；当前选择始终保留，输入关键词不会切换任务。",
            )
            self.start = server.gui.add_button("开始 / 重播", color="green")
            self.pause = server.gui.add_button("暂停", disabled=True)
            self.status = server.gui.add_markdown("")

        @search.on_update
        def search_changed(event):
            if event.client is not None:
                with self.lock:
                    self.query = search.value
                    self.refresh_options()

        @scope.on_update
        def scope_changed(event):
            if event.client is not None:
                with self.lock:
                    self.split = scopes[scope.value]
                    self.refresh_options()

        @obj.on_update
        def object_changed(event):
            if event.client is not None:
                with self.lock:
                    self.object = objects[obj.value]
                    self.refresh_options()

        @self.dropdown.on_update
        def selected(event):
            if event.client is not None:
                task = self.catalog.by_label.get(self.dropdown.value)
                if task is not None:
                    with self.lock:
                        self.selected = task
                        self.pending = task.key
                    self.status.content = f"正在切换到 **{task.name}**…"

        @self.start.on_click
        def start(_event):
            with self.lock:
                self.pending = self.selected.key

        @self.pause.on_click
        def pause(_event):
            with self.lock:
                self.paused = not self.paused
                self.pause.label = "继续" if self.paused else "暂停"

        self.activate(current)

    def activate(self, current):
        with self.lock:
            self.current = current
            # Completing an older load must never overwrite a newer GUI choice.
            self.set_running(False)
            self.refresh_options()
            self.set_status("已就绪，点击 **开始 / 重播**；选择其他任务会自动开始执行。")

    def refresh_options(self):
        selected = self.catalog.by_label.get(self.dropdown.value)
        if selected is not None:
            self.selected = selected
        results = self.catalog.search(self.query, self.split, self.object)
        # Keep the selected value valid even when the search/filter excludes it.
        # Otherwise updating options replaces the selection with a placeholder.
        shown = [self.selected, *(task for task in results if task != self.selected)][:200]
        labels = tuple(task.label for task in shown)
        self.dropdown.options = labels
        self.dropdown.value = self.selected.label
        self.dropdown.disabled = not bool(results)
        suffix = "；显示前 200 项，请继续输入关键词缩小范围" if len(results) > 200 else ""
        self.matches.content = f"匹配 **{len(results)} / {len(self.catalog.tasks)}** 个片段{suffix}。"

    def set_status(self, text):
        self.status.content = (
            f"**当前：{self.current.name}**  \n{self.current.object} · {self.current.split}  \n\n{text}"
        )

    def set_running(self, value):
        with self.lock:
            self.running = value
            self.paused = False
            self.pause.disabled = not value
            self.pause.label = "暂停"


class ViewerLifetime:
    """Retain the native viewer while individual physics scenes are rebuilt."""

    def __init__(self):
        self.viewer = None
        lifetime = self

        class TaskVisualizer(ViserVisualizer):
            def _setup_isaaclab_sidebar(self, server):
                # Simulation pause/reset is handled by TaskPanel at safe step boundaries.
                if self._viewer._live_plots_folder is None:
                    server.gui.set_panel_label("Weave · Policy eval")
                    self._viewer._live_plots_folder = server.gui.add_folder("Live plots", expand_by_default=False)

            def _create_viewer(self, record_to_viser, metadata=None):
                if lifetime.viewer is None:
                    super()._create_viewer(record_to_viser, metadata)
                    lifetime.viewer = self._viewer
                    return
                self._viewer = lifetime.viewer
                viewer = self._viewer
                viewer._metadata = metadata or {}
                viewer.set_model(self.backend.model)
                # Replace scene-specific plots without touching the task controls,
                # panel placement, search input or the browser's selected value.
                for child in tuple(viewer._live_plots_folder._children.values()):
                    child.remove()
                viewer._per_plot_folders.clear()
                self._setup_isaaclab_sidebar(viewer._server)
                viewer.set_visible_worlds(self._resolved_visible_env_ids)
                viewer.set_world_offsets((0.0, 0.0, 0.0))
                self._set_viser_camera_view(self._resolve_initial_camera_pose())
                self._sim_time = 0.0

            def close(self):
                # The parent owns the viewer/server, not the disposable simulation.
                self._viewer = None
                super().close()

        self.visualizer_type = TaskVisualizer

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def render_idle(env, viewer):
    """Keep connected browsers interactive without GPU polling when nobody is viewing."""
    if viewer._server.get_clients():
        env.unwrapped.sim.render()
    time.sleep(0.05)


def run_task_browser(env_cfg, agent_cfg, args, app, checkpoint, evaluator_type):
    catalog = TaskCatalog(args.viser_catalog)
    initial = catalog.find(env_cfg.commands.motion.motion_files)
    panel = TaskPanel(catalog, checkpoint)
    lifetime = ViewerLifetime()
    env = None
    active_scene = None
    policy_nn = None
    next_key, autoplay = initial.key, args.viser_no_wait
    output_root = Path(args.output_dir) if args.output_dir else REPO / "outputs/viser" / Path(checkpoint).stem / "tasks"
    print(
        f"[VISER] Searchable catalog: {len(catalog.tasks)} clips, {len({t.object for t in catalog.tasks})} objects.",
        flush=True,
    )
    try:
        while app.is_running():
            task = catalog.by_key[next_key]
            if env is not None:
                panel.set_status(f"正在加载 **{task.name}**…")
            motion, eye, lookat = catalog.extract(task)
            support = args.task.endswith("Pickup-v0") and task.needs_support
            scene_key = task.object, support
            rebuild_scene = env is None or active_scene != scene_key
            if rebuild_scene:
                if env is not None:
                    env.close()
                    env = None
                    gc.collect()
                cfg = copy.deepcopy(env_cfg)
                cfg.scene.num_envs = 1
                cfg.commands.motion.motion_files = [str(motion)]
                cfg.viewer.eye, cfg.viewer.lookat = eye, lookat
                configure_viser(cfg, args)
                cfg.sim.visualizer_cfgs[0].class_type = lifetime.visualizer_type
                # Only the original requested white-chair pickup has a fitted support.
                task_id = args.task
                if args.task.endswith("Pickup-v0") and not support:
                    task_id = "G1-Inspire-HOI-v0"
                    cfg.scene.support_table = None
                env = RslRlVecEnvWrapper(gym.make(task_id, cfg=cfg), clip_actions=agent_cfg.clip_actions)
                active_scene = scene_key
                if policy_nn is None:
                    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                    runner.load(checkpoint)
                    policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
                    policy_nn.eval()
                    del runner
            else:
                command = env.unwrapped.command_manager.get_term("motion")
                command.motion = MotionLoader([str(motion)], device=env.unwrapped.device)
                command.cfg.motion_files = [str(motion)]
                command.env_clip.zero_()
                command._eval_env_clip.zero_()
                command.time_steps.zero_()
                command._refresh_caches()
                visualizer = env.unwrapped.sim.visualizers[0]
                visualizer._set_viser_camera_view((eye, lookat))
            if panel.server is None:
                panel.attach(lifetime.viewer._server, task)
            else:
                panel.activate(task)
            print(f"[VISER] Task ready: {task.key} · http://localhost:{lifetime.viewer._server.get_port()}", flush=True)
            obs, _ = env.reset()
            policy_nn.reset()
            evaluator = evaluator_type(env, [task.name])
            terminal = TerminalScene(env.unwrapped)
            pending = None

            def disconnected():
                return not args.viser_no_wait and not lifetime.viewer._server.get_clients()

            try:
                while app.is_running() and not autoplay:
                    pending = panel.pop_request()
                    if pending is not None:
                        if pending == task.key:
                            autoplay, pending = True, None
                        break
                    render_idle(env, lifetime.viewer)
                if pending is None and app.is_running():
                    panel.set_running(True)
                    for step in range(evaluator.max_steps):
                        pending = panel.pop_request()
                        if pending is not None or not app.is_running():
                            break

                        if panel.paused or disconnected():
                            while app.is_running() and (panel.paused or disconnected()):
                                pending = panel.pop_request()
                                if pending is not None:
                                    break
                                render_idle(env, lifetime.viewer)
                            if pending is not None or not app.is_running():
                                break
                        started = time.monotonic()
                        # Metrics survive between episodes and are reset in-place.
                        # no_grad keeps them writable when the next task is loaded.
                        with torch.no_grad():
                            evaluator.collect()
                            actions = policy_nn.act_inference(obs)
                            obs, _, dones, extras = env.step(actions)
                            policy_nn.reset(dones)
                            evaluator.update(dones, extras["time_outs"])
                        if step % 10 == 0:
                            panel.set_status(f"正在执行：**{min(100, 100 * (step + 1) / task.frames):.1f}%**")
                        if evaluator.done:
                            break
                        remaining = env.unwrapped.step_dt - (time.monotonic() - started)
                        if args.real_time and remaining > 0:
                            time.sleep(remaining)
                if evaluator.done:
                    result = evaluator.results()
                    result.update(checkpoint=str(checkpoint), motion_files=[str(motion)], reference=task.key)
                    directory = output_root / task.split / task.object / task.name
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "metrics.json").write_text(json.dumps(result, indent=2))
                    clip = result["clips"][0]
                    outcome = "完整成功" if clip["success"] else "失败"
                    panel.set_status(
                        f"**{outcome}** · 完成 {100 * clip['progress']:.1f}% · `{clip['cause']}`  \n"
                        "保留终止姿态；可以重播或直接选择其他任务。"
                    )
                    print(
                        f"[VISER] Finished {task.key}: {outcome}, {clip['progress']:.3f}, {clip['cause']}", flush=True
                    )
            finally:
                terminal.restore()
                panel.set_running(False)
            if args.viser_exit and evaluator.done:
                break
            while app.is_running() and pending is None:
                pending = panel.pop_request()
                if pending is None:
                    render_idle(env, lifetime.viewer)
            if pending is not None:
                next_key, autoplay = pending, True
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            lifetime.close()
