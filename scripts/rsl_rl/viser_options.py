# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Browser visualization options shared by policy playback and evaluation."""

import importlib.util
import socket
import threading
import time


def run_and_close(function, app):
    """Keep the original exception visible when Lab 3 closes by raising SystemExit."""
    import inspect
    import traceback

    status = 0
    try:
        function()
    except KeyboardInterrupt:
        status = 130
    except BaseException as error:
        status = error.code if isinstance(error, SystemExit) and isinstance(error.code, int) else 1
        if status:
            traceback.print_exc()
    finally:
        if "exit_code" in inspect.signature(app.close).parameters:
            app.close(exit_code=status)
        else:
            app.close()
    if status:
        raise SystemExit(status)


def add_viser_args(parser):
    group = parser.add_argument_group("Viser browser visualization")
    group.add_argument("--viser", action="store_true", help="Show the live policy in the Isaac Lab 3 Viser viewer.")
    group.add_argument("--viser-port", type=int, default=8080, help="Viser HTTP/WebSocket port (default: 8080).")
    group.add_argument("--viser-host", default="127.0.0.1", help="Bind address (default: localhost, suitable for SSH).")
    group.add_argument("--viser-no-wait", action="store_true", help="Start stepping without waiting for a browser.")
    group.add_argument("--viser-exit", action="store_true", help="Close the browser server when evaluation finishes.")


def prepare_viser_args(args, parser):
    """Select a headless Kit runtime with the standalone Viser visualizer."""
    if not args.viser:
        return
    if importlib.util.find_spec("isaaclab_visualizers") is None:
        parser.error("--viser requires the upgraded environment: source scripts/reproduce/activate_isaac62.sh")
    if not 1 <= args.viser_port <= 65535:
        parser.error("--viser-port must be between 1 and 65535")
    if args.video:
        parser.error("Use --viser and --video in separate runs; the browser viewer does not need RTX cameras")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((args.viser_host, args.viser_port))
        except OSError as error:
            parser.error(f"Viser port {args.viser_host}:{args.viser_port} is unavailable: {error}")
    args.headless = True
    args.visualizer = ["viser"]
    args.real_time = True


def configure_viser(env_cfg, args):
    if not args.viser:
        return
    from isaaclab_visualizers.viser import ViserVisualizerCfg

    env_cfg.sim.visualizer_cfgs = [
        ViserVisualizerCfg(
            port=args.viser_port,
            bind_address=args.viser_host,
            display_address="localhost",
            open_browser=False,
            eye=tuple(env_cfg.viewer.eye),
            lookat=tuple(env_cfg.viewer.lookat),
            max_visible_envs=12,
            randomly_sample_visible_envs=False,
        )
    ]


def _server(env):
    for visualizer in env.sim.visualizers:
        if visualizer.cfg.visualizer_type == "viser":
            return visualizer._viewer._server
    raise RuntimeError("Isaac Lab did not initialize the requested Viser visualizer")


def wait_for_viser(env, args, app):
    if not args.viser or args.viser_no_wait:
        return
    server = _server(env)
    started = threading.Event()
    button = server.gui.add_button("Start policy", color="green")

    @button.on_click
    def begin(_event):
        button.disabled = True
        started.set()

    print(f"[VISER] Open http://localhost:{server.get_port()} and click 'Start policy'.", flush=True)
    while app.is_running() and not (server.get_clients() and started.is_set()):
        # Pump visualization without integrating physics or advancing the reference.
        env.sim.render()
        time.sleep(0.05)


def show_eval_result(env, args, result):
    if not args.viser:
        return
    lines = ["### Evaluation finished", "", "| Clip | Result | Progress | End reason |", "| --- | --- | ---: | --- |"]
    for clip in result["clips"]:
        lines.append(
            f"| {clip['name']} | {'SUCCESS' if clip['success'] else 'FAIL'} | "
            f"{100 * clip['progress']:.1f}% | {clip['cause']} |"
        )
    lines += ["", "The scene shows the terminal state. Restart the command to replay or select another task."]
    _server(env).gui.add_markdown("\n".join(lines))


class TerminalScene:
    """Save each evaluated environment before automatic reset for the final browser view."""

    def __init__(self, env):
        self.env = env
        self.state = None
        self.original = env.recorder_manager.record_pre_reset
        self.finished = set()
        env.recorder_manager.record_pre_reset = self.capture

    def capture(self, env_ids, *args, **kwargs):
        ids = [int(i) for i in env_ids if int(i) not in self.finished]
        if ids:
            fresh = self.env.scene.get_state()
            if self.state is None:
                self.state = fresh
            else:
                for kind, assets in fresh.items():
                    for name, fields in assets.items():
                        for field, tensor in fields.items():
                            self.state[kind][name][field][ids] = tensor[ids]
            self.finished.update(ids)
        return self.original(env_ids, *args, **kwargs)

    def restore(self):
        self.env.recorder_manager.record_pre_reset = self.original
        if self.state is not None:
            self.env.scene.reset_to(self.state)
            self.env.sim.forward()
            self.env.scene.update(0.0)
            self.env.sim.render()


def hold_viser(env, args, app):
    if not args.viser or args.viser_exit:
        return
    print("[VISER] Evaluation finished. The scene remains available; press Ctrl+C to close.", flush=True)
    while app.is_running():
        env.sim.render()
        time.sleep(0.05)
