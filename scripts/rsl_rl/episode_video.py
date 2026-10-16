"""Record a single evaluation episode, including its state before automatic reset."""
from pathlib import Path
from contextlib import contextmanager

import imageio.v2 as imageio
import numpy as np


def physics_poses(env):
    """Read directly from PhysX, bypassing the scene's cached state tensors."""
    return (
        env.scene["robot"].root_physx_view.get_link_transforms().cpu().numpy().copy(),
        env.scene["object"].root_physx_view.get_transforms().cpu().numpy().copy(),
    )


@contextmanager
def render_without_physics(env):
    import carb

    before = physics_poses(env)
    settings = carb.settings.get_settings()
    previous = settings.get("/app/player/playSimulations")
    settings.set_bool("/app/player/playSimulations", False)
    try:
        yield
    finally:
        settings.set_bool("/app/player/playSimulations", bool(previous))
    after = physics_poses(env)
    if any(not np.allclose(a, b, rtol=0.0, atol=1e-6) for a, b in zip(before, after)):
        raise RuntimeError("Rendering changed a robot/object physics pose; evaluation video is invalid")


class FirstEpisodeVideo:
    def __init__(self, env, path, fps, frame_source=None):
        if env.num_envs != 1:
            raise ValueError("Episode video requires --num_envs 1 and a single reference clip")
        self.env = env
        self.frame_source = frame_source or capture_rgb
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(self.path), fps=fps, codec="libx264", quality=8, macro_block_size=1,
            ffmpeg_params=["-threads", "2"],
        )
        self.frames = 0
        self.finished = False
        self._original_pre_reset = env.recorder_manager.record_pre_reset
        env.recorder_manager.record_pre_reset = self._before_reset

    def capture(self):
        if self.finished:
            return
        frame = np.asarray(self.frame_source(self.env))
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise RuntimeError(f"Invalid RGB frame: {frame.shape}, {frame.dtype}")
        if not frame.any():
            raise RuntimeError("RTX returned an entirely black frame; refusing to write a misleading video")
        self.writer.append_data(frame)
        self.frames += 1

    def _before_reset(self, env_ids, *args, **kwargs):
        # Isaac Lab resets done environments inside step(). Capture the terminal
        # pose here, so the last video frame is not a jump to a new episode.
        if not self.finished and 0 in env_ids:
            self.capture()
            self.finished = True
        return self._original_pre_reset(env_ids, *args, **kwargs)

    def close(self):
        self.env.recorder_manager.record_pre_reset = self._original_pre_reset
        self.writer.close()


def warm_up_renderer(env):
    """Initialize the headless render graph without advancing physics time."""
    with render_without_physics(env):
        _warm_up_renderer(env)


def _warm_up_renderer(env):
    import carb
    import omni.replicator.core as rep

    carb.settings.get_settings().set_bool("/exts/omni.replicator.core/Orchestrator/enabled", True)
    for _ in range(10):
        frame = env.render(recompute=True)
    if np.asarray(frame).any():
        return
    # Isaac Sim 5.1 can leave the headless annotator graph inactive until the
    # orchestrator initializes a capture. Explicitly initialize at time zero.
    rep.settings.set_render_rtx_realtime(antialiasing="FXAA")
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=2, pause_timeline=False, delta_time=0.0)
        frame = env.render(recompute=True)
    print(f"[VIDEO] Render warmup: shape={np.asarray(frame).shape}, max={np.asarray(frame).max()}", flush=True)
    if not np.asarray(frame).any():
        raise RuntimeError("Headless RTX capture remained black after render-graph initialization")


def capture_rgb(env):
    """Schedule a fresh annotator capture at the current physics pose."""
    import omni.replicator.core as rep

    with render_without_physics(env):
        env.sim.forward()
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False, delta_time=0.0)
        frame = env._rgb_annotator.get_data()[:, :, :3].copy()
    return frame

