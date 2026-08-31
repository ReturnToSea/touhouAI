# 7 · The environment

A Gymnasium wrapper around one hooked `th07.exe`: construct, auto-navigate into
Stage 1, freeze the snapshot, then `reset()` / `step()` drive episodes.

!!! note "Draft"
    **To write.** The cross-process build lock (concurrent D3D init races). The
    per-session audio mute and the launch-audio saga (endpoint mute vs. the
    4-second title dwell). Faithful reset via state snapshot. `hard_reset=True`
    and why the survey needs it. `dll_obs` vs. the Python obs path.
