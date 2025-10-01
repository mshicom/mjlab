# Multi-GPU Integration Plan

This plan describes the steps required to integrate robust multi-GPU training support into the `mjlab` training entrypoints.

## Objectives

- Detect when the training script is launched under `torchrun` and configure devices based on the distributed environment variables.
- Ensure that logging artifacts (directories, YAML dumps, and optional media) are created exactly once by the main process while the other ranks reuse the generated paths.
- Prevent non-zero ranks from performing side effects such as Weights & Biases artifact downloads or video recording that should only be executed by the main process.
- Keep the existing single-GPU/CPU behaviour unchanged when not launched under a distributed runner.

## Implementation Steps

1. **Distributed utilities**
   - Create a helper module (e.g., `mjlab.utils.distributed`) that inspects `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` to expose a lightweight `DistributedContext` dataclass.
   - Provide utility functions to (a) resolve the correct CUDA device when multi-GPU training is requested, (b) query whether the current process is the main rank, and (c) perform safe synchronization helpers (e.g., polling for files when `torch.distributed` is not yet initialised).

2. **Training script updates**
   - Use the distributed utilities to select the correct device (`cuda:{LOCAL_RANK}`) before constructing the environment or runner when running with multiple GPUs.
   - Derive a shared logging directory name on the main rank, persist it to disk (e.g., inside `logs/rsl_rl/<experiment>/.latest_run`), and let the other ranks poll for this file so every process uses the same folder without racing on directory creation.
   - Restrict operations with side effects (Weights & Biases artifact download, YAML dumps, directory creation, and optional video recording) to the main process while the remaining ranks wait for the resources to appear.

3. **Runner compatibility checks**
   - Verify that existing `VelocityOnPolicyRunner` and `MotionTrackingOnPolicyRunner` subclasses remain compatible with the distributed runner; guard any custom logging/export logic so it only executes on the main rank if required by the updated script logic.

4. **Testing**
   - Validate the behaviour by launching the provided command `uv run torchrun --nproc_per_node=3 src/mjlab/scripts/train.py ...` to ensure distributed initialisation succeeds and that non-main processes reuse the generated logging directory.

