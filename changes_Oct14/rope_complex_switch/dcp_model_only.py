"""Read only the MODEL weights out of the step-46400 DCP, and cache them.

The converter's ``_load_dcp`` uses ``_EmptyStateDictLoadPlanner()`` with no key
filter, so it materializes optimizer state too: 1110 of the 1230 entries in this
checkpoint's metadata are ``optimizer.*``. Across 6144 shards that is ~90% of
the read thrown away, and it did not finish inside 50 minutes.

``_EmptyStateDictLoadPlanner(keys=...)`` filters at plan time, so only shards
holding the 111 model tensors are touched. The result is cached as a single
torch file so the later forward-pass comparison does not pay the read twice.
"""
import pathlib
import time

import torch

DCP = pathlib.Path(
    "/lus/flare/projects/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/"
    "outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288/step-46400"
)
CACHE = pathlib.Path(
    "/lus/flare/projects/datascience/seonghapark/titanx/_scratch_rope/"
    "step46400_model_only.pt"
)


def load_model_only(verbose: bool = True) -> dict[str, torch.Tensor]:
    """Return the DCP's model tensors, reading from the cache when present."""
    if CACHE.exists():
        if verbose:
            print(f"  cache hit: {CACHE}", flush=True)
        return torch.load(CACHE, map_location="cpu", weights_only=True)

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import (
        _EmptyStateDictLoadPlanner,
    )
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    reader = dcp.FileSystemReader(str(DCP))
    all_keys = reader.read_metadata().state_dict_metadata.keys()
    model_keys = {
        k
        for k in all_keys
        if not k.startswith(("optimizer", "optim", "lr_scheduler",
                             "dataloader", "train_state"))
    }
    if verbose:
        print(f"  {len(model_keys)} model keys of {len(all_keys)} total",
              flush=True)

    sd: dict[str, torch.Tensor] = {}
    t0 = time.time()
    _load_state_dict(
        sd,
        storage_reader=reader,
        planner=_EmptyStateDictLoadPlanner(keys=model_keys),
        no_dist=True,
    )
    sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    if verbose:
        print(f"  read {len(sd)} tensors in {time.time() - t0:.1f}s", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".pt.tmp")
    torch.save(sd, tmp)
    tmp.rename(CACHE)
    if verbose:
        print(f"  cached to {CACHE}", flush=True)
    return sd


if __name__ == "__main__":
    sd = load_model_only()
    print(f"\n{len(sd)} tensors")
    for k in sorted(sd)[:4]:
        print(f"  {k:52s} {tuple(sd[k].shape)} {sd[k].dtype}")
