"""Isolated reproducer hooks; no synchronization or tensor-value reads."""

import ctypes
import inspect
import json
import os
from collections import Counter
from functools import wraps
from math import prod
from pathlib import Path
import runpy
import socket
import sys
import threading
import time


def tensor_metadata(tensor):
    storage = tensor.untyped_storage()
    return dict(
        address=hex(tensor.data_ptr()),
        storage_address=hex(storage.data_ptr()),
        storage_offset=tensor.storage_offset(),
        storage_bytes=storage.nbytes(),
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        numel=tensor.numel(),
        bytes=tensor.numel() * tensor.element_size(),
    )


def main():
    import torch
    import torch.distributed as dist
    from torch.distributed import _functional_collectives as funcol
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives as fc
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg
    from torchtitan.components.loss import _LossParallelCrossEntropy

    root = Path(os.environ["RS_TRACE_DIR"])
    rank = next(int(os.environ[k]) for k in
                ("PALS_RANKID", "PMI_RANK", "PMIX_RANK", "RANK") if k in os.environ)
    native = ctypes.CDLL(str(root / "rs_event_trace.so"))
    native.rs_trace_start.argtypes = [ctypes.c_char_p]
    native.rs_trace_start.restype = ctypes.c_uint
    native.rs_trace_mark.argtypes = [ctypes.c_uint, ctypes.c_uint]
    native.rs_trace_mark.restype = None
    native.rs_trace_dtensor.argtypes = [ctypes.c_uint, ctypes.c_uint]
    native.rs_trace_dtensor.restype = None
    native.rs_trace_begin_detail.argtypes = []
    native.rs_trace_begin_detail.restype = None
    native.rs_trace_ce_enter.argtypes = []
    native.rs_trace_ce_enter.restype = None
    rc = native.rs_trace_start(os.fsencode(root / f"ze_{rank:05d}.tsv"))
    if rc:
        raise RuntimeError(f"rank {rank}: Level Zero tracing setup failed: 0x{rc:x}")

    state = threading.local()
    counts = Counter()
    log = (root / f"rs_{rank:05d}.jsonl").open("x", buffering=1)
    def emit(event, **fields):
        print(json.dumps(dict(event=event, rank=rank, time_ns=time.time_ns(),
                              tid=threading.get_native_id(), **fields)), file=log)

    emit("initialized", pid=os.getpid(), host=socket.gethostname(),
         torch_version=torch.__version__, device_mask=os.environ.get("ZE_AFFINITY_MASK"))
    original_foreach = fc.foreach_reduce
    signature = inspect.signature(original_foreach)
    original_pack = fc.foreach_reduce_scatter_copy_in
    original_rs = dist.reduce_scatter_single
    loss_forward = _LossParallelCrossEntropy.forward
    detail_started = False
    ce_entered = False
    dtensor_sequence = 0
    dual_path = os.environ.get("RS_OWNER_TRACE") == "1"

    def begin_detail(trigger):
        nonlocal detail_started
        if not detail_started:
            emit("detail_begin", trigger=trigger)
            native.rs_trace_begin_detail()
            detail_started = True

    @wraps(loss_forward)
    def trace_loss_forward(*args, **kwargs):
        nonlocal ce_entered
        if not ce_entered:
            native.rs_trace_ce_enter()
            ce_entered = True
        if not dual_path:
            begin_detail("vocab_parallel_cross_entropy")
        return loss_forward(*args, **kwargs)

    functional_rs = funcol.reduce_scatter_single

    @wraps(functional_rs)
    def dtensor_reduce(*args, **kwargs):
        nonlocal dtensor_sequence, detail_started
        caller = sys._getframe(1).f_globals.get("__name__")
        if caller != "torch.distributed.tensor.placement_types":
            return functional_rs(*args, **kwargs)
        if not dual_path and not detail_started:
            emit("detail_begin", trigger="dtensor_reduce_scatter")
            detail_started = True
        dtensor_sequence += 1
        sequence = dtensor_sequence
        field = dict(collective_sequence=sequence, pre_ce=not ce_entered)
        emit("dtensor_before", **field)
        native.rs_trace_dtensor(sequence, 0)
        try:
            result = functional_rs(*args, **kwargs)
        except BaseException as exc:
            native.rs_trace_dtensor(sequence, 2)
            emit("dtensor_error", error=repr(exc), **field)
            raise
        native.rs_trace_dtensor(sequence, 1)
        emit("dtensor_returned", **field)
        return result

    @wraps(original_foreach)
    def foreach(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        state.params = bound["fsdp_params"]
        state.packing = None
        try:
            return original_foreach(*args, **kwargs)
        finally:
            state.params = None
            state.packing = None

    @wraps(original_pack)
    def pack(grads, output, world_size):
        params = state.params
        if len(params) != len(grads):
            raise RuntimeError("FSDP trace: parameter / gradient packing count mismatch")
        chunks = []
        offset = 0
        for param, grad in zip(params, grads):
            padded = ((grad.size(0) + world_size - 1) // world_size) * world_size
            per_shard = (padded // world_size) * prod(grad.shape[1:])
            chunks.append(dict(name=param._param_fqn, placement_dim=param.fsdp_placement.dim,
                               sharded_size=list(param.sharded_size),
                               sharded_stride=list(param.contiguous_sharded_stride),
                               packed_offset_elements=offset, packed_elements_per_shard=per_shard,
                               gradient=tensor_metadata(grad)))
            offset += per_shard
        state.packing = dict(segments=chunks, packed_elements_per_rank=offset)
        return original_pack(grads, output, world_size)

    @wraps(original_rs)
    def reduce(output, input, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if getattr(state, "params", None) is None:
            return original_rs(output, input, op, group, async_op)
        if not detail_started and not dual_path:
            raise RuntimeError("FSDP trace: expected loss hook did not start native detail")
        counts[group] += 1
        sequence = counts[group]
        emit("before", members=dist.get_process_group_ranks(group),
             group_name=group.group_name, group_sequence=sequence,
             op=str(op), async_op=async_op, stream=str(torch.xpu.current_stream(input.device)),
             input=tensor_metadata(input), output=tensor_metadata(output), packing=state.packing)
        native.rs_trace_mark(sequence, 0)
        try:
            result = original_rs(output, input, op, group, async_op)
        except BaseException as exc:
            native.rs_trace_mark(sequence, 2)
            emit("error", group_name=group.group_name, group_sequence=sequence, error=repr(exc))
            raise
        native.rs_trace_mark(sequence, 1)
        emit("returned", group_name=group.group_name, group_sequence=sequence)
        return result

    if pg.foreach_reduce is not original_foreach:
        raise RuntimeError("FSDP trace: installed foreach_reduce call site has changed")
    fc.foreach_reduce_scatter_copy_in = pack
    pg.foreach_reduce = foreach
    dist.reduce_scatter_single = reduce
    if dual_path:
        funcol.reduce_scatter_single = dtensor_reduce
    _LossParallelCrossEntropy.forward = staticmethod(trace_loss_forward)

    entry = Path(os.environ["RS_TRAIN_ENTRY"])
    sys.path.insert(0, str(entry.parent))
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
