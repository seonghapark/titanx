import json
import os
import time
import traceback

import gdb


rank = int(os.environ["RS_OWNER_GDB_RANK"])
log = open(os.path.join(os.environ["RS_TRACE_DIR"], f"owner_{rank:05d}.jsonl"), "x", buffering=1)
inferior = lambda: gdb.selected_inferior()
probes = []
window_probes = []
installed = False
generation = {}
destroy = {}
lists = {}
wrappers = {}
impls = {}
active_enqueue = {}
collectives = {}
ipc_exchanges = {}
store_id = 0
window_epoch = 0
current_window = 0
stack_states = {}
conversion_id = 0


def thread_id():
    return gdb.selected_thread().ptid[1]


def emit(kind, **fields):
    thread = gdb.selected_thread()
    tid = thread.ptid[1] if thread else None
    log.write(json.dumps(dict(kind=kind, rank_id=rank, ns=int(time.time() * 1_000_000_000),
                              tid=tid, collective=collectives.get(tid),
                              ipc_exchange=ipc_exchanges.get(tid),
                              **fields)) + "\n")


def reg(name):
    return int(gdb.parse_and_eval("$" + name))


def words(address, count):
    if not count:
        return []
    data = bytes(inferior().read_memory(address, 8 * count))
    return [int.from_bytes(data[i:i + 8], "little") for i in range(0, len(data), 8)]


def native(handle, record=None):
    if record is None:
        record = generation.get(handle, {})
    return dict(event_handle=handle, event_generation=record.get("event_generation"),
                ze_pool_handle=record.get("ze_pool_handle"), alive=record.get("alive"),
                oneccl_pool_id=record.get("oneccl_pool_id"))


def vector(addr, width=1):
    begin, end, _ = words(addr, 3)
    if end < begin or (end - begin) % (8 * width):
        raise ValueError(f"invalid vector bounds: {begin:#x} .. {end:#x}")
    count = (end - begin) // (8 * width)
    return begin, count, words(begin, min(64, count) * width)


def observe_list(path, owner, addr, source, copied=None):
    global store_id
    width = 1 if path == "oneCCL" else 2
    storage, count, values = vector(addr, width)
    key = path, owner, addr
    old = lists.get(key, {})
    current = {}
    storage_stack = None
    for slot in range(min(64, count)):
        identity = tuple(values[slot * width:(slot + 1) * width])
        previous = old.get(slot)
        if previous is not None and tuple(previous["identity"]) == identity:
            current[slot] = dict(previous, slot_address=storage + slot * 8 * width)
            continue
        if previous is not None:
            emit("dep_not_in_slot", **previous,
                 stack=sample_stack(("slot_replace", path, source)))
        impl = impls.get(identity[0]) if width == 2 else None
        at_observation = native(identity[0]) if width == 1 else None
        reference = impl["reference"] if impl and source == "after_depends_on" else None
        origin = "event_impl_binding" if reference else None
        source_store_id = None
        if width == 1:
            if source == "entry_constructor_return":
                prior = copied.get(slot) if copied else None
                if prior and prior["identity"] == identity and prior["reference"]:
                    reference = prior["reference"]
                    origin = "source_slot"
                    source_store_id = prior["store_id"]
                else:
                    reference = at_observation
                    origin = "native_at_copy"
        store_id += 1
        current[slot] = dict(path=path, owner_id=owner, list_id=addr, slot=slot,
                             slot_address=storage + slot * 8 * width,
                             identity=identity, reference=reference, store_id=store_id,
                             stored_generation=reference["event_generation"] if reference else None,
                             observation=source, generation_origin=origin,
                             native_at_observation=at_observation, source_store_id=source_store_id)
        if current_window == 2 and storage_stack is None:
            storage_stack = sample_stack(("slot_add", path, source, origin),
                                         reference is not None)
        emit("dep_stored", **current[slot], stack=storage_stack)
    for slot in old.keys() - current.keys():
        emit("dep_not_in_slot", **old[slot],
             stack=sample_stack(("slot_remove", path, source)))
    lists[key] = current
    return dict(list_id=addr, storage=storage, count=count, slots=list(current.values()))


def list_owner_end(path, owner):
    for key in list(lists):
        if key[:2] == (path, owner):
            records = lists.pop(key)
            emit("dep_owner_end", path=path, owner_id=owner, list_id=key[2],
                 last_observed=list(records.values()),
                 stack=sample_stack(("owner_end", path),
                                    any(dep["stored_generation"] is not None
                                        for dep in records.values())))


def dep_candidates(handle):
    return [dep for slots in lists.values() for dep in slots.values()
            if (dep["path"] == "oneCCL" and dep["identity"][0] == handle)
            or (dep["reference"] and dep["reference"]["event_handle"] == handle)]


def check_slot(dep):
    width = 1 if dep["path"] == "oneCCL" else 2
    try:
        storage, count, values = vector(dep["list_id"], width)
    except (gdb.error, ValueError) as exc:
        return "unreadable", None, str(exc)
    slot = dep["slot"]
    if slot >= count:
        return "removed", None, None
    actual = tuple(values[slot * width:(slot + 1) * width])
    return ("unchanged" if actual == dep["identity"] else "replaced",
            storage + slot * 8 * width, None)


def destroy_slot_states(handle, before):
    seen = {dep["store_id"]: dep for dep in before}
    seen.update((dep["store_id"], dep) for dep in dep_candidates(handle))
    result = []
    for dep in seen.values():
        key = dep["path"], dep["owner_id"], dep["list_id"]
        if key in lists:
            state, address, error = check_slot(dep)
        else:
            state, address, error = "owner_no_longer_tracked", None, None
        result.append(dict(dep, state=state, current_slot_address=address, read_error=error))
    return result


def correlate_wait(handle, index, address, enqueue):
    current = native(handle)
    candidates = dep_candidates(handle)
    exact = []
    for dep in candidates:
        if dep["path"] == "oneCCL":
            state, slot_address, error = check_slot(dep)
            if error is not None:
                emit("dep_slot_unreadable", store_id=dep["store_id"], error=error)
            if state == "unchanged" and slot_address == address:
                exact.append(dict(dep, slot_address=slot_address))
    ur_candidates = [slot for slot in enqueue["slots"] if slot["reference"]
                     and slot["reference"]["event_handle"] == handle] if enqueue else []
    conversion = enqueue.get("conversion") if enqueue else None
    converted = None
    if conversion and index < len(conversion["slots"]):
        slot = conversion["slots"][index]
        if slot["slot_address"] == address and slot["event_handle"] == handle:
            converted = slot
    sources = set(dep["stored_generation"] for dep in exact
                  if dep["stored_generation"] is not None)
    if converted and converted["stored_generation"] is not None:
        sources.add(converted["stored_generation"])
    stored = next(iter(sources)) if len(sources) == 1 else None
    appended = current["event_generation"]
    comparison = ("unknown" if stored is None or appended is None
                  else "same_generation" if stored == appended else "different_generation")
    if comparison == "different_generation":
        outcome = "address_reused"
    elif comparison == "same_generation" and current["alive"] is not None:
        outcome = "same_live_generation" if current["alive"] else "destroyed_same_generation"
    else:
        outcome = "unknown_provenance"
    return dict(**current, slot=index, slot_address=address,
                append_generation=appended, current_generation=appended,
                stored_generation=stored, conversion_slot=converted,
                stored_generation_conflict=len(sources) > 1,
                comparison=comparison, outcome=outcome,
                exact_storage=exact, address_candidates=candidates,
                enclosing_ur_address_candidates=ur_candidates)


def seed_native_generations():
    path = os.path.join(os.environ["RS_TRACE_DIR"], f"ze_{rank:05d}.tsv")
    pending = {}
    with open(path) as stream:
        for row in stream:
            cols = row.rstrip("\n").split("\t")
            api, phase, rc = cols[3:6]
            if api == "LIFECYCLE_LIMIT":
                generation.clear()
                emit("generation_seed_incomplete", reason=api)
                return
            if api == "EventCreate" and rc == "0x0":
                generation[int(cols[7], 16)] = dict(
                    event_generation=int(cols[11]), ze_pool_handle=int(cols[6], 16), alive=True)
            elif api == "EventDestroy":
                call = int(cols[11])
                if phase == "enter":
                    handle = int(cols[6], 16)
                    pending[call] = (handle, generation.get(handle), ())
                else:
                    _, old, _ = pending.pop(call, (None, None, ()))
                    if old is not None and rc == "0x0":
                        old["alive"] = False
    destroy.update(pending)


def set_window(window):
    global installed, current_window
    current_window = window
    if window:
        seed_native_generations()
        if not installed:
            install()
            installed = True
        for probe in window_probes:
            probe.enabled = True
    else:
        for probe in window_probes:
            probe.enabled = False
        # Active finish probes may still return; their old window must not mutate the new one.
        for state in (generation, destroy, lists, wrappers, impls, active_enqueue,
                      collectives, ipc_exchanges):
            state.clear()
    emit("owner_window", window=window, seeded_generations=len(generation))


class Probe(gdb.Breakpoint):
    def __init__(self, spec, kind, callback):
        super().__init__(spec, internal=True)
        self.kind, self.callback = kind, callback

    def stop(self):
        try:
            self.callback()
        except (gdb.error, TypeError, ValueError, OSError) as exc:
            log_probe_error(self.kind, "entry", exc)
        return False


class ReturnProbe(gdb.FinishBreakpoint):
    def __init__(self, kind, callback, **context):
        super().__init__(gdb.newest_frame(), internal=True)
        self.kind, self.callback = kind, callback
        self.context = context
        self.epoch = window_epoch

    def stop(self):
        if self.epoch != window_epoch:
            return False
        returned = None
        try:
            returned = reg("rax")
            self.callback(returned)
        except (gdb.error, TypeError, ValueError, OSError) as exc:
            log_probe_error(self.kind, "return", exc, returned=returned, **self.context)
        return False

    def out_of_scope(self):
        if self.epoch == window_epoch:
            emit("return_not_seen", probe=self.kind, **self.context)


def log_probe_error(probe, phase, exc, **context):
    emit("probe_error", probe=probe, phase=phase, error_type=type(exc).__name__,
         error=str(exc), traceback=traceback.format_exc(), context=context,
         tracked_generations=len(generation), pending_destroys=len(destroy))


def frame_stack():
    return gdb.execute("bt 18", to_string=True)


def sample_stack(kind, state=None):
    collective = collectives.get(thread_id()) or {}
    key = (window_epoch, collective.get("path"), collective.get("sequence"), kind)
    if key in stack_states and stack_states[key] == state:
        return None
    stack = frame_stack()
    stack_states[key] = state
    return stack


def ur_import():
    handle, props, result = reg("rdi"), reg("rdx"), reg("rcx")
    event = native(handle)
    # Keep the properties address and the standard base header/top-level ownership flag.
    prop_bytes = bytes(inferior().read_memory(props, 17)).hex() if props else None
    def finish(rc):
        wrapper = words(result, 1)[0] if rc == 0 else None
        if wrapper:
            wrappers[wrapper] = event
        emit("ur_import", rc=rc, wrapper=wrapper, properties=prop_bytes, **event)

    ReturnProbe("ur_import", finish)


def ur_get():
    wrapper, output = reg("rdi"), reg("rsi")
    ReturnProbe("ur_get", lambda rc: emit(
        "ur_get", wrapper=wrapper, rc=rc,
        **(native(words(output, 1)[0]) if rc == 0 else {})))


def ur_enqueue():
    thread = thread_id()
    if thread not in collectives:
        return
    queue, kernel = reg("rdi"), reg("rsi")
    _, _, count, dep, _ = words(reg("rsp"), 5)
    count &= 0xffffffff
    args = words(dep, min(64, count))
    slots = []
    for i, wrapper in enumerate(args):
        reference = wrappers.get(wrapper)
        source_slots = [slot for key, observed in lists.items() if key[0] == "SYCL"
                        for slot in observed.values()
                        if impls.get(slot["identity"][0], {}).get("wrapper") == wrapper
                        and slot["reference"] is not None and slot["reference"] is reference]
        slots.append(dict(slot=i, wrapper=wrapper, reference=reference,
                          sycl_source_candidates=source_slots))
    record = dict(queue=queue, kernel=kernel, list_id=dep, count=count, slots=slots)
    emit("ur_enqueue", **record)
    previous = active_enqueue.get(thread)
    active_enqueue[thread] = record

    def finish(rc):
        if previous is None:
            active_enqueue.pop(thread, None)
        else:
            active_enqueue[thread] = previous
        emit("ur_enqueue_return", rc=rc, list_id=dep)

    ReturnProbe("ur_enqueue", finish, list_id=dep)


def ur_convert(return_pc):
    global conversion_id
    enqueue = active_enqueue.get(thread_id())
    if enqueue is None or words(reg("rsp"), 1)[0] != return_pc:
        return
    owner, count, inputs, queue = (reg(r) for r in ("rdi", "esi", "rdx", "rcx"))
    count &= 0xffffffff
    if (inputs, count, queue) != (enqueue["list_id"], enqueue["count"], enqueue["queue"]):
        return
    conversion_id += 1
    cid = conversion_id
    enqueue.pop("conversion", None)
    input_slots = []
    for slot in enqueue["slots"]:
        wrapper = slot["wrapper"]
        try:
            event = native(words(wrapper + 0x48, 1)[0]) if wrapper else None
        except gdb.error as exc:
            emit("ur_wait_input_unreadable", conversion_id=cid, wrapper=wrapper, error=str(exc))
            event = None
        input_slots.append(dict(slot, native_before_conversion=event,
                                stored_generation=slot["reference"]["event_generation"]
                                if slot["reference"] else None))
    emit("ur_wait_convert_enter", conversion_id=cid, list_id=owner,
         input_list_id=inputs, input_slots=input_slots)

    def finish(rc):
        slots = []
        if (rc & 0xffffffff) == 0:
            ze_array, ur_array, length = words(owner, 3)
            count_out = length & 0xffffffff
            for index, (handle, wrapper) in enumerate(zip(
                    words(ze_array, min(64, count_out)), words(ur_array, min(64, count_out)))):
                inputs_for_wrapper = [slot for slot in input_slots
                                      if wrapper and slot["wrapper"] == wrapper]
                reference = inputs_for_wrapper[0]["reference"] if len(inputs_for_wrapper) == 1 else None
                current = native(handle)
                slots.append(dict(conversion_id=cid, list_id=owner, slot=index,
                                  slot_address=ze_array + 8 * index, event_handle=handle,
                                  wrapper=wrapper, input_slots=inputs_for_wrapper,
                                  stored_generation=reference["event_generation"]
                                  if reference and reference["event_handle"] == handle else None,
                                  current_generation=current["event_generation"], alive=current["alive"]))
            enqueue["conversion"] = dict(conversion_id=cid, list_id=owner,
                                         native_array=ze_array, count=count_out, slots=slots)
        emit("ur_wait_convert_return", conversion_id=cid, list_id=owner,
             rc=rc & 0xffffffff, slots=slots)

    ReturnProbe("ur_wait_convert", finish, conversion_id=cid)


def ipc_exchange():
    tid = thread_id()
    exchange = dict(comm=reg("rdi"), stream=reg("rsi"), observed_ns=int(time.time() * 1e9))
    previous = ipc_exchanges.get(tid)
    ipc_exchanges[tid] = exchange
    emit("ipc_exchange_enter", stack=frame_stack())

    def finish(_):
        emit("ipc_exchange_return")
        if previous is None:
            ipc_exchanges.pop(tid, None)
        else:
            ipc_exchanges[tid] = previous

    ReturnProbe("ipc_exchange", finish)


def sycl_impl_create():
    impl, wrapper = reg("rdi"), reg("rsi")
    impls[impl] = dict(wrapper=wrapper, reference=wrappers.get(wrapper))
    emit("sycl_impl_create", impl=impl, **impls[impl])


def sycl_impl_end():
    impl = reg("rdi")
    emit("sycl_impl_end", impl=impl, last_binding=impls.pop(impl, None))


def sycl_dep():
    handler = reg("rdi")
    handler_impl = words(handler + 8, 1)[0]
    addr = handler_impl + 0x1f8
    observe_list("SYCL", handler, addr, "before_depends_on")
    ReturnProbe("sycl_dep", lambda _: observe_list("SYCL", handler, addr, "after_depends_on"),
                owner_id=handler, list_id=addr)


def sycl_native():
    impl = reg("rdi")
    ReturnProbe("sycl_native", lambda handle: emit("sycl_native", impl=impl, **native(handle)))


def ccl_pool_get():
    pool = reg("rdi")

    def record_get(handle):
        if handle in generation:
            generation[handle]["oneccl_pool_id"] = pool
        emit("pool_get", **dict(native(handle), oneccl_pool_id=pool))

    ReturnProbe("pool_get", record_get, oneccl_pool_id=pool)


def ccl_pool_put():
    pool, handle = reg("rdi"), reg("rsi")
    emit("pool_put", **dict(native(handle), oneccl_pool_id=pool))


def ccl_entry_create():
    entry, schedule, source = (reg(r) for r in ("rdi", "rsi", "rdx"))
    emit("entry_create", entry=entry, schedule=schedule, source_list_id=source)
    known_sources = [slots for (path, _, addr), slots in lists.items()
                     if path == "oneCCL" and addr == source]
    copied = known_sources[0] if len(known_sources) == 1 else None
    # Build-specific vector member, confirmed by both list accessor disassemblies.
    ReturnProbe("entry_create", lambda _: observe_list("oneCCL", entry, entry + 0x100,
                                                      "entry_constructor_return", copied),
                owner_id=entry)


def install():
    maps = {}
    with open(f"/proc/{inferior().pid}/maps") as stream:
        for line in stream:
            fields = line.rstrip().split(maxsplit=5)
            if len(fields) == 6 and fields[2] == "00000000" and fields[5].startswith("/"):
                maps[os.path.basename(fields[5])] = (int(fields[0].split("-")[0], 16), fields[5])

    def add(lib, offset, symbol, kind, callback):
        if lib not in maps:
            emit("probe_unavailable", probe=kind, library=lib)
            return
        base, path = maps[lib]
        addr = base + offset
        actual = gdb.execute(f"info symbol {addr:#x}", to_string=True).strip()
        if not actual.startswith((symbol + "(", symbol + " in section ")):
            emit("probe_unavailable", probe=kind, library=path, expected=symbol, actual=actual)
            return
        probe = Probe(f"*{addr:#x}", kind, callback)
        probes.append(probe)
        if kind != "mpi_abort":
            window_probes.append(probe)
        emit("probe_ready", probe=kind, library=path, address=addr, symbol=actual)

    ccl = "libccl.so.1.0"
    add(ccl, 0x2d8c540, "do_ipc_exchange", "ipc_exchange", ipc_exchange)
    mpi = "libmpi.so.12.5.0" if "libmpi.so.12.5.0" in maps else "libmpi.so.12"
    add(mpi, 0x25f7720, "MPIR_Abort", "mpi_abort",
        lambda: emit("mpi_abort", stack=frame_stack()))
    add(ccl, 0x2718590, "ccl::host_event_impl::~host_event_impl", "owner_end",
        lambda: emit("owner_end", owner=reg("rdi")))
    add(ccl, 0x27c5f00, "ccl::recycle_storage::store_events", "recycle_store",
        lambda: emit("recycle_store", storage=reg("rdi"), oneccl_pool_id=reg("rsi"),
                     events=[words(reg(r), 2) for r in ("rdx", "rcx")]))
    add(ccl, 0x291bce0, "ccl::ze::dynamic_event_pool::get_event", "pool_get", ccl_pool_get)
    add(ccl, 0x291d1c0, "ccl::ze::dynamic_event_pool::put_event", "pool_put", ccl_pool_put)
    add(ccl, 0x28a7a80, "ze_base_entry::ze_base_entry", "entry_create",
        ccl_entry_create)
    for offset, fn in [(0x29340f0, "get_comp_list"), (0x29341a0, "get_copy_list")]:
        add(ccl, offset, f"ccl::ze::list_manager::{fn}", fn,
            lambda name=fn: emit(name, manager=reg("rdi"), entry=reg("rsi"),
                                 **observe_list("oneCCL", reg("rsi"), reg("rdx"), name)))
    add(ccl, 0x28a8140, "ze_base_entry::~ze_base_entry", "entry_end",
        lambda: list_owner_end("oneCCL", reg("rdi")))

    sycl = "libsycl.so.9" if "libsycl.so.9" in maps else "libsycl.so.9.0.0"
    add(sycl, 0x331c20, "sycl::_V1::handler::depends_on", "sycl_dep", sycl_dep)
    add(sycl, 0x329dd0, "sycl::_V1::handler::~handler", "sycl_handler_end",
        lambda: list_owner_end("SYCL", reg("rdi")))
    add(sycl, 0x20e310, "sycl::_V1::detail::event_impl::~event_impl", "sycl_impl_end", sycl_impl_end)
    add(sycl, 0x20ecd0, "sycl::_V1::detail::event_impl::event_impl", "sycl_impl_create", sycl_impl_create)
    add(sycl, 0x2105c0, "sycl::_V1::detail::event_impl::getNative", "sycl_native", sycl_native)

    for stem, offsets in (
        ("libur_adapter_level_zero.so.0", (0xde6d0, 0xddc60, 0xd7b50, 0xdcf10, 0x12d5c0)),
        ("libur_adapter_level_zero_v2.so.0", (0x101170, 0x100fe0, 0xffa00, 0xffa10, 0x120020)),
    ):
        lib = stem if stem in maps else stem + ".12.0"
        if lib not in maps:
            continue
        for offset, fn, hook in zip(offsets,
            ("EventCreateWithNativeHandle", "EventGetNativeHandle", "EventRetain", "EventRelease",
             "EnqueueKernelLaunch"),
            (ur_import, ur_get, lambda: emit("ur_retain", wrapper=reg("rdi")),
             lambda: emit("ur_release", wrapper=reg("rdi")), ur_enqueue)):
            add(lib, offset, f"ur::level_zero::ur{fn}", fn, hook)
        if stem == "libur_adapter_level_zero.so.0":
            base, path = maps[lib]
            layout = ((0x12e3d8, "4c8db424a00000004c89f7"),
                      (0x12e3fa, "e89155faff"),
                      (0xd46cc, "488b4848488b7c2440488b178b74243448890cf2488b4f08488904f1"))
            if all(bytes(inferior().read_memory(base + offset, len(bytes.fromhex(code))))
                   == bytes.fromhex(code) for offset, code in layout):
                add(lib, 0xd3990, "ur_ze_event_list_t::createAndRetainUrZeEventList",
                    "ur_wait_convert", lambda pc=base + 0x12e3ff: ur_convert(pc))
            else:
                emit("probe_unavailable", probe="ur_wait_convert", library=path,
                     reason="unrecognized adapter layout")
    if not any(lib.startswith("libur_adapter_level_zero") for lib in maps):
        emit("probe_unavailable", probe="ur", library="Level Zero UR adapter")


def site():
    global window_epoch
    step, call, rc, obj, a, wait = (reg(r) for r in ("rdi", "rsi", "rdx", "rcx", "r8", "r9"))
    if step == 0:
        window_epoch += 1
        stack_states.clear()
        set_window(call)
    elif step == 1:
        if rc == 0:
            generation[a] = dict(event_generation=call, ze_pool_handle=obj, alive=True)
        emit("ze_create", api_call_id=call, rc=rc,
             **dict(native(a), ze_pool_handle=obj), stack=sample_stack("create", rc))
    elif step == 2:
        dependents = dep_candidates(obj) if current_window == 2 else ()
        destroy[call] = (obj, generation.get(obj), dependents)
        emit("ze_destroy_enter", api_call_id=call, **native(obj),
             dependent_slots=dependents, stack=sample_stack("destroy", bool(dependents)))
    elif step == 3:
        handle, old, dependents = destroy.pop(call, (None, None, ()))
        if rc == 0 and old is not None:
            old["alive"] = False
        emit("ze_destroy_return", api_call_id=call, rc=rc, **native(handle, old or {}),
             last_observed_dependents=dep_candidates(handle),
             dependent_slots_after=destroy_slot_states(handle, dependents)
             if current_window == 2 and handle is not None else [])
    elif step == 4:
        handles = words(wait, min(64, rc))
        enqueue = active_enqueue.get(thread_id())
        slots = [correlate_wait(h, i, wait + 8 * i, enqueue) for i, h in enumerate(handles)]
        wait_state = (rc, tuple((slot["outcome"], slot["alive"]) for slot in slots))
        emit("ze_collective_launch", api_call_id=call, append_sequence=call,
             command_list=obj, signal=native(a), wait_list=wait,
             slots=slots,
             wait1_address_candidates=slots[1]["address_candidates"] if len(slots) > 1 else [],
             wait1_same_storage=slots[1]["exact_storage"] if len(slots) > 1 else [],
             enclosing_ur_enqueue=enqueue,
             stack=sample_stack("append", wait_state))
    elif step == 5:
        emit("ze_collective_return", api_call_id=call, rc=rc)
    elif step in (6, 7):
        tid = thread_id()
        if rc == 0:
            collectives[tid] = dict(path="FSDP" if step == 6 else "DTENSOR", sequence=call)
        emit("collective_marker", phase=rc)
        if rc != 0:
            collectives.pop(tid, None)


def stopped(event):
    if isinstance(event, gdb.SignalEvent):
        emit("signal", signal=event.stop_signal, stack=frame_stack())


gdb.events.stop.connect(stopped)
probes.append(Probe("rs_owner_site", "native", site))
emit("started")
