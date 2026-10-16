# SIGSEGV re-fault loop in SYCL device-image finalization at process exit (48/48 ranks, Aurora / PVC)

**Status:** failure site confirmed by live stacks; **ownership and root cause deliberately left open** pending input from Intel.
**Reporter:** ALCF AuroraGPT / datascience, Aurora.
**Date observed:** 2026-10-14. **Reference job:** PBS 8929331 (also 8924552, 8924583, 8924940, 8928962).

---

## 1. Summary

At normal process exit, after a distributed PyTorch training run has completed
all work and written its checkpoint, every rank enters a **SIGSEGV that repeats
in a loop rather than terminating the process**. The faulting thread is the main
thread, inside `free()` called from the SYCL runtime's device-binary-image
teardown, reached from a `libtorch-xpu-ops` kernel library's `descriptor_unreg`
during `_dl_fini`.

The loop consumes increasing CPU (up to ~9.7 cores per rank) and does not end on
its own. It is terminated only by an external timeout. Depending on whether the
job's timeout fires first, the same defect presents either as **exit 139
(SIGSEGV)** or as an apparent **multi-minute "hang"** at shutdown. We previously
tracked these as two separate problems; live stacks show they are one.

**Impact.** Training results and checkpoints are unaffected (verified, §5).
The practical cost is that every multi-node job ends in a stall of several
minutes and a non-zero exit status, which wastes allocation and makes genuine
shutdown failures indistinguishable from this one.

---

## 2. Environment

| Component | Version / value |
|---|---|
| System | Aurora (ALCF), Intel Data Center GPU Max 1550 |
| Kernel | `6.4.0-150700.53.78-default` (SLES 15) |
| Frameworks module | `frameworks/2026.1.0` (`aurora_frameworks-2026.1.0`) |
| PyTorch | `pytorch_2.13.0_patched_08_02_2026` |
| oneAPI runtime | `intel-oneapi-runtime-2026.1.0-sesoxjw`, `libsycl.so.9` |
| oneCCL | `/opt/aurora/26.181.0/oneapi/ccl/latest`, `libccl.so.1` |
| Faulting library | `torch/lib/libtorch-xpu-ops-sycl-ReduceNormKernel.so` (dated Sep 24 09:56) |
| gdb | GNU gdb (SUSE SLE 15) 14.2 |
| Scale | 4 nodes x 12 ranks = **48 ranks** (1 rank per tile, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`) |
| Launcher | PALS / `mpiexec`, `CCL_PROCESS_LAUNCHER=hydra` |
| Workload | TorchTitan fork, 2B-parameter model, FSDP2/DTensor, DCP checkpoint |

Relevant environment per rank (from census):

```
CCL_CONFIGURATION=cpu_gpu_dpcpp  CCL_PROCESS_LAUNCHER=hydra  CCL_LOG_LEVEL=info
FI_PROVIDER=cxi,tcp;ofi_rxm      FI_CXI_RX_MATCH_MODE=hardware
ZE_FLAT_DEVICE_HIERARCHY=FLAT    ZE_ENABLE_API_TRACING=1
PALS_PMI=pmix                    PALS_LOCAL_SIZE=12
```

Each rank maps **223 distinct `libtorch-xpu-ops-sycl-*.so` kernel libraries**.

---

## 3. The observed failure site

Captured with `gdb -p <pid> -batch -ex "info threads" -ex "thread apply all bt"`
against **live, running ranks during the stall** (not from a core file):

```
Thread 1 "python" received signal SIGSEGV, Segmentation fault.

#0  0x0000151a322b6f00 in ?? ()
#1  0x0000151aaf99b5fa in free () from /lib64/libc.so.6
#2  0x0000151a95d9ac2e in sycl::_V1::detail::CompressedRTDeviceBinaryImage::~CompressedRTDeviceBinaryImage()
                           from .../intel-oneapi-runtime-2026.1.0-sesoxjw/lib/libsycl.so.9
#3  0x0000151a95e20a3c in sycl::_V1::detail::ProgramManager::removeImages(sycl_device_binaries_struct*)
                           from .../intel-oneapi-runtime-2026.1.0-sesoxjw/lib/libsycl.so.9
#4  0x0000151a7b73452d in sycl.descriptor_unreg ()
                           from .../torch/lib/libtorch-xpu-ops-sycl-ReduceNormKernel.so
#5  0x0000151aafbfd6ca in _dl_call_fini ()        from /lib64/ld-linux-x86-64.so.2
#6  0x0000151aafc01666 in _dl_fini ()             from /lib64/ld-linux-x86-64.so.2
#7  0x0000151aaf93e1b6 in __run_exit_handlers ()  from /lib64/libc.so.6
#8  0x0000151aaf93e2fc in exit ()                 from /lib64/libc.so.6
#9  0x0000151aaf924ef3 in __libc_start_call_main ()
#10 0x0000151aaf924fb5 in __libc_start_main_impl ()
#11 0x000055c0abee86c7 in _start ()
```

A core dump from the earlier job 8924552 gives the same stack with the free
path expanded, showing the fault is a **heap-consistency failure inside
`free()`**, not a bad pointer at the call site — and with a *different*
kernel library at the `descriptor_unreg` frame:

```
#1  sbrk ()                        from /lib64/libc.so.6
#2  __default_morecore@GLIBC_2.2.5 from /lib64/libc.so.6
#3  systrim.constprop ()           from /lib64/libc.so.6
#4  _int_free ()                   from /lib64/libc.so.6
#5  free ()                        from /lib64/libc.so.6
#6  sycl::_V1::detail::CompressedRTDeviceBinaryImage::~CompressedRTDeviceBinaryImage()
#7  sycl::_V1::detail::ProgramManager::removeImages(sycl_device_binaries_struct*)
#8  sycl.descriptor_unreg ()
       from .../torch/lib/libtorch-xpu-ops-sycl-ForeachBinaryOpScalarKernels.so
#9  _dl_call_fini () / #10 _dl_fini ()
```

`systrim` means glibc was returning heap to the OS and found its own
bookkeeping inconsistent — consistent with a double free or a free of a
pointer the allocator does not own. Full backtraces for both retained cores:
`core_evidence_8924552/*.gdb.txt`.

Consistency across the capture set:

| Check | Result |
|---|---|
| Captures taken | **48** |
| Captures showing `SIGSEGV` on Thread 1 | **48 / 48** |
| Captures with `~CompressedRTDeviceBinaryImage` at frame #2 | **48 / 48** |
| Captures with `descriptor_unreg` at frame #4 | **48 / 48** |
| Library at frame #4 | `libtorch-xpu-ops-sycl-ReduceNormKernel.so` in **all 48** |
| Frame #0 PC symbolizable | **0 / 48** (always `?? ()`) |
| Distinct ranks | 12 (ranks 9,10,11,21,22,23,33,34,35,45,46,47) |
| Distinct nodes | 4 of 4 |

Frame #0 never resolves to a mapped symbol. Combined with `free()` at #1, this
is consistent with a call through a pointer into memory that is already freed or
whose library has already been unmapped — but we have not proven which, and we
are not asserting a mechanism (see §7).

### It is a fault *loop*, not a deadlock

The same rank, sampled across successive capture rounds:

| Round | Time without forward progress | CPU in last 5s poll |
|---|---|---|
| 1 | 25 s | 503 jiffies (~1.0 core) |
| 2 | 117 s | **4877 jiffies (~9.7 cores)** |
| 3 | 186 s | 4848 jiffies (~9.7 cores) |

Same program counter, same stack, every round. The process is faulting, the
fault is being handled, and it re-faults — consuming more CPU over time while
making no progress. It never self-terminates; an external timeout ends it.

This is why the symptom is bimodal: if the harness timeout is shorter than the
loop, the job looks "hung"; otherwise it reports SIGSEGV.

---

## 4. oneCCL helper threads are idle — CCL does not appear involved

We initially suspected a communication-library busy-wait. The captures rule that
out. In the same processes, at the same instant, the non-main threads are all in
ordinary blocking waits:

```
Thread 2 "jemalloc_bg_thd"  __futex_abstimed_wait_common  <- pyarrow jemalloc bg thread
Thread 3 "pt_xccl_heartbt"  __futex_abstimed_wait_common
                            c10d::HeartbeatMonitorXCCL::runLoop()
                            (ProcessGroupXCCLMonitor.cpp:55, normal wait_for)
Thread 4 "python"           poll()  <- server::run() / kvs_server_init() in libccl.so.1
Thread 5 "python"           epoll_wait()  <- PMIx progress_engine
                            (pmix_progress_threads.c:230)
```

Per-thread kernel wait channels from `/proc/<tid>/wchan` agree:

```
tid=<main>  state=R  wchan=0                     comm=python        <- faulting
tid=...     state=S  wchan=futex_wait_queue      comm=jemalloc_bg_thd
tid=...     state=S  wchan=futex_wait_queue      comm=pt_xccl_heartbt
tid=...     state=S  wchan=do_sys_poll           comm=python
tid=...     state=S  wchan=do_epoll_wait         comm=python
```

Only the main thread is runnable. The XCCL heartbeat monitor is in a normal
timed condition wait and the CCL KVS server thread in a routine `poll()`.
**We see no evidence that oneCCL is implicated**, and we note this explicitly so
the report is not mis-routed to the CCL team.

---

## 5. Work and checkpoint are complete and valid before the fault

The fault occurs strictly after all useful work. From job 8929331:

```
'Training completed' lines              : 48 / 48
distinct ranks reaching final atexit    : 48 / 48   (Python finalization done)
last training step logged               : step 3, loss 12.53470, 19:14:27
teardown stall                          : ~505 s
launcher exit status                    : 124       (our own timeout, not the defect)
```

Checkpoint validation, run as a separate phase in the same job after the stall:

```
checkpoint under test:
  outputs/agpt_2b_20261014_191255/agpt-2b-AdamW-pg19-n4-gbs96/step-3
  48 .distcp shards (one per rank)
  RELOAD OK
checkpoint durable + reloadable : YES
```

The DCP checkpoint reloads cleanly. **No data loss is attributable to this
defect.** The ordering is unambiguous: training completes -> checkpoint is
written and flushed -> all 48 ranks clear their final Python `atexit` handler ->
`exit()` runs -> `_dl_fini` -> fault.

---

## 6. Reproduction

Reproduced on 4 of 5 attempts at this scale. (The one non-reproduction, job
8929205, died earlier for an unrelated reason and never reached exit.)

1. Aurora, `module load frameworks/2026.1.0`.
2. 4 nodes, 12 ranks/node, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `CCL_PROCESS_LAUNCHER=hydra`.
3. Run any multi-node PyTorch XPU training that exercises enough distinct
   `libtorch-xpu-ops` kernels to map a large set of SYCL device images
   (we map 223) and completes normally.
4. Let the process exit normally.
5. Observe: ranks do not exit. Attach gdb during the stall.

Observed stall durations across jobs: **354 s, 466 s, 505 s** — none ended on
its own.

Note for anyone reproducing: **a core dump will not be produced while the
process is looping**, so "no core file" does not mean the fault did not occur.
Live attach is required. On Aurora `ptrace_scope` is absent (no yama LSM), so
`gdb -p` works on compute nodes without special privileges.

Also note that a stall detector keyed on *idle* CPU will not see this — the
ranks are burning CPU, not parked. We detect it by fingerprinting per-thread
`(state, wchan)` and firing on absence of change.

---

## 7. What we are NOT claiming

We are reporting a reproducible **failure site**, not a diagnosis. Specifically
left open for Intel to confirm:

- **Ownership.** Frame #4 is in `libtorch-xpu-ops-sycl-ReduceNormKernel.so` and
  frames #2-#3 are in `libsycl.so.9`. We cannot tell from the stack whether the
  defect is in the SYCL runtime's image bookkeeping, in how `libtorch-xpu-ops`
  registers/unregisters descriptors, or in the destructor ordering between the
  223 kernel libraries and `libsycl` at `_dl_fini`.
- **Mechanism.** A double-free, a free of a pointer owned elsewhere, and a call
  into an already-unmapped library are all consistent with what we see. We have
  not instrumented the allocator to distinguish them. The core-dump stack above
  narrows it somewhat: the fault occurs in `systrim`, while glibc is returning
  heap to the OS, which points at corrupted allocator bookkeeping rather than
  at a call into unmapped code.
- **~~Why `ReduceNormKernel` specifically.~~ Resolved 2026-10-15: it is not
  specific.** It is the library at frame #4 in all 48 live-gdb captures of job
  8929331, but a core dump from job 8924552
  (`core.x4116c7s3b0n0.1762491`, retained) faults with frame #4 in
  **`libtorch-xpu-ops-sycl-ForeachBinaryOpScalarKernels.so`** instead, with
  frames #2-#3 unchanged in `libsycl.so.9`. The faulting library is therefore
  whichever one `_dl_fini` happens to finalize into the already-freed image,
  not a property of any one kernel. This supports the destructor-ordering
  reading of the Ownership bullet above and argues against a defect localized
  to a single kernel library.
- **Why the fault repeats** instead of terminating the process. We have not
  determined what re-enters the faulting path.

Questions that would help us narrow it:

1. Is `ProgramManager::removeImages` expected to be safe when many device-image
   descriptors are unregistered during `_dl_fini`, in unspecified order?
2. Is there a supported way to force SYCL image teardown *before* `exit()`, so
   finalization does not run at `_dl_fini` time?
3. Is a build of `libsycl.so.9` / `libtorch-xpu-ops` with allocator debugging
   available for Aurora that would let us characterize the `free()` argument?

---

## 8. Artifacts

All paths on Aurora, readable by the `datascience` project:

```
Run directory
  /lus/flare/projects/datascience/seonghapark/xpu_launcher/stall_gdb_20261014_191243/

  stackdumps/<node>/gdb_rSTALLED-SPINNING_rank<N>_pid<PID>_round<R>.txt
      48 live captures: `info threads` + `thread apply all bt`.
      Header records rank, pid, timestamp, classification, CPU delta,
      and time without forward progress.

  stackdumps/<node>/census_round<R>.txt
      16 per-node censuses: for every rank, its pid, per-thread state and
      kernel wchan, the mapped SYCL libraries, and the PALS/CCL/FI environment.

  stackdumps/watch_<node>.log
      Per-poll instrumentation: phase, rank count, CPU delta min/med/max,
      max time without forward progress, classification histogram.

  phase1_repro.log      full training log (48 ranks)
  ptrace_permissions.txt  gdb/ptrace availability on the compute node

Job stdout
  /lus/flare/projects/datascience/seonghapark/xpu_launcher/agpt_stall_gdb.o8929331

Earlier evidence
  Job 8924552 produced 47 core dumps (529.8 GB, 11.3 GB mean) showing this
  same stack, before any diagnostic tooling existed in the run. Two of those
  cores are retained at
  /lus/flare/projects/datascience/seonghapark/core_evidence_8924552/
  with a manifest and full extracted backtraces. All 47 were a single
  4-second fault window on 2026-10-13 (hosts s3/s4/s5/s6, 11/12/12/12); the
  other 45 were deleted on 2026-10-15 after the stacks were extracted, since
  they carried no evidence the retained pair does not.
```

**The diagnostic tooling did not cause this.** The identical stack was first
obtained from core dumps in job 8924552, prior to any instrumentation. In job
8929331 the Python `faulthandler` installed by our tracing shim never fired
(0 dumps), and the shim performs no allocation or device work at exit.
