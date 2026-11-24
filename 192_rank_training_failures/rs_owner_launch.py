import os
from pathlib import Path
import sys


root = Path(__file__).resolve().parent
rank = next((int(os.environ[k]) for k in ("PALS_RANKID", "PMI_RANK", "PMIX_RANK", "RANK")
             if k in os.environ), None)
if rank is None:
    raise RuntimeError("RS ownership diagnostic requires an MPI rank")

training = [sys.executable, str(root / "rs_event_train.py"), *sys.argv[1:]]
if rank not in (128, 150, 156, 183):
    os.execv(training[0], training)

os.environ["RS_OWNER_GDB_RANK"] = str(rank)
commands = (
    "set pagination off", "set confirm off", "set breakpoint pending on",
    "set auto-load off", "set debuginfod enabled off", "set print thread-events off",
    "set disable-randomization off", "handle SIGSEGV stop print nopass",
    f"source {root / 'rs_owner_gdb.py'}", "run",
)
argv = ["/usr/bin/gdb", "-nx", "-batch", "-return-child-result"]
for command in commands:
    argv.extend(("-ex", command))
os.execv(argv[0], [*argv, "--args", *training])
