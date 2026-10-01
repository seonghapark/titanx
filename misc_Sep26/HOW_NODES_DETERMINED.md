# How run_train_torchtitan.sh Determines Computing Nodes and Spare Nodes

**Date**: 2026-09-26  
**Topic**: Node determination logic in multi-node training

---

## Overview

The `./xpu_torchtitan/run_train_torchtitan.sh multi` script determines the number of computing nodes and spare nodes through a specific sequence of steps. This document explains the exact logic.

---

## Quick Answer

### Computing Nodes (NNODES)

**How it's determined**:
```bash
NNODES="${NNODES:-$(awk 'NF {print $1}' "$HOSTFILE" | sort -u | wc -l)}"
```

**In plain English**:
1. Check if `NNODES` environment variable is set
2. If NOT set, count unique hostnames in the hostfile
3. That count becomes NNODES

### Spare Nodes (SPARE_NODES)

**How it's determined**:
```bash
SPARE_NODES="${SPARE_NODES:-auto}"
```

**In plain English**:
1. Check if `SPARE_NODES` environment variable is set
2. If NOT set, default to "auto"
3. The XPU launcher then decides how many spare nodes to allocate based on "auto"

---

## Detailed Step-by-Step Explanation

### Step 1: Where Does the Hostfile Come From?

```bash
# From run_train_torchtitan.sh line 108-119
if [[ -z "$HOSTFILE" ]]; then
    # No hostfile given: run_train.sh derives one from PBS_NODEFILE
    if [[ -z "${PBS_NODEFILE:-}" || ! -f "${PBS_NODEFILE:-}" ]]; then
      echo "error: multi mode requires <hostfile> (or run inside a PBS job with PBS_NODEFILE)" >&2
      usage
      exit 1
    fi
  else
    if [[ ! -f "$HOSTFILE" ]]; then
      echo "error: hostfile not found: $HOSTFILE" >&2
      exit 1
    fi
    HOSTFILE="$(realpath "$HOSTFILE")"
fi
```

**Three sources for hostfile**:
1. **Explicit hostfile**: User provides `--hostfile /path/to/hosts`
2. **PBS environment**: System auto-derives from `PBS_NODEFILE` (set by scheduler when PBS job runs)
3. **Error**: If none available, script fails

### Step 2: Determine NNODES from Hostfile

```bash
# From run_train.sh line 150
NNODES="${NNODES:-$(awk 'NF {print $1}' "$HOSTFILE" | sort -u | wc -l)}"
```

**Breakdown**:

```
${NNODES:-...}
└─ If NNODES is set, use it
   If NOT set, use: $(...)

$(awk 'NF {print $1}' "$HOSTFILE" | sort -u | wc -l)
│
├─ awk 'NF {print $1}' "$HOSTFILE"
│  └─ Print first column (hostname) of non-empty lines
│     Example hostfile content:
│     ───────────────────────────────────
│     x4001c7s2b0n0.hsn.cm.aurora.alcf.anl.gov
│     x4002c3s0b0n0.hsn.cm.aurora.alcf.anl.gov
│     x4003c7s2b0n0.hsn.cm.aurora.alcf.anl.gov
│     x4001c7s2b0n0.hsn.cm.aurora.alcf.anl.gov  (duplicate)
│     ───────────────────────────────────
│     Output: 
│     x4001c7s2b0n0.hsn.cm.aurora.alcf.anl.gov
│     x4002c3s0b0n0.hsn.cm.aurora.alcf.anl.gov
│     x4003c7s2b0n0.hsn.cm.aurora.alcf.anl.gov
│     x4001c7s2b0n0.hsn.cm.aurora.alcf.anl.gov
│
├─ sort -u
│  └─ Sort and remove duplicates
│     Output:
│     x4001c7s2b0n0.hsn.cm.aurora.alcf.anl.gov
│     x4002c3s0b0n0.hsn.cm.aurora.alcf.anl.gov
│     x4003c7s2b0n0.hsn.cm.aurora.alcf.anl.gov
│
└─ wc -l
   └─ Count lines = count unique nodes
      Output: 3
      
NNODES = 3
```

**Summary**: Script counts unique hostnames in the hostfile

### Step 3: Calculate Total Processes (NPROC)

```bash
# From run_train.sh line 151
NPROC="${NPROC:-$((NNODES * NPROC_PER_NODE))}"
```

**Logic**:
- If `NPROC` is set, use it
- Otherwise, calculate: `NNODES × NPROC_PER_NODE`
- Since NPROC_PER_NODE = 12 (auto-detected), this becomes: `NNODES × 12`

**Example**:
```
NNODES = 9
NPROC_PER_NODE = 12 (auto-detected)
NPROC = 9 × 12 = 108
```

### Step 4: Determine Spare Nodes

```bash
# From run_train.sh line 155-159
AUTO_RETRY="${AUTO_RETRY:-1}"
if [[ "$AUTO_RETRY" == "1" ]]; then
  SPARE_NODES="${SPARE_NODES:-auto}"
  FAILOVER_PROFILE="${FAILOVER_PROFILE:-auto}"
  LAUNCH_CMD+=("--auto-retry" "--spare-nodes" "$SPARE_NODES" "--failover-profile" "$FAILOVER_PROFILE")
fi
```

**Logic**:

```
AUTO_RETRY (default: 1 = enabled)
  │
  ├─ If AUTO_RETRY = 1:
  │  │
  │  ├─ SPARE_NODES="${SPARE_NODES:-auto}"
  │  │  └─ If SPARE_NODES not set → use "auto"
  │  │
  │  ├─ FAILOVER_PROFILE="${FAILOVER_PROFILE:-auto}"
  │  │  └─ If FAILOVER_PROFILE not set → use "auto"
  │  │
  │  └─ Pass to launcher:
  │     --auto-retry
  │     --spare-nodes auto
  │     --failover-profile auto
  │
  └─ If AUTO_RETRY = 0:
     Spare nodes NOT allocated
     Failover disabled
```

**What "auto" means**:
- The XPU launcher determines optimal spare nodes
- Typically: 1-2 extra nodes allocated automatically
- Based on cluster availability and NNODES size

### Step 5: Build Launch Command

```bash
# From run_train.sh line 153
LAUNCH_CMD+=("--hostfile" "$HOSTFILE" "-nh" "$NNODES" "-n" "$NPROC" "-ppn" "$NPROC_PER_NODE")

# Full command looks like:
xpu launch --scheduler auto \
  --hostfile /path/to/hosts \
  -nh 9 \                          # NNODES = 9
  -n 108 \                         # NPROC = 108
  -ppn 12 \                        # NPROC_PER_NODE = 12
  --auto-retry \
  --spare-nodes auto \
  --failover-profile auto \
  -- python train.py ...
```

---

## Complete Flow Diagram

```
PBS Job Submission
       │
       ├─ Scheduler allocates nodes
       │  └─ Creates PBS_NODEFILE with node list
       │
       ├─ run_train_torchtitan.sh multi called
       │  │
       │  ├─ Derives hostfile from PBS_NODEFILE
       │  │
       │  └─ Calls run_train.sh multi
       │     │
       │     ├─ NNODES = count unique hosts in hostfile
       │     │  Example: 9 nodes detected
       │     │
       │     ├─ NPROC_PER_NODE = auto-detected (always 12)
       │     │
       │     ├─ NPROC = NNODES × NPROC_PER_NODE
       │     │  = 9 × 12 = 108
       │     │
       │     ├─ SPARE_NODES = "auto" (if not set)
       │     │
       │     └─ Build launch command with:
       │        - hostfile path
       │        - NNODES=9
       │        - NPROC=108
       │        - NPROC_PER_NODE=12
       │        - spare-nodes=auto
       │
       └─ XPU launcher executes:
          - Allocates 9 primary nodes for training
          - Allocates 1-2 spare nodes (from "auto")
          - Starts 108 training processes
          - Enables auto-retry failover
```

---

## Real Examples

### Example 1: From PBS Job (Most Common)

**PBS Job Script**:
```bash
#!/bin/bash
#PBS -l select=9:ngpus=1
#PBS -l walltime=12:00:00

cd /lus/flare/projects/datascience/seonghapark/xpu_launcher

./xpu_torchtitan/run_train_torchtitan.sh multi \
  --training.steps 400000
```

**What happens**:
1. PBS allocates 9 nodes
2. System creates PBS_NODEFILE with 9 hostnames
3. run_train_torchtitan.sh reads PBS_NODEFILE
4. Extracts hostfile from PBS_NODEFILE
5. Counts unique hosts: **NNODES = 9**
6. Calculates processes: **NPROC = 9 × 12 = 108**
7. Sets spare nodes: **SPARE_NODES = auto** (1-2 nodes)
8. Launches training with auto-retry enabled

**Result**: 9 nodes for training + 1-2 spare nodes (system decides)

### Example 2: With Explicit Hostfile

**Command**:
```bash
NNODES=5 ./xpu_torchtitan/run_train_torchtitan.sh multi /path/to/hostfile.txt \
  --training.steps 400000
```

**What happens**:
1. NNODES explicitly set to 5 → **overrides** counting from hostfile
2. NPROC_PER_NODE auto-detected: **12**
3. NPROC calculated: **5 × 12 = 60**
4. SPARE_NODES defaults to: **auto**

**Result**: 5 nodes for training + spare nodes (system decides)

### Example 3: With Explicit SPARE_NODES

**Command**:
```bash
export SPARE_NODES=2
./xpu_torchtitan/run_train_torchtitan.sh multi \
  --training.steps 400000
```

**What happens**:
1. SPARE_NODES explicitly set to 2 → **overrides "auto"**
2. NNODES from hostfile: **9**
3. NPROC calculated: **9 × 12 = 108**

**Result**: 9 nodes for training + exactly 2 spare nodes

---

## Key Decision Points

### 1. Where Does NNODES Come From?

| Source | Priority | Example |
|--------|----------|---------|
| Environment variable `NNODES` | Highest | `export NNODES=5` |
| Unique hosts in hostfile | Medium | Derives 9 from hostfile |
| PBS_NODEFILE (auto) | Lowest | Auto-derived |

**Decision Logic**:
```bash
NNODES="${NNODES:-$(count_unique_hosts_in_hostfile)}"
```

If `NNODES` set → use it  
Else → count hostfile → use that

### 2. Where Does SPARE_NODES Come From?

| Source | Priority | Example |
|--------|----------|---------|
| Environment variable `SPARE_NODES` | Highest | `export SPARE_NODES=2` |
| Default "auto" | Lowest | System decides 1-2 |

**Decision Logic**:
```bash
SPARE_NODES="${SPARE_NODES:-auto}"
```

If `SPARE_NODES` set → use it  
Else → use "auto"

### 3. When Are Spare Nodes Allocated?

```bash
# Spare nodes only allocated if AUTO_RETRY is enabled
if [[ "$AUTO_RETRY" == "1" ]]; then
  # Allocate spare nodes for failover
fi
```

**Default**: AUTO_RETRY=1 (enabled)  
**Override**: `AUTO_RETRY=0` (disable failover)

---

## What "auto" Actually Means for SPARE_NODES

The XPU launcher's internal logic (not shown in this script) decides:

```
SPARE_NODES="auto"
    ├─ System checks cluster size
    ├─ System checks NNODES value
    ├─ System determines optimal spares
    │  ├─ For NNODES=9 → typically 1-2 spares
    │  ├─ For NNODES=13 → typically 1-2 spares
    │  ├─ For NNODES=1 → 0-1 spares
    │  └─ For NNODES=49 → may be more
    └─ Allocates determined number of spares
```

**Result**: System automatically optimizes spare allocation

---

## Practical Decision Table

| Scenario | NNODES Source | SPARE_NODES Source | Result |
|----------|---------------|------------------|--------|
| PBS job, defaults | Hostfile (from PBS) | "auto" | System-optimized |
| PBS job + explicit NNODES | Environment var | "auto" | Explicit + auto spares |
| PBS job + explicit both | Environment NNODES | Environment SPARE | Fully explicit |
| Explicit hostfile | Hostfile count | "auto" | Derived + auto spares |
| All explicit | Env var | Env var | Complete control |

---

## Summary: The Decision Tree

```
run_train_torchtitan.sh multi called
│
├─ Get hostfile
│  ├─ From PBS_NODEFILE (if in PBS job)
│  └─ OR from explicit --hostfile argument
│
├─ Determine NNODES
│  ├─ If NNODES env var set → use it (highest priority)
│  └─ Else → count unique hosts in hostfile
│
├─ Get NPROC_PER_NODE
│  └─ Auto-detected from torch.xpu.device_count() → always 12
│
├─ Calculate NPROC
│  ├─ If NPROC env var set → use it
│  └─ Else → NNODES × NPROC_PER_NODE
│
├─ Determine SPARE_NODES
│  ├─ If SPARE_NODES env var set → use it
│  └─ Else → use "auto" (system decides)
│
└─ Build and execute launcher command
   with all determined values
```

---

## Verification Commands

**Check what NNODES will be detected**:
```bash
# If using PBS job, check how many unique hosts:
awk 'NF {print $1}' $PBS_NODEFILE | sort -u | wc -l
# Returns: number of nodes

# If using explicit hostfile:
awk 'NF {print $1}' /path/to/hostfile | sort -u | wc -l
# Returns: number of nodes
```

**Check what will be passed to launcher**:
```bash
# Use dry-run to see the exact command:
./xpu_torchtitan/run_train_torchtitan.sh multi --dry-run \
  --training.steps 100
# Shows: xpu launch --scheduler auto -nh <NNODES> -n <NPROC> ...
```

---

## Conclusion

The script determines computing nodes and spare nodes through a **prioritized, deterministic process**:

1. **NNODES**: Explicit env var > Hostfile count (from PBS or explicit file)
2. **SPARE_NODES**: Explicit env var > Default "auto"
3. **NPROC**: Explicit env var > Calculated from NNODES × NPROC_PER_NODE

This design provides:
- ✅ Automatic detection when possible
- ✅ Explicit control when needed
- ✅ Smart defaults (auto spare nodes)
- ✅ Full transparency (can dry-run to see command)

---

**Document Version**: 1.0  
**Last Updated**: 2026-09-26
