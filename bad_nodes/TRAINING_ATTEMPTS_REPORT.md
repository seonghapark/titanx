# Training Attempts Report - 2026-09-26

## Summary
Multiple attempts to launch 32-49 node training jobs failed due to widespread resource unavailability in the x4012 cabinet.

## Attempts

### Attempt 1: 49 Nodes (39 computing + 10 spare)
- **Hostfile**: hostfile_48nodes.txt
- **Config**: NNODES=39, SPARE_NODES=10 (20% spare)
- **Status**: ❌ FAILED
- **Error**: x4012c0s0b0n0 - "Resource temporarily unavailable"

### Attempt 2: 46 Nodes (37 computing + 9 spare)
- **Hostfile**: hostfile_46nodes.txt (excluded x4012c0s0 and x4012c5s1)
- **Config**: SPARE_NODES_PERCENTAGE=20
- **Status**: ❌ FAILED
- **Error**: x4012c0s1b0n0 and x4012c5s0b0n0 - "Resource temporarily unavailable"

### Attempt 3: 32 Nodes (25 computing + 6 spare)
- **Hostfile**: hostfile_32nodes.txt (only c1-c4 cabinets)
- **Config**: NNODES=25, SPARE_NODES=6 (20% spare)
- **Total Processes**: 300 (25 × 12)
- **Status**: ❌ FAILED
- **Error**: x4012c1s0b0n0 and x4012c4s2b0n0 - "Resource temporarily unavailable"

## Root Cause Analysis

The entire x4012 cabinet appears to be experiencing resource constraints:
- Consistent "Resource temporarily unavailable" errors
- Affects multiple cabinets (c0-c6)
- Issue persists across different node subsets
- Auto-retry failover unable to find healthy nodes

## Possible Causes

1. **Cluster Maintenance**: Scheduled maintenance on x4012 cabinet
2. **Node Issues**: Hardware problems affecting entire cabinet
3. **Resource Contention**: High cluster load from other jobs
4. **Network Issues**: Connectivity problems in the x4012 segment

## Recommendations

### Option 1: Check Cluster Status
```bash
# Check Aurora cluster status
sinfo -N -o "NodeList,State,Gres"

# Check specific cabinet
qstat -f | grep x4012
```

### Option 2: Wait and Retry
- Wait 15-30 minutes for cluster to stabilize
- Retry same command: `./run_training_32nodes.sh`

### Option 3: Use Different Cabinet (if available)
- Submit request to use x4013, x4014, or other cabinets
- Contact HPC team for available resources

### Option 4: Smaller Job
- Try with 8-16 nodes to test if any resources are available
- Smaller job less likely to hit unavailable nodes

### Option 5: Run on Different Date
- Submit job to queue for later execution
- Avoid current high-load period

## Script Improvements Made

Updated `run_train.sh` to support:
- `SPARE_NODES_PERCENTAGE` for dynamic spare node calculation
- Correct calculation: `total_nodes - (total_nodes × percentage)`
- Backward compatible with explicit NNODES/SPARE_NODES

## Next Steps

1. **Check cluster status** with commands above
2. **Contact HPC team** if x4012 is down
3. **Try different node set** if other cabinets are available
4. **Retry** when cluster is healthy

---

**Report Generated**: 2026-09-26 15:04  
**Last Attempt**: 32 nodes (c1-c4 only)  
**Status**: Cluster resource unavailable
