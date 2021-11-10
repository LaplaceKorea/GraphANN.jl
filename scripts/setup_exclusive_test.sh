#!/bin/sh

# Configures huge pages and environment variables to perform the JULIA_EXCLUSIVE NUMA-local
# tests.

echo "Clearing Existing Hugepages"
sudo hugeadm --pool-pages-min=1G:0

# Note - `--pool-pages-min` is not exclusive to each NUMA node - rather it refers to
# the global number of pages.
#
# So, when allocating on node 1, we actually need to add the number of pages we want
# on that node to the current total>
echo "Allocating 2 1GiB huge pages on numa node 0 and 1 1GiB huge page on numa node 1"
sudo numactl --cpunodebind=0 --membind=0 hugeadm --obey-mempolicy --pool-pages-min=1G:2
sudo numactl --cpunodebind=1 --membind=1 hugeadm --obey-mempolicy --pool-pages-min=1G:3
sudo hugeadm --create-mounts

# Set environment variables
export JULIA_EXCLUSIVE=1
export JULIA_NUM_THREADS=$(nproc)
