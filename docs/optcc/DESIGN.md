# OptCC changes to NCCL: design log

One entry per change: what it does, why it is built this way, what was
rejected, how to turn it off. Newest last.

## 1. Receive-side rate limit (`NCCL_NET_RX_MAX_MBPS`)

Branch `optcc-rxlimit`, file `src/transport/net.cc`.

**Goal.** Every emulated node must be a 25 Gb/s *full-duplex* NIC. The rig caps
TX in hardware (SR-IOV `max_tx_rate` on each container's VF). RX has no
hardware cap: on kernel 5.15 / mlx5 legacy SR-IOV, both `ip link ... vf` and
`devlink port function rate` expose only `tx_max`/`tx_share`. The ring never
notices, because each rank receives from one capped sender. OptCC does notice:

- the straggler receives uploads from several healthy ranks;
- a healthy rank can receive from its ring predecessor and from the straggler at once.

Without an RX cap, those receives would run faster than the model allows.

**Mechanism.**
- A process-wide token bucket, filled at `NCCL_NET_RX_MAX_MBPS` (payload Mbit/s).
- `recvProxyProgress` posts an `irecv` only after taking that post's bytes from the bucket. If the bucket is short, the steps stay unposted and are retried on the next progress call.
- IB/RoCE senders can only RDMA-write into a buffer the receiver has posted (the CTS goes through the sender's FIFO). So posting at rate R bounds bytes received by R*t plus one post.
- One rank per container means one bucket per emulated node.

**Parameters.**
- `NCCL_NET_RX_MAX_MBPS`: default 0, which is off and exactly upstream behaviour.
- `NCCL_NET_RX_BURST_BYTES`: bucket depth, default one post (Simple: stepSize x sliceSteps, about 1 MiB).
- A 25 Gb/s wire corresponds to about 23500 payload Mbit/s (RoCE headers, measured on the rig). Use that value so RX matches the VF's TX cap.

**Why here.**
- The proxy thread is the only place that decides when a receive buffer is offered, for every algorithm alike.
- Ring and OptccRing therefore run under the same limit, which keeps the comparison fair.
- It also lives next to where the OptCC receive arbiter will go (one sender at a time per receiver), so both RX rules are in one function.

**Rejected.**
- tc ingress police: software in the kernel, and RoCE bypasses it.
- switchdev + offloaded police on the representor: rebuilds the whole rig (VF networking, RoCE path); whether mlx5 on 5.15 offloads RX-direction police is unverified.
- Limiting senders per destination: no such knob exists.

**Limits.**
- The cap is on posting, so a single burst of up to one post (plus buffers posted before the bucket emptied) can arrive at line rate.
- Averages over >= 1 ms follow R.
- Profiler "recv step start" events can be recorded twice for a step that was throttled and retried. This does not matter with the profiler off.

**Off switch.** Unset `NCCL_NET_RX_MAX_MBPS` (or set it to 0).

**Checks.**
- Ring AllReduce beta unchanged with the cap at 23500 (only one sender per receiver).
- nccl-tests `alltoall` (4 senders into each receiver): per-rank RX rate = busbw drops to <= the cap, versus above the cap without it.

### 1a. Throttled is not idle (fix)

First rig run: ring AllReduce at 8 MiB went from 4.4 ms to 11.2 ms with the cap
on (3 of 3 runs), while 256 MiB lost only ~1%. A fixed stall per collective,
not a rate effect. Cause: when every post in a progress call is held back by
the bucket, the op reports idle, and the proxy loop `sched_yield()`s; on the
container's few NUMA-bound cores that costs a scheduler slice before the post
is retried. Fix: a call that was throttled marks the op busy at the very end
of `recvProxyProgress` (not at the throttle point: `idle = 0` there would take
the early `return` and skip the completion checks). The proxy thread then
spins on the bucket instead of yielding.
