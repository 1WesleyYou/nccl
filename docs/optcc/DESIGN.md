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

### 1b. Charge bytes received, not buffers posted (fix)

1a did not change the 8 MiB number (still 11.2 ms, 3/3), so the stall was not
the yield. What the data fit instead: a post reserves a whole step buffer
(Simple: stepSize x sliceSteps, about 1 MiB) while a step of a small message
carries far less, so the bucket charged several times the real bytes and a
single-sender 8 MiB ring was held to ~9.6 Gb/s; at 256 MiB steps are full and
the error vanishes (the measured +1%). Fix: remember the posted size per step
slot (`recvNetResources::rxPosted`) and refund posted - received when the step
completes. The cap now counts bytes received. 1a is kept: it is still right
that waiting for tokens is not idleness.

## 2. OptccRing AllReduce kernel (l >= 2 schedule, one straggler)

Branch `optcc-kernel` (on top of `optcc-rxlimit`).

**Use.**
- `NCCL_ALGO=allreduce:OptccRing` turns it on, and `NCCL_OPTCC_STRAGGLER=<rank>` names the straggler (default 2, the old hard-coded value).
- Without that, nothing changes:
  - `ncclOptccRequested()` is false;
  - the algorithm stays disabled and has no cost;
  - the eager `ncclTransportOptccConnect` is skipped.

**Device (`src/device/all_reduce_optcc.h`, new).**
- `RunWorkColl<AllReduce, OPTCCRING, SIMPLE>`.
- **Segments:** a channel's loop is one segment of nh = p-1 sections, and healthy index i owns section i.
- **Two orderings, by channel parity:**
  - Even channels run S1 S2 S3 S4: `send`/`recvReduceSend`/`recvReduceCopy` over the healthy ring, then `sendFromOutput` + `recv` with the straggler, then the ring allgather.
  - Odd channels run S3 S1 S4 S2. The straggler sends raw section j to the rank that starts j's chain (healthy index j+1), which takes it with `recvReduceSend` on its first hop. The owner ends reduce-scatter with the global sum, allgathers it, and sends it back last.
- **Stages as scoped primitives:** each stage is its own `Primitives<..., FanAsymmetric<1,1>, ...>` in a scope.
  - The destructor writes the connection step back, so stages chain on the connections the OptCC connect built.
  - FanAsymmetric, because the straggler's send-only and recv-only stages pass -1. `FanSymmetric` stores the recv count for both sides and would silently drop the send.
- **Concurrency:** channels run concurrently, which is where the ring of one channel overlaps the straggler link of another. The channel count plays the role of the paper's pattern count.

**Host.**
- `device.h`: `ncclOptccRing` gains `straggler`, `healthyIndex` and a fixed `healthy[8]`. The healthy order is canonical (the channel ring rotated to its lowest rank, stragglers skipped), so every rank agrees on which rank owns which section.
- `optcc.cc`: fills those fields; defines `ncclOptccRequested()` (NCCL_ALGO names it, not as a `^` exclusion) and `NCCL_OPTCC_STRAGGLER`.
- `tuning.cc`: when requested, AllReduce/Simple takes the ring's bandwidth and latency as a placeholder cost; otherwise the algorithm is disabled as before.
- `enqueue.cc`:
  - selection guard only when not requested;
  - `ncclPatternOptcc` with `nstepsPerLoop = 1`, `nchunksPerLoop = p-1`;
  - the ring's chunk/slice steps and the extra sync warp.
- `proxy.cc`: `ncclPatternOptcc` scales `nsteps` per connection:
  - healthy: 2(nh-1) to next / from prev, 1 to / 1 from the straggler;
  - straggler: 1 each way per healthy rank.
  - This is the part that hangs if it disagrees with the kernel.
- `device.h ncclDevFuncId` + `generate.py`: AllReduce gets a 7th algorithm slot (OPTCCRING is id 7; PAT = 6 is not an AllReduce algorithm). `generate.py` already builds only SIMPLE for non-ring/tree algorithms.

**Checks done (2026-09-24, 5-container rig).**
- `Connected optcc rings` on all 5 ranks, i.e. the algorithm was selected.
- `ring_allreduce.cpp` exact at 1-64 MiB, l = 1 and 2.
- nccl-tests `all_reduce_perf -c 1`, 1-64 MiB at l = 2 with RX caps: 0 wrong in two runs.
- **Open:** one earlier nccl-tests run hung at its first size (1 MiB) and was not reproduced in the next three runs. The campaign retries that gate once and times out single runs.

**Not done.** l < 2 bubble filling; a real cost model; more than one straggler; LL/LL128.

## 3. Why OptccRing missed the bound, and the fix: finer segments

**Symptom.** With default buffers (NCCL_BUFFSIZE 4 MiB), the l = 2 slope was 1.468 x beta_1 at 4 channels (bound 1.25), and l = 1 was 1.317 (fluid ideal 1.094 = 1.75n / 1.6n: each healthy NIC sends RS 0.75n + AG 0.75n + upload 0.25n).

**Cause: stage serialisation inside a channel.**
- A section is one NCCL chunk (Simple: stepSize 512 KiB x chunkSteps 4 = 2 MiB), and a channel runs RS -> straggler link -> AG of a segment strictly in order.
- Ordering-1 channels leave the ring idle while the straggler serves four uploads in turn.
- Ordering-2 channels use the straggler link one direction at a time: sends first, receives later.
- With t = one healthy-NIC section time and l = 2, a segment costs about 14t (ordering 1) and 22t (ordering 2) against the ideal 8t. That model predicts 1.72 for 2 channels; measured 1.745.
- Other channels can hide these bubbles only if segments are short relative to the collective. With 2 MiB sections, a channel has just 8 segments at 256 MiB.

**Fix: `NCCL_BUFFSIZE=524288`.**
- Chunks are 256 KiB and there are 8x more segments per channel. The kernel does not change.
- Measured (8-256 MiB, 3 rounds, 2026-09-25, `results/20260925_optcc_kernel_b512`):
  - l = 2, 4 channels: **1.279 [1.268, 1.291]**, 2.3% above the bound. 256 MiB: 187.7 ms vs ring 293.9 ms (bound 184.1 ms).
  - l = 1, 4 channels: **1.096**, the fluid ideal.
  - Ring with the same buffer: 0.998 (l = 1) and 2.009 (l = 2), i.e. unchanged. The comparison stays fair.
- Search at 128/256 MiB, l = 2: 2 MiB -> 1 MiB -> 512 KiB buffers improve monotonically, and 4-12 channels are all within noise at 512 KiB. Smaller buffers were not tried.

**Possible follow-up.**
- Give OptccRing its own smaller chunk in `calcCollChunking` instead of a comm-wide buffer size, so the ring keeps its default. Not done: the buffer knob already leaves the ring unchanged here, and a code default needs the same validation again.
- Overlap the next segment's ring stage with the current straggler stage inside one channel (split the block into two thread groups), the structural version of the same fix.

**Intermittent hang: found and fixed (`0501bbff`).**
- **Symptom:** about 4% of OptccRing runs (5 of ~120) hung in their first collective.
- **What the dumps showed:**
  - With per-sub steps in the proxy dump (`24c42d79`), exactly one rank's GPU had done nothing on any channel: data had arrived (`r8 t8`) but was never consumed (`d0`), and nothing had been sent. Every other rank was waiting on it.
  - That rank was vn2 in one hang and vn0 in the next: the two ranks that share GPU1 through MPS.
  - gdb from the host showed every rank's main thread past the launches, in `cudaStreamSynchronize`, so the kernels had been issued.
- **Isolation, 40 runs each:**
  - OptccRing with MPS off: 0 hangs, but 3x slower, because vn0 and vn2 time-slice GPU1;
  - ring with MPS on: 0;
  - OptccRing with eager connect (`NCCL_RUNTIME_CONNECT=0`): 0.
- **Cause:** the lazy connect path. At the first OptccRing collective, `ncclTransportOptccConnect` runs, including `ncclTransportP2pSetup` and its device-side copies. On an MPS-shared GPU this can leave one client's first kernel unable to run while the other client's kernel, already resident, waits for it.
- **Fix:** when `NCCL_ALGO` names OptccRing, its connections are built at init also in runtime-connect mode, so the first collective has no setup work.
- **Validation:** 80 of 80 runs clean with MPS on and default runtime connect, 64 MiB median 46.5 ms (unchanged). The chance of 0/80 at the old 4% rate is 0.96^80, about 4%.
- **Logs:** `optcc results/20260925_rca/hang/` (DVC).

## 4. Receive cap fidelity: keep the bucket shallow (follow-up to 1)

No code change; this fixes how entry 1 is used. `NCCL_NET_RX_BURST_BYTES`
must stay near one grant plus a bandwidth-delay product. The rig uses
**256 KiB**; earlier campaigns used 4 MiB.

**Finding (profiler traces, 64 MiB).** With a 4 MiB bucket, a receiver banks
the credit of every idle moment and spends it later. The emulated NIC then
runs faster than its cap whenever it has bubbles:
- OptccRing straggler: 11.9-13.2 Gb/s against an 11.75 cap. At 16 MiB, 4 MiB is a quarter of the traffic.
- Ring: 23.6-23.8 Gb/s against a 23.5 cap.

A real half-speed port cannot store capacity, so this favours OptccRing: its
bubbles are exactly the idle time that gets banked.

With 256 KiB (2 grants at b512, 8 at b128):
- the ring sits at 23.45 Gb/s;
- the straggler sits at 11.77 Gb/s;
- OptccRing l = 2 (8 channels x 128 KiB buffers) is 1.27x the ring at 128 MiB (was 1.26-1.27 with 4 MiB).

A bucket of exactly one grant (`0`) under-delivers slightly: ring -1.6%.

**Validation: the receive cap is a faithful stand-in for a slower link.**

| Test | Receive cap only | Send cap only | Both caps |
|---|---|---|---|
| Ring l = 1, every rank, 128 MiB, 3 reps | 72.66 ms (VF 100G, RX 23500) | 72.96 ms (VF 25G, no RX cap) | 73.01 ms |
| OptccRing l = 2, straggler only, 128 MiB, 3 reps | 93.77 ms | 93.06 ms | 95.10 ms |

- The ring has one inbound flow per receiver; the straggler has several (8 channels x 4 healthy senders).
- For OptccRing both directions carry n, so each cap alone must give the same time.
- `results/20260925_rxab`, `20260925_stragab`.

**How this relates to multi-tenant practice.**
- The cap is receiver-driven admission. NCCL's CTS is the grant, as with pHost/Homa/NDP grants and PicNIC's receiver-side ingress envelopes.
- It is lossless: RoCE error counters and port drops were 0 in every checked run.
- The alternatives, and why they were not used:
  - Drop-based ingress policing (the AWS style): go-back-N RoCE collapses under loss.
  - Hardware rate control at the senders driven by the receiver (EyeQ / Gatekeeper / Harmonic): see 4c.

### 4c. Hardware alternative: receiver-forged CNPs (tried, blocked)

Harmonic (NSDI'24) throttles RDMA senders by forging CNPs, which drives the
senders' DCQCN rate limiters. A PoC (`cnp_send.py`, scapy RoCE CNP from node1's
PF with vn3's addresses, to vn2's QPs paired with vn3's):
- 119,836 CNPs sent; node0 `rp_cnp_handled` +119,836, `rp_cnp_ignored` 0, so the NIC accepts them.
- The ring hop vn2 -> vn3 did not slow (73.2 ms per call before, during and after).
- It also did not slow with `cc_params/rp_rate_to_set_on_first_cnp = rp_max_rate = 5000`.
- Nor at the reduction-monitor period: a C raw-socket sender (`cnp_blast.c`) pushed 1.05 M CNPs in 3 s (350 k/s, one per QP every ~6 us); all were handled, none ignored, and the ring still ran at 73.0 ms.

Cause: firmware runs the non-legacy CC algorithm (`ROCE_CC_LEGACY_DCQCN=False`,
`mstconfig -d 81:00.1 q`). The debugfs `cc_params` are the legacy DCQCN knobs,
and `USER_PROGRAMMABLE_CC=False`.

Path, if wanted: `mstconfig set ROCE_CC_LEGACY_DCQCN=1` on both BF2s plus a
portal power cycle, then a controller in the straggler's host that meters its
VF's RX and sends CNPs. That is not doable unattended; the credit gate stays.
