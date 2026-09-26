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

## 5. Segment pipelining inside a channel (branch `optcc-skew`)

**Problem.** A channel ran its segments strictly one after another:
- a healthy rank could not start segment L+1's reduce-scatter until segment L's allgather had finished;
- that allgather itself waited for the straggler's answer.

With few, large segments the straggler link idles for about one segment per
loop. This is the paper's (k+1)/k head and tail repeated on every segment
instead of once per collective. Fixed-k runs (Task 3, `NCCL_BUFFSIZE = n/(2k)`)
showed it: k = 8 on 4 channels took 66 ms at 64 MiB, 1.45x the straggler's
link bound, where the paper's schedule gives 1.125x.

**Change.** Each segment is split where it waits on the other side of the
straggler link:

| Role, ordering | front(L) | back(L) |
|---|---|---|
| healthy, 1 | RS + send partial to S | recv sum from S + AG |
| healthy, 2 | fold S's raw section + RS | AG + send sum to S |
| straggler, 1 | the fused `recvReduceCopySend` per section | nothing |
| straggler, 2 | send raw sections | recv sums |

The issue order is `front(0); for L>=1: front(L), back(L-1); back(last)`. This
is the paper's composite schedule, in which one segment's straggler exchange
hides behind the next segment's healthy-ring work.

**Why it is safe.**
- Every rank issues the same sequence, so each connection still carries one fixed order of chunks. The ring, for example, sees A(0) A(1) D(0) A(2) D(1) ...
- Per-connection step counts are unchanged: the healthy ordering-1 link stage becomes a send-only and a recv-only primitive, same steps.
- At most 2 chunks are ever outstanding on a connection, which is its FIFO:
  - The straggler sends raw(L+1) only after all sums of L-1 arrived, and every starter consumed raw(L) before sending them.
  - A healthy rank sends partial(L+1) only after the straggler answered L-1.
- Off switch: `optcc-kernel` without this commit.

**Status (2026-09-25 04:15).**
- Data check passes: nccl-tests out-of-bounds 0 at 8-128 MiB and 12/24/48/96 MiB, for 4 x 512K, 8 x 128K and 4 x 4M.
- Speed-up at l = 2, 128 MiB, interleaved with the original (4 rounds):

  | Configuration | Change |
  |---|---|
  | k = 8 on 4 channels | -14% |
  | k = 16 on 4 channels | -7% |
  | 4 x 512K | -1.4% |
  | 8 x 128K | 0% (already at the bound's edge) |
  | one segment per channel | 0% (nothing to overlap) |

- Hangs: 2 of about 250 runs, both k = 16 on 4 channels at 8 MiB. Neither reproduced in 120 dedicated attempts, back to back or alternating with another configuration.
- The original kernel also hung once tonight, in about 810 runs (6 x 128K, 16 MiB), with the same symptom: nothing after the NCCL banner.
- So the hang may be the residual first-collective race rather than this reordering. Root cause open; not merged.

## 6. Pattern stagger (branch `optcc-stagger`, experiment)

**Why.** With one segment per channel (the paper's k = 4 on 4 channels), all four channels used to start together:
- the two ordering-1 channels (A, C) run their reduce-scatter at the same time, then hit the straggler at the same time;
- B and D do the same.

The paper instead starts C and D one "body" after A and B (Fig. 6). A body is the time for the straggler to move one segment's four sections, `l * n/k` at the healthy rate.

**Knob.**
- `NCCL_OPTCC_STAGGER_NS` (default 0) stored in `ncclOptccRing::staggerNs`.
- Every thread of a channel in the upper half of `[channelLo, channelHi]` spins on `%globaltimer` for that long before `front(0)`.
- No barrier, so there is no divergence problem.
- The data check passes with it on (0.5 ms, 4 x 256K, 8-128 MiB).

**First results.** 128 MiB, 5 reps, pipelined kernel, time / straggler link bound:

| k (segments), 4 channels | No stagger | Stagger one body | Paper (k+1)/k |
|---|---|---|---|
| k = 4 | 1.50 | 1.31 | 1.25 |
| k = 8 | 1.37 | 1.14 | 1.125 |

- Fine segments (4 x 256K): no effect, as expected, since the offset there is a fraction of one of many segments.
- Missing piece: the body length depends on n and the rates. A real implementation would derive it, or synchronise C on A's progress instead of a timer.

## 7. Straggler flow arbiter (branch `optcc-serial`, experiment)

**Why.** The paper lets a NIC carry one flow at a time. OptccRing's channels are independent CTAs, and the straggler posts receive buffers for every healthy peer up front. So the straggler's NIC serves several peers at once: in about half of the paper's time slots it exchanges data with two or more different healthy ranks (profiler, `scripts/prof_overlap.py`). A symmetric ring never does (0%).

**Mechanism** (`src/transport/net.cc`, straggler's proxy only, `NCCL_OPTCC_SERIAL`: bit 0 receive, bit 1 send, default 0).
- One token per direction. The straggler may post receive buffers (Stage 2), or issue sends (Stage 3), only for the (channel, healthy peer) section that holds the token. A section is one chunk, `chunkSteps` steps.
- The token passes when the section completes: on receive when all its steps have landed, on send when all its sends have completed.
- Orders, taken from the paper's schedule (Fig. 6):
  - receive, per loop: ordering-1 channels (A, C), then ordering-2 channels (B, D);
  - send, per loop: ordering-2 channels (B, D), then ordering-1 channels (A, C);
  - within a channel, the kernel's section order: healthy index 0..nh-1; an ordering-2 raw send goes to healthy[i+1].
- A flow in either order only depends on flows earlier in the two orders:
  - A's results need A's receives;
  - B's results need B's raw sends and the healthy ring;
  - B's next raw section comes after C's results.

  So the orders cannot deadlock. A flow whose data is late holds its link idle (a head-of-line wait).
- Collectives are served in opCount order. An op starts only when all `nChannels x (nRanks - 1)` subs are registered, because the straggler's subs of one collective reach the proxy in several ProxyArgs.
- Safety valve: if the current section makes no progress for `NCCL_OPTCC_SERIAL_VALVE_MS` (default 5000), the arbiter warns and switches itself off for the rest of the process.

**Bug found on the way.** The first version started an op from the subs registered so far. It served one section, took the op for finished, and held the other 15 sections forever, so the next collective stalled. The valve caught it ("1 sections served").

**Checks.**
- nccl-tests `-c 1` (varied data), 8-128 MiB, l = 2: 0 wrong for 4 x 8 MiB, 4 x 256K and 8 x 128K buffers, strict (3) and lookahead (7), on both this pipelined kernel and the original kernel (`optcc-serial-orig`, same net.cc).
- The valve never fired after the registration fix.
- The straggler really serves one peer at a time. On 4 x 256K, receive time with two or more peers on the wire at once falls from 9% to 0 (`prof_overlap.py`, instant share).

**Results.** 64 MiB profiles (straggler span, bound 45.69 ms):

| | off | strict | lookahead |
|---|---|---|---|
| original kernel, 4 x 256K | 48.48 +- 0.97 | 46.45 +- 0.10 | 46.32 +- 0.10 |
| pipelined kernel, 4 x 256K | 46.85 +- 0.39 | 63.63 +- 0.34 | 64.43 +- 1.34 |
| original kernel, k = 4 | 56.94 +- 1.42 | 84.18 +- 2.38 | 70.23 +- 4.96 |

**The kernel has to be the original one.**
- The original kernel runs a channel's segments strictly in turn, like the paper's patterns.
- In the pipelined kernel, an ordering-2 channel's results for segment j come after its front(j+1), which needs the straggler's raw data of segment j+1.
- The receive order puts B_j before A_{j+1}. So A waits with its data ready, and the link idles about 17 ms per direction at 64 MiB.

**Timing** (4 rounds, original kernel, time / ring l = 1 of the same round):
- 4 x 256K at 128 MiB:
  - off 1.294 +- 0.019;
  - strict 1.258 +- 0.003;
  - lookahead 1.255 +- 0.005, i.e. 1.009 x the link bound;
  - both p < 0.01 against off.
- 4 x 256K at 64 MiB: 1.307 -> 1.264 / 1.256.
- 8 x 128K at 128 MiB: 1.278 -> 1.264.
- Coarse segments got slower (k = 4 at 128 MiB: 1.93 -> 2.21, p = 0.03): a late 4 MiB section idles the link for long.
- The "off" numbers there profit from the pre-posting leak of the RX cap (DESIGN 4, T3b page). Serial mode posts one section at a time, which closes most of it.
- 16 MiB: noisy, no gain; 8 x 128K strict is 15-27% slower (p about 0.06).
- Lookahead helps only a little (the hand-over round trip is small next to head-of-line waits).

**Status.** Correct and deadlock-free in every run so far (156 timed runs, no hang). Useful for 4 x 256K at 64 MiB and above; off by default.

## 8. RX cap: pace the hand-over to the GPU, not the posting (fix to 1 and 4)

**Leak.** Entry 1 charged the bucket when an irecv was posted. Posting goes on while no data flows, so an idle receiver banks credit as outstanding receive buffers.
- In the k = 4 runs (one segment per channel), the straggler waits about 17 ms for the first partial sums. In that time it posted 24-26 MiB of buffers.
- The healthy senders then wrote into them at line rate: 14 MiB in the first 5 ms after the first byte, 23.5 Gbit/s against an 11.75 cap.
- The staggered k = 4 run finished in 58.4 ms, below its own faithful bound of 62.4 ms (idle head + 64 MiB at the cap).
- The shallow bucket of entry 4 does not help, because the credit sits in posted buffers, not in the bucket.
- Fine segments are barely affected: their head is about 0.5 ms, so about 1 MiB gets pre-posted.

**Fix** (`NCCL_NET_RX_PACE_DELIVER`, default 1; 0 restores entry 1).
- Buffers are posted as upstream does.
- A landed step is held from the GPU (recvTail) until the bucket has its bytes. The bucket keeps the same rate and depth: `NCCL_NET_RX_BURST_BYTES`, at least one step.
- Data may still land early at line rate, but the kernel sees it at the cap. Idle time can bank at most the bucket depth.
- A completed flush request is cleared at the hand-over, so it is never tested twice while the step waits.
- The OptCC arbiter (entry 7) counts a receive section as done at the hand-over.

**Why at the hand-over.** A real half-speed port backpressures the senders on the wire. The rig can only withhold credit, and credit can pile up while nothing flows. Pacing when data becomes usable is the closest observable: the receiver's kernel cannot run ahead of the cap. The senders' writes do complete earlier than on a real port, so a sender's own FIFO frees up sooner. It still cannot get further ahead than the receiver's posted buffers (4 slices per connection).

**Profiler.** A receive step now has three times: landing (`RecvWait` end), hand-over (`RecvGpuWait` begin), and GPU done. `prof_timeline.py` counts a receive at the hand-over.
