# Clash evaluator v2 supplement

This version preserves `clash-fixed8-v1`: fixed eight-card level-11 mirror deck,
4 Hz paused decisions, 7,024 public int32 values, 2,307 legal-mask entries, all
200 entity rows, 180-second regulation plus 120-second overtime and natural end
delays. Native rules/data/source pins and all historical receipts remain intact.
It does not change the weak entrant starter or import an external engine.

Entrants retain `clash-ar-bundle-v1`, finite float32 safe NPZ, at most one million
parameters, stateless per-seat neural inference and deterministic masked greedy
actions. `readiness_contract.admit()` verifies the existing binding and direct
`return model(observation,mask)[0].argmax(-1)` entry, rejects common IO/state/RNG
escapes, and requires owning source review. Official repeated-input, permutation
and single-row probability/action probes add evidence of stateless behavior;
they do not prove arbitrary Python purity. The existing unprivileged child,
no-new-privileges, network-blocked supervisor and 15-second pipe deadline remain.

Trusted opponents have a **separate** inference declaration: CPU-only public
single-seat arrays, at most 200 ms per decision, no engine forks/search, no enemy
hand/RNG access. Teacher/intermediate scripts are exempt from entrant neural
parameter rules, not from visibility, masks, resources, timing or match rules.
Official CPU qualification uses eight fixed CPU cores and 32 GiB RAM. NN batch
latency is measured after admission warmup; cold import/probes remain charged in
the complete provider interval. The original qualification receipt's cold-latency
label was corrected in a supplemental note, without changing its measurements.

`visible_teacher.PublicTeacher(kind="teacher"|"intermediate", data=task_data)`
accepts only one `(7024,)` observation and `(2307,)` legal mask and returns a legal
integer. The teacher combines the pinned standard bot's economic foundation with
visible tank/air defense, tank support, lane pressure, tower finishing and spell
value. The intermediate uses the same game economy and a reduced tactical
repertoire at a declared one-second decision cadence. Source snapshots and the
eight qualification seeds are public development context in the ignored
campaign release. No private final bank has been opened for new RL selection.

The upstream method audit is pinned to
[ClashRoyaleAi 0dbccbb](https://github.com/itzik123/ClashRoyaleAi/blob/0dbccbb283f46afe33b2647f6739279e9d94f995/python_ai/opponents/teacher.py).
Its simulation-ranked recurrent-policy system has a different engine,
observation/action space and parameter count. Only tactical/curriculum ideas are
adapted; its weights and privileged engine forks are not drop-in opponents here.

Report natural wins/ties separately from administrative caps, policy deadlines,
illegal actions, visible refusal observations, simulator/unsupported faults and
infrastructure faults. An incomplete pair has no official score. The 95% interval
resamples complete paired public seed units; eight seeds provide limited
calibration. Keep per-opponent/seat points, full latency/resource accounting and
separate harness, learned-reference, strong-panel and superhuman evidence flags.
Reference normalization stays null until independently calibrated anchors exist.
