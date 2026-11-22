# Doudizhu fixed-role research v2

`benchmark.json` is the versioned discoverable task declaration. `contract.py`
provides enforced runtime limits; release manifests bind the supplied runtime
and public instructions. Rules/source/deal locks and historical records are unchanged.

The task has all 54 cards, three fixed roles and 20/17/17-card hands. No bidding,
spring or bid multiplier. Bombs/rocket affect secondary economic scores. Natural
empty-hand outcomes alone supply terminal win/loss targets. The 162-decision
guard is an unscored simulator fault. Public-only static/history arrays and the
complete sorted ragged legal action surface remain frozen. GPU scratch overflow
is a fault; candidate chunking never clips legal actions.

Every worker, training or practice, is exactly one L40S, four physical CPU cores
and 16 GiB. A session runs in one of two budget modes, recorded in its ledger and
never changed on resume. Seconds mode (the default): the research allowance is
7200 allocated worker seconds, one active worker at a time. Dollar mode: the
allowance is $5.81 of worker time at the frozen list price ($0.0008064 per worker
second, the same 7200 seconds), with up to four jobs and two GPUs at once (so two
workers), each queued or running job holding its whole grant until it ends. The
separately authorized smoke session has a cumulative 600-second ($0.48) cap, with
a 20-second GPU teardown reserve excluded from new grants. Failed allocations
count. No warm pool or automatic GPU retry. Host provider timestamps settle
charges; unresolved leases retain holds and count against the concurrency limit. CPU agent
editing and isolated CPU grading use no attached GPU and are reported separately.
Supplied engine image build occurs before GPU allocation; candidate compilation
and GPU initialization count. Preparation alone provisions no worker.

Starter: random 128x3/LSTM64 three-role DMC with RMSprop, terminal WP targets,
pure self-play, no replay/Adam/population/scheduling. Agent may replace only
policy.py, learner.py, train.py and recipes. Network state <=500,000 tensor elements
and 128 MiB. Exact source, architecture, tensor checkpoint, captured recipe/seed
and frozen runtime digest are bound into immutable submission identity.

Primary raw score is 100 times equal-weight role/opponent win probability.
Each deal is replayed with the candidate separately controlling each role; the
panel controls both other seats, including peasant teammate. This does not
measure joint peasant-team control. Report per-role/opponent win rates, complete-
deal bootstrap intervals and separate per-role economic scores. Ordinary full
practice uses 32 deals; private grade uses 256. Campaign practice uses 8; its
independent private smoke uses 16 new deals. Small panels do not calibrate difficulty.
Reference-normalized reward remains null until equal-budget three-seed anchor
calibration qualifies. Illegal/malformed actions, policy timeouts, simulator
faults, budget-aborted waves and incomplete schedules are distinct failures,
never ordinary natural losses or partially scored schedules.

Host-private seeds/panel and engine live in a privileged Linux parent; candidate
inference runs as another UID with no network and framed public arrays only.
Submissions at one-third/two-thirds allocation clock remain immutable. Final
uses the latest explicit valid submission; active/unresolved jobs prevent finalization.
A session directory is never reopened with a new allowance; reopening one for
host grading keeps the same ledger and settles interrupted leases from provider receipts.
Representative exact checkpoint continuation is tested in the frozen SDK;
the minimal editable starter intentionally starts fresh on each registered run.
