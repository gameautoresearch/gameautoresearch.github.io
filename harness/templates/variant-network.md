## What is graded: your network

This is a **network** session. Your answer is a trained network produced by one
of your own runs in this session.

- `ar submit NAME` registers the completed run `NAME` as a submission. A
  submission binds that run's exact source, effective configuration and weights
  and is immutable. You may submit several runs; the latest submission before
  `ar finalize` is final.
- All training happens inside your research allocation of {research}. There is
  no later retraining: the host evaluates the submitted network itself after the
  session closes.
- Only a registered, completed run can be submitted. An unfinished or unsubmitted
  run never replaces your latest submission, so submit a tested network early and
  replace it when you have better evidence. If the session ends with nothing
  submitted, the host registers your newest complete `configs/train.json` run.
