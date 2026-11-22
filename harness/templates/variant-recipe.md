## What is graded: your recipe

This is a **recipe** session. Your answer is the source and configuration you
leave behind, not any network you trained here.

- `ar submit` registers the current contents of `training/` and
  `configs/train.json` (plus any other declared config) as your recipe. You may
  submit again; the latest submission before `ar finalize` is final. If the
  session ends with nothing submitted, the host registers the files of your
  newest complete `configs/train.json` run instead.
- After the session closes, the host trains your final recipe **from scratch** on
  a fresh worker for {final_train} with `configs/train.json`, then evaluates the
  resulting network itself. Networks trained during this session are development
  evidence only and are discarded.
- Your recipe must therefore train reliably within that fresh budget on its own:
  no dependence on checkpoints, caches or files from this session, and no manual
  intervention. A recipe that crashes or produces no policy scores as a failure.
- Use your research allocation of {research} to find out what trains best within
  a {final_train} fresh run, including at least one end-to-end run of the exact
  recipe you intend to submit.
