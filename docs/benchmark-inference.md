# Hero roster inference

**Historical (superseded 2026-11-11).** The model client described below,
`tools/benchmark_inference.py`, was removed with the earlier harnesses; its source is
in git history at `ba5c0d5`. Research sessions now reach models through Harbor's
Terminus-2 and LiteLLM; see [the harness guide](../harness/README.md). The roster
smoke results remain valid as a record of which model routes worked on November 9.

All ten selected models passed a real read-only tool call plus a nonce-based
continuation on November 9. The check recreates each client before the second
request, verifying native reasoning state survives a controller restart.
See [the readout](../reports/benchmark-inference-smoke-2026-11-09.md) and
[machine-readable receipts](../reports/benchmark-inference-smoke-2026-11-09.json).

`tools/benchmark_inference.py` is the shared trusted-controller client:

- OpenAI uses native `/v1/responses`, retaining complete output items, phases and
  encrypted reasoning. Anthropic uses native `/v1/messages`, adaptive thinking
  and fixed `output_config.effort`, retaining signed thinking blocks.
- OpenRouter uses Chat Completions with reasoning retained. The selected endpoint
  tag is pinned with `only`, `allow_fallbacks=false`, `require_parameters=true`
  and price ceilings. No model substitution or automatic inference POST retry.
- Research calls have a **65,536 total output-token ceiling**, including
  reasoning, and a 600-second absolute HTTP deadline bounded by the persistent
  eight-hour research deadline. The smoke used 8,192 tokens to keep checks tiny.
  This ceiling is not a request to generate that many tokens each time.
- Every request gets a durable SQLite reservation. Paid calls include a 5.5%
  credit-fee allowance; unknown requests retain their full hold. Admission checks
  both $15 per model/game and the shared $450 inference bucket atomically.
  A quote breach blocks further requests. Qualification costs reduce the named
  final cell's remaining allowance; they are not additional funds.
- Native responses live in private host-side state files (mode 0600), matched to
  retained transcript turn IDs. A research attempt cannot change model, tools,
  output limit, state location or budget on resume, or renew a lost deadline.
- Credentials stay in the trusted controller, then the HTTP child's stdin.
  No OAuth files, management keys, client keys or gateway URLs enter game tools.
  HTTP errors never echo provider response bodies into user-facing exceptions.

## Use

From the repository root, these commands never allocate a GPU:

```sh
python3 tools/benchmark_rollout.py           # offline routing and 60 driver bindings
python3 tools/benchmark_rollout.py --catalog # read-only access/endpoint/price metadata
python3 tools/benchmark_rollout.py --status  # persistent inference spending/request summary
python3 -m unittest tools.tests.test_benchmark_inference -q
```

The reviewed bindings are saved in
[benchmark-inference-bindings-v1.json](benchmark-inference-bindings-v1.json).
The game-specific `run(...)` functions (Polytopia: `campaign.session.research`)
accept an optional `inference` argument. For example, the Generals host driver
receives `model='gpt-6.1-sol', model_dollar_cap=15,
inference={'model_id':'sol61'}`. Pokémon/Polytopia use `model_id`; Clash,
Pokémon and Polytopia use `model_cap`. Each game still owns its tool schemas,
workspace, controller and worker allocation rules. Historical launch commands
keep their historical inference defaults unless this opt-in binding is passed.
Do not mix hero inference with historical model/cash campaign adapters.

`--live --models <ids> --out <fresh-directory>` is an explicit serving check:
maximum two calls per selected model and $1 aggregate smoke cash. It only reads
a toy budget tool; it cannot execute model-generated code, train, allocate a
worker, or grade. The completed check does not need to be rerun. `--repair` is a
bounded explicit diagnostic option for a repaired failed row, allowing at most
four lifetime requests per model and 30 aggregate, retaining all old rows and
the same $1 cash envelope. This is not a researcher retry or allowance reset.

## Host setup

Run the trusted controller on the Mac hosting CLIProxyAPI. Its default client
base is `http://127.0.0.1:8317/v1`; no public tunnel is needed. Remote Modal game
sandboxes only execute tools and experiments. A controller running inside Modal
cannot reach Mac loopback: that configuration is rejected.

Proxy credentials come from `CLIPROXY_API_KEY` or the existing owner-controlled
`~/.cli-proxy-api/client-keys.json` (`CLIPROXY_KEY_FILE` overrides the path).
OpenRouter uses `OPENROUTER_API_KEY` or an owner-only
`~/.config/openrouter/api-key` (`OPENROUTER_KEY_FILE` overrides the path).
Neither provider can fall back to a direct paid OpenAI/Anthropic API.

During qualification, the service's direct ChatGPT route timed out, and Claude's
saved token was expired. Using the existing macOS outbound proxy repaired the
route and allowed the Claude token to refresh; all ten models then passed.
This local service setting and its rollback receipt remain outside the repo.
Browser consent alone is not proof that the subsequent OAuth token exchange
succeeded. If reconnection is later necessary and port 3000 is occupied, use
`cliproxyapi -claude-login -oauth-callback-port 54545`.

Hunyuan's highest native mode is `high`, per
[Tencent's documentation](https://cloud.tencent.com/document/product/1823/131208).
Native reasoning retention follows the
[OpenAI reasoning guide](https://developers.openai.com/api/docs/guides/reasoning),
[Claude thinking guide](https://platform.claude.com/docs/en/build-with-claude/thinking)
and [OpenRouter reasoning guide](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

The inference layer is ready. Sustained subscription quota, qualified from-scratch
game releases, grading runtimes, and compute/global $2,000 budget enforcement
remain separate hero-launch gates. No campaign was launched by this work.
