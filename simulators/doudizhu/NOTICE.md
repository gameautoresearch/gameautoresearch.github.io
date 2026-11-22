This integration uses the unchanged DouZero CPU game engine and observation/action
encoding, and adapts its three-role LSTM/MLP architecture and Deep Monte Carlo loss.

Upstream: Daochen Zha et al., Kwai Inc., https://github.com/kwai/DouZero,
commit `718a5c920bf3361e34178a38f3b80458e176b351`.
The upstream Apache-2.0 license is preserved in `LICENSE-DouZero` and in the checkout.
See the paper, “DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement
Learning,” ICML 2021, https://proceedings.mlr.press/v139/zha21a.html.

Integration changes: deterministic domain-separated dealing; public-only observation
objects; canonical legal actions; once-per-observation history inference; bounded
candidate transfers; persistent CPU worker option; safe checkpoint/resume; explicit
frozen/history mixture; paired evaluation; full allocation-time metering and manifests.

`native/engine.cpp` is a new C++ port derived from the pinned DouZero game's
transition, move generator, detector, selector and public codec implementations.
It preserves their Apache-2.0 attribution and license in `LICENSE-DouZero`.
The unchanged Python upstream remains its independent differential rules oracle.
`cuda/rules.h` ports the same complete rules and public codec to a shared C++/CUDA
implementation; `cuda/engine.cu` provides device batching, canonical sorting,
public encoding, legal selection and trajectory storage. The pinned Apache-2.0
license and authorship above apply to these derived rule implementations as well.
The direct public encoder, packed worker protocol, bounded CUDA inference graphs,
native thread pool and array-based DMC rollout/learner input are integration changes.

No upstream pretrained weights, human-play data, RLCard data files, or pickled evaluation
datasets are included. This notice does not establish a license for external weights.
