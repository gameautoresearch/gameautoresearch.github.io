"""Researcher agents the harness can run, and the box adaptations each one needs.

Terminus-2 is the default. It drives the box over the terminal and reaches the
model through Harbor's host-side LiteLLM, so no credential enters the box and
the box stays offline. An installed CLI agent runs inside the box instead:
Claude Code is baked into the image at build time, its phase gets the egress it
needs for model calls, and Harbor delivers the model key into the box.

A game opts in through `agents` in its SPEC. A game that does not say is
Terminus-2 only, so the adapter is inert everywhere else.
"""
import json

TERMINUS = 'terminus-2'
CLAUDE_CODE = 'claude-code'

ADAPTERS = {
    TERMINUS: dict(
        harbor='terminus-2',
        key_env='OPENROUTER_API_KEY',
        api_base_kwarg='api_base',
        model_info=True,
        agent_network=None,
        installs=False,
        summary='terminal researcher; the model is called on the host through Harbor',
        instruction='''## Terminal limits

Each turn you are shown at most about 10,000 bytes of new terminal output; the
middle of anything longer is cut out. Read large files in pieces and bound run
output with `ar log NAME --tail N`. To interrupt a blocking command, send `C-c` as
its own keystroke, the last of a turn, with nothing after it.''',
    ),
    CLAUDE_CODE: dict(
        harbor='claude-code',
        key_env='ANTHROPIC_API_KEY',
        api_base_kwarg='base_url',
        model_info=False,
        # The box baseline stays offline; only the agent phase reaches the network.
        agent_network='public',
        installs=True,
        summary='Claude Code CLI inside the box; the model key enters the box',
        instruction='''## Shell limits

Your shell tool stops a command that runs too long, so do not block on `ar wait`:
poll with `ar status` and `ar log NAME`, or repeat `ar wait --timeout 60`.''',
    ),
}


def allowed(spec):
    """The agents a game's SPEC enables; Terminus-2 unless the game says otherwise."""
    return tuple(spec.get('agents') or (TERMINUS,))


def get(agent=None):
    agent = agent or TERMINUS
    if agent not in ADAPTERS:
        raise ValueError(f'Unknown researcher agent: {agent}')
    return dict(ADAPTERS[agent], name=agent)


def resolve(spec, agent=None):
    """The adapter for one session, refusing an agent this game did not enable."""
    adapter = get(agent)
    if adapter['name'] not in allowed(spec):
        raise ValueError(f"{spec['game']} does not enable the {adapter['name']} agent; "
                         f"choose one of: {', '.join(allowed(spec))}")
    return adapter


def command_kwargs(adapter, *, api_base=None, reasoning_effort=None, model_info=None, extra=()):
    """Harbor `--ak` arguments, named the way this agent's option model expects."""
    kwargs = list(extra)
    if api_base:
        kwargs.append(adapter['api_base_kwarg'] + '=' + api_base)
    if reasoning_effort:
        kwargs.append('reasoning_effort=' + reasoning_effort)
    if model_info and adapter['model_info']:
        kwargs.append('model_info=' + json.dumps(model_info))
    if adapter['name'] == CLAUDE_CODE:
        # The agent phase has egress for model calls; the researcher still must
        # not reach the open web for information.
        kwargs.append('disable_web_search=true')
    return kwargs


# Installed at build time, as the researcher, so agent setup needs no network.
# Harbor finds `$HOME/.local/bin/claude` and skips its own install; its run
# command already puts that directory on PATH.
CLAUDE_CODE_INSTALL = '''
# Claude Code is baked in at build time so the runtime box needs no network for
# agent setup. It installs into the researcher's home, where Harbor looks for it.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \\
    && rm -rf /var/lib/apt/lists/* \\
    && runuser -u researcher -- env HOME=/home/researcher bash -c 'curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh | bash -s -- stable' \\
    && runuser -u researcher -- env HOME=/home/researcher bash -c '"$HOME/.local/bin/claude" --version'
'''


def box_install(adapter):
    """The Dockerfile layer an installed agent needs, or an empty string."""
    return CLAUDE_CODE_INSTALL if adapter['installs'] else ''
