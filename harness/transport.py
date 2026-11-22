"""Transport to the researcher's box. The box has no network: the host reaches in.

Harbor creates the box with its own Modal client, which is newer than the
client each game pins for its workers, and an older client cannot see boxes a
newer one creates. So in a live session the broker reaches the box through a
small bridge process run by Harbor's interpreter (`python -m harness.transport`),
speaking one JSON line per call.
"""
import json
from pathlib import Path
import re
import subprocess
import sys

REQUEST = re.compile(r'[0-9a-f]{32}\.json')
LIMIT = 6 * 1024 * 1024
LIST = ("import json,pathlib\nout={}\nfor p in sorted(pathlib.Path('/ar/requests').glob('*.json'))[:16]:\n"
        " out[p.name]=p.read_text() if p.stat().st_size<=%d else ''\nprint(json.dumps(out))" % LIMIT)
# The researcher's files as they are in the box right now (a pause keeps them): text files up to SNAPSHOT_LIMIT.
SNAPSHOT_LIMIT = 1024 * 1024
SNAPSHOT = ("import json,pathlib,sys\nroot=pathlib.Path('/workspace')\nout={}\nfor name in json.loads(sys.argv[1]):\n"
            " p=root/name\n try:\n  if p.is_file() and not p.is_symlink() and p.stat().st_size<=%d: out[name]=p.read_text()\n"
            " except Exception: pass\nprint(json.dumps(out))" % SNAPSHOT_LIMIT)
REPLY = ("import pathlib,sys\nname=sys.argv[1]\nt=pathlib.Path('/ar/replies')/(name+'.tmp')\nt.write_text(sys.stdin.read())\n"
         "t.chmod(0o444)\nt.replace(pathlib.Path('/ar/replies')/name)\npathlib.Path('/ar/requests',name).unlink(missing_ok=True)")


class LocalBox:
    """A directory standing in for /ar: tests and offline rehearsals."""

    def __init__(self, root):
        self.root = Path(root)
        for name in ('requests', 'replies'):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def pending(self):
        return {p.name: p.read_text() for p in sorted((self.root / 'requests').glob('*.json'))}

    def reply(self, name, value):
        temporary = self.root / 'replies' / (name + '.tmp')
        temporary.write_text(json.dumps(value))
        temporary.replace(self.root / 'replies' / name)
        (self.root / 'requests' / name).unlink(missing_ok=True)

    def alive(self):
        return True

    def terminate(self):
        return False

    def snapshot(self, names):
        """{name: text} of the workspace files that exist (the workspace stands beside this /ar directory)."""
        out = {}
        for name in names:
            path = self.root.parent / 'workspace' / name
            if path.is_file() and not path.is_symlink() and path.stat().st_size <= SNAPSHOT_LIMIT:
                try:
                    out[name] = path.read_text()
                except (OSError, UnicodeDecodeError):
                    pass
        return out


class ModalBox:
    """The Harbor-created Modal sandbox, found by the session tag Harbor attached to it."""

    def __init__(self, sandbox):
        self.sandbox = sandbox

    @classmethod
    def find(cls, app_name, session, environment):
        import modal
        try:
            app = modal.App.lookup(app_name, environment_name=environment, create_if_missing=False)
        except Exception:
            return None  # Harbor has not created its app yet
        live = [sandbox for sandbox in modal.Sandbox.list(app_id=app.app_id, tags={'ar_session': session})
                if sandbox.poll() is None]
        if len(live) > 1:  # never guess: serving or terminating another session's box corrupts both sessions
            raise RuntimeError(f'{len(live)} live boxes carry the session tag {session!r}')
        return cls(live[0]) if live else None

    def _exec(self, code, *args, stdin=None):
        process = self.sandbox.exec('python3', '-I', '-c', code, *args, timeout=60)
        if stdin is not None:
            data = stdin.encode()
            for start in range(0, len(data), 1 << 20):  # one large write overflows the stdin buffer
                process.stdin.write(data[start:start + (1 << 20)])
                process.stdin.drain()
            process.stdin.write_eof()
            process.stdin.drain()
        output = process.stdout.read()
        if process.wait():
            raise RuntimeError('Box command failed: ' + process.stderr.read()[-500:])
        return output

    def pending(self):
        return json.loads(self._exec(LIST) or '{}')

    def reply(self, name, value):
        self._exec(REPLY, name, stdin=json.dumps(value))

    def alive(self):
        return self.sandbox.poll() is None

    def snapshot(self, names):
        return json.loads(self._exec(SNAPSHOT, json.dumps(list(names))) or '{}')

    def terminate(self):
        if self.sandbox.poll() is None:
            self.sandbox.terminate()
            return True
        return False


class Bridge:
    """The box as seen through a bridge subprocess. Same interface as the boxes above."""

    def __init__(self, python, arguments, env=None, cwd=None):
        self.process = subprocess.Popen([str(python), '-m', 'harness.transport', *arguments], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, env=env, cwd=cwd)

    def _call(self, **message):
        if self.process.poll() is not None:
            raise RuntimeError('Box bridge exited')
        self.process.stdin.write(json.dumps(message) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Box bridge closed its output')
        reply = json.loads(line)
        if 'error' in reply:
            raise RuntimeError('Box bridge: ' + reply['error'])
        return reply['result']

    def find(self):
        """This bridge once the box exists, else None."""
        return self if self._call(call='find') else None

    def pending(self):
        return self._call(call='pending')

    def reply(self, name, value):
        self._call(call='reply', name=name, value=value)

    def alive(self):
        return bool(self._call(call='alive'))

    def snapshot(self, names):
        return self._call(call='snapshot', names=list(names))

    def terminate(self):
        """Stop the box if Harbor left it running. No box outlives its session."""
        return bool(self._call(call='find')) and bool(self._call(call='terminate'))

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def bridge_main(argv=None):
    """Serve find/pending/reply/alive/snapshot/terminate for one session's box over stdin/stdout."""
    import argparse
    parser = argparse.ArgumentParser(description=bridge_main.__doc__)
    parser.add_argument('--app')
    parser.add_argument('--session')
    parser.add_argument('--environment')
    parser.add_argument('--local', help='a directory standing in for /ar (tests)')
    args = parser.parse_args(argv)
    box = LocalBox(args.local) if args.local else None
    out = sys.stdout
    sys.stdout = sys.stderr  # nothing but replies may reach the pipe
    for line in sys.stdin:
        try:
            message = json.loads(line)
            call = message['call']
            if call == 'find':
                if box is None or not box.alive():
                    box = ModalBox.find(args.app, args.session, args.environment)
                result = box is not None
            elif box is None:
                raise RuntimeError('No box yet')
            elif call == 'pending':
                result = box.pending()
            elif call == 'reply':
                result = box.reply(message['name'], message['value'])
            elif call == 'alive':
                result = box.alive()
            elif call == 'snapshot':
                result = box.snapshot(message['names'])
            elif call == 'terminate':
                result = box.terminate()
            else:
                raise ValueError('Unknown call')
            reply = {'result': result}
        except Exception as exc:
            reply = {'error': type(exc).__name__ + ': ' + str(exc)[:400]}
        out.write(json.dumps(reply) + '\n')
        out.flush()


def serve(box, broker):
    """Answer every pending request once. Returns how many were handled."""
    handled = 0
    for name, text in box.pending().items():
        if not REQUEST.fullmatch(name):
            continue
        try:
            if not text:
                raise ValueError('Request exceeds the size limit')
            request = json.loads(text)
        except ValueError as exc:
            reply = {'error': 'Malformed request: ' + str(exc)[:200]}
        else:
            reply = broker.handle(request)
        box.reply(name, reply)
        handled += 1
    return handled


if __name__ == '__main__':
    bridge_main()
