"""MoHex 2.0 (benzene-vanilla-cmake) as a GTP subprocess: the evaluation opponent and 0-Elo anchor.

Jones (2021) fixes MoHex's play at 0 Elo on boards up to 9x9, where it is close to perfect, and runs it
through boardlaw's `MoHexAgent()`: default search plus the parallel DFPN solver. On this 13x13 board the
same program is a strong reference, far from perfect play. MoHex's own
conventions match the engine's: black moves first and joins north to south, swap is
off unless enabled, and a cell is written column letter then 1-based row (`e5`).

A level is a list of `param_*` configuration lines. Search is bounded by simulated
games (`max_games`) with a generous wall-clock backstop, so a level's strength does
not depend on how busy the evaluation machine is. Levels are defined in
`simulators/hex/autoresearch/contract.py`.
"""
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time

from .engine import BLACK, SIZE, cell_name, parse_cell

COLOUR = {BLACK: 'b', 2: 'w'}


class MoHexFault(RuntimeError):
    """MoHex died, timed out, refused a command or answered something that is not a legal cell."""


def locate(binary=None):
    """The MoHex binary: an explicit path, $HEX_MOHEX, the worker image's /opt/mohex/mohex, or `mohex` on PATH."""
    for candidate in (binary, os.environ.get('HEX_MOHEX'), '/opt/mohex/mohex'):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    from shutil import which
    found = which('mohex')
    if not found:
        raise FileNotFoundError('MoHex binary not found; build vendor/engines/benzene (simulators/hex/build_mohex.sh)')
    return found


class MoHex:
    def __init__(self, params=(), seed=1, binary=None, timeout=600.0):
        self.timeout = float(timeout)
        self._config = tempfile.NamedTemporaryFile('w', prefix='mohex-config-', suffix='.htp', delete=False)
        self._config.write('\n'.join(params) + '\n')
        self._config.close()
        self._stderr = tempfile.TemporaryFile()
        command = [locate(binary), '--use-logfile=0', f'--config={self._config.name}', f'--seed={int(seed)}']
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self._stderr, start_new_session=True)
        self._buffer = b''
        self.query(f'boardsize {SIZE} {SIZE}')

    def _line(self, deadline):
        while b'\n' not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MoHexFault('MoHex did not answer before its deadline')
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise MoHexFault(f'MoHex exited (code {self.process.poll()}): {self.stderr_tail()}')
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b'\n', 1)
        return line.decode(errors='replace').rstrip('\r')

    def query(self, command, timeout=None):
        """Send one GTP command; return the answer text without the leading '='."""
        if self.process.poll() is not None:
            raise MoHexFault('MoHex is not running')
        try:
            self.process.stdin.write((command + '\n').encode())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MoHexFault(f'MoHex pipe closed: {exc}') from exc
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        lines = []
        while True:
            line = self._line(deadline)
            if line == '' and lines:
                break
            if line == '' and not lines:
                continue
            lines.append(line)
        text = '\n'.join(lines)
        if not text.startswith('='):
            raise MoHexFault(f'MoHex refused {command!r}: {text}')
        return text[1:].strip()

    def new_game(self):
        self.query('clear_board')

    def play(self, colour, cell):
        self.query(f'play {COLOUR[colour]} {cell_name(cell)}')

    def genmove(self, colour):
        """MoHex chooses and plays a move for `colour`; returns its cell."""
        answer = self.query(f'genmove {COLOUR[colour]}').lower()
        try:
            return parse_cell(answer)
        except (ValueError, IndexError):
            raise MoHexFault(f'MoHex answered {answer!r} instead of a cell') from None

    def stderr_tail(self, limit=2000):
        self._stderr.seek(0)
        return self._stderr.read()[-limit:].decode(errors='replace')

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b'quit\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
                self.process.wait()
        self._stderr.close()
        Path(self._config.name).unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
