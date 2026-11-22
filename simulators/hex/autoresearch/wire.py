"""Length-prefixed public arrays: boards, side to move and legal masks. No pickle, seeds or engine handles."""
import io
import json
import struct

import numpy as np

CELLS = 169  # 13 x 13; kept beside the wire so this module imports nothing of the engine
MAX_BYTES = 4 * 1024 * 1024


def encode(board, to_play, legal):
    board = np.asarray(board, dtype=np.int8)
    to_play = np.asarray(to_play, dtype=np.int8)
    legal = np.asarray(legal, dtype=bool)
    f = io.BytesIO()
    np.savez(f, board=board, to_play=to_play, legal=legal)
    value = f.getvalue()
    if len(value) > MAX_BYTES:
        raise ValueError('Public inference batch too large')
    return struct.pack('>I', len(value)) + value


def decode(value):
    if len(value) > MAX_BYTES:
        raise ValueError('Oversized request')
    with np.load(io.BytesIO(value), allow_pickle=False) as data:
        if set(data.files) != {'board', 'to_play', 'legal'}:
            raise ValueError('Unexpected public fields')
        board, to_play, legal = data['board'], data['to_play'], data['legal']
    n = len(to_play)
    if board.shape != (n, CELLS) or board.dtype != np.int8 or not np.isin(board, (0, 1, 2)).all():
        raise ValueError('Invalid board array')
    if to_play.shape != (n,) or to_play.dtype != np.int8 or not np.isin(to_play, (1, 2)).all():
        raise ValueError('Invalid side-to-move array')
    if legal.shape != (n, CELLS) or legal.dtype != bool or not legal.any(1).all():
        raise ValueError('Invalid legal mask')
    return board, to_play, legal


def read_exact(stream, n):
    parts = []
    while n:
        part = stream.read(n)
        if not part:
            raise EOFError('Policy pipe closed')
        parts.append(part)
        n -= len(part)
    return b''.join(parts)


def read_frame(stream):
    head = stream.read(4)
    if not head:
        return None
    if len(head) != 4:
        head += read_exact(stream, 4 - len(head))
    size = struct.unpack('>I', head)[0]
    if size > MAX_BYTES:
        raise ValueError('Policy message exceeds cap')
    return read_exact(stream, size)


def reply(cells):
    raw = json.dumps([int(x) for x in cells], separators=(',', ':')).encode()
    return struct.pack('>I', len(raw)) + raw
