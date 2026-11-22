"""Batch the frozen visible scripts without changing their opponent choices.

The supplied immutable script module remains the oracle and metadata source.
Pending choices use that oracle directly. Ordinary recruit actions are evaluated
as NumPy batches, including all position/duplicate/tribe and action-mask rules.
No native game state or hidden information is accessed.
"""
import numpy as np


def minion_values(rows):
    attack = np.maximum(0., rows[..., 1] * 10)
    health = np.maximum(0., rows[..., 2] * 10)
    # Preserve the scalar script's float32 arithmetic and accumulation order.
    value = .9 * attack + health + .25 * np.sqrt(attack * health) + rows[..., 4] * 6 + rows[..., 5] * 4
    value += (5 * rows[..., 9] + 4 * rows[..., 11] + 4 * rows[..., 12]
              + 6 * rows[..., 13] + 3 * rows[..., 10] + 6 * rows[..., 17])
    return np.where(rows[..., 0] != 0, value, 0.).astype(np.float64)


def priorities(rows):
    return (rows[..., 1] * 10 + 18 * rows[..., 17] + 12 * rows[..., 12]
            + 7 * rows[..., 9] + 4 * rows[..., 10] - 2 * rows[..., 8]).astype(np.float64)


class BatchedVisibleScript:
    def __init__(self, oracle, style):
        if style not in oracle.STYLES:
            raise ValueError(style)
        self.oracle, self.style = oracle, style
        self.metadata = None

    def tables(self):
        if self.metadata is None:
            o = self.oracle
            size = max(o.catalog()) + 1
            words = [('end of your turn', 'whenever', 'battlecry', 'rally', 'deathrattle'),
                     ('gold', 'refresh', 'tavern spell', 'sell'),
                     ('give', 'gain', 'blood gem'), ('permanent', 'each', 'end of'),
                     ('gold', 'discount', 'refresh')]
            table = np.zeros((size, 6), np.float64)
            for card, meta in o.catalog().items():
                table[card, 0] = float(meta.get('tier', 0))
                for j, group in enumerate(words, 1):
                    table[card, j] = sum(word in o.text(card) for word in group)
            self.metadata = table
        return self.metadata

    def remote_act(self, obs, indices):
        indices = np.asarray(indices, np.int64)
        actions = np.empty(len(indices), np.int64)
        if not len(indices):
            return actions, np.zeros(0, np.float32), np.zeros(0, np.float32)
        masks = obs['action_mask'][indices].astype(bool)
        if not masks.any(1).all():
            raise ValueError('No action requested for a waiting seat')
        pending = masks[:, 30:46].any(1)
        for row in np.flatnonzero(pending):
            i = indices[row]
            actions[row] = self.oracle.act_visible(obs['features'][i], obs['cards'][i], obs['action_mask'][i], self.style)
        ordinary = ~pending
        if ordinary.any():
            ids = indices[ordinary]
            actions[ordinary] = self.recruit(obs['features'][ids], obs['cards'][ids], masks[ordinary])
        return actions, np.zeros(len(actions), np.float32), np.zeros(len(actions), np.float32)

    def recruit(self, f, c, masks):
        o, style = self.oracle, self.style
        n = len(f)
        board = f[:, o.BF:o.HF].reshape(n, 7, o.MF)
        hand = f[:, o.HF:o.SF].reshape(n, 10, o.IF)
        shop = f[:, o.SF:o.CF].reshape(n, 8, o.IF)
        bv = minion_values(board)
        nb, nh = (board[..., 0] != 0).sum(1), (hand[..., 0] != 0).sum(1)
        turn, tier, gold = f[:, 0] * 20, f[:, 10] * 6, f[:, 7] * 10
        health = f[:, 5] * 40 + f[:, 6] * 20
        owned = np.concatenate((c[:, o.BC:o.BC+7], c[:, o.HC:o.HC+10]), axis=1)
        races = board[..., 20:31].sum(1)
        table = self.tables()

        def item_values(rows, cards):
            metadata = table[cards]
            value = minion_values(rows[..., 9:])
            duplicates = (owned[:, None, :] == cards[..., None]).sum(-1)
            duplicates = np.where(cards != 0, duplicates, 0)
            value += 5 * np.minimum(duplicates, 1) + 15 * (duplicates >= 2)
            if style == 'synergy':
                # Same float32 dot product as the row oracle.
                dot = np.matmul(rows[..., 29:40], races[..., None]).squeeze(-1)
                value += 3 * dot.astype(np.float64)
                value += 5 * metadata[..., 1]
            if style == 'economy':
                value += 5 * metadata[..., 2]
            spell = 3 + metadata[..., 0]
            spell += np.minimum(nb, 4)[:, None] * metadata[..., 3]
            if style == 'synergy':
                spell += 3 * metadata[..., 4]
            if style == 'economy':
                spell += 4 * metadata[..., 5]
            return np.where(rows[..., 1] > .5, value, spell)

        sv = np.where(shop[..., 0] != 0, item_values(shop, c[:, o.SC:o.SC+8]), -1000.)
        hv = np.where(hand[..., 0] != 0, item_values(hand, c[:, o.HC:o.HC+10]), -1000.)
        weakest = np.where(nb != 0, np.where(board[..., 0] > .5, bv, np.inf).argmin(1), 0)
        weak_value = bv[np.arange(n), weakest]
        best_hand = np.where((hand[..., 0] != 0) & (hand[..., 1] > .5), hv, -1000.).max(1)
        legal_buys = masks[:, 8:16]
        best_shop = np.where(legal_buys & (shop[..., 1] > .5), sv, -1000.).max(1)
        scores = np.full((n, 245), -1000., np.float64)
        scores[:, 0], scores[:, 7] = 0., -5.
        duplicates = (owned[:, None, :] == c[:, o.SC:o.SC+8, None]).sum(-1)
        skip_buy = ((shop[..., 1] > .5) & ((nb + (hand[..., 1] > .5).sum(1))[:, None] >= 7)
                    & (duplicates < 2) & (sv < weak_value[:, None] + 4))
        # Scalar multiplication by .3 occurs on a float32 scalar in the oracle.
        scores[:, 8:16] = np.where(skip_buy, -1000., 4 + sv * .15 - shop[..., 2] * .3)
        best = np.maximum(best_hand, best_shop)
        sell = ((nb == 7)[:, None] & (np.arange(7) == weakest[:, None]) & (best[:, None] > bv + 4))
        scores[:, 16:23] = np.where(sell, 9 + (best[:, None] - bv) * .05, -1000.)
        scores[:, 23:30] = 3.5 + (1. if style == 'synergy' else 0.)
        bp, hp = priorities(board), priorities(hand[..., 9:])
        present = np.arange(7)[None, :] < nb[:, None]
        desired = ((bp[:, None, :] > hp[..., None]) & present[:, None, :]).sum(-1)
        play = 15 + hv[..., None] * .12 - np.abs(np.arange(8) - desired[..., None]) * .1
        scores[:, 46:126] = play.reshape(n, 80)
        targeted = 16 + hv[..., None] * .12 + .02 * bv[:, None, :]
        scores[:, 126:196] = targeted.reshape(n, 70)
        # Stable ordering includes card ID and original position, as in sorted().
        order = np.lexsort((np.broadcast_to(np.arange(7), (n, 7)), c[:, o.BC:o.BC+7],
                            np.where(present, -bp, np.inf)), axis=1)
        mismatch = (order != np.arange(7)) & present
        first = mismatch.argmax(1)
        reorder = mismatch.any(1) & (nb > 1)
        rows = np.flatnonzero(reorder)
        scores[rows, 196 + order[rows, first[rows]] * 7 + first[rows]] = .3
        upgrade = f[:, 11] * 10
        min_turn = 2 * tier + (1 if style == 'tempo' else -1 if style == 'economy' else 0)
        # int(turn) in the oracle truncates toward zero.
        can_upgrade = ((turn >= min_turn) & ((nb >= np.minimum(5, turn.astype(int) + 1)) | (upgrade <= 2)) & (health > 8))
        scores[can_upgrade, 3] = 10 if style == 'economy' else 6.5
        for slot in range(2):
            cost = f[:, 17 + 3*slot] * 5
            scores[:, 4+slot] = 2.5 + (1 if style == 'economy' else 0) - .1 * cost
        scores[(nb < 7) & (health > 12), 6] = 3.
        promising = sv.max(1) > np.maximum(8., np.where(nb == 7, weak_value + 6, 8.))
        scores[~legal_buys.any(1) & promising & (f[:, 15] < .5), 2] = .2
        scores[~promising & (f[:, 15] > .5), 2] = .1
        scores[(gold >= np.maximum(1., f[:, 12] * 2)) & (health > 8) & (nh < 8), 1] = .5
        return np.where(masks, scores, -np.inf).argmax(1).astype(np.int64)
