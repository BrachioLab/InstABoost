"""Two rules, one boosted and one not, with K further rules added as context.

The whole sample at ``K = 0``::

    [ a -> X ]   [init {a,b,c}]   [ b -> Y ]   [restate init]
      +B on this                   not boosted

Four rows.  One rule derives ``X`` and is boosted; one derives ``Y`` and is not.  Both antecedents
are single facts drawn from the initial state, so both fire at the first step, and at ``K = 0``
both fire essentially always -- the baseline is 1.000, not something the construction has already
eroded.  ``K`` then counts every other rule in the prompt, so the sequence grows from 3 rows to
``3 + K`` and nothing is competing for attention that we did not put there.

The two rules are not identical -- they cannot be, since two rules concluding the same variable
would be one rule -- but they are **exchangeable**.  Both are drawn as ``<one fact of the initial
state> -> <one otherwise unused variable>``, from a single random permutation, so swapping which
draw carries the bias gives an equally likely sample.  Nothing distinguishes them except the bias.
Hence **at ``B = 0`` their firing rates must agree**, and any gap at ``B > 0`` is the bias.

Samples are **paired across cells**.  Every sample is generated from a seed derived from its
index, and the competing rows come from a fixed-size pool drawn in a fixed order, of which the
first ``K`` are used.  So raising ``K`` *adds* rows without disturbing the ones already there, and
every bias sees byte-identical prompts.  Without that, each cell drew an independent dataset and
adjacent points on a curve had uncorrelated binomial error -- measured at rms 0.0075 against a
predicted 0.0070, i.e. all of the visible roughness.  Pairing removes it at no extra cost, and
makes a difference between two biases exactly attributable to the bias.

``applicable_fraction`` splits the K added rules:

* an *applicable* row is satisfied by the initial state but concludes a fact that is already true,
  so it competes for attention without changing the proof;
* an *inert* row carries a variable in its antecedent that is never derivable, so it can never
  fire at all.

Each pool row is independently applicable with that probability, so the first ``K`` of them carry
a Binomial(``K``, ``f``) count -- nested in ``K``, and free of the staircase a fixed
``round(f*K)`` would impose.

Growing past the 33-row training length is sound for these checkpoints because positional
embeddings are zeroed and frozen (``models/small_models.py``), so the attention paid to a rule row
depends on its content alone.
"""

from __future__ import annotations

from itertools import combinations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def hot(i, n):
    return F.one_hot(torch.as_tensor(i), n)


class BoostTradeoffDataset(Dataset):
    """One boosted rule, one unboosted rule, and ``K`` competing rules added as context."""

    def __init__(
        self,
        reasoner_dataset: Dataset,
        num_competing: int,
        dataset_len: int,
        applicable_fraction: float = 0.0,
        num_init_facts: int = 3,
        seed: int = 0,
        pool_size: int = 256,
    ):
        if int(num_competing) < 0:
            raise ValueError(f"num_competing must be >= 0, got {num_competing}")
        if not 0.0 <= float(applicable_fraction) <= 1.0:
            raise ValueError(f"applicable_fraction must be in [0, 1], got {applicable_fraction}")
        if int(num_init_facts) < 1:
            raise ValueError(f"num_init_facts must be >= 1, got {num_init_facts}")
        if int(num_competing) > int(pool_size):
            raise ValueError(
                f"num_competing={num_competing} exceeds pool_size={pool_size}; raise pool_size so "
                "every K draws from the same pool, otherwise cells stop being nested."
            )

        self.num_vars = reasoner_dataset.num_vars
        self.hot_prob = reasoner_dataset.exp_hots / self.num_vars
        self.num_competing = int(num_competing)
        self.applicable_fraction = float(applicable_fraction)
        self.num_init_facts = int(num_init_facts)
        self.dataset_len = dataset_len
        self.seed = int(seed)
        # Drawn in full for every K so the random stream -- and hence the first K rows -- does not
        # depend on K.  Cost is flat in K and equal to the largest cell's.
        self.pool_size = int(pool_size)
        # The boosted rule is the single row before the initial-state token.
        self.rulebook_len = 1

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        n, K, j, pool = self.num_vars, self.num_competing, self.num_init_facts, self.pool_size
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(idx))

        v = torch.randperm(n, generator=g)
        init_facts, X, Y, dead = v[:j], v[j], v[j + 1], v[j + 2]
        s0 = hot(init_facts, n).sum(0)

        # One permutation draws both antecedents and both consequents, so the two rules are
        # exchangeable: swapping them yields an equally likely sample.
        ante = init_facts[torch.randperm(j, generator=g)[:2]] if j > 1 else init_facts.repeat(2)
        x_rule = torch.cat([hot(ante[0], n), hot(X, n)])
        y_rule = torch.cat([hot(ante[1 % ante.numel()], n), hot(Y, n)])

        # --- the pool: always drawn at full size, so the first K rows are the same at every K ---
        is_app = torch.rand(pool, generator=g) < self.applicable_fraction
        opts = [(a, c) for size in range(1, j + 1)
                for a in combinations(range(j), size)
                for c in range(j) if c not in a]
        app_pick = torch.randint(len(opts), (pool,), generator=g)
        inert_ante = (torch.rand(pool, n, generator=g) < self.hot_prob).long()
        inert_cons = (torch.rand(pool, n, generator=g) < self.hot_prob).long()
        inert_ante[:, dead] = 1          # never derivable, so the row can never fire
        inert_ante[:, init_facts] = 0    # nor even partially match the state

        rows = [y_rule.view(1, -1).long()]
        for r in range(K):
            if is_app[r]:
                a_idx, c_idx = opts[int(app_pick[r])]
                rows.append(torch.cat([hot(init_facts[list(a_idx)], n).sum(0),
                                       hot(init_facts[c_idx], n)]).view(1, -1))
            else:
                rows.append(torch.cat([inert_ante[r], inert_cons[r]]).view(1, -1))
        suffix = torch.cat(rows, dim=0)

        # The state token goes last because that is the format the model predicts from.  It
        # appears exactly once: an earlier version also carried one between the boosted rule and
        # the suffix, inherited from the attack datasets where it separates the rulebook from the
        # appended rows.  Here the boosted row is picked out explicitly by `bias_positions`, so
        # that separator had no job -- dropping it moved every firing rate by <= 0.005.
        init_token = torch.cat([torch.zeros(n, dtype=torch.long), s0]).view(1, -1)
        tokens = torch.cat([x_rule.view(1, -1), suffix, init_token], dim=0)

        # Both rules fire at step 1; nothing after that changes the state.  The y-rule is kept at
        # the front of the suffix rather than shuffled in: row order provably does not affect this
        # model (positional embeddings are zeroed), and a fixed position keeps the nesting exact.
        s1 = ((s0 + hot(X, n) + hot(Y, n)) > 0).long()
        return {
            "tokens": tokens.float(),
            "labels": torch.stack([s1, s1, s1]),
            "x_index": X.clone(),
            "y_index": Y.clone(),
            "y_position": torch.tensor(0),
            "num_applicable": is_app[:K].sum() if K else torch.tensor(0),
        }
