"""Attack datasets that APPEND the adversarial suffix instead of replacing base rules.

The upstream datasets in `my_datasets.attack_datasets` carve the suffix out of the base
rulebook (`num_rules = reasoner_dataset.num_rules - num_attack_tokens`). That caps k at
`num_rules - 6` (26 for the nr=32 checkpoints, 8 for coerce_state, which spends 3k+1), and
it confounds the sweep: raising k also deletes legitimate rules, so at k=26 the model is
left reasoning over 6 rules instead of 31.

These subclasses keep the base rulebook whole. The suffix is built and concatenated by the
runner, so the only thing that has to change is `num_rules`, which `__getitem__` reads. k is
then bounded only by the model's context window. This lives here rather than upstream so the
shared `tf_logic` package is untouched and published results reproduce exactly.
"""

from my_datasets.attack_datasets import (
    CoerceStateDataset,
    FactAmnesiaDataset,
    SuppressRuleDataset,
)


class _KeepRulebookMixin:
    """Restore the full base rulebook after the parent subtracts the attack tokens."""

    def __init__(self, reasoner_dataset, num_attack_tokens: int, dataset_len: int):
        super().__init__(
            reasoner_dataset,
            num_attack_tokens=num_attack_tokens,
            dataset_len=dataset_len,
        )
        self.num_attack_tokens = num_attack_tokens
        self.num_rules = reasoner_dataset.num_rules


class AppendSuppressRuleDataset(_KeepRulebookMixin, SuppressRuleDataset):
    pass


class AppendFactAmnesiaDataset(_KeepRulebookMixin, FactAmnesiaDataset):
    pass


class AppendCoerceStateDataset(_KeepRulebookMixin, CoerceStateDataset):
    pass


APPEND_DATASETS = {
    "suppress_rule": AppendSuppressRuleDataset,
    "fact_amnesia": AppendFactAmnesiaDataset,
    "coerce_state": AppendCoerceStateDataset,
}
