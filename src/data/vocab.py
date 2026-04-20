from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence
import json


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    mask: str = "<mask>"
    unk: str = "<unk>"


class Vocab:
    """
    Minimal token<->id mapping with a few special tokens.
    """

    def __init__(self, token2id: Dict[str, int], specials: SpecialTokens = SpecialTokens()):
        self.specials = specials
        self.token2id = token2id
        self.id2token = {i: t for t, i in token2id.items()}

        # sanity checks
        for s in [specials.pad, specials.bos, specials.eos, specials.mask, specials.unk]:
            if s not in self.token2id:
                raise ValueError(f"Missing special token in vocab: {s}")

        self.pad_id = self.token2id[specials.pad]
        self.bos_id = self.token2id[specials.bos]
        self.eos_id = self.token2id[specials.eos]
        self.mask_id = self.token2id[specials.mask]
        self.unk_id = self.token2id[specials.unk]

    def __len__(self) -> int:
        return len(self.token2id)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.token2id.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.id2token.get(int(i), self.specials.unk) for i in ids]

    def to_json(self) -> Dict:
        return {"token2id": self.token2id, "specials": self.specials.__dict__}

    @staticmethod
    def from_json(obj: Dict) -> "Vocab":
        specials = SpecialTokens(**obj["specials"])
        return Vocab(token2id={k: int(v) for k, v in obj["token2id"].items()}, specials=specials)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            return Vocab.from_json(json.load(f))


def build_vocab_from_token_stream(
    token_stream: Iterable[Sequence[str]],
    *,
    specials: SpecialTokens = SpecialTokens(),
    min_freq: int = 1,
) -> Vocab:
    """
    Build vocab from an iterable of token sequences.
    Keeps all tokens with frequency >= min_freq.
    """
    freq: Dict[str, int] = {}
    for seq in token_stream:
        for t in seq:
            freq[t] = freq.get(t, 0) + 1

    # deterministic ordering: specials first, then sorted tokens
    tokens: List[str] = [
        specials.pad, specials.bos, specials.eos, specials.mask, specials.unk
    ]
    for t in sorted(freq.keys()):
        if t in tokens:
            continue
        if freq[t] >= min_freq:
            tokens.append(t)

    token2id = {t: i for i, t in enumerate(tokens)}
    return Vocab(token2id=token2id, specials=specials)
