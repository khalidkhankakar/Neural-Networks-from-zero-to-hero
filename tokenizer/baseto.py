"""
base_tokenizer.py
-----------------
Abstract base class for tokenizers.

Subclasses must implement the four abstract methods:
    - tokenize(text)       -> list[str]
    - encode(text)         -> list[int]
    - decode(token_ids)    -> str
    - get_vocab()          -> dict[str, int]

Everything else (vocab_size, special-token management, batch helpers,
save / load) is provided here for free.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union


class BaseTokenizer(ABC):
    """Abstract base class that every tokenizer should extend.

    Attributes
    ----------
    special_tokens : dict[str, int]
        Mapping of special-token string → token ID.
        Common keys: ``<pad>``, ``<unk>``, ``<bos>``, ``<eos>``.
    max_length : int | None
        Hard limit on the number of tokens produced per call.
        ``None`` means no limit.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        special_tokens: Optional[Dict[str, int]] = None,
        max_length: Optional[int] = None,
    ) -> None:
        self.special_tokens: Dict[str, int] = special_tokens or {}
        self.max_length: Optional[int] = max_length

    # ------------------------------------------------------------------ #
    #  Abstract interface — every subclass must implement these            #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Split *text* into a list of token strings.

        Parameters
        ----------
        text:
            Raw input string.

        Returns
        -------
        list[str]
            Human-readable token pieces (e.g. ``["Hello", "▁world"]``).
        """

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert *text* into a list of integer token IDs.

        Parameters
        ----------
        text:
            Raw input string.

        Returns
        -------
        list[int]
            Token IDs that can be fed directly to a model.
        """

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Reconstruct a string from a sequence of token IDs.

        Parameters
        ----------
        token_ids:
            Sequence of integer token IDs.

        Returns
        -------
        str
            Decoded text.
        """

    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        """Return the full vocabulary as ``{token_str: token_id}``."""

    # ------------------------------------------------------------------ #
    #  Derived / convenience properties                                    #
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary (including special tokens)."""
        return len(self.get_vocab())

    def token_to_id(self, token: str) -> Optional[int]:
        """Look up the integer ID for a token string.

        Returns ``None`` when the token is not in the vocabulary.
        """
        return self.get_vocab().get(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Look up the token string for an integer ID.

        Returns ``None`` when the ID is not in the vocabulary.
        """
        return {v: k for k, v in self.get_vocab().items()}.get(token_id)

    # ------------------------------------------------------------------ #
    #  Special-token helpers                                               #
    # ------------------------------------------------------------------ #

    def add_special_tokens(self, tokens: Dict[str, int]) -> None:
        """Register additional special tokens.

        Parameters
        ----------
        tokens:
            ``{token_str: token_id}`` pairs to add.
        """
        self.special_tokens.update(tokens)

    def get_special_token_id(self, name: str) -> Optional[int]:
        """Return the ID of a named special token (e.g. ``"<pad>"``).

        Returns ``None`` if the token has not been registered.
        """
        return self.special_tokens.get(name)

    # Convenience shortcuts for the most common special tokens
    @property
    def pad_token_id(self) -> Optional[int]:
        return self.special_tokens.get("<pad>")

    @property
    def unk_token_id(self) -> Optional[int]:
        return self.special_tokens.get("<unk>")

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.special_tokens.get("<bos>")

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.special_tokens.get("<eos>")

    # ------------------------------------------------------------------ #
    #  Batch helpers                                                       #
    # ------------------------------------------------------------------ #

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a list of strings, returning a list of ID sequences."""
        return [self.encode(text) for text in texts]

    def decode_batch(self, batch: List[List[int]]) -> List[str]:
        """Decode a list of ID sequences, returning a list of strings."""
        return [self.decode(ids) for ids in batch]

    def __call__(
        self,
        text: Union[str, List[str]],
        *,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        """Callable interface for encoding text or a batch of texts.

        Parameters
        ----------
        text:
            A single string or a list of strings.
        max_length:
            Overrides ``self.max_length`` for this call only.
        truncation:
            Truncate sequences that exceed *max_length*.
        padding:
            Pad shorter sequences to *max_length* (requires ``<pad>`` token).

        Returns
        -------
        list[int] | list[list[int]]
            Encoded token IDs (single or batch).
        """
        limit = max_length or self.max_length
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]

        encoded = self.encode_batch(texts)

        if truncation and limit is not None:
            encoded = [ids[:limit] for ids in encoded]

        if padding and limit is not None:
            pad_id = self.pad_token_id
            if pad_id is None:
                raise ValueError("Padding requested but '<pad>' special token is not set.")
            encoded = [
                ids + [pad_id] * (limit - len(ids)) if len(ids) < limit else ids
                for ids in encoded
            ]

        return encoded if is_batch else encoded[0]

    # ------------------------------------------------------------------ #
    #  Iteration helpers                                                   #
    # ------------------------------------------------------------------ #

    def iter_tokens(self, text: str) -> Iterator[str]:
        """Yield token strings one at a time."""
        yield from self.tokenize(text)

    def iter_ids(self, text: str) -> Iterator[int]:
        """Yield token IDs one at a time."""
        yield from self.encode(text)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, directory: str) -> None:
        """Persist tokenizer metadata to *directory*.

        Writes ``tokenizer_config.json`` containing ``special_tokens``
        and ``max_length``.  Subclasses should call ``super().save(directory)``
        and then write their own vocabulary files.
        """
        os.makedirs(directory, exist_ok=True)
        config = {
            "special_tokens": self.special_tokens,
            "max_length": self.max_length,
            "tokenizer_class": type(self).__name__,
        }
        config_path = os.path.join(directory, "tokenizer_config.json")
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str) -> "BaseTokenizer":
        """Restore tokenizer metadata from *directory*.

        Loads ``tokenizer_config.json`` and passes ``special_tokens`` /
        ``max_length`` to the constructor.  Subclasses should override
        this to also reload vocabulary files, then call ``super().load()``.

        Note
        ----
        Because ``BaseTokenizer`` is abstract, calling this method directly
        will raise a ``TypeError``.  Call it on a concrete subclass instead:
        ``MyTokenizer.load(directory)``.
        """
        config_path = os.path.join(directory, "tokenizer_config.json")
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        return cls(
            special_tokens=config.get("special_tokens"),
            max_length=config.get("max_length"),
        )

    # ------------------------------------------------------------------ #
    #  Dunder helpers                                                      #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.vocab_size

    def __contains__(self, token: str) -> bool:
        return token in self.get_vocab()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"vocab_size={self.vocab_size}, "
            f"max_length={self.max_length}, "
            f"special_tokens={list(self.special_tokens.keys())})"
        )