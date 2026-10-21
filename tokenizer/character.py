"""
character_tokenizer.py
----------------------
Concrete character-level tokenizer that extends BaseTokenizer.

Each Unicode character (or special token) is one token.  The vocabulary
is built either from a user-supplied corpus / character set or loaded
from a saved directory.

Key features
------------
* Corpus-based vocab building  (``from_corpus`` / ``from_texts``)
* Configurable case folding and whitespace normalisation
* BOS / EOS / PAD / UNK special tokens, all optional
* Full save / load round-trip (JSON vocabulary file)
* Inherits batch helpers, truncation, padding, and iteration from
  BaseTokenizer — nothing extra needed here.

Quick start
-----------
>>> tok = CharacterTokenizer.from_corpus("hello world")
>>> tok.encode("low")
[10, 13, 21]        # IDs will vary by corpus
>>> tok.decode(tok.encode("low"))
'low'
"""

from __future__ import annotations

import json
import os
import unicodedata
from typing import Dict, Iterable, List, Optional, Set

from baseto import BaseTokenizer


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

_DEFAULT_SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


def _build_vocab(
    chars: Iterable[str],
    special_token_names: Iterable[str],
) -> Dict[str, int]:
    """Assign integer IDs: special tokens first (in order), then sorted chars."""
    vocab: Dict[str, int] = {}
    for name in special_token_names:
        if name not in vocab:
            vocab[name] = len(vocab)
    for ch in sorted(set(chars)):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


# --------------------------------------------------------------------------- #
#  CharacterTokenizer                                                          #
# --------------------------------------------------------------------------- #

class CharacterTokenizer(BaseTokenizer):
    """Character-level tokenizer.

    Every Unicode code-point is a distinct token.  Special tokens
    (``<pad>``, ``<unk>``, ``<bos>``, ``<eos>``) occupy the lowest IDs
    so that ordinary characters always get IDs ≥ ``len(special_tokens)``.

    Parameters
    ----------
    vocab : dict[str, int]
        Complete ``{token: id}`` mapping including special tokens.
    special_tokens : dict[str, int], optional
        Subset of *vocab* that are special tokens.  If omitted, any key
        that starts and ends with ``<`` / ``>`` is treated as special.
    max_length : int, optional
        Passed through to ``BaseTokenizer``.
    lowercase : bool
        Fold all input to lowercase before tokenising.  Default ``False``.
    normalize_unicode : bool
        Apply NFC normalisation before tokenising.  Default ``True``.
    add_bos : bool
        Prepend a ``<bos>`` token on every ``encode`` call.
    add_eos : bool
        Append an ``<eos>`` token on every ``encode`` call.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        vocab: Dict[str, int],
        *,
        special_tokens: Optional[Dict[str, int]] = None,
        max_length: Optional[int] = None,
        lowercase: bool = False,
        normalize_unicode: bool = True,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> None:
        # Derive special_tokens from vocab if not provided explicitly
        if special_tokens is None:
            special_tokens = {
                k: v for k, v in vocab.items()
                if k.startswith("<") and k.endswith(">")
            }
        super().__init__(special_tokens=special_tokens, max_length=max_length)

        self._vocab: Dict[str, int] = vocab
        self._id_to_token_map: Dict[int, str] = {v: k for k, v in vocab.items()}

        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.add_bos = add_bos
        self.add_eos = add_eos

    # ------------------------------------------------------------------ #
    #  Factory / class-method constructors                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_corpus(
        cls,
        corpus: str,
        *,
        special_token_names: Iterable[str] = _DEFAULT_SPECIAL_TOKENS,
        **kwargs,
    ) -> "CharacterTokenizer":
        """Build a tokenizer whose vocabulary covers every character in *corpus*.

        Parameters
        ----------
        corpus : str
            Any string; every unique character becomes a vocabulary entry.
        special_token_names : iterable[str]
            Special tokens to prepend to the vocabulary (in order).
        **kwargs
            Forwarded to ``__init__`` (e.g. ``lowercase``, ``add_bos``).
        """
        vocab = _build_vocab(corpus, special_token_names)
        special_tokens = {n: vocab[n] for n in special_token_names if n in vocab}
        return cls(vocab, special_tokens=special_tokens, **kwargs)

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        *,
        special_token_names: Iterable[str] = _DEFAULT_SPECIAL_TOKENS,
        **kwargs,
    ) -> "CharacterTokenizer":
        """Build a tokenizer whose vocabulary covers all characters in *texts*.

        Parameters
        ----------
        texts : iterable[str]
            Any iterable of strings (list, generator, file lines …).
        special_token_names : iterable[str]
            Special tokens to prepend to the vocabulary.
        **kwargs
            Forwarded to ``__init__``.
        """
        all_chars: Set[str] = set()
        for text in texts:
            all_chars.update(text)
        vocab = _build_vocab(all_chars, special_token_names)
        special_tokens = {n: vocab[n] for n in special_token_names if n in vocab}
        return cls(vocab, special_tokens=special_tokens, **kwargs)

    @classmethod
    def from_charset(
        cls,
        charset: Iterable[str],
        *,
        special_token_names: Iterable[str] = _DEFAULT_SPECIAL_TOKENS,
        **kwargs,
    ) -> "CharacterTokenizer":
        """Build a tokenizer from an explicit set of characters.

        Parameters
        ----------
        charset : iterable[str]
            Characters to include (e.g. ``string.ascii_letters``).
        special_token_names : iterable[str]
            Special tokens to prepend to the vocabulary.
        **kwargs
            Forwarded to ``__init__``.
        """
        return cls.from_texts(["".join(charset)], special_token_names=special_token_names, **kwargs)

    # ------------------------------------------------------------------ #
    #  Pre-processing                                                      #
    # ------------------------------------------------------------------ #

    def _preprocess(self, text: str) -> str:
        """Apply normalisation steps before character splitting."""
        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = text.lower()
        return text

    # ------------------------------------------------------------------ #
    #  BaseTokenizer abstract methods                                      #
    # ------------------------------------------------------------------ #

    def tokenize(self, text: str) -> List[str]:
        """Split *text* into individual characters.

        Unknown characters are replaced with the ``<unk>`` string if
        ``<unk>`` is in the vocabulary, otherwise they are dropped.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        list[str]
            One entry per character (after pre-processing).
        """
        text = self._preprocess(text)
        unk = "<unk>" if "<unk>" in self._vocab else None
        tokens: List[str] = []
        for ch in text:
            if ch in self._vocab:
                tokens.append(ch)
            elif unk is not None:
                tokens.append(unk)
            # unknown chars with no <unk> token are silently dropped
        return tokens

    def encode(self, text: str) -> List[int]:
        """Convert *text* to a list of token IDs.

        Prepends ``<bos>`` and / or appends ``<eos>`` when those flags
        are set and the corresponding special token exists in the vocab.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        list[int]
            Token IDs.
        """
        ids: List[int] = [self._vocab[ch] for ch in self.tokenize(text)]

        if self.add_bos and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        if self.add_eos and self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]

        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Reconstruct a string from a list of token IDs.

        Special tokens (``<bos>``, ``<eos>``, ``<pad>``) are stripped
        from the output.  Unknown IDs are replaced with ``?``.

        Parameters
        ----------
        token_ids : list[int]
            Sequence of integer token IDs.

        Returns
        -------
        str
            Decoded text, with special tokens removed.
        """
        skip_ids = set(self.special_tokens.values())
        chars: List[str] = []
        for tid in token_ids:
            if tid in skip_ids:
                continue
            chars.append(self._id_to_token_map.get(tid, "?"))
        return "".join(chars)

    def get_vocab(self) -> Dict[str, int]:
        """Return the full ``{token: id}`` vocabulary dictionary."""
        return dict(self._vocab)

    # ------------------------------------------------------------------ #
    #  Persistence  (override to also save / load the vocabulary)         #
    # ------------------------------------------------------------------ #

    def save(self, directory: str) -> None:
        """Save config + vocabulary to *directory*.

        Writes two files:
        * ``tokenizer_config.json`` — metadata (via super)
        * ``vocab.json``            — ``{token: id}`` vocabulary

        Parameters
        ----------
        directory : str
            Target directory (created if it does not exist).
        """
        super().save(directory)  # writes tokenizer_config.json

        # Augment config with CharacterTokenizer-specific flags
        config_path = os.path.join(directory, "tokenizer_config.json")
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        config.update(
            {
                "lowercase": self.lowercase,
                "normalize_unicode": self.normalize_unicode,
                "add_bos": self.add_bos,
                "add_eos": self.add_eos,
            }
        )
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, ensure_ascii=False)

        # Save vocabulary
        vocab_path = os.path.join(directory, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as fh:
            json.dump(self._vocab, fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str) -> "CharacterTokenizer":
        """Restore a ``CharacterTokenizer`` from *directory*.

        Reads ``tokenizer_config.json`` and ``vocab.json``.

        Parameters
        ----------
        directory : str
            Directory previously created by ``save()``.

        Returns
        -------
        CharacterTokenizer
            Fully initialised tokenizer.
        """
        config_path = os.path.join(directory, "tokenizer_config.json")
        vocab_path = os.path.join(directory, "vocab.json")

        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        with open(vocab_path, "r", encoding="utf-8") as fh:
            vocab = json.load(fh)

        special_tokens = config.get("special_tokens", {})
        return cls(
            vocab=vocab,
            special_tokens=special_tokens,
            max_length=config.get("max_length"),
            lowercase=config.get("lowercase", False),
            normalize_unicode=config.get("normalize_unicode", True),
            add_bos=config.get("add_bos", False),
            add_eos=config.get("add_eos", False),
        )

    # ------------------------------------------------------------------ #
    #  Extra character-level utilities                                     #
    # ------------------------------------------------------------------ #

    def char_frequency(self, text: str) -> Dict[str, int]:
        """Count how often each character token appears in *text*.

        Parameters
        ----------
        text : str
            Input string (pre-processing is applied first).

        Returns
        -------
        dict[str, int]
            ``{character: count}`` sorted by descending frequency.
        """
        freq: Dict[str, int] = {}
        for ch in self.tokenize(text):
            freq[ch] = freq.get(ch, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

    def coverage(self, text: str) -> float:
        """Fraction of characters in *text* that are in-vocabulary.

        Parameters
        ----------
        text : str
            Input string (pre-processing is applied first).

        Returns
        -------
        float
            Value in ``[0.0, 1.0]``.  ``1.0`` means every character is known.
        """
        text = self._preprocess(text)
        if not text:
            return 1.0
        known = sum(1 for ch in text if ch in self._vocab)
        return known / len(text)

    def extend_vocab(self, new_chars: Iterable[str]) -> int:
        """Add new characters to the vocabulary in-place.

        Parameters
        ----------
        new_chars : iterable[str]
            Characters to add (duplicates and existing chars are ignored).

        Returns
        -------
        int
            Number of new entries actually added.
        """
        added = 0
        for ch in new_chars:
            if ch not in self._vocab:
                new_id = max(self._vocab.values()) + 1
                self._vocab[ch] = new_id
                self._id_to_token_map[new_id] = ch
                added += 1
        return added