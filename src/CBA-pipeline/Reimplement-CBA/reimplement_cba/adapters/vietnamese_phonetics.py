"""Vietnamese syllable-to-phone conversion and dialect-aware phone similarity.

The inventory and dialect rules are loaded from the paper-derived JSON files in
``resources/phonetics``. This is a practical scoring converter, not a full IPA
pronunciation dictionary: unrecognized syllables are retained as marked
grapheme tokens so their errors are visible instead of silently discarded.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TONE_MARKS = {
    "\u0300": "tone_huyen",
    "\u0301": "tone_sac",
    "\u0303": "tone_nga",
    "\u0309": "tone_hoi",
    "\u0323": "tone_nang",
}
_VOWEL_GRAPHEMES = set("aăâeêioôơuưy")
_DIPHTHONGS = {
    "iê": ("i", "ə"), "yê": ("i", "ə"),
    "ia": ("i", "ə"), "ya": ("i", "ə"),
    "uô": ("u", "ə"), "ua": ("u", "ə"),
    "ươ": ("ɯ", "ə"), "ưa": ("ɯ", "ə"),
}


def _default_resource_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "resources" / "phonetics"


def _fold_label(value: str) -> str:
    value = unicodedata.normalize("NFD", value.casefold().strip())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("đ", "d")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def _strip_tone_marks(text: str) -> tuple[str, str]:
    """Return a tone-free NFC spelling and the tone carried by its vowel mark."""
    output: list[str] = []
    tones: set[str] = set()
    for char in unicodedata.normalize("NFD", text):
        if char in _TONE_MARKS:
            tones.add(_TONE_MARKS[char])
        else:
            output.append(char)
    if len(tones) > 1:
        raise ValueError(f"Syllable has conflicting tone marks: {text!r}")
    return unicodedata.normalize("NFC", "".join(output)), next(iter(tones), "tone_ngang")


def _ipa_tokens(ipa: str, symbols: tuple[str, ...]) -> tuple[str, ...]:
    """Split a compact IPA sequence using longest known symbols first."""
    tokens: list[str] = []
    position = 0
    while position < len(ipa):
        char = ipa[position]
        if tokens and (
            unicodedata.category(char) == "Mn" or char == "ː" or char in {"ʷ", "ᵐ", "ᵖ"}
        ):
            tokens[-1] += char
            position += 1
            continue
        symbol = next((item for item in symbols if ipa.startswith(item, position)), None)
        if symbol is None:
            # Keep an unfamiliar IPA codepoint as its own unit; the metric should
            # remain usable when a paper rule adds a symbol not in the inventory.
            symbol = ipa[position]
        tokens.append(symbol)
        position += len(symbol)
    return tuple(tokens)


def _edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class ParsedSyllable:
    spelling: str
    onset_spelling: str
    coda_spelling: str
    onset: str | None
    glide: str | None
    nucleus: tuple[str, ...]
    coda: str | None
    tone: str
    supported: bool

    @property
    def phones(self) -> tuple[str, ...]:
        result: list[str] = []
        if self.onset:
            result.append(self.onset)
        if self.glide:
            result.append(self.glide)
        result.extend(self.nucleus)
        if self.coda:
            result.append(self.coda)
        result.append(f"@{self.tone}")
        return tuple(result)


class VietnamesePhoneticEncoder:
    """Parse Quốc ngữ syllables and apply the article's region-level rules."""

    def __init__(
        self,
        *,
        vocabulary_path: str | Path | None = None,
        transformations_path: str | Path | None = None,
    ):
        resource_dir = _default_resource_dir()
        self.vocabulary_path = Path(vocabulary_path or resource_dir / "vi_phoneme_vocabulary.json")
        self.transformations_path = Path(
            transformations_path or resource_dir / "vi_dialect_transformations.json"
        )
        self.vocabulary = json.loads(self.vocabulary_path.read_text(encoding="utf-8"))
        self.transformations = json.loads(self.transformations_path.read_text(encoding="utf-8"))
        inventory = self.vocabulary["inventories"]
        self._onsets = {
            spelling: item["ipa"]
            for item in inventory["onsets"]["items"]
            for spelling in item["spellings"]
        }
        self._codas = {
            spelling: item["ipa"]
            for item in inventory["codas"]["items"]
            for spelling in item["spellings"]
        }
        self._glide = inventory["glides"][0]["ipa"]
        self._monophthongs = {
            spelling: item["ipa"]
            for item in inventory["vowel_nuclei"]["monophthongs"]
            for spelling in item["spellings"]
            if not (spelling == "a" and item["ipa"] == "ă")
        }
        self._onset_spellings = tuple(sorted(self._onsets, key=len, reverse=True))
        self._coda_spellings = tuple(sorted(self._codas, key=len, reverse=True))
        phones = {
            item["ipa"] for item in inventory["onsets"]["items"]
        } | {
            item["ipa"] for item in inventory["codas"]["items"]
        } | {
            item["ipa"] for item in inventory["glides"]
        } | {
            item["ipa"] for item in inventory["vowel_nuclei"]["monophthongs"]
        }
        phones.update(phone for pair in _DIPHTHONGS.values() for phone in pair)
        phones.update({"ɨ", "ɜ", "ɤ", "ʌ", "ʊ", "ɑ", "æ", "w", "j", "ː", "̃", "ᵐ", "ᵖ", "ʷ"})
        self._ipa_symbols = tuple(sorted(phones, key=len, reverse=True))
        self._region_index = self._build_region_index()
        self._rules = self.transformations["rules"]

    def _build_region_index(self) -> dict[str, tuple[str, str | None]]:
        profiles = self.transformations["dialect_profiles"]
        index: dict[str, tuple[str, str | None]] = {}
        for macro, profile in profiles.items():
            for label in profile.get("paper_regions", []):
                index[_fold_label(label)] = (macro, None)
            for name, labels in profile.get("subdialects", {}).items():
                scope = f"{macro}.{name}"
                for label in labels:
                    index[_fold_label(label)] = (macro, scope)
        for label in profiles["northern"].get("special_l_n_merger_locations", []):
            index[_fold_label(label)] = ("northern", None)
        index[_fold_label("Nghệ Tĩnh")] = ("central", "central.nghe_tinh")
        # Common labels in ViMD exports and research notes.
        for label in ("north", "northern", "mien bac", "bac bo"):
            index[_fold_label(label)] = ("northern", None)
        for label in ("central", "mien trung", "trung bo"):
            index[_fold_label(label)] = ("central", None)
        for label in ("south", "southern", "mien nam", "nam bo"):
            index[_fold_label(label)] = ("southern", None)
        return index

    def resolve_region(self, region: str | None) -> tuple[str | None, str | None, str | None]:
        if region is None or not str(region).strip():
            return None, None, None
        label = str(region).strip()
        normalized = _fold_label(label)
        if normalized in self._region_index:
            macro, subdialect = self._region_index[normalized]
            return macro, subdialect, label
        # Accept explicit scope IDs from the transformation vocabulary.
        explicit_scopes = {
            "central nghe tinh": ("central", "central.nghe_tinh"),
            "central quang tri": ("central", "central.quang_tri"),
            "central quang binh": ("central", "central.quang_binh"),
            "central hue": ("central", "central.hue"),
            "southern quang nam quang ngai": ("southern", "southern.quang_nam_quang_ngai"),
            "southern mekong delta": ("southern", "southern.mekong_delta"),
        }
        if normalized in explicit_scopes:
            macro, subdialect = explicit_scopes[normalized]
            return macro, subdialect, label
        return None, None, label

    def parse_syllable(self, spelling: str) -> ParsedSyllable:
        spelling = unicodedata.normalize("NFC", spelling.strip().lower())
        plain, tone = _strip_tone_marks(spelling)
        onset_spelling = next((s for s in self._onset_spellings if plain.startswith(s)), "")
        onset = self._onsets.get(onset_spelling)
        rhyme = plain[len(onset_spelling):]

        coda_spelling = next((s for s in self._coda_spellings
                              if len(rhyme) > len(s) and rhyme.endswith(s)), "")
        coda = self._codas.get(coda_spelling)
        if coda_spelling:
            rhyme = rhyme[:-len(coda_spelling)]

        glide = None
        # q + u is always the initial /k/ followed by the paper's /u̯/ glide.
        if onset_spelling == "q" and rhyme.startswith("u") and len(rhyme) > 1:
            glide = self._glide
            rhyme = rhyme[1:]
        elif (len(rhyme) > 1 and rhyme[0] in {"u", "o"}
              and rhyme[1] in _VOWEL_GRAPHEMES
              and not any(rhyme.startswith(diphthong) for diphthong in _DIPHTHONGS)):
            glide = self._glide
            rhyme = rhyme[1:]

        # The spelling a before final y represents the short vowel /ă/ in the paper.
        if rhyme == "a" and coda_spelling == "y":
            nucleus = ("ă",)
        elif rhyme == "y":
            nucleus = ("i",)
        elif rhyme in _DIPHTHONGS:
            nucleus = _DIPHTHONGS[rhyme]
        else:
            nucleus_value = self._monophthongs.get(rhyme)
            if nucleus_value is not None:
                nucleus = (nucleus_value,)
            else:
                # Common open-syllable y is the orthographic alternative for /i/.
                fallback_spelling = "i" if rhyme == "y" else rhyme
                fallback_value = self._monophthongs.get(fallback_spelling)
                nucleus = (fallback_value,) if fallback_value else ()

        supported = bool(nucleus)
        if not supported:
            # Keep unsupported characters as distinct units. This ensures a
            # foreign/invalid syllable cannot disappear from phonetic similarity.
            fallback = tuple(f"?{char}" for char in plain if char.isalpha())
            return ParsedSyllable(
                spelling=spelling, onset_spelling=onset_spelling,
                coda_spelling=coda_spelling, onset=None, glide=None,
                nucleus=fallback or ("?",), coda=None,
                tone=tone, supported=False,
            )

        return ParsedSyllable(
            spelling=spelling, onset_spelling=onset_spelling,
            coda_spelling=coda_spelling, onset=onset, glide=glide,
            nucleus=nucleus, coda=coda, tone=tone, supported=True,
        )

    def _onset_for_region(self, syllable: ParsedSyllable, macro: str | None,
                          region_label: str | None) -> str | None:
        onset = syllable.onset
        raw = syllable.onset_spelling
        if onset is None or macro is None:
            return onset
        if macro == "northern":
            if raw in {"x", "s"}:
                return "s"
            if raw in {"ch", "tr"}:
                return "c͡ɕ"
            if raw in {"gi", "r", "d"}:
                return "z"
            sublabels = {
                _fold_label(label)
                for label in self.transformations["dialect_profiles"]["northern"]
                .get("special_l_n_merger_locations", [])
            }
            if raw in {"l", "n"} and region_label and _fold_label(region_label) in sublabels:
                return "l_n"
        elif macro == "central":
            if raw in {"gi", "d"}:
                return "z"
            if raw == "tr":
                return "t"
            if raw == "ch":
                return "c"
        elif macro == "southern":
            if raw in {"gi", "d", "v"}:
                return "j"
            if raw == "q":
                return "w"
            if raw == "tr":
                special = {
                    _fold_label(label)
                    for label in self.transformations["dialect_profiles"]["southern"]
                    .get("tr_to_t_locations", [])
                }
                return "t" if region_label and _fold_label(region_label) in special else "c"
            if raw == "ch":
                return "c"
            if raw in {"s", "x"}:
                return "s"
        return onset

    @staticmethod
    def _replace_sequence(phones: list[str], input_phones: tuple[str, ...],
                          output_phones: tuple[str, ...]) -> list[str]:
        if not input_phones:
            return phones
        result = list(phones)
        position = 0
        while position <= len(result) - len(input_phones):
            if tuple(result[position:position + len(input_phones)]) == input_phones:
                result[position:position + len(input_phones)] = output_phones
                position += len(output_phones)
            else:
                position += 1
        return result

    def _apply_rhyme_rules(self, phones: list[str], syllable: ParsedSyllable,
                           scopes: tuple[str, ...]) -> list[str]:
        result = list(phones)
        for rule in self._rules:
            scope = rule.get("scope")
            applicable = set(scope if isinstance(scope, list) else [scope]) & set(scopes)
            if not applicable or rule.get("component") not in {"rhyme", "nucleus", "coda"}:
                continue
            if rule["id"] == "quang_nam_quang_ngai_open_close_vowels" and syllable.coda:
                continue
            if rule["id"] == "quang_nam_quang_ngai_labial_context_vowels":
                if syllable.coda not in {"m", "p"} and syllable.coda_spelling not in {"u", "o"}:
                    continue
            if rule["id"] == "quang_nam_quang_ngai_velar_context_vowels" and syllable.coda not in {"ŋ", "k"}:
                continue
            # The Mekong rules are more specific than the general Southern
            # palatalization rule for /i, e/ before velars.
            if (rule["id"] == "south_front_vowel_velar_coda_palatalization"
                    and "southern.mekong_delta" in scopes
                    and syllable.nucleus[0] in {"i", "e"}):
                continue

            variants = rule.get("rules", [])
            for variant in variants:
                input_ipa, output_ipa = variant.get("input_ipa"), variant.get("output_ipa")
                if not input_ipa or output_ipa is None:
                    continue
                condition = variant.get("condition", "")
                if "coda is /m/" in condition and syllable.coda != "m":
                    continue
                if "Only locations_with_tr_to_t" in condition:
                    continue
                result = self._replace_sequence(
                    result,
                    _ipa_tokens(input_ipa, self._ipa_symbols),
                    _ipa_tokens(output_ipa, self._ipa_symbols),
                )
        return result

    def _merged_tone(self, tone: str, scopes: tuple[str, ...]) -> str:
        for rule in self._rules:
            scope = rule.get("scope")
            applicable = set(scope if isinstance(scope, list) else [scope]) & set(scopes)
            if rule.get("component") != "tone" or not applicable:
                continue
            group = rule.get("equivalence_group", [])
            if tone in group:
                return "tone:merged:" + ":".join(sorted(group))
        return tone

    def encode(self, text: str, region: str | None = None) -> tuple[str, ...]:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        macro, subdialect, region_label = self.resolve_region(region)
        scopes = tuple(scope for scope in (macro, subdialect) if scope)
        words = re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", text.lower()), re.UNICODE)
        encoded: list[str] = []
        for word in words:
            syllable = self.parse_syllable(word)
            if not syllable.supported:
                phones = list(syllable.phones[:-1])
            else:
                onset = self._onset_for_region(syllable, macro, region_label)
                phones = []
                if onset:
                    phones.append(onset)
                if syllable.glide and not (macro == "southern" and syllable.onset_spelling == "q"):
                    phones.append(syllable.glide)
                phones.extend(syllable.nucleus)
                if syllable.coda:
                    phones.append(syllable.coda)
                phones = self._apply_rhyme_rules(phones, syllable, scopes)
            tone = self._merged_tone(syllable.tone, scopes)
            encoded.extend(phones)
            encoded.append(f"@{tone}")
            encoded.append("|")
        return tuple(encoded)

    def similarity(self, reference: str, hypothesis: str,
                   region: str | None = None) -> float:
        reference_phones = self.encode(reference, region)
        hypothesis_phones = self.encode(hypothesis, region)
        if not reference_phones:
            return 0.0
        if not hypothesis_phones:
            return 0.0
        denominator = max(len(reference_phones), len(hypothesis_phones))
        return max(0.0, 1.0 - _edit_distance(reference_phones, hypothesis_phones) / denominator)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "ViMD regional phonetic similarity",
            "algorithm": "1 - normalized Levenshtein distance over atomic IPA phones, tone IDs, and syllable boundaries",
            "vocabulary_path": str(self.vocabulary_path),
            "transformations_path": str(self.transformations_path),
            "region_policy": "apply matching paper macro/subdialect rules; canonical phones if region is unknown",
        }
