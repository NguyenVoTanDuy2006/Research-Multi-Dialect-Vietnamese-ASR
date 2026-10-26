# Vietnamese phonetic vocabulary

This directory contains a structured transcription of the phonetic inventories and regional pronunciation rules in *Phonetic Modeling of Dialectal Variation in Vietnamese Speech* (local copy: [`docs/2605.24451v1.pdf`](../../../../docs/2605.24451v1.pdf)). The JSON files are data specifications; [`VietnamesePhoneticEncoder`](../reimplement_cba/adapters/vietnamese_phonetics.py) loads them for the phonetic SemScore component.

## Files

- [`vi_phoneme_vocabulary.json`](../resources/phonetics/vi_phoneme_vocabulary.json) defines the paper's onset, glide, vowel, coda, and tone units, plus Quốc ngữ spelling rules.
- [`vi_dialect_transformations.json`](../resources/phonetics/vi_dialect_transformations.json) defines the Northern, Central, Southern, and listed subdialect transformations.

## Conversion flow

The encoder treats each whitespace-delimited Quốc ngữ syllable as the paper's basic input unit. It normalizes text to Unicode NFC, preserves tone and vowel marks, and parses each syllable into:

```text
onset? + glide? + vowel nucleus + coda? + tone
```

Resolve orthographic alternatives such as `k/c/q`, `g/gh`, and `ng/ngh` using their context rules. Then apply the macro-dialect rules followed by a matching subdialect rule. Keep the canonical phoneme sequence and the dialectal surface sequence separately. For SemScore, compare atomic phoneme tokens; do not split IPA strings into Unicode characters or discard tone marks.

Example at the level shown in the paper: `x` and `s` are distinct canonical onsets, but the Northern rule maps both to the same surface phone. They should therefore be equivalent for a Northern pronunciation comparison while remaining distinguishable in the canonical representation.

## Source decisions and caveats

1. **Onset inventory:** Appendix B claims 22 initials and 26 Quốc ngữ writing forms, but the printed mapping list omits `h`. Figure 6 decomposes `hoàng` with `/h/`, so `/h/` is included and marked `shown_in_figure_only` in the JSON. The paper's count of 22 and spelling variants otherwise reconcile when this onset is present.
2. **Inventory sizes:** Appendix B's 22 onsets, 15 nuclei, 10 codas, and 6 tones describe phonetic units. The paper separately gives 27 initial tokens and 240 rhyme tokens for its decoder. Do not use those counts as if they were the same inventory.
3. **IPA notation:** Some PDF glyphs do not survive text extraction, and some surface realizations use superscripts or length/rounding marks. The JSON preserves clearly printed symbols where possible and marks uncertain transcriptions rather than guessing. In particular, review Northern rounded-vowel/velar-coda outputs and Mekong rounded-vowel outputs before using them as exact acoustic targets.
4. **Dialect and tone mergers:** A merger is represented as an equivalence group, not as a directional rewrite. The PDF's contour glyphs for regional tone mergers are malformed; the tone IDs are associated with the Appendix B contour descriptions and explicitly flagged for checking before collapsing tone labels.
5. **Province coverage:** The province lists reproduce the article and contain omissions, duplicates, and names that may not align with current administrative boundaries. Use ViMD's own region metadata as the primary source. If a label does not match an explicit paper profile, keep the canonical sequence and flag the sample instead of guessing.
6. **English spans:** These vocabularies only describe Vietnamese phonetics. If following the paper's transcript policy for bilingual audio, remove non-Vietnamese transcript tokens in preprocessing while retaining the audio; that policy is separate from regional phoneme conversion.

## Scope for implementation

The current adapter exposes these operations:

```python
encoder = VietnamesePhoneticEncoder()
syllable = encoder.parse_syllable("người")
phones = encoder.encode("người", region="Hà Nội")
similarity = encoder.similarity("người", "người", region="Hà Nội")
```

The encoder applies explicit JSON rules and the onset mergers for the selected profile. It does not guess IPA outputs for rules marked unresolved in the JSON; those acoustic details still need review. Unsupported syllables are retained as marked grapheme tokens. Keep the canonical sequence and dialect surface sequence separate when extending this implementation.

Source page numbers in these files refer to the paper's printed page numbers (Appendix B pp. 12–14; Appendix C pp. 15–19; Appendix F/Figure 9 p. 20).
