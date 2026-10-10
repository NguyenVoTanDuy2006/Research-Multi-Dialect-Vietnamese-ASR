from __future__ import annotations

from types import SimpleNamespace

from mdv_asr.analysis.lexicon import LexicalSignificanceTester, LexicalStatisticsBuilder


def test_speaker_table_uses_all_province_speakers():
    records = [
        SimpleNamespace(province_name="A", speaker_id="a1", text="tui nói"),
        SimpleNamespace(province_name="A", speaker_id="a2", text="tui đây"),
        SimpleNamespace(province_name="B", speaker_id="b1", text="tui nói"),
        SimpleNamespace(province_name="B", speaker_id="b2", text="kia kia"),
    ]
    stats = LexicalStatisticsBuilder().build(records)
    table = LexicalSignificanceTester.build_speaker_table(
        "nói", "A", stats["speaker_presence"], stats["province_speakers"]
    )
    assert table == (1, 1, 1, 1)


def test_fdr_is_bounded_and_monotone_by_rank():
    adjusted = LexicalSignificanceTester.adjust_fdr([0.001, 0.02, 0.5])
    assert all(0 <= value <= 1 for value in adjusted)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
