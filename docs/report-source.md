# Canonical research source — Dialect-preserving Vietnamese ASR

## Metadata

- Audience: Vietnamese university student/researcher planning a 4–6 month ASR thesis.
- Research date: 2026-09-04.
- Scope: voice input; main output is verbatim dialect-preserving Vietnamese text; optional standardized text and phonetic realization; three macro-regions first; offline ASR; code-switching and streaming excluded from the MVP.
- Assumptions: modest academic compute; public data plus a small newly annotated evaluation set; no direct experiments were run for this research report.
- User-facing deliverable: `bao-cao-huong-nghien-cuu-asr-tieng-viet-phuong-ngu.md`.

## Direct answer

The recommended thesis direction is **Dialect-Preserving Vietnamese ASR with Joint Orthographic and Phonetic Modeling**. The main transcript must retain dialect vocabulary, while a separate phonetic head represents regional pronunciation. Standardization into common Vietnamese must be optional and downstream; otherwise it destroys the evidence the project is meant to preserve.

The smallest defensible contribution is a trustworthy, speaker-disjoint dialect-rich gold test set plus an evaluation protocol. The preferred model contribution is a shared speech encoder with raw-text and initial–rhyme–tone heads, tested against pooled and region-balanced PhoWhisper/Wav2Vec2 baselines.

## Analysis

### Output contract

Use three explicitly separated layers:

1. `raw_dialect_text`: lexical words actually used, including mô/răng/rứa/mần/hổng.
2. `standard_text`: optional semantic normalization for retrieval or non-dialect readers.
3. `surface_phone`: audio-derived IPA or structured initial–rhyme–tone labels.

Vietnamese orthography does not uniquely or safely encode every regional surface realization. Therefore a listener's regional realization of an otherwise ordinary word should not be represented by deliberately misspelling the main transcript. Orthographic variants without a single convention require strict and flexible scoring, with accepted variants or multiple references.

### Evidence synthesis and research gap

- ViMD is the closest public task/data source: 102.56 hours, 18,949 clips, 12,955 speaker IDs and 63 province/city labels under the data collection scheme. It uses local provincial-station interviews, removes presenters, preserves vernacular forms in transcription, and has speaker-exclusive splits. Its weaknesses are broadcast-interview style, roughly 98 minutes per local label, male skew, and residual errors on local words/place names. Best reported fine-tuned WER is much worse for Central speech (17.15%) than North (12.17%) and South (13.54%). The dataset card currently says CC BY-NC-ND 4.0; downstream modification and release rights require project-specific checking.
- PhoASR shows that diverse, balanced pooling can help: with 64 hours from each macro-region, pooled Whisper-small reaches 12.97 N-WER overall and beats region-only models, but Central remains worse (16.56 versus 12.18 North and 12.36 South). This complements rather than contradicts ViMD's finding that naive pooling produces only limited gains: pooling is useful, but does not solve the dialect gap alone.
- PhoWhisper supplies a practical strong encoder-decoder baseline trained on about 844 hours of Vietnamese. Its decoder can also normalize rare dialect tokens or hallucinate, so lexical erasure must be measured explicitly.
- Wav2Vec2/XLS-R character CTC provides a complementary, acoustically grounded baseline and is easier to extend with a phonetic auxiliary head.
- The May 2026 phonetic-ASR preprint reports 13.05 WER and 8.36 PER on ViMD using dialect-aware phonetic vocabulary and structured initial–rhyme–tone decoding. This motivates the main hypothesis, but is not settled evidence: it is a preprint, code was not available at the research date, phone labels are rule-derived rather than all expert audio annotations, and phone-to-text reconstruction is ambiguous.
- ViDia2Std contributes 13,657 manually annotated dialect–standard text pairs across 63 local labels. It is useful for lexicon extraction, prompts and contextual biasing, not acoustic training.
- vPhon can bootstrap Hanoi/Hue/Saigon IPA labels from orthography, but it does not listen to audio and does not exhaust within-region variation. Generated labels must be manually corrected on a gold subset.

### Research questions

1. How do current Vietnamese ASR systems fail by macro-region and specifically on dialect vocabulary?
2. Does a phonetic/tone auxiliary objective improve dialect-term retention and worst-region accuracy without degrading overall accuracy?
3. What is the resource/performance trade-off among balanced sampling, region conditioning and parameter-efficient regional adapters?

### Experiment ladder

- B0: off-the-shelf PhoWhisper-small, a Vietnamese Wav2Vec2/XLS-R CTC checkpoint, and PhoASR Whisper-small where available.
- B1: pooled fine-tuning on raw ViMD transcripts.
- B2: region-balanced sampling.
- B3: region token or auxiliary macro-region classifier.
- B4: main proposal, joint raw-text and initial–rhyme–tone objective.
- B5: regional residual adapters, comparing oracle and predicted region routing.
- B6: dialect lexicon/contextual biasing.

Proposed objective: `L = L_raw_text + lambda_phone * L_phone + lambda_region * L_region`. Standard-text normalization is trained/evaluated separately. Accent-invariant adversarial learning is an ablation at most because erasing region information can conflict with preserving dialectal realizations.

### Data and evaluation

For an MVP, collect 5–10 hours from 30–60 speakers balanced across North/Central/South; a stronger thesis target is 15–30 hours from 60–120 speakers. Mix designed dialect prompts/minimal pairs, natural conversation, and parallel reading. Record birthplace, formative residence, present residence history, self-identified accent, age/gender, device, environment and speaking style. Preserve originals, derive 16 kHz mono training audio, mark overlaps/laughter/code-switching, and split by speaker; optionally hold out localities.

Use two independent transcribers plus adjudication. Keep raw, standard and phonetic layers distinct. vPhon may initialize phone labels, but 1–3 hours should be manually corrected as a phone-gold evaluation subset.

Primary metrics: raw WER, raw CER, dialect-term precision/recall/F1, dialect-erasure rate, PER and initial/rhyme/tone error. Report substitutions/deletions/insertions and hallucination; disaggregate by region, locality, gender, age, style, device and noise; include macro average, worst group, max–min gap, ratio and speaker-level bootstrap confidence intervals. Never strip Vietnamese tone marks in the primary metric. Lock the test set before model iteration.

### Roadmap

- Weeks 1–2: finalize output contract, annotation guide and dialect lexicon; adjudicate 50 examples.
- Weeks 3–4: audit 200 ViMD clips; run B0; build scoring and error taxonomy.
- Weeks 5–6: collect/annotate and freeze the dialect-rich gold test set.
- Weeks 7–8: run pooled and balanced baselines on PhoWhisper and CTC.
- Weeks 9–11: implement and ablate region and phonetic heads.
- Weeks 12–13: choose exactly one extension: adapters or contextual biasing.
- Weeks 14–15: repeat seeds, bootstrap intervals, subgroup/error analysis.
- Week 16: write, package reproducibility artifacts and document license/ethics constraints.

## Limitations

- No model was trained and no user hardware/data were inspected, so collection hours and model size must be adjusted after the 200-clip pilot.
- Public corpora have heterogeneous provenance, metadata and license terms. The report records published/card labels but is not legal advice.
- Three macro-regions are only an operational baseline; substantial within-region phonetic variation remains.
- The closest joint phonetic result is a 2026 preprint and needs independent replication.
- A filter driven by upstream ASR confidence can preferentially discard difficult accents/environments; filtered corpora are not automatically fair benchmarks.

## Recommendations

1. Make `raw_dialect_text` the contractual output and never replace it with normalized Vietnamese.
2. Spend the first month on a 200-clip audit, baseline error table, lexicon v0 and a 300–500-item dialect-rich seed benchmark.
3. Use PhoWhisper-small plus one CTC baseline; do not train from scratch or build 63 routed models first.
4. Run balanced pooling before architectural novelty, then test the phonetic/tone head as the primary contribution.
5. If time permits only one contribution, prioritize the gold benchmark and dialect-preserving scoring protocol.

## Claim-to-source ledger

| Claim | Source | Source role / note |
|---|---|---|
| ViMD design, size, provincial labels, vernacular transcription, splits, baseline gaps and limitations | https://aclanthology.org/2024.emnlp-main.426/ | Peer-reviewed EMNLP 2024 primary paper |
| ViMD current distribution license label | https://huggingface.co/datasets/nguyendv02/ViMD_Dataset | Official dataset card; verify before reuse/release |
| Balanced 3-region pooled data and Central-region gap; refinement-pipeline caveat | https://aclanthology.org/2026.findings-eacl.345/ | Peer-reviewed Findings EACL 2026 primary paper |
| PhoASR code/checkpoint availability | https://github.com/qualcomm-ai-research/PhoASR | Official author repository |
| PhoWhisper training scale and model family | https://arxiv.org/abs/2406.02555 | Author preprint and primary model report |
| Vietnamese dialect phonetic variation | https://pubmed.ncbi.nlm.nih.gov/27172848/ | Peer-reviewed phonology review |
| Joint dialect-aware phonetic model, WER/PER, label/reconstruction caveats | https://arxiv.org/abs/2605.24451 | May 2026 preprint; replicate independently |
| ViDia2Std size and coverage | https://ojs.aaai.org/index.php/AAAI/article/view/40247 | Peer-reviewed AAAI 2026 primary paper |
| vPhon capabilities, license and stated coverage limits | https://github.com/kirbyj/vPhon | Official tool repository |
| Wav2Vec2 low-label self-supervised transfer | https://arxiv.org/abs/2006.11477 | Primary model paper |
| XLS-R multilingual pretraining scale | https://arxiv.org/abs/2111.09296 | Primary model paper |
| Parameter-efficient ASR accent adapters | https://research.google/pubs/residual-adapters-for-parameter-efficient-asr-adaptation-to-atypical-and-accented-speech/ | Primary EMNLP 2021 publication page; non-Vietnamese evidence |
| Single-reference WER can mishandle dialect orthography | https://arxiv.org/abs/1709.07484 | Dialect-ASR metric proposal |
| Multiple references for ASR evaluation | https://aclanthology.org/anthology-files/pdf/W/W15/W15-3213.pdf | Peer-reviewed evaluation method |
| CER robustness evidence | https://aclanthology.org/2025.findings-naacl.277/ | Peer-reviewed NAACL Findings 2025; complementary, not sufficient alone |

## Research process and stopping reason

Research prioritized primary papers, official dataset cards and author repositories. Search lanes covered Vietnamese multi-dialect corpora/phonology and ASR methods/evaluation, followed by reconciliation of conflicting-looking evidence on pooled training. Research stopped after the central claims had at least one authoritative source, the most recent directly relevant 2026 work had been reviewed, and further searches were yielding adjacent corpora rather than changing the recommended task definition or experiment design.
