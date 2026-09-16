# Hướng nghiên cứu: nhận dạng tiếng Việt đa vùng miền và bảo toàn từ phương ngữ

**Đối tượng đọc:** sinh viên/nghiên cứu viên bắt đầu một đề tài ASR tiếng Việt  
**Ngày tổng hợp:** 04/09/2026  
**Phạm vi giả định:** luận văn hoặc dự án nghiên cứu 4–6 tháng; input là giọng nói, output chính là chữ; ưu tiên ba vùng lớn trước, chưa đặt streaming và code-switching làm yêu cầu bắt buộc.

## Kết luận ngắn

Hướng phù hợp nhất không phải chỉ là **“ASR chịu được giọng địa phương”**, mà là:

> **Dialect-preserving Vietnamese ASR:** nhận dạng tiếng Việt đa vùng miền, giữ nguyên từ phương ngữ trong bản chép, đồng thời dùng một nhánh âm vị để biểu diễn cách phát âm vùng miền.

Tên đề tài gợi ý:

> **Nhận dạng tiếng Việt đa vùng miền có bảo toàn từ phương ngữ bằng mô hình hóa đồng thời văn bản và âm vị**  
> *Dialect-Preserving Vietnamese ASR with Joint Orthographic and Phonetic Modeling*

Đóng góp khả thi nhất là xây dựng một tập kiểm thử nhỏ nhưng đáng tin cậy, có nhãn sâu, rồi chứng minh rằng mô hình đa nhiệm giảm việc “xóa” từ phương ngữ và giảm chênh lệch lỗi giữa các vùng. Không nên bắt đầu bằng việc huấn luyện mô hình từ đầu hoặc phân loại đủ 63 giọng tỉnh.

## 1. Trước hết phải định nghĩa “hiện đúng phát âm”

Cần tách hai hiện tượng:

- **Khác từ vựng:** người nói dùng “mô/răng/rứa/mần/hổng” thay cho “đâu/sao/vậy/làm/không”. Bản chép nguyên dạng phải giữ đúng từ địa phương.
- **Khác cách phát âm của cùng một từ:** cùng chữ “vội vã”, “bạn”, “cát” có thể được hiện thực khác nhau theo vùng. Nếu cố viết lại bằng chữ Quốc ngữ theo tai nghe, kết quả có thể thành một từ khác hoặc một cách viết không ổn định. Lớp này nên được biểu diễn bằng IPA hoặc cấu trúc `âm đầu – âm đệm – âm chính – âm cuối – thanh`.

Ví dụ thiết kế đầu ra:

| Tầng | Ví dụ | Mục đích |
|---|---|---|
| `raw_dialect_text` | “Mi đi mô rứa?” | Output ASR chính; giữ từ người nói đã dùng |
| `standard_text` | “Bạn đi đâu vậy?” | Tùy chọn, phục vụ tìm kiếm hoặc người đọc không biết phương ngữ |
| `surface_phone` | IPA/phone được gán theo âm thanh | Ghi cách phát âm thực; dùng cho PER và phân tích thanh/âm |

Vì cách viết phương ngữ không luôn được chuẩn hóa, một câu có thể có nhiều bản chép hợp lệ. Nghiên cứu về ASR phương ngữ đã chỉ ra rằng WER một tham chiếu dễ phạt oan các biến thể chính tả; có thể dùng nhiều tham chiếu hoặc danh sách biến thể được chấp nhận ([WERd, Ali et al.](https://arxiv.org/abs/1709.07484); [Multi-Reference WER](https://aclanthology.org/anthology-files/pdf/W/W15/W15-3213.pdf)).

**Quyết định nên chốt:** output chính là `raw_dialect_text`; `standard_text` là nhánh hậu xử lý riêng; `surface_phone` là output nghiên cứu/giải thích. Không cho bộ chuẩn hóa can thiệp vào bản chép nguyên dạng.

## 2. Câu hỏi nghiên cứu nên trả lời

Ba câu hỏi đủ rõ cho một luận văn:

1. Các mô hình ASR tiếng Việt hiện có sai khác như thế nào theo Bắc–Trung–Nam và riêng trên các từ phương ngữ?
2. Học đa nhiệm với nhãn âm vị/thanh có giúp giữ từ phương ngữ và giảm lỗi vùng khó mà không làm giảm độ chính xác chung không?
3. Cân bằng dữ liệu, conditioning theo vùng hoặc adapter vùng nào cho tỷ lệ hiệu quả/tài nguyên tốt nhất?

Giả thuyết chính:

- Fine-tune trên transcript nguyên dạng và sampling cân bằng theo vùng sẽ giảm raw WER/CER và tăng recall từ phương ngữ.
- Nhánh phone/thanh sẽ giảm nhầm lẫn có hệ thống ở âm đầu, vần và thanh, đặc biệt ở miền Trung.
- Language model chỉ học văn bản chuẩn có thể làm WER chung đẹp hơn nhưng tăng **dialect erasure rate**; lexicon/ngôn ngữ mô hình phải chứa văn bản phương ngữ.

## 3. Bằng chứng hiện có và khoảng trống

Tiếng Việt thường được khái quát thành ba vùng Bắc–Trung–Nam, nhưng mỗi vùng vẫn có biến thể cấp tỉnh và nhỏ hơn; hệ phụ âm, nguyên âm và thanh không giống nhau hoàn toàn ([Phạm & McLeod, 2016](https://pubmed.ncbi.nlm.nih.gov/27172848/)). Do đó, ba vùng là baseline hợp lý, không phải chân lý ngôn ngữ học cuối cùng.

[ViMD, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.426/) là nguồn gần nhất với đề tài: 102,56 giờ, 18.949 đoạn, 12.955 speaker ID và 63 nhãn tỉnh/thành theo cấu trúc dữ liệu lúc thu thập. Dữ liệu lấy từ phỏng vấn người địa phương trên đài tỉnh, loại giọng phát thanh viên, chia train/dev/test không trùng người nói. Hướng dẫn gán nhãn yêu cầu giữ nguyên từ địa phương; bài báo minh họa “làm… 50 ha” thành transcript nguyên dạng “mần… năm chục héc ta”. Tuy nhiên, mỗi nhãn tỉnh/thành chỉ khoảng 98 phút, nam nhiều hơn nữ khoảng ba lần, phong cách vẫn là phỏng vấn truyền hình, và chính tác giả ghi nhận lỗi ở từ địa phương/địa danh. Dataset card hiện ghi giấy phép **CC BY-NC-ND 4.0**, nên cần xác nhận quyền dùng, sửa nhãn và phát hành mô hình trước khi phân phối sản phẩm ([ViMD dataset card](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset)).

Kết quả ViMD cho thấy sau fine-tune, WER tốt nhất được báo cáo khoảng **12,17% Bắc, 17,15% Trung và 13,54% Nam**; miền Trung khó nhất. Phân loại ba vùng đạt F1-macro trên 85%, nhưng phân loại 63 tỉnh tốt nhất chỉ khoảng 41,07%. Vì vậy, không nên xây 63 bộ giải mã hoặc 63 adapter ngay từ đầu. Một mô hình chung, conditioning ba vùng và đánh giá chi tiết theo tỉnh là phạm vi hợp lý hơn.

[PhoASR, Findings EACL 2026](https://aclanthology.org/2026.findings-eacl.345/) hợp nhất và lọc nhiều corpus tiếng Việt thành 502,67 giờ. Khi cân bằng 64 giờ mỗi vùng và fine-tune Whisper-small trên tập gộp 192 giờ, N-WER chung đạt 12,97%, tốt hơn từng mô hình chỉ học một vùng; miền Trung vẫn cao hơn (16,56% so với 12,18% Bắc và 12,36% Nam). Hai kết quả không mâu thuẫn: dữ liệu đa vùng có ích, nhưng gộp dữ liệu đơn thuần chưa xóa được khoảng cách vùng. PhoASR cũng cảnh báo lọc dữ liệu bằng ASR có thể loại không cân xứng các giọng hoặc điều kiện âm thanh mà model nguồn xử lý kém.

Một preprint rất sát đề tài, [Phonetic Modeling of Dialectal Variation in Vietnamese Speech, 2026](https://arxiv.org/abs/2605.24451), đề xuất tokenizer âm vị theo vùng và decoder dự đoán đồng thời âm đầu–vần–thanh. Trên ViMD, mô hình báo cáo WER 13,05% và PER 8,36%. Đây là tín hiệu mạnh rằng mô hình hóa âm vị đáng thử, nhưng bài còn là preprint, code chưa được phát hành tại thời điểm tổng hợp, nhãn phone được sinh từ quy tắc theo vùng chứ không phải toàn bộ được chuyên gia nghe–gán, và tác giả thừa nhận ánh xạ phone ngược về chữ còn mơ hồ, đặc biệt ở miền Nam. Kết quả này nên được tái lập độc lập, không xem như kết luận đã đóng.

Khoảng trống rõ nhất hiện nay là chưa có corpus công khai quy mô đáng kể đồng thời có:

- audio hội thoại tự nhiên;
- thông tin người nói/vùng đáng tin cậy;
- transcript nguyên dạng giữ phương ngữ;
- bản chuẩn hóa nghĩa song song;
- nhãn phone/IPA bám theo âm thanh cho ít nhất một tập con.

Đây chính là chỗ đề tài có thể đóng góp.

## 4. Dữ liệu nên dùng

| Nguồn | Dùng để làm gì | Điểm mạnh | Hạn chế quyết định |
|---|---|---|---|
| [ViMD](https://aclanthology.org/2024.emnlp-main.426/) | Corpus chính để fine-tune và benchmark | 102,56 giờ; 63 nhãn tỉnh/thành lúc thu thập; speaker-disjoint; giữ từ địa phương | Broadcast interview; ít phút mỗi nhãn địa phương; nhãn phone không có; CC BY-NC-ND |
| [PhoASR pipeline/model](https://github.com/qualcomm-ai-research/PhoASR) | Checkpoint mạnh và quy trình làm sạch/align | Pipeline tái lập; checkpoint Whisper-small tiếng Việt; timestamp | Không phải benchmark chuyên biệt về giữ phương ngữ; region một phần do classifier dự đoán |
| [LSVSC](https://www.mdpi.com/2079-9292/13/5/977) | Bổ sung speech tự nhiên và domain | 100,5 giờ; transcript thủ công; nhiều chủ đề | 88,1% giọng Bắc; paper không nêu giấy phép dữ liệu chuẩn rõ ràng |
| [Common Voice vi v23](https://dev.datacollective.mozillafoundation.org/datasets/cmflnn484oaiy1nwkq2dp76ig) | Sanity test/augmentation sạch | 22 giờ, 6,3 giờ validated; 361 người; CC0; có trường accent tùy chọn | Đọc câu soạn sẵn; accent tự khai và thiếu nhiều |
| [ViDia2Std](https://ojs.aaai.org/index.php/AAAI/article/view/40247) | Trích từ điển phương ngữ, tạo prompt và LM | 13.657 cặp dialect–standard, 63 tỉnh, người gán nhãn thủ công | Chỉ có text, không huấn luyện acoustic trực tiếp; CC BY-NC 4.0 |
| [vPhon](https://github.com/kirbyj/vPhon) | Khởi tạo IPA giả cho ba giọng đại diện | Xuất IPA cho Hà Nội–Huế–Sài Gòn; GPL-3.0 | Là G2P từ chữ, không “nghe” audio; tác giả cảnh báo Trung không đồng nhất và nhiều biến thể chưa biểu diễn |

VIVOS/FLEURS có thể dùng làm sanity test cho speech đọc chuẩn. Các bộ hàng trăm hoặc hàng nghìn giờ nhưng thiếu speaker/region metadata chỉ nên dùng để tiếp tục pretrain hoặc huấn luyện nền; không dùng làm gold dialect test.

### Tập dữ liệu mới tối thiểu nên tự tạo

Nếu nguồn lực hạn chế, ưu tiên **chất lượng benchmark** hơn số giờ:

- MVP đánh giá: 30–60 người, cân bằng ba vùng, 5–10 giờ, tối thiểu 300–500 câu có từ phương ngữ.
- Mức luận văn tốt: 60–120 người, 15–30 giờ, có nhiều thiết bị và môi trường nhưng mỗi điều kiện phải có metadata.
- Nội dung: khoảng 40% prompt có chủ đích chứa từ phương ngữ/minimal pairs, 40% hội thoại tự nhiên, 20% speech đọc song song cùng nội dung để tách ảnh hưởng phát âm khỏi lựa chọn từ.
- Thu quê quán, nơi lớn lên, lịch sử cư trú, vùng tự nhận, tuổi, giới, thiết bị, môi trường và kiểu nói. “Tỉnh đang sống” không đủ để khẳng định giọng.
- Giữ file gốc; tạo bản 16 kHz mono cho huấn luyện. Cắt câu khoảng 2–20 giây, đánh dấu chồng tiếng, không rõ, tiếng cười và code-switch.
- Mỗi câu có hai người chép độc lập; bất đồng được người thứ ba phân xử. Với IPA, dùng vPhon làm nhãn khởi tạo rồi chuyên gia nghe và sửa một tập gold 1–3 giờ; không coi output vPhon là ground truth.
- Chia theo `speaker_id`, không theo đoạn; nếu đủ dữ liệu, giữ riêng vài tỉnh chưa thấy để đo generalization.
- Có đồng thuận sử dụng giọng nói, phương án ẩn danh/xóa dữ liệu và giấy phép phát hành rõ từ đầu.

Schema tối thiểu:

```text
audio_path, speaker_id, province, macro_region,
raw_dialect_text, standard_text, dialect_term_spans,
surface_phone_optional, speech_style, device, noise,
code_switch, annotator_ids, adjudication_status
```

## 5. Kiến trúc và thí nghiệm

### Baseline bắt buộc

1. **PhoWhisper-small**: dễ có baseline mạnh; PhoWhisper được fine-tune trên khoảng 844 giờ tiếng Việt đa giọng ([PhoWhisper paper](https://arxiv.org/abs/2406.02555)). Phải đo riêng hallucination và việc decoder tự sửa từ hiếm thành từ phổ thông.
2. **Wav2Vec2 tiếng Việt hoặc XLS-R + CTC ký tự**: đầu ra bám âm thanh và bảng chữ dễ kiểm soát; thuận lợi thêm head phone/thanh. Tự giám sát giúp giảm nhu cầu nhãn lớn ([wav2vec 2.0](https://arxiv.org/abs/2006.11477)); XLS-R mở rộng pretraining cho 128 ngôn ngữ ([XLS-R](https://arxiv.org/abs/2111.09296)).
3. **PhoASR-whisper-small**: strong zero-shot checkpoint hiện hành để biết mức trần thực dụng, nhưng không mặc định nó bảo toàn phương ngữ.

PhoWhisper và CTC phải cùng train/dev/test và cùng quy tắc chuẩn hóa. Không kết luận kiến trúc nào thắng từ benchmark khác miền.

### Mô hình đề xuất

```text
audio
  │
shared speech encoder
  ├── raw-text head ──> bản chép giữ từ phương ngữ
  ├── phone/tone head ──> âm đầu, vần, thanh hoặc IPA
  └── region head (phụ) ──> Bắc / Trung / Nam

raw-text ──> bộ chuẩn hóa riêng, tùy chọn ──> standard text
```

Hàm mất mát có thể là:

`L = L_raw_text + λ_phone L_phone + λ_region L_region`

Đi từ đơn giản đến phức tạp:

| ID | Hệ thống | Câu hỏi trả lời |
|---|---|---|
| B0 | Model có sẵn, không fine-tune | Baseline zero-shot và mức thiên lệch ban đầu |
| B1 | Fine-tune pooled multi-dialect | Fine-tune thông thường giúp bao nhiêu? |
| B2 | B1 + sampling cân bằng vùng/từ phương ngữ | Lợi ích đến từ phân phối dữ liệu hay kiến trúc? |
| B3 | B2 + region token/auxiliary classifier | Biết vùng có giúp mà không cần nhiều model? |
| B4 | B2 + head phone/thanh | Đóng góp chính: cấu trúc âm vị có giúp không? |
| B5 | B4 + adapter riêng ba vùng | Có đạt tốt hơn với ít tham số cập nhật? |
| B6 | B4/B5 + lexicon/contextual biasing | Từ phương ngữ hiếm có được giữ tốt hơn không? |

Adapter là lựa chọn hợp lý nếu compute hạn chế: nghiên cứu EMNLP 2021 cho thấy residual adapter có thể đạt mức cải thiện gần full fine-tuning khi cập nhật dưới 0,5% tham số, dù kết quả đó chưa phải trên tiếng Việt ([Tomanek et al.](https://research.google/pubs/residual-adapters-for-parameter-efficient-asr-adaptation-to-atypical-and-accented-speech/)). Cần so sánh `oracle region`, region do classifier dự đoán và một shared adapter cho vùng chưa thấy.

Không nên lấy domain-adversarial/accent-invariant làm hướng chính vì nó chủ ý xóa tín hiệu vùng khỏi representation, có thể xung đột với mục tiêu giữ phương ngữ. Có thể dùng làm đối chứng.

## 6. Metric đúng với mục tiêu

Không chọn mô hình chỉ bằng một con số normalized WER.

### Chỉ số chính

- **Raw WER và raw CER:** không đổi từ phương ngữ sang từ phổ thông, không bỏ dấu thanh.
- **Dialect-term precision/recall/F1:** tính riêng trên span “mô/răng/rứa/mần/hổng/…” và các cụm trong lexicon.
- **Dialect erasure rate:** tỷ lệ từ phương ngữ bị thay bằng từ phổ thông tương đương.
- **PER:** trên tập phone gold; phân rã lỗi âm đầu, âm chính/vần, âm cuối và thanh.
- **Theo nhóm:** WER/CER/PER cho từng vùng, tỉnh, giới, tuổi, kiểu nói, thiết bị và mức nhiễu; báo cáo macro-average, worst-group và chênh lệch max–min.
- **Loại lỗi:** substitution/deletion/insertion, hallucination, lỗi địa danh/tên riêng, lỗi code-switch nếu có.

CER nên bổ sung cho WER vì đánh giá đa ngôn ngữ cho thấy CER ổn định hơn trước khác biệt ranh giới từ và tương quan tốt với đánh giá con người hơn trong tập khảo sát của nghiên cứu ([NAACL 2025](https://aclanthology.org/2025.findings-naacl.277/)). Tuy nhiên CER vẫn không tự nói lỗi nằm ở thanh, âm đầu hay việc xóa từ địa phương, nên không thay thế PER và dialect-term recall.

### Quy tắc đánh giá

- Định nghĩa một phiên bản normalization cố định và công bố code.
- Raw score không được lowercase/strip dấu theo cách làm mất thông tin mục tiêu; punctuation có thể báo riêng.
- Với cách viết phương ngữ không thống nhất, dùng multi-reference hoặc `acceptable_variants` và báo cả strict lẫn flexible score.
- Bootstrap theo người nói để tạo khoảng tin cậy; không xem hàng nghìn đoạn của cùng một người là quan sát độc lập.
- Chốt test set trước khi tinh chỉnh và không dùng nó để làm lexicon/bias list.

## 7. Những gì cần học, theo thứ tự

1. **Ngữ âm–âm vị tiếng Việt:** IPA; phoneme, phone, allophone; cấu trúc âm tiết; âm đầu/vần/thanh; merger theo vùng; khác nhau giữa accent và dialect vocabulary.
2. **Nền tảng ASR:** waveform, sample rate, log-Mel; CTC; attention encoder–decoder; Whisper; beam search và language model; WER/CER/PER.
3. **Transfer learning:** fine-tune pretrained model, freezing, LoRA/adapter, augmentation, balanced sampling và catastrophic forgetting.
4. **Dữ liệu tiếng nói:** VAD/segmentation, resampling, loudness/noise, forced alignment, speaker-disjoint split, phát hiện leakage và transcript QA.
5. **NLP phương ngữ:** lexicon phương ngữ–toàn dân, chuẩn hóa văn bản nhưng không phá bản raw, contextual biasing và multi-reference evaluation.
6. **Thực nghiệm:** ablation, seed lặp lại, confidence interval theo speaker, error analysis theo vùng, theo từ và theo thành phần âm tiết.
7. **Tái lập và đạo đức:** version dataset/model, data card, consent, quyền rút giọng, ẩn danh, giấy phép nguồn và giới hạn phát hành.

## 8. Kế hoạch 16 tuần

| Tuần | Việc chính | Sản phẩm kiểm chứng |
|---|---|---|
| 1–2 | Chốt task/output; đọc ViMD, PhoWhisper, PhoASR, vPhon; lập lexicon ban đầu | Đặc tả nhãn 2–3 trang và 50 ví dụ đã phân xử |
| 3–4 | Tải/audit dữ liệu; chạy B0 trên ViMD; viết scorer theo vùng/từ | Báo cáo lỗi baseline, danh sách top confusion |
| 5–6 | Làm gold test nhỏ; hai người chép; kiểm tra agreement | Test set speaker-disjoint bị khóa; annotation guideline v1 |
| 7–8 | Fine-tune B1, B2 trên PhoWhisper và một CTC model | Bảng raw WER/CER, dialect recall và worst-region |
| 9–11 | Xây B3/B4: region auxiliary + phone/tone head | Ablation cho λ và từng thành phần âm tiết |
| 12–13 | Adapter hoặc contextual biasing, chỉ chọn một nếu thời gian ít | B5 hoặc B6 và phân tích over-bias |
| 14–15 | Lặp seed, bootstrap theo speaker, phân tích lỗi thủ công | Bảng kết quả cuối + confidence intervals |
| 16 | Viết luận văn, model/data card, đóng gói code | Repo tái lập, checklist giấy phép và giới hạn |

### Việc cần làm trong 30 ngày đầu

- Viết một trang xác định ba output và điều gì **không** làm: chưa streaming, chưa 63 model, chưa code-switch nếu không phải nhu cầu chính.
- Tải ViMD, đọc ngẫu nhiên/nghe ít nhất 200 đoạn theo ba vùng; đếm tỷ lệ từ phương ngữ thật, lỗi nhãn và điều kiện nhiễu.
- Chạy PhoWhisper-small/PhoASR và một Wav2Vec2 CTC trên đúng 200 đoạn đó; không fine-tune trước khi hiểu lỗi.
- Tạo lexicon v0 từ ViDia2Std + phụ lục ViMD, rồi gán `dialect_term_spans` cho 300–500 câu test.
- Viết scorer cho raw WER, CER, dialect-term recall, erasure và WER theo vùng.
- Fine-tune baseline pooled và balanced. Chỉ sau đó mới quyết định phone head, adapter hay biasing là đóng góp chính.

## 9. Tiêu chí quyết định sau pilot

- Nếu off-the-shelf WER thấp nhưng erasure cao: tập trung dialect lexicon, raw labels và objective giữ từ.
- Nếu lỗi chủ yếu theo âm đầu/vần/thanh: ưu tiên phone/tone auxiliary head.
- Nếu miền Trung kém rõ nhưng từng vùng riêng tốt: thử balanced sampling và adapter/conditioning.
- Nếu transcript ViMD sai nhiều ở từ/địa danh: không mở rộng model trước; xây gold test và sửa nhãn tập con.
- Nếu IPA agreement thấp: thu hẹp phone inventory hoặc chỉ gán `initial/rhyme/tone`, thay vì narrow IPA quá chi tiết.
- Nếu không có GPU lớn: PhoWhisper-small + LoRA/adapter hoặc Wav2Vec2-base; đóng góp nằm ở dữ liệu, objective và metric, không cần model tỷ tham số.

## 10. Rủi ro cần tránh

- Gọi mọi khác biệt là “phát âm sai”; giọng địa phương là biến thể có hệ thống, không phải noise.
- Dùng transcript chuẩn hóa làm nhãn duy nhất rồi tuyên bố đã bảo toàn phương ngữ.
- Bỏ dấu khi tính metric chính; điều này xóa thông tin thanh.
- Chia train/test theo file thay vì speaker, gây leakage.
- Dùng vPhon-generated IPA làm ground truth không kiểm tra bằng tai/chuyên gia.
- Dùng dữ liệu lớn không rõ speaker/provenance làm test set.
- Chỉ báo WER trung bình; kết quả có thể tốt lên trong khi vùng yếu vẫn bị bỏ lại.
- Phát hành lại audio/model mà chưa kiểm tra giấy phép từng nguồn, đặc biệt các bộ NC/ND hoặc dữ liệu lấy từ video công khai.

## Khuyến nghị cuối

Phiên bản đề tài vừa đủ mới, vừa khả thi là:

> **Fine-tune PhoWhisper-small và một Wav2Vec2-CTC trên ViMD bằng transcript nguyên dạng; xây tập gold giàu từ phương ngữ; đề xuất head phụ âm đầu–vần–thanh; đánh giá bằng raw WER/CER, dialect-term recall, dialect erasure, PER và worst-region gap.**

Nếu thời gian chỉ đủ cho một đóng góp, hãy ưu tiên **tập gold + protocol đánh giá dialect-preserving**. Một benchmark nhỏ, gán nhãn rõ và speaker-disjoint có giá trị khoa học hơn thêm một model phức tạp nhưng đánh giá vẫn chỉ bằng WER trung bình.

## Nguồn cốt lõi

- Nguyen Van Dinh et al., 2024, [Multi-Dialect Vietnamese: Task, Dataset, Baseline Models and Challenges](https://aclanthology.org/2024.emnlp-main.426/).
- Thi Vu et al., 2026, [Vietnamese Automatic Speech Recognition: A Revisit / PhoASR](https://aclanthology.org/2026.findings-eacl.345/).
- Thanh-Thien Le et al., 2024, [PhoWhisper](https://arxiv.org/abs/2406.02555).
- Quan Ngoc Hoang et al., 2026, [Phonetic Modeling of Dialectal Variation in Vietnamese Speech](https://arxiv.org/abs/2605.24451) — preprint, chưa qua phản biện chính thức tại thời điểm tổng hợp.
- Khoa Anh Ta et al., 2026, [ViDia2Std](https://ojs.aaai.org/index.php/AAAI/article/view/40247).
- James Kirby, [vPhon 2.1.1](https://github.com/kirbyj/vPhon).
- Ben Phạm & Sharynne McLeod, 2016, [Consonants, vowels and tones across Vietnamese dialects](https://pubmed.ncbi.nlm.nih.gov/27172848/).
- Alexei Baevski et al., 2020, [wav2vec 2.0](https://arxiv.org/abs/2006.11477).
- Arun Babu et al., 2021/2022, [XLS-R](https://arxiv.org/abs/2111.09296).

## Giới hạn của báo cáo

Thông tin được rà đến 04/09/2026. Một số corpus công khai thiếu mô tả giấy phép/provenance thống nhất; báo cáo không thay thế tư vấn pháp lý hoặc phê duyệt đạo đức của trường. Kết quả phonetic-ASR sát nhất là preprint tháng 5/2026 và cần được tái lập. Chưa có thử nghiệm trực tiếp trên dữ liệu hoặc phần cứng của người thực hiện, nên số giờ thu và model size cần điều chỉnh sau pilot 200 đoạn.
