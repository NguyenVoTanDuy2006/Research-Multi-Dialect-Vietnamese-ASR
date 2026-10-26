# -*- coding: utf-8 -*-
# Python script with VS Code/Jupyter-style cells.

# %% [markdown]
# # Chạy thử CBA trên ViMD small
#
# Script này dùng trực tiếp `reimplement_cba.workflow.run`: baseline PhoWhisper, fine-tune AdaLoRA, lọc curriculum, chọn checkpoint trên validation rồi đánh giá test. Mặc định chạy **1 vòng CL × 1 epoch** trên tối đa 3 mẫu mỗi split để kiểm tra pipeline; đổi biến số mẫu hoặc hai biến vòng/epoch ở ô cấu hình để mở rộng. Mỗi lần tạo cấu hình sẽ dùng một thư mục output mới.
#
# Mở file `.py` trong VS Code/Kaggle để chạy từng ô `# %%`, hoặc chạy cả file bằng Python 3.11. Trên Kaggle, ô đầu clone source và cài dependency; output được lưu riêng trong `/kaggle/working/outputs/cba_whisper`. Dữ liệu ViMD small phải được gắn dưới Kaggle Input; script không tải dữ liệu.

# %% [markdown]
# ## 0. Clone source và cài thư viện
#
# Ô này dùng checkout riêng `/kaggle/working/mdv_asr_source`, tương tự
# `src/nb/benchmark.py`. Chạy lại ô sẽ clone lại source đã nhận diện là Git;
# artifacts nằm ngoài checkout nên không bị xóa theo source.

# %%
from importlib.util import find_spec
from pathlib import Path
import importlib
import shutil
import subprocess
import sys

REPO_URL = 'https://github.com/NguyenVoTanDuy2006/Research-Multi-Dialect-Vietnamese-ASR.git'
REPO_BRANCH = 'research-CBA-Whisper'
REPO_DIR = Path('/kaggle/working/mdv_asr_source')
KAGGLE_WORKING = Path('/kaggle/working')
PROJECT_RELATIVE = Path('src/CBA-pipeline/Reimplement-CBA')
PROJECT_DIR_OVERRIDE = None

script_path = Path(__file__).resolve() if '__file__' in globals() else None
running_from_clone = script_path is not None and REPO_DIR in script_path.parents
if KAGGLE_WORKING.is_dir():
    if not running_from_clone:
        if REPO_DIR.is_symlink():
            raise ValueError(f'Không xóa symlink source: {REPO_DIR}')
        if REPO_DIR.exists():
            if not REPO_DIR.is_dir() or not (REPO_DIR / '.git').is_dir():
                raise ValueError(f'Checkout cũ không phải Git repo, kiểm tra thủ công: {REPO_DIR}')
            legacy_outputs = (REPO_DIR / 'outputs', REPO_DIR / PROJECT_RELATIVE / 'outputs')
            if any(path.exists() for path in legacy_outputs):
                raise RuntimeError(
                    f'Checkout cũ còn output bên trong; chuyển output ra ngoài trước khi clone lại: {REPO_DIR}'
                )
            changes = subprocess.run(
                ['git', '-C', str(REPO_DIR), 'status', '--porcelain'],
                check=True, capture_output=True, text=True,
            )
            if changes.stdout.strip():
                raise RuntimeError(f'Checkout cũ có thay đổi chưa lưu, không xóa: {REPO_DIR}')
            shutil.rmtree(REPO_DIR)
            print('Đã xóa source clone cũ:', REPO_DIR)
        subprocess.run(
            ['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH,
             REPO_URL, str(REPO_DIR)],
            check=True,
        )
    project_dir = REPO_DIR / PROJECT_RELATIVE
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '-r',
         str(REPO_DIR / 'requirements-kaggle.txt')],
        check=True,
    )
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', '-e',
         f'{REPO_DIR}[semantic]'],
        check=True,
    )
    importlib.invalidate_caches()
elif PROJECT_DIR_OVERRIDE:
    project_dir = Path(PROJECT_DIR_OVERRIDE).expanduser().resolve()
else:
    locations = [script_path.parents[1]] if script_path else []
    locations += [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    options = [location / 'src/CBA-pipeline/Reimplement-CBA' for location in locations]
    options += [location / 'CBA-pipeline/Reimplement-CBA' for location in locations]
    options += locations
    project_dir = next((path for path in options if (path / 'reimplement_cba/workflow.py').is_file()), None)
if project_dir is None or not (project_dir / 'reimplement_cba/workflow.py').is_file():
    raise FileNotFoundError('Không tìm thấy Reimplement-CBA; hãy đặt PROJECT_DIR_OVERRIDE.')
workspace_dir = project_dir.parents[2]
for package_dir in (project_dir, workspace_dir / 'src'):
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))

required_modules = ('numpy', 'soundfile', 'datasets', 'torch', 'transformers', 'peft', 'accelerate', 'bert_score')
missing = [name for name in required_modules if find_spec(name) is None]
print('Project:', project_dir)
print('Python:', sys.version.split()[0])
print('Thiếu thư viện:', missing or 'không')
if missing:
    raise RuntimeError(
        f"Cài dependency từ {workspace_dir}: python -m pip install -e '.[data,train,semantic]'"
    )
import torch
print('CUDA:', torch.cuda.is_available())

# %% [markdown]
# ## 2. Cấu hình chạy thử
#
# Đặt `VIMD_ROOT` thành thư mục ViMD small trên máy. Nếu chạy ở Kaggle và đường dẫn trong dataset config đã tồn tại, giữ `None`. Đổi `CONFIG_FILENAME` sang `vimd_phowhisper_large.json` để chạy PhoWhisper-large. Script giữ backend `transformers` và AdaLoRA.

# %%
from copy import deepcopy
from datetime import datetime
import json
from uuid import uuid4
from reimplement_cba.config import CBAConfig

VIMD_ROOT = None  # Ví dụ: '/data/vimd/small-vimd'
CONFIG_FILENAME = 'vimd_phowhisper_small.json'  # Đổi thành 'vimd_phowhisper_large.json' khi cần.
SMOKE_ROUNDS = 1
SMOKE_EPOCHS = 1
SMOKE_SAMPLES_PER_SPLIT = 3

base_config = CBAConfig.from_file(project_dir / 'configs' / CONFIG_FILENAME)
values = deepcopy(base_config.values)
if VIMD_ROOT is not None:
    values['dataset']['root_override'] = str(Path(VIMD_ROOT).expanduser().resolve())
values['curriculum']['max_rounds'] = SMOKE_ROUNDS
values['training']['epochs'] = SMOKE_EPOCHS
values['inference']['backend'] = 'transformers'
run_label = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:6]
values['run_id'] = f'script-{Path(CONFIG_FILENAME).stem}-{run_label}'
OUTPUT_ROOT = (KAGGLE_WORKING if KAGGLE_WORKING.is_dir() else workspace_dir) / 'outputs' / 'cba_whisper'
values['output_dir'] = str(OUTPUT_ROOT / values['run_id'])
config = CBAConfig(base_config.path, values)
config.validate()
OUTPUT_DIR = config.output_dir
if REPO_DIR == OUTPUT_DIR or REPO_DIR in OUTPUT_DIR.parents:
    raise ValueError(f'Output phải nằm ngoài source clone: {OUTPUT_DIR}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
(OUTPUT_DIR / 'resolved-config.json').write_text(
    json.dumps(values, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
)
print('Dữ liệu:', config.dataset_config_path)
print('Root override:', values['dataset']['root_override'])
print('Output:', OUTPUT_DIR)
print('CL rounds:', SMOKE_ROUNDS, '| epochs/round:', SMOKE_EPOCHS)
print('Số mẫu tối đa mỗi split:', SMOKE_SAMPLES_PER_SPLIT)

# %% [markdown]
# ## 3. Kiểm tra ViMD trước khi huấn luyện
#
# Ô này đọc index và xác nhận đủ train, validation, test. Nếu báo thiếu dữ liệu, sửa `VIMD_ROOT` rồi chạy lại từ ô cấu hình.

# %%
from collections import Counter
from reimplement_cba.adapters.vimd import ViMDAdapter

with ViMDAdapter(config) as vimd:
    print('Cấu hình nguồn:', vimd.validate_config())
    print('Kiểm tra split:', vimd.validate_splits())
    records = vimd.open().index.records
    print('Số mẫu theo split:', dict(Counter(record.split for record in records)))
    print('Ba mẫu đầu:')
    for record in records[:3]:
        print(record.sample_id, record.split, round(record.duration_seconds, 2), str(record.text)[:120])

# %% [markdown]
# ## 4. Chạy workflow
#
# Ô này fine-tune trên 3 mẫu train hợp lệ đầu tiên, dùng 3 mẫu đầu của validation và test, rồi tính NLI, BERTScore, phonetic similarity. Đây chỉ là smoke run; metrics trên 3 mẫu không đại diện cho toàn bộ ViMD small. Workflow ghi từng giai đoạn vào `output_dir`; hãy dùng một cấu hình/output mới khi chạy lại.

# %%
from reimplement_cba.workflow import run

summary = run(config, max_samples_per_split=SMOKE_SAMPLES_PER_SPLIT)
print('Trạng thái:', summary['status'])
print('Số vòng CL:', len(summary['rounds']))
print('Checkpoint được chọn:', summary['final']['checkpoint'])
print('Validation:', summary['final']['validation'])
print('Test:', summary['final']['test'])
print('Baseline test:', summary['final']['baseline_test'])
print('Báo cáo:', config.output_dir / 'run-summary.json')

# %% [markdown]
# ## 5. Xem kết quả đã lưu
#
# Ô này đọc lại báo cáo JSON. WER ở dạng tỉ lệ: `0.20` nghĩa là 20%. SemScore ở thang 0–100 và hiện cần hiệu chuẩn trên ViMD.

# %%
import json

summary_path = config.output_dir / 'run-summary.json'
report = json.loads(summary_path.read_text(encoding='utf-8'))
print('Status:', report['status'])
print('Train ban đầu:', report['initial_train_count'])
print('Mẫu loại ở bước đầu:', report['initial_train_rejected_count'])
for round_report in report['rounds']:
    print('Round', round_report['round_index'],
          '| checkpoint:', round_report['selected_checkpoint'],
          '| validation:', round_report['selected_validation'],
          '| curriculum:', round_report['curriculum'])
if 'final' in report:
    print('Final test:', report['final']['test'])

# %% [markdown]
# ## 6. Dọn source clone khỏi Kaggle Output
#
# Chạy sau khi workflow và phần xem báo cáo đã kết thúc. Chỉ xóa checkout
# `/kaggle/working/mdv_asr_source`; giữ `OUTPUT_DIR` và các checkpoint trong
# `/kaggle/working/outputs/cba_whisper/` để tải về từ Kaggle Output.

# %%
from mdv_asr_module.runtime.cleanup import NotebookOutputCleaner

if KAGGLE_WORKING.is_dir():
    legacy_outputs = (REPO_DIR / 'outputs', REPO_DIR / PROJECT_RELATIVE / 'outputs')
    if any(path.exists() for path in legacy_outputs):
        raise RuntimeError(
            f'Source còn output bên trong; chuyển output ra ngoài trước khi dọn: {REPO_DIR}'
        )
    removed = NotebookOutputCleaner(working_dir=KAGGLE_WORKING).remove_source_checkout(
        REPO_DIR, preserve_paths=[OUTPUT_DIR]
    )
    print('Đã dọn source clone:' if removed else 'Source đã được dọn trước đó:', REPO_DIR)
    print('Artifacts được giữ tại:', OUTPUT_DIR)
else:
    print('Đang chạy ngoài Kaggle; giữ nguyên source tại:', workspace_dir)
