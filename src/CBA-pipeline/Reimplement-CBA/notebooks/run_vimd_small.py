# -*- coding: utf-8 -*-
# Python script with VS Code/Jupyter-style cells.

# %% [markdown]
# # Chạy thử CBA trên ViMD small
#
# Script này dùng trực tiếp `reimplement_cba.workflow.run`: baseline PhoWhisper, fine-tune AdaLoRA, lọc curriculum, chọn checkpoint trên validation rồi đánh giá test. Mặc định chạy **1 vòng CL × 1 epoch** để kiểm tra pipeline; đổi hai biến ở ô cấu hình thành `2` để chạy cấu hình 2 vòng × 2 epoch. Mỗi lần tạo cấu hình sẽ dùng một thư mục output mới.
#
# Mở file `.py` trong VS Code để chạy từng ô `# %%`, hoặc chạy cả file bằng Python 3.11. Từ thư mục gốc của workspace, cài các thư viện cần cho backend Transformers: `python -m pip install -e '.[data,train,semantic]'`. Dữ liệu ViMD small phải có sẵn trên máy; script không tải dữ liệu.

# %% [markdown]
# ## 1. Tìm project và kiểm tra thư viện

# %%
from importlib.util import find_spec
from pathlib import Path
import sys

# Nếu chạy từng ô từ một thư mục khác workspace, có thể điền đường dẫn ở đây.
PROJECT_DIR_OVERRIDE = None

if PROJECT_DIR_OVERRIDE:
    project_dir = Path(PROJECT_DIR_OVERRIDE).expanduser().resolve()
else:
    script_path = globals().get('__file__')
    locations = [Path(script_path).resolve().parents[1]] if script_path else []
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
# Đặt `VIMD_ROOT` thành thư mục ViMD small trên máy. Nếu chạy ở Kaggle và đường dẫn trong dataset config đã tồn tại, giữ `None`. Script giữ backend `transformers`; phần fine-tune vẫn dùng PhoWhisper-medium và AdaLoRA như cấu hình project.

# %%
from copy import deepcopy
from datetime import datetime
from uuid import uuid4
from reimplement_cba.config import CBAConfig

VIMD_ROOT = None  # Ví dụ: '/data/vimd/small-vimd'
SMOKE_ROUNDS = 1
SMOKE_EPOCHS = 1

base_config = CBAConfig.from_file(project_dir / 'configs/vimd_phowhisper_small.json')
values = deepcopy(base_config.values)
if VIMD_ROOT is not None:
    values['dataset']['root_override'] = str(Path(VIMD_ROOT).expanduser().resolve())
values['curriculum']['max_rounds'] = SMOKE_ROUNDS
values['training']['epochs'] = SMOKE_EPOCHS
values['inference']['backend'] = 'transformers'
run_label = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid4().hex[:6]
values['run_id'] = f'script-vimd-small-{run_label}'
values['output_dir'] = str(project_dir / 'outputs' / values['run_id'])
config = CBAConfig(base_config.path, values)
config.validate()
print('Dữ liệu:', config.dataset_config_path)
print('Root override:', values['dataset']['root_override'])
print('Output:', config.output_dir)
print('CL rounds:', SMOKE_ROUNDS, '| epochs/round:', SMOKE_EPOCHS)

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
# Ô này thực sự fine-tune model và tính NLI, BERTScore, phonetic similarity. Với PhoWhisper-medium, thời gian chạy và bộ nhớ GPU có thể đáng kể dù đang dùng ViMD small. Workflow ghi từng giai đoạn vào `output_dir`; hãy dùng một cấu hình/output mới khi chạy lại.

# %%
from reimplement_cba.workflow import run

summary = run(config)
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
