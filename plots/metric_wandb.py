import wandb
import json
import yaml

from collections import defaultdict
from pathlib import Path

# plots_dir 설정
plots_dir = Path("large-diverse-3/norm/0.003-2")
# plots_dir = Path("large-diverse-1/base/0.005-base")
plots_dir.mkdir(parents=True, exist_ok=True)

# config.yaml 파일 읽기 (plots_dir 내부에서 가져옴)
config_yaml_path = plots_dir / 'config.yaml'
with open(config_yaml_path, 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)

# JSON 파일로 저장 (plots_dir 내부에 저장)
config_json_path = plots_dir / 'config.json'
with open(config_json_path, 'w') as json_file:
    json.dump(yaml_data, json_file, indent=2)

api = wandb.Api()
run = api.run("sjrhee/IQL-antmaze-large-diverse-v2-3-0.003-norm-grad-True/6201ee89f70b422eaee74770f063ce12")

# metrics 데이터를 가져오기
history = run.scan_history(keys=['epoch', 'opex_average_return', 'eval_average_return', 'action_gap']) # for normal
# history = run.scan_history(keys=['epoch', 'opex_average_return', 'action_gap']) # for baseline
# JSON 형식으로 변환
metrics_json = defaultdict(lambda: {"steps": [], "timestamps": [], "values": []})

for row in history:
    for key in row.keys():
        # _timestamp와 _step은 제외
        if key not in ['_timestamp', '_step']:
            # epoch 값을 안전하게 가져오기
            step_value = row.get('epoch', None)
            if step_value is not None:
                metrics_json[key]["steps"].append(step_value)
            else:
                metrics_json[key]["steps"].append(None)  # 기본값 None으로 설정
            # 나머지 값들 추가
            metrics_json[key]["timestamps"].append(row.get('_timestamp', ''))
            metrics_json[key]["values"].append(row.get(key, None))

# metrics.json 파일 경로 설정
output_file = plots_dir / 'metrics.json'
# JSON 파일로 저장
with open(output_file, 'w') as f:
    json.dump(metrics_json, f, indent=2)

print(f"Metrics JSON saved to: {output_file}")
print(f"Config JSON saved to: {config_json_path}")
