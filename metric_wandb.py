import wandb
import json
import yaml

from collections import defaultdict

# YAML 파일 읽기
with open('config.yaml', 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)

# JSON 파일로 저장
with open('config.json', 'w') as json_file:
    json.dump(yaml_data, json_file, indent=2)

api = wandb.Api()
run = api.run("sjrhee/IQL-antmaze-umaze-diverse-v2-1-0.5-norm-grad-False/96513b796e7a4a0387bbb195d1723faa")
# run = api.run("/home/wisrl/sjrhee/IE801_project/experiment_output/cc632aa92c0a4d159f3fe8c30ca2b221")
# save the metrics for the run to a csv file
# metrics_dataframe = run.history(keys=['epoch', 'opex_average_return', 'eval_average_return', 'action_gap'])
# metrics_dataframe.to_csv("metrics.csv")

history = run.scan_history(keys=['epoch', 'opex_average_return', 'eval_average_return', 'action_gap'])

# JSON 형식으로 변환
metrics_json = defaultdict(lambda: {"steps": [], "timestamps": [], "values": []})

for row in history:
    for key in row.keys():
        # _timestamp와 _step은 제외
        if key not in ['_timestamp', '_step']:
            # _step 값을 안전하게 가져오기
            step_value = row.get('epoch', None)
            if step_value is not None:
                metrics_json[key]["steps"].append(step_value)
            else:
                metrics_json[key]["steps"].append(None)  # 기본값 None으로 설정
            # 나머지 값들 추가
            metrics_json[key]["timestamps"].append(row.get('_timestamp', ''))
            metrics_json[key]["values"].append(row.get(key, None))


# JSON 파일로 저장
with open('metrics.json', 'w') as f:
    json.dump(metrics_json, f, indent=2)