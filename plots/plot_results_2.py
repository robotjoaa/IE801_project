import argparse
from collections import defaultdict
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ALPHA = 0.2
THRESHOLD_FOR_NUM_ALGS_UNTIL_LEGEND_BELOW_PLOT = 6
THRESHOLD_FOR_ALG_NAME_LENGTH_UNTIL_LEGEND_BELOW_PLOT = 20

# 기존 BETA_COLORS를 유지하되 opex/eval 결과를 구분하기 위해 메트릭별로 색상을 다르게 사용
BETA_COLORS = {
    0.5: "tab:blue",
    0.3: "tab:blue",
    0.05: "tab:orange",
    0.03: "tab:orange",
    0.005: "tab:green",
    0.003: "tab:green",
}

# eval metric일 때는 더 연한 색을 사용할 것이므로 정의
EVAL_COLOR_VARIATION = {
    0.5: "cornflowerblue",
    0.3: "cornflowerblue",
    0.05: "lightsalmon",
    0.03: "lightsalmon",
    0.005: "lightgreen",
    0.003: "lightgreen",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to directory containing (multiple) results",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="opex_average_return",
        help="Metric to plot",
    )
    parser.add_argument(
        "--filter_by_algs",
        nargs="+",
        default=[],
        help="Filter results by algorithm names. Only showing results for algorithms that contain any of the specified strings in their names.",
    )
    parser.add_argument(
        "--filter_by_envs",
        nargs="+",
        default=[],
        help="Filter results by environment names. Only showing results for environments that contain any of the specified strings in their names.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=Path.cwd(),
        help="Path to directory to save plots to",
    )
    parser.add_argument(
        "--y_min",
        type=float,
        default=None,
        help="Minimum value for y-axis",
    )
    parser.add_argument(
        "--y_max",
        type=float,
        default=None,
        help="Maximum value for y-axis",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Use log scale for y-axis",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=None,
        help="Smoothing window for data",
    )
    parser.add_argument(
        "--best_per_alg",
        action="store_true",
        help="Plot only best performing config per alg",
    )
    return parser.parse_args()


def extract_alg_name_from_config(config):
    return config["algo_cfg"]["value"]["name"]


def extract_env_name_from_config(config):
    env = config["env"]
    if "value" in config["env"]:
        env_name = config["env"]["value"]
    elif "key" in config["env_args"]:
        env_name = config["env_args"]["key"]
    else:
        env_name = None
    return f"{env_name}"

def load_results(path, metrics_to_load):
    """
    Load results from the given path for the specified metrics.
    If multiple metrics are given (like opex and eval), load both if available.
    """
    path = Path(path)
    metrics_files = path.glob("**/metrics.json")  # 모든 하위 폴더의 metrics.json 검색

    # 데이터 구조:
    # data[(env_name, beta, num_steps)][alg_name][config_str][metric_name] = list of (config, steps, values)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for file in metrics_files:
        with open(file, "r") as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                warnings.warn(f"Could not load metrics from {file} --> skipping")
                continue

        # config.json 파일 확인
        config_file = file.parent / "config.json"
        if not config_file.exists():
            warnings.warn(f"No config file found for {file} --> skipping")
            continue
        else:
            with open(config_file, "r") as f:
                config = json.load(f)

        alg_name = config.get("algo_cfg", {}).get("value", {}).get("name", "unknown")
        env_name = config.get("env", {}).get("value", "unknown")
        beta = config.get("algo_cfg", {}).get("value", {}).get("training", {}).get("opex_beta", "unknown")
        num_steps = config["algo_cfg"]["value"]["training"]["num_steps"]

        # 지정된 metric들에 대해 결과를 로드
        for m in metrics_to_load:
            if m in metrics:
                steps = np.array(metrics[m]["steps"])
                values = np.array(metrics[m]["values"])
                data[(env_name, beta, num_steps)][alg_name][str(config)][m].append((config, steps, values))
            else:
                # metric이 없는 경우 경고를 내보낼 수도 있지만, 여기서는 그냥 넘어감.
                continue
    return data


def filter_results(data, filter_by_algs, filter_by_envs):
    """
    Filter data to only contain results for algorithms and envs that contain any of the specified strings in their names.
    :param data: dict with results
    :param filter_by_algs: list of strings to filter algorithms by
    :param filter_by_envs: list of strings to filter environments by
    :return: filtered data
    """
    filtered_data = {}
    for env_key, env_data in data.items():
        env_name, _, _ = env_key
        if filter_by_envs and not any(env in env_name for env in filter_by_envs):
            continue
        new_env_data = {}
        for alg_name, alg_data in env_data.items():
            if filter_by_algs and not any(alg in alg_name for alg in filter_by_algs):
                continue
            new_env_data[alg_name] = alg_data
        if new_env_data:
            filtered_data[env_key] = new_env_data
    return filtered_data


def aggregate_results(alg_data):
    """
    Aggregate results over runs for each config and each metric.
    alg_data: dict[config_str][metric_name] = list of (config, steps, values)
    return: dict[config_str] = (config, steps, {metric_name: (means, stds)})
    """
    aggregated_data = {}
    for config_key, metrics_dict in alg_data.items():
        # metrics_dict: {metric_name: [(config, steps, values), ...]}
        # 모든 metric에 대해 aggregate
        all_metrics_aggregated = {}
        config_sample = None
        common_steps = None

        for m, runs in metrics_dict.items():
            # runs: list of (config, steps, values)
            if not runs:
                continue
            config_sample = runs[0][0]
            max_len = max(len(r[1]) for r in runs)  # steps 길이 최대치
            all_steps = []
            all_values = []
            for (c, steps, values) in runs:
                if len(steps) != max_len:
                    # 길이 맞추기 (nan padding)
                    diff = max_len - len(steps)
                    steps = np.concatenate([steps, np.full(diff, np.nan)])
                    values = np.concatenate([values, np.full(diff, np.nan)])
                all_steps.append(steps)
                all_values.append(values)
            agg_steps = np.nanmean(np.stack(all_steps), axis=0)
            agg_means = np.nanmean(np.stack(all_values), axis=0)
            agg_stds = np.nanstd(np.stack(all_values), axis=0)

            if common_steps is None:
                common_steps = agg_steps
            else:
                # steps는 모든 metric에 대해 동일하다고 가정
                pass

            all_metrics_aggregated[m] = (agg_means, agg_stds)
        if config_sample is not None and common_steps is not None:
            aggregated_data[config_key] = (config_sample, common_steps, all_metrics_aggregated)
    return aggregated_data


def smooth_data(alg_data, window_size):
    """
    Apply window smoothing to aggregated data.
    alg_data: dict[config_str] = (config, steps, {metric_name: (means, stds)})
    return: same structure with smoothed data
    """
    for config_key, (config, steps, metrics_data) in alg_data.items():
        smoothed_metrics = {}
        length = len(steps)
        for m, (means, stds) in metrics_data.items():
            if length < window_size:
                # smoothing 불가능하면 그냥 유지
                smoothed_metrics[m] = (means, stds)
                continue

            smoothed_steps = []
            smoothed_means = []
            smoothed_stds = []
            for i in range(length - window_size + 1):
                smoothed_steps.append(np.mean(steps[i : i + window_size]))
                smoothed_means.append(np.mean(means[i : i + window_size]))
                smoothed_stds.append(np.mean(stds[i : i + window_size]))
            smoothed_metrics[m] = (np.array(smoothed_means), np.array(smoothed_stds))
        alg_data[config_key] = (
            config,
            np.array(smoothed_steps),
            smoothed_metrics,
        )
    return alg_data


def _get_unique_keys(dicts):
    """
    Get all keys from a list of dicts that do not have identical values across all dicts
    """
    keys_to_check = set()
    for config in dicts:
        keys_to_check.update(config.keys())

    unique_keys = []
    for key in keys_to_check:
        if key == "hypergroup":
            continue
        if any(key not in d for d in dicts):
            unique_keys.append(key)
            continue
        if any(isinstance(d[key], (dict, list)) for d in dicts):
            continue
        if len(set(d[key] for d in dicts)) > 1:
            unique_keys.append(key)
    return unique_keys


def shorten_config_names(alg_data):
    """
    Shorten config names of algorithm to only include hyperparam values that differ across configs
    alg_data: dict[config_str] = (config, steps, {metric_name: (means, stds)})
    return: dict with shortened_config_str -> (config, steps, {metric_name: (means, stds)})
    """
    configs = [config for config, _, _ in alg_data.values()]
    unique_keys_across_configs = _get_unique_keys(configs)

    shortened_data = {}
    for config_key, (config, steps, metrics_data) in alg_data.items():
        key_names = []
        for key in unique_keys_across_configs:
            if key not in config:
                continue
            value = config[key]
            if isinstance(value, float):
                value = round(value, 4)
            key_names.append(f"{key}={value}")
        shortened_config_name = "_".join(key_names)
        shortened_data[shortened_config_name] = (config, steps, metrics_data)
    return shortened_data


def _filter_best_per_alg(alg_data, main_metric):
    """
    Filter data to only contain best performing config per alg based on main_metric mean.
    alg_data: dict[config_key] = (config, steps, {metric_name:(means,stds)})
    """
    means = {}
    for config_key, (config, steps, metrics_data) in alg_data.items():
        if main_metric in metrics_data:
            means[config_key] = np.mean(metrics_data[main_metric][0])
        else:
            means[config_key] = -np.inf
    best_key = max(means, key=means.get)
    return {best_key: alg_data[best_key]}


def plot_results(data, metrics_to_load, save_dir, y_min, y_max, log_scale):
    """
    data structure: data[(env_name, beta, num_steps)][alg_name][config_key] = (config, steps, {metric_name:(means,stds)})
    """
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # 한 figure 안에 동일한 env_name, num_steps, beta에 대해 opex와 eval 둘 다 있으면 둘 다 plot
    for (env_name, beta, num_steps), env_data in data.items():
        plt.figure(figsize=(10, 6))
        num_plots = 0
        max_label_len = 0

        # metric이 하나인 경우와 두 개인 경우를 구분
        if len(metrics_to_load) == 2:
            # 두 metric이 모두 있는 경우
            # 각 알고리즘, 각 config에 대해 opex와 eval 모두 plot
            for alg_name, alg_configs in env_data.items():
                for config_key, (config, steps, metrics_data) in alg_configs.items():
                    # opex plot (조건문 제거)
                    means, stds = metrics_data["opex_average_return"]
                    color = BETA_COLORS.get(beta, "gray")
                    plt.plot(steps, means, label=f"{alg_name} (opex)", color=color)
                    plt.fill_between(steps, means - stds, means + stds, alpha=ALPHA, color=color)
                    num_plots += 1
                    max_label_len = max(max_label_len, len(f"{alg_name}(opex)"))

                    # eval plot (조건문 제거)
                    means, stds = metrics_data["eval_average_return"]
                    eval_color = EVAL_COLOR_VARIATION.get(beta, "lightblue")
                    plt.plot(steps, means, label=f"{alg_name} (eval)", color=eval_color)
                    plt.fill_between(steps, means - stds, means + stds, alpha=ALPHA, color=eval_color)
                    num_plots += 1
                    max_label_len = max(max_label_len, len(f"{alg_name}(eval)"))
        else:
            # metric이 하나인 경우(기존 로직)
            main_metric = metrics_to_load[0]
            for alg_name, alg_configs in env_data.items():
                for config_key, (config, steps, metrics_data) in alg_configs.items():
                    if main_metric not in metrics_data:
                        continue
                    means, stds = metrics_data[main_metric]
                    color = BETA_COLORS.get(beta, "gray")
                    plt.plot(steps, means, label=alg_name, color=color)
                    plt.fill_between(steps, means - stds, means + stds, alpha=ALPHA, color=color)
                    num_plots += 1
                    max_label_len = max(max_label_len, len(alg_name))

        # 그래프 설정
        title = f"{env_name} | Num Steps: {num_steps} | Beta: {beta}"
        if len(metrics_to_load) == 2:
            # 두 metric을 plot하는 경우
            title += " | opex & eval"
        else:
            title += f" | {metrics_to_load[0]}"

        plt.title(title, fontsize=14)
        plt.xlabel("Timesteps [1e3]", fontsize=12)
        plt.ylabel("Return", fontsize=12)
        if log_scale:
            plt.yscale("log")
        if y_min is not None or y_max is not None:
            plt.ylim(y_min, y_max)

        # 범례 위치
        if (num_plots > THRESHOLD_FOR_NUM_ALGS_UNTIL_LEGEND_BELOW_PLOT
            or max_label_len > THRESHOLD_FOR_ALG_NAME_LENGTH_UNTIL_LEGEND_BELOW_PLOT):
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
        else:
            plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=10)

        plt.tight_layout()

        if save_dir is not None:
            filename = f"{env_name}-num_step-{num_steps}"
            if len(metrics_to_load) == 2:
                filename += "-opex_eval"
            else:
                filename += f"-{metrics_to_load[0]}"
            filename += ".pdf"
            plt.savefig(save_dir / filename, bbox_inches="tight")
        plt.close()



def main():
    args = parse_args()
    # metric이 opex_average_return나 eval_average_return이면 두 metric 모두 plot
    if args.metric in ["opex_average_return", "eval_average_return"]:
        metrics_to_load = ["opex_average_return", "eval_average_return"]
    else:
        metrics_to_load = [args.metric]

    data = load_results(args.path, metrics_to_load)
    data = filter_results(data, args.filter_by_algs, args.filter_by_envs)

    # aggregate
    for env_key, env_data in data.items():
        for alg_name, alg_data in env_data.items():
            data[env_key][alg_name] = aggregate_results(alg_data)

    # smoothing
    if args.smoothing_window is not None:
        for env_key, env_data in data.items():
            for alg_name, alg_data in env_data.items():
                data[env_key][alg_name] = smooth_data(alg_data, args.smoothing_window)

    # shorten config names
    for env_key, env_data in data.items():
        for alg_name, alg_data in env_data.items():
            data[env_key][alg_name] = shorten_config_names(alg_data)

    # best per alg
    if args.best_per_alg:
        main_metric = args.metric if args.metric in metrics_to_load else metrics_to_load[0]
        best_data = defaultdict(dict)
        for env_key, env_data in data.items():
            for alg_name, alg_data in env_data.items():
                best_data[env_key][alg_name] = _filter_best_per_alg(alg_data, main_metric)
        data = best_data

    plot_results(data, metrics_to_load, Path(args.save_dir), args.y_min, args.y_max, args.log_scale)


if __name__ == "__main__":
    main()
