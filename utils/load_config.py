import yaml
from pathlib import Path
import pprint

def merge_dicts(base: dict, override: dict) -> dict:
    """递归合并两个字典，override 会覆盖 base 中的值"""
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            base[k] = merge_dicts(base.get(k, {}), v)
        else:
            base[k] = v
    return base

def load_config(config_path: str | Path) -> dict:
    """加载 YAML 配置文件，支持 _base_ 字段继承"""
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 处理 _base_（支持相对路径）
    if "_base_" in cfg:
        base_path = config_path.parent / cfg["_base_"]
        base_cfg = load_config(base_path)
        del cfg["_base_"]
        cfg = merge_dicts(base_cfg, cfg)

    return cfg

def print_config(cfg: dict, title: str = "CONFIG"):
    """打印格式化的配置内容"""
    print("=" * 40)
    print(f"{title}")
    print("=" * 40)
    pp = pprint.PrettyPrinter(indent=2, width=120, compact=False)
    pp.pprint(cfg)
    print("=" * 40 + "\n")

# 示例调用
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_config.py <path_to_config.yaml>")
    else:
        cfg = load_config(sys.argv[1])
        print_config(cfg)