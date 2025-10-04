import argparse, yaml
from .config import AppConfig
from .runner import ExperimentRunner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = AppConfig.from_dict(cfg_dict)

    runner = ExperimentRunner(cfg)
    runner.run_all()

if __name__ == "__main__":
    main()
