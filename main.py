import hydra

@hydra.main(config_path="cfg", config_name="config", version_base="1.2")
def main(cfg):
    PROJECT_ROOT = cfg.paths.project_root
    print(f"Project root is: {PROJECT_ROOT}")

if __name__ == "__main__":
    main()