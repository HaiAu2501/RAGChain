import hydra
from dotenv import load_dotenv
from src.builder import DatabaseBuilder

load_dotenv()

@hydra.main(config_path="cfg", config_name="config", version_base="1.2")
def main(cfg):
    builder = DatabaseBuilder(cfg)
    builder.build_database()

if __name__ == "__main__":
    main()