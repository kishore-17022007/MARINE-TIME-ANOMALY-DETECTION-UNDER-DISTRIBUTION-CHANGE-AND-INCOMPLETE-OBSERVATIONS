import logging

from src.utils.paths import project_path


def configure_logging(name: str = "maritime", log_file: str = "artifacts/logs/pipeline.log") -> logging.Logger:
    destination = project_path(log_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(destination, encoding="utf-8")],
        force=True,
    )
    return logging.getLogger(name)