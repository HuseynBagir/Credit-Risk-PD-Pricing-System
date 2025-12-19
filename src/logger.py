import logging
import json
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "credit_risk.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("credit_risk")

def log_decision(pd, decision, apr, channel="streamlit", model_version="v1.0"):
    payload = {
        "event": "credit_decision",
        "pd": round(pd, 6),
        "decision": decision,
        "apr": None if apr is None else round(apr, 6),
        "channel": channel,
        "model_version": model_version,
    }
    logger.info(json.dumps(payload))
