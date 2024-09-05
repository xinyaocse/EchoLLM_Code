import sys
sys.path.append('/root/code_project/audio/asr_pre_dataset/exp/')
from exp_config import config_adapter


class ExpConfig:
    JSON_PATH = r"./ASR/exp/exp_wer_result.json"
    VALUE_LIST = ["distance", "volume"]
