import sys
sys.path.append('/root/code_project/audio/asr_pre_dataset/exp/')
from exp_config import config_adapter


class ExpConfig:
    JSON_PATH = r"/root/public/dev8T/jtang/ASR/exp/exp_wer_result.json"
    VALUE_LIST = ["distance", "volume"]
