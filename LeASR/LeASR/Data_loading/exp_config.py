import os


class ExpConfig:

    def __init__(self, config_list, json_path, pre_audio_path, raw_json=None):
        self.config_list = config_list
        self.key_list = config_list.keys()
        self.json_path = json_path
        self.pre_audio_path = pre_audio_path
        self.path_dict = None
        if raw_json:
            self.raw_json = raw_json
        else:
            self.raw_json = r"/root/public/dev8T/username/ASR/exp/exp_test.json"
        # for key in self.key_list:
        #     self.path_list[key] = ""

    def get_list_by_key(self, key):
        if key in self.config_list:
            return self.config_list[key]
        return []

    def set_path_dict(self, path_dict):
        self.path_dict = path_dict

    def get_path_list(self, key):
        if key in self.path_dict:
            return self.path_dict[key]
        return []


exp_controlled_experiments_config = ExpConfig({"Volume": ['50', '60', '70', '80', '90', '100'], 
                            "Angle": ["15", '30', '45', '60', '75'],
                            "Distance": ['40', '60', '80', '100', '120'],
                            "Motion": ['static', "FB", "LR", "UD"]},
                            "EchoLLM/Datasets/Controlled_Experiments/Controlled_Experiments.json",
                            "EchoLLM/Datasets/Controlled_Experiments/")
exp_controlled_experiments_config.set_path_dict({
                            "Volume": "EchoLLM/Datasets/Controlled_Experiments/Volume/",
                            "Angle": "EchoLLM/Datasets/Controlled_Experiments/Angle/",
                            "Distance": "EchoLLM/Datasets/Controlled_Experiments/Distance/",
                            "Motion": "EchoLLM/Datasets/Controlled_Experiments/Motion/"})

exp_ablation_config = ExpConfig({
    "Ablation": ['Normal', 'w/o-BRE', 'w/o-MC', 'w/o-AE'],
    "Context": ['Caller-Callee', 'Callee-Caller', 'w/o-Caller']},
    "/root/public/dev8T/username/ASR/exp/exp_ablation_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_ablation_dataset/")
exp_ablation_config.set_path_dict({
    "Ablation": "/root/public/dev8T/username/ASR/exp/exp_ablation/Ablationt/",
    "Context": "/root/public/dev8T/username/ASR/exp/exp_context/Context/"})

exp_robustness_config = ExpConfig({
    "Noise": ['30', '40', '50', '60', '70'],
    "Env": ['UE', 'CS', 'OZ', 'SA'],
    "Headphone": ["SZ-ORP", "SZ-OM", "NK-RP4"]},
    "/root/public/dev8T/username/ASR/exp/exp_robustness_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_robustness_dataset/")
exp_robustness_config.set_path_dict({
    "Noise": "/root/public/dev8T/username/ASR/exp/exp_diff_noise/Noise/",
    "Env": "/root/public/dev8T/username/ASR/exp/exp_diff_env/Env/",
    "Headphone": "/root/public/dev8T/username/ASR/exp/exp_controlled/Headphone/",
    })



exp_user_diversity_config = ExpConfig({
    "Human": ['Slow_F_Y', 'Moderate_F_Y', 'Fast_F_Y', 'Slow_M_Y', 'Moderate_M_Y', 'Fast_M_Y', 'Slow_F_M', 'Moderate_F_M', 'Fast_F_M', 'Slow_M_M', 'Moderate_M_M', 'Fast_M_M', 'Slow_F_O', 'Moderate_F_O', 'Fast_F_O', 'Slow_M_O', 'Moderate_M_O', 'Fast_M_O']},
    "/root/public/dev8T/username/ASR/exp/exp_user_diversity_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_user_diversity_dataset/",
    "/root/public/dev8T/username/ASR/exp/exp_user_diversity_eval.json")
exp_user_diversity_config.set_path_dict({
    "Human": "/root/public/dev8T/username/ASR/exp/exp_user_diversity/Human/"})

exp_phone_config = ExpConfig({"Distance": ['50', '100', '150', '200']},
                             "/root/public/dev8T/username/ASR/exp/exp_phone.json",
                             "/root/public/dev8T/username/ASR/exp/exp_phone_dataset/")
exp_phone_config.set_path_dict({"Distance": r"/root/public/dev8T/username/ASR/exp/exp_phone/Distance/"})

exp_headphones_config = ExpConfig({"Headphones": ['BCH', 'In-ear', 'Over-ear']},
                             "/root/public/dev8T/username/ASR/exp/exp_headphones.json",
                             "/root/public/dev8T/username/ASR/exp/exp_headphones_dataset/")
exp_headphones_config.set_path_dict({"Headphones": r"/root/public/dev8T/username/ASR/exp/exp_headphones/Headphones/"})

exp_digit_config = ExpConfig({
    "Digit": ['digit']},
    "/root/public/dev8T/username/ASR/exp/exp_digit_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_digit_dataset/")
exp_digit_config.set_path_dict({
    "Digit": "/root/public/dev8T/username/ASR/exp/exp_digit/Digit/"})

exp_sensitive_config = ExpConfig({
    "Sensitive": ['digit']},
    "/root/public/dev8T/username/ASR/exp/exp_sensitive_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_sensitive_dataset/")
exp_sensitive_config.set_path_dict({
    "Sensitive": "/root/public/dev8T/username/ASR/exp/exp_sensitive/Sensitive/"})






class ConfigAdapter:
    def __init__(self):
        self.config = None

    def set_config(self, config):
        self.config = config


config_adapter = ConfigAdapter()
config_adapter.set_config(exp_phone_config)
