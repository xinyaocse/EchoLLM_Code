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


exp_config = ExpConfig({"distance": ['40', '60', '80', '90'], "volume": ['50', '60', '70', '80', '90', '100']},
                       "", "")
exp_v2_config = ExpConfig({"volume_d30": ['50', '60', '70', '80', '90', '100']},
                          "/root/public/dev8T/username/ASR/exp/exp_result_d30_volume.json",
                          "/root/public/dev8T/username/ASR/exp/exp_v2_pre_dataset/")
exp_v2_config.set_path_dict({"volume_d30": r"/root/public/dev8T/username/ASR/exp/exp_v2/volume_d30/"})

exp_v3_config = ExpConfig({"volume": ['50', '60', '70', '80', '90', '100'], "distance": ['40', '60', '80', '90']},
                          "/root/public/dev8T/username/ASR/exp/exp_v3_result.json",
                          "/root/public/dev8T/username/ASR/exp/exp_v3_pre_dataset/")
exp_v3_config.set_path_dict({"volume": r"/root/public/dev8T/username/ASR/exp/exp_v3/volume/",
                             "distance": r"/root/public/dev8T/username/ASR/exp/exp_v3/distance/"})

exp_v4_config = ExpConfig({"distance_v80": ['40', '60', '80', '100']},
                          "/root/public/dev8T/username/ASR/exp/exp_result_v80_distance.json",
                          "/root/public/dev8T/username/ASR/exp/exp_v4_pre_dataset/")
exp_v4_config.set_path_dict({"distance_v80": r"/root/public/dev8T/username/ASR/exp/exp_v4/distance_v80/"})

exp_v5_config = ExpConfig({"volume": ['40', '50', '60', '70', '80', '90', '100'], "Angle": ['30', '45', '60', '75'],
                           "headphone": ["type2"]},
                          "/root/public/dev8T/username/ASR/exp/exp_v5_result.json",
                          "/root/public/dev8T/username/ASR/exp/exp_v5_pre_dataset/")
exp_v5_config.set_path_dict({"volume": r"/root/public/dev8T/username/ASR/exp/exp_v5/volume/",
                             "Angle": r"/root/public/dev8T/username/ASR/exp/exp_v5/Angle/",
                             "headphone": "/root/public/dev8T/username/ASR/exp/exp_v5/headphone/"})

exp_v6_config = ExpConfig({"volume": ['50', '60', '70', '80', '90', '100'], "Angle": ["15", '30', '45', '60', '75'],
                           "headphone": ["type2"], "distance_v80": ['40', '60', '80', '100'],
                           "Motion": ['static', "FB", "LR", "UD"]},
                          "/root/public/dev8T/username/ASR/exp/exp_v6_result.json",
                          "/root/public/dev8T/username/ASR/exp/exp_v6_pre_dataset/")
exp_v6_config.set_path_dict({"volume": r"/root/public/dev8T/username/ASR/exp/exp_v6/volume/",
                             "Angle": r"/root/public/dev8T/username/ASR/exp/exp_v6/Angle/",
                             "headphone": "/root/public/dev8T/username/ASR/exp/exp_v6/headphone/",
                             "distance_v80": r"/root/public/dev8T/username/ASR/exp/exp_v4/distance_v80/",
                             "Motion": r"/root/public/dev8T/username/ASR/exp/exp_v6/Motion/"})

exp_v8_config = ExpConfig({"Angle": ['30', '45'],
                           "headphone": ["type2", "type3"], "distance": ['100', "120"]},
                          "/root/public/dev8T/username/ASR/exp/exp_v8_result.json",
                          "/root/public/dev8T/username/ASR/exp/exp_v8_pre_dataset/")
exp_v8_config.set_path_dict({
    "Angle": r"/root/public/dev8T/username/ASR/exp/exp_v8/Angle/",
    "headphone": "/root/public/dev8T/username/ASR/exp/exp_v8/headphone/",
    "distance": r"/root/public/dev8T/username/ASR/exp/exp_v8/distance/"})

exp_v9_config = ExpConfig({
    "Motion": ['STATIC', "FB", "LR", "UD"]},
    "/root/public/dev8T/username/ASR/exp/exp_v9_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v9_pre_dataset/")
exp_v9_config.set_path_dict({"Motion": r"/root/public/dev8T/username/ASR/exp/exp_v8/Motion/"})

exp_v10_config = ExpConfig({
    "headphone": ["type2", "type3"], "distance": ['120', "140"]},
    "/root/public/dev8T/username/ASR/exp/exp_v10_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v10_pre_dataset/")
exp_v10_config.set_path_dict({
    "headphone": "/root/public/dev8T/username/ASR/exp/exp_v10/headphone/",
    "distance": r"/root/public/dev8T/username/ASR/exp/exp_v10/distance/"})

exp_v11_config = ExpConfig({
    "Motion": ['static'], "volume": ['80'], },
    "/root/public/dev8T/username/ASR/exp/exp_v11_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v11_pre_dataset/")
exp_v11_config.set_path_dict({
    "Motion": "/root/public/dev8T/username/ASR/exp/exp_v11/Motion/",
    "volume": "/root/public/dev8T/username/ASR/exp/exp_v11/volume/"})

exp_v12_config = ExpConfig({
    "Motion": ['static'], "volume": ['80'], },
    "/root/public/dev8T/username/ASR/exp/exp_v12_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v12_pre_dataset/")
exp_v12_config.set_path_dict({
    "Motion": "/root/public/dev8T/username/ASR/exp/exp_v12/Motion/",
    "volume": "/root/public/dev8T/username/ASR/exp/exp_v12/volume/"})

exp_v13_config = ExpConfig({
    "Motion": ['static'], "volume": ['80'], },
    "/root/public/dev8T/username/ASR/exp/exp_v13_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v13_pre_dataset/")
exp_v13_config.set_path_dict({
    "Motion": "/root/public/dev8T/username/ASR/exp/exp_v13/Motion/",
    "volume": "/root/public/dev8T/username/ASR/exp/exp_v13/volume/"})

exp_v14_config = ExpConfig({
    "digit": ['digit']},
    "/root/public/dev8T/username/ASR/exp/exp_v14_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v14_pre_dataset/",
    "/root/public/dev8T/username/ASR/exp/digit_exp/raw_audio.json")
exp_v14_config.set_path_dict({
    "digit": "/root/public/dev8T/username/ASR/exp/exp_v14/digit/"})

exp_v15_config = ExpConfig({
    "digit": ['digit']},
    "/root/public/dev8T/username/ASR/exp/exp_v15_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v15_pre_dataset/",
    "/root/public/dev8T/username/ASR/exp/digit_exp/digit_eval.json")
exp_v15_config.set_path_dict({
    "digit": "/root/public/dev8T/username/ASR/exp/exp_v14/digit/"})

exp_v16_config = ExpConfig({
    "digit": ['digit']},
    "/root/public/dev8T/username/ASR/exp/exp_v16_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v16_pre_dataset/",
    "/root/public/dev8T/username/ASR/exp/digit_exp/digit_eval.json")
exp_v16_config.set_path_dict({
    "digit": "/root/public/dev8T/username/ASR/exp/exp_v17/digit/"})

exp_v17_config = ExpConfig({
    "Human": ['Slow_F_Y', 'Moderate_F_Y', 'Fast_F_Y', 'Slow_M_Y', 'Moderate_M_Y', 'Fast_M_Y']},
    "/root/public/dev8T/username/ASR/exp/exp_v17_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v17_pre_dataset/",
    "/root/public/dev8T/username/ASR/exp/digit_exp/Human_eval.json")
exp_v17_config.set_path_dict({
    "Human": "/root/public/dev8T/username/ASR/exp/exp_v17/Human/"})

exp_v18_config = ExpConfig({
    "Human": ['Slow_F_M', 'Moderate_F_M', 'Fast_F_M', 'Slow_M_M', 'Moderate_M_M', 'Fast_M_M']},
    "/root/public/dev8T/username/ASR/exp/exp_v18_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v18_pre_dataset/",
    "/root/public/dev8T/username/ASR/exp/digit_exp/Human_eval.json")
exp_v18_config.set_path_dict({
    "Human": "/root/public/dev8T/username/ASR/exp/exp_v18/Human/"})

exp_v19_config = ExpConfig({
    "Human": ['Slow_F_O', 'Moderate_F_O', 'Fast_F_O', 'Slow_M_O', 'Moderate_M_O', 'Fast_M_O']},
    "/root/public/dev8T/username/ASR/exp/exp_v19_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v19_pre_dataset/",
    "/root/public/dev8T/username/ASR/exp/digit_exp/Human_eval.json")
exp_v19_config.set_path_dict({
    "Human": "/root/public/dev8T/username/ASR/exp/exp_v19/Human/"})

exp_v20_config = ExpConfig({
    "Env": ['UE', 'CS', 'OZ', 'SA']},
    "/root/public/dev8T/username/ASR/exp/exp_v20_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v20_pre_dataset/",
    "/root/public/dev8T/username/ASR/exp/digit_exp/Env_eval.json")
exp_v20_config.set_path_dict({
    "Human": "/root/public/dev8T/username/ASR/exp/exp_20/Env/"})

exp_v21_config = ExpConfig({
    "Noise": ['30', '40', '50', '60', '70']},
    "/root/public/dev8T/username/ASR/exp/exp_v21_result.json",
    "/root/public/dev8T/username/ASR/exp/exp_v21_pre_dataset/",
    "/root/public/dev8T/username/ASR/exp/digit_exp/Noise_eval.json")
exp_v21_config.set_path_dict({
    "Human": "/root/public/dev8T/username/ASR/exp/exp_v21/Noise/"})

exp_phone_config = ExpConfig({"distance": ['50', '100', '150', '200']},
                             "/root/public/dev8T/username/ASR/exp/exp_phone_distance.json",
                             "/root/public/dev8T/username/ASR/exp/exp_phone_pre_dataset/")
exp_phone_config.set_path_dict({"distance": r"/root/public/dev8T/username/ASR/exp/exp_phone/distance/"})


class ConfigAdapter:
    def __init__(self):
        self.config = None

    def set_config(self, config):
        self.config = config


config_adapter = ConfigAdapter()
config_adapter.set_config(exp_phone_config)
