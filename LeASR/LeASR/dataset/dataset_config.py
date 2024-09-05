import os

NO_BOS_PATH = "/root/public/jtang/ast_datasets/fmcw_audio_asr_no_bof/"
FLAG_PATH = "/root/public/jtang/ast_datasets/fmcw_audio_asr_flag/"
NOISED_B_PATH = "/root/public/jtang/ast_datasets/fmcw_audio_asr_noised/"
noisy_audio_B_add_bos = "/root/public/dev8T/jtang/ASR/datasets/fmcw_audio_asr_noised_add_bos/"
noisy_audio_B_add_bos_real = "/root/public/dev8T/jtang/ASR/datasets/fmcw_audio_asr_noised_add_bos_real/"


class ASRDataset: 
    train_path = r"/root/public/dev8T/jtang/ASR/datasets/asr_test_228/asr_train.json"
    val_path = r"/root/public/dev8T/jtang/ASR/datasets/asr_test_228/asr_test.json"

    def __init__(self, label_key, audio_key, audio_path):
        self.label_key = label_key
        self.audio_key = audio_key
        self.audio_path = audio_path

    def get_label(self, item):
        return item[self.label_key]

    def get_audio(self, item):
        return item[self.audio_key]


class DatasetAdapter:
    def __init__(self):
        self.dataset = None

    def set_datasets(self, dataset):
        self.dataset = dataset


class EvalAdapter:
    def __init__(self):
        self.eval_dataset = None

    def set_datasets(self, dataset):
        self.eval_dataset = dataset


no_bof_dataset = ASRDataset("transcript", "noisy_audio_no_bof", NO_BOS_PATH)
flag_dataset = ASRDataset("transcript_flag", "noisy_audio_add_flag", FLAG_PATH)
noised_b_dataset = ASRDataset("current_text", "noisy_audio_B", NOISED_B_PATH)
ablation_dataset = ASRDataset("current_text_bos", "noisy_audio_B_add_bos", noisy_audio_B_add_bos)
model_b_dataset = ASRDataset("current_text", "real_no_pre_audio_path", NOISED_B_PATH)
real_no_context = ASRDataset("current_text_bos", "noisy_audio_add_noised_b_flag", noisy_audio_B_add_bos_real)
real_with_context = ASRDataset("transcript_flag", "noisy_audio_add_flag", noisy_audio_B_add_bos_real)


class EvalConfig:

    def __init__(self, val_path, val_key, dataset):
        self.val_path = val_path
        self.val_key = val_key
        self.dataset = dataset

    def get_label(self, item):
        return self.dataset.get_label(item)

    def get_audio(self, item):
        return self.dataset.get_audio(item)


flag_eval = EvalConfig("/root/public/dev8T/jtang/ASR/datasets/asr_test_228/asr_eval.json", "flag_asr_text", flag_dataset)
ablation_eval = EvalConfig("/root/public/dev8T/jtang/ASR/exp/exp_ablation.json", "asr_result_no_context", ablation_dataset)
ablation_flag_eval = EvalConfig("/root/public/dev8T/jtang/ASR/exp/exp_ablation.json", "asr_result_with_context", flag_dataset)
ablation_flag_real_eval = EvalConfig("/root/public/dev8T/jtang/ASR/exp/exp_ablation_result.json", "asr_result_with_context_real", real_with_context)
ablation_no_flag_real_eval = EvalConfig("/root/public/dev8T/jtang/ASR/exp/exp_ablation_result.json", "asr_result_no_context_real", real_no_context)
real_b_eval = EvalConfig("/root/public/dev8T/jtang/ASR/datasets/asr_test_223/asr_test.json", "model_B_asr_text", model_b_dataset)