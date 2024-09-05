import datasets
import torch
from datasets import load_dataset, DatasetDict, Split
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        # print(features[0])
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # decoder_features = [{"input_ids": feature["decoder_input_ids"]} for feature in features]


        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # decoder_batch = self.processor.tokenizer.pad(decoder_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # decoder_input_ids = decoder_batch["input_ids"]

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        # if (decoder_input_ids[:, 0] == self.decoder_start_token_id).all().cpu().item():
        #     decoder_input_ids = decoder_input_ids[:, 1:]

        batch["labels"] = labels
        # batch['decoder_input_ids'] = decoder_input_ids
        return batch


@dataclass
class DataCollatorBARTSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        # print(features[0])
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # decoder_features = [{"input_ids": feature["decoder_input_ids"]} for feature in features]


        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])
        # self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # decoder_batch = self.processor.tokenizer.pad(decoder_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # decoder_input_ids = decoder_batch["input_ids"]

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        # if (decoder_input_ids[:, 0] == self.decoder_start_token_id).all().cpu().item():
        #     decoder_input_ids = decoder_input_ids[:, 1:]

        batch["labels"] = labels
        # batch['decoder_input_ids'] = decoder_input_ids
        return batch


def load_data(is_train=True, is_val=False):
    raw_datasets = DatasetDict()
    if is_train:
        data_train = load_dataset(path="/root/code_project/audio/LLM_datasets/libri_pre_16k_noised.py",
                                  cache_dir="/root/public/dev8T/jtang/ASR/datasets/libri_pre_asr_test/cache/",
                                  split=Split.TRAIN)
        raw_datasets["train"] = data_train

    if is_val:
        data_val = load_dataset(path="/root/code_project/audio/LLM_datasets/libri_pre_16k_noised.py",
                                cache_dir="/root/public/dev8T/jtang/ASR/datasets/libri_pre_asr_test/cache/",
                                split=Split.VALIDATION)
        raw_datasets["eval"] = data_val
    # dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    # if dataset_sampling_rate != feature_extractor.sampling_rate:
    #     raw_datasets = raw_datasets.cast_column(
    #         data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    #     )
    # raw_datasets = raw_datasets.cast_column(
    #     "audio", datasets.features.Audio(sampling_rate=16000)
    # )
    return raw_datasets


def load_data_by_path(data_py_path, cache_path, is_train=True, is_val=False):
    raw_datasets = DatasetDict()
    if is_train:
        data_train = load_dataset(path=data_py_path,
                                  cache_dir=cache_path,
                                  split=Split.TRAIN)
        raw_datasets["train"] = data_train

    if is_val:
        data_val = load_dataset(path=data_py_path,
                                cache_dir=cache_path,
                                split=Split.VALIDATION)
        raw_datasets["eval"] = data_val
    # dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    # if dataset_sampling_rate != feature_extractor.sampling_rate:
    #     raw_datasets = raw_datasets.cast_column(
    #         data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    #     )
    # raw_datasets = raw_datasets.cast_column(
    #     "audio", datasets.features.Audio(sampling_rate=16000)
    # )
    return raw_datasets


def get_eval_datasets(is_train=True, is_val=False, is_add_pre=False):
    raw_datasets = DatasetDict()
    if is_add_pre:
        path = "/root/code_project/audio/LLM_datasets/libri_pre_16k_noised_one_sentence_pre.py"
    else:
        path = "/root/code_project/audio/LLM_datasets/libri_pre_16k_noised_one_sentence_no_pre.py"
    if is_train:
        data_train = load_dataset(path=path,
                                  cache_dir="/root/public/dev8T/jtang/ASR/datasets/libri_pre_asr_test_eval/cache/",
                                  split=Split.TRAIN)
        raw_datasets["train"] = data_train

    if is_val:
        data_val = load_dataset(path=path,
                                cache_dir="/root/public/dev8T/jtang/ASR/datasets/libri_pre_asr_test_eval/cache/",
                                split=Split.VALIDATION)
        raw_datasets["eval"] = data_val
    # dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    # if dataset_sampling_rate != feature_extractor.sampling_rate:
    #     raw_datasets = raw_datasets.cast_column(
    #         data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    #     )
    # raw_datasets = raw_datasets.cast_column(
    #     "audio", datasets.features.Audio(sampling_rate=16000)
    # )
    return raw_datasets


def get_eval_datasets_bart2(is_train=True, is_val=False, is_add_pre=False):
    raw_datasets = DatasetDict()
    if is_add_pre:
        path = "/root/code_project/audio/asr_pre_dataset/libri_pre_16k_noised_dialog_eval_with_pre_real.py"
    else:
        path = "/root/code_project/audio/asr_pre_dataset/libri_pre_16k_noised_dialog_eval_with_no_pre_real.py"
    if is_train:
        data_train = load_dataset(path=path,
                                  cache_dir="/root/public/dev8T/jtang/ASR/datasets/libri_pre_asr_test_eval/cache4/",
                                  split=Split.TRAIN)
        raw_datasets["train"] = data_train

    if is_val:
        data_val = load_dataset(path=path,
                                cache_dir="/root/public/dev8T/jtang/ASR/datasets/libri_pre_asr_test_eval/cache4/",
                                split=Split.VALIDATION)
        raw_datasets["eval"] = data_val
    # dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    # if dataset_sampling_rate != feature_extractor.sampling_rate:
    #     raw_datasets = raw_datasets.cast_column(
    #         data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    #     )
    # raw_datasets = raw_datasets.cast_column(
    #     "audio", datasets.features.Audio(sampling_rate=16000)
    # )
    return raw_datasets


def get_ve_dataset(feature_extractor, tokenizer, is_train=True, is_val=False):

    def prepare_dataset(batch):
        # process audio
        sample = batch['audio']
        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=False
        )
        # process audio length
        batch['input_values'] = inputs.get("input_values")[0]
        batch["input_length"] = len(sample["array"])
        # if False:
        #     batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = batch["text_upper"].lower()
        pre_str = batch["text_pre"].lower()
        batch["labels"] = tokenizer(input_str).input_ids
        batch["decoder_input_ids"] = tokenizer(pre_str, add_special_tokens=False).input_ids
        return batch

    raw_datasets = DatasetDict()
    if is_train:
        data_train = load_dataset(path="/root/code_project/audio/LLM_datasets/libri_pre_16k_noised.py",
                                  cache_dir="/root/public/dev8T/jtang/ASR/datasets/libri_pre_asr_test/cache/",
                                  split=Split.TRAIN)
        raw_datasets["train"] = data_train

    if is_val:
        data_val = load_dataset(path="/root/code_project/audio/LLM_datasets/libri_pre_16k_noised.py",
                                cache_dir="/root/public/dev8T/jtang/ASR/datasets/libri_pre_asr_test/cache/",
                                split=Split.VALIDATION)
        raw_datasets["eval"] = data_val
    # dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    # if dataset_sampling_rate != feature_extractor.sampling_rate:
    #     raw_datasets = raw_datasets.cast_column(
    #         data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    #     )

    # raw_datasets = raw_datasets.cast_column(
    #     "audio", datasets.features.Audio(sampling_rate=16000)
    # )
    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        num_proc=8,
        desc="preprocess train dataset",
    )
    return vectorized_datasets


if __name__ == '__main__':
    # raw_data = load_data()
    # temp = next(iter(raw_data.values())).column_names
    # raw_data = raw_data.cast_column(
    #     "audio", datasets.features.Audio(sampling_rate=16000)
    # )
    # print(raw_data['train'][0])
    raw_datasets = load_data_by_path("/root/code_project/speech_asr/dataset/libri_pre_16k_noised_dialog_eval_exp_data.py",
                                     "/root/public/dev8T/jtang/ASR/datasets/asr_test_223/dialog_cache",
                                     False, True)
    print(raw_datasets['train'][2])