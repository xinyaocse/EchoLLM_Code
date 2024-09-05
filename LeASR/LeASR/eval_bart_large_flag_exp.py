#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.
import json
import logging
import os
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import sys
sys.path.append('/root/code_project/audio/asr_pre_dataset/exp/')
from exp_config import config_adapter
import datasets
import evaluate
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed, SpeechEncoderDecoderModel,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.38.0.dev0")
#
# require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")
from dataset.dataset_config import EvalAdapter, flag_eval
from dataset.exp_config import ExpConfig
from dataset.fmcw_dataloader import load_data, DataCollatorSpeechSeq2SeqWithPadding, get_eval_datasets, \
    load_data_by_path

logger = logging.getLogger(__name__)

def write_result(path, result_):
    with open(path, "w") as f:
        f.write(json.dumps(result_))


def get_result(path):
    with open(path, "r") as f:
        result = f.read()
    return json.loads(result)


def write_json_result(json_path, text_list):
    files = get_result(json_path)
    index = 0
    for item in files:
        for key in config_adapter.config.key_list:
            for value in files[item][key]:
                files[item][key][value]['asr_result_large'] = text_list[index]
                index += 1
    write_result(json_path, files)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
        , default='/root/public/dev8T/jtang/dataset_text_audio/pretrained_models/models/mode_hu-bart/'
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."}
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                "will always be a token of index 123."
            )
        },
    )
    suppress_tokens: List[int] = field(
        default=None, metadata={"help": "A list of tokens that will be suppressed at generation."}
    )
    apply_spec_augment: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply *SpecAugment* data augmentation to the input features. This is currently only relevant for Wav2Vec2, HuBERT, WavLM and Whisper models."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default="clean", metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=16,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    train_split_name: str = field(
        default="train.100",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    # parser.add_argument("--dataset_name", type=str, default="librispeech_asr")
    # parser.add_argument("--model_name_or_path", type=str,
    #                     default="/root/public/dev8T/jtang/dataset_text_audio/pretrained_models/models/mode_hu-bart/")
    # parser.add_argument("--dataset_config_name", type=str, default="clean")
    # parser.add_argument("--train_split_name", type=str, default="train.100")
    # parser.add_argument("--eval_split_name", type=str, default="validation")
    # parser.add_argument("--output_dir", type=str, default="/root/public/dev8T/jtang/ASR/output_dir/")
    # # parser.add_argument("--preprocessing_num_workers", type=int, default=16)
    # parser.add_argument("--length_column_name", type=str, default="input_length")
    # parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    # parser.add_argument("--num_train_epochs", type=int, default=5)
    # parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    # parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # parser.add_argument("--learning_rate", type=float, default=3e-4)
    # parser.add_argument("--warmup_steps", type=int, default=400)
    # parser.add_argument("--evaluation_strategy", type=str, default="steps")
    # # parser.add_argument("--text_column_name", type=str, default="text")
    # parser.add_argument("--save_steps", type=int, default=400)
    # parser.add_argument("--eval_steps", type=int, default=400)
    # parser.add_argument("--logging_steps", type=int, default=10)
    # parser.add_argument("--save_total_limit", type=int, default=1)
    # # parser.add_argument("--freeze_feature_encoder", type=bool, default=False)
    # parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    # parser.add_argument("--fp16", type=bool, default=True)
    # parser.add_argument("--group_by_length", type=bool, default=True)
    # parser.add_argument("--predict_with_generate", type=bool, default=True)
    # parser.add_argument("--generation_max_length", type=int, default=40)
    # parser.add_argument("--generation_num_beams", type=int, default=1)
    # parser.add_argument("--do_train", type=bool, default=False)
    # parser.add_argument("--do_eval", type=bool, default=True)
    # parser.add_argument("--do_lower_case", type=bool, default=True)
    # parser.add_argument("--do_lower_case", type=bool, default=True)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_speech_recognition_seq2seq", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)



    # if data_args.audio_column_name not in next(iter(raw_datasets.values())).column_names:  
    #     raise ValueError(
    #         f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
    #         "Make sure to set `--audio_column_name` to the correct audio column - one of "
    #         f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
    #     )
    #
    # if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names: 
    #     raise ValueError(
    #         f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
    #         "Make sure to set `--text_column_name` to the correct text column - one of "
    #         f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
    #     )

    # 5. Load pretrained model, tokenizer, and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        training_args.output_dir,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    config.update({"forced_decoder_ids": model_args.forced_decoder_ids, "suppress_tokens": model_args.suppress_tokens})

    # SpecAugment for whisper models
    if getattr(config, "model_type", None) == "whisper":
        config.update({"apply_spec_augment": model_args.apply_spec_augment})

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        training_args.output_dir,
        cache_dir=model_args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.output_dir,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer
    )
    model = SpeechEncoderDecoderModel.from_pretrained(
        training_args.output_dir,
        config=config,
        cache_dir=model_args.cache_dir
    )

    if model.config.decoder_start_token_id is None:  
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()
    # if model_args.freeze_feature_encoder:
    # for param in model.feature_extractor.parameters():
    #     param.requires_grad = False

    if model_args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    if data_args.language is not None:  # 学习一下tokenizer
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)

    # 6. Resample speech dataset if necessary
    # dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    # if dataset_sampling_rate != feature_extractor.sampling_rate:
    #     raw_datasets = raw_datasets.cast_column(
    #         data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    #     )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = data_args.do_lower_case
    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    # if data_args.max_train_samples is not None:
    #     raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
    #
    # if data_args.max_eval_samples is not None:
    #     raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

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
        # pre_str = batch["text_pre"].lower()
        batch["labels"] = tokenizer(input_str).input_ids
        # batch["decoder_input_ids"] = tokenizer(pre_str, add_special_tokens=False).input_ids
        return batch
    cache_path = r"/root/public/dev8T/jtang/ASR/datasets/asr_test_228/dialog_cache_flag_real_eval"
    if os.path.exists(cache_path):
        try:
            shutil.rmtree(cache_path)
            print(f"Folder '{cache_path}' has been successfully deleted.")
        except Exception as e:
            print(f"An error occurred while deleting the folder: {e}")
    else:
        print(f"Folder '{cache_path}' does not exist.")
    raw_datasets = load_data_by_path("/root/code_project/speech_asr/dataset/libri_pre_16k_noised_dialog_eval_exp_data.py",
                                     cache_path,
                                     False, True)
    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=8,
            desc="preprocess train dataset",
        )
    print("vectorized_datasets", "ready")
    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    # def is_audio_in_length_range(length):
    #     return length > min_input_length and length < max_input_length
    #
    # vectorized_datasets = vectorized_datasets.filter(
    #     is_audio_in_length_range,
    #     num_proc=num_workers,
    #     input_columns=["input_length"],
    # )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    # 8. Load Metric
    metric = evaluate.load("/root/code_project/speech_seq_to_seq/metrics/wer.py", cache_dir=model_args.cache_dir)

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        pred_ids[pred_ids == -100] = tokenizer.pad_token_id
        # pred.label_ids[pred.label_ids == -100] = tokenizer.eos_token_id
        # pred_ids[pred_ids == -100] = tokenizer.eos_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        print(pred_str)
        write_json_result(config_adapter.config.json_path, pred_str)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        wer = metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # 9. Create a single speech processor
    # make sure all processes wait until data is saved
    # with training_args.main_process_first():
    #     # only the main process saves them
    #     if is_main_process(training_args.local_rank):
    #         # save feature extractor, tokenizer and config
    #         feature_extractor.save_pretrained(training_args.output_dir)
    #         tokenizer.save_pretrained(training_args.output_dir)
    #         config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )

    # 11. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # # 12. Training
    # if training_args.do_train:
    #     checkpoint = None
    #     if training_args.resume_from_checkpoint is not None:
    #         checkpoint = training_args.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     trainer.save_model()  # Saves the feature extractor too for easy upload
    #
    #     metrics = train_result.metrics
    #     max_train_samples = (
    #         data_args.max_train_samples
    #         if data_args.max_train_samples is not None
    #         else len(vectorized_datasets["train"])
    #     )
    #     metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))
    #     trainer.log_metrics("train", metrics)
    #     trainer.save_metrics("train", metrics)
    #     trainer.save_state()

    # 13. Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 14. Write Training Stats
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "automatic-speech-recognition"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()