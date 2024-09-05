import os
import shutil
import json
import sys
sys.path.append('/root/code_project/audio/asr_pre_dataset/exp/')
from exp_config import config_adapter
from dataset.dataset_config import EvalAdapter, flag_eval, ablation_eval, ablation_flag_eval
from transformers import (AutoFeatureExtractor, WhisperProcessor, SpeechT5Processor, SpeechT5ForSpeechToText)
from datasets import Audio
from transformers.trainer_utils import get_last_checkpoint
from dataset.fmcw_dataloader import load_data, DataCollatorSpeechSeq2SeqWithPadding, load_data_by_path
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import Seq2SeqTrainingArguments

normalizer = BasicTextNormalizer()
from transformers import Seq2SeqTrainer

def write_result(path, result_):
    with open(path, "w") as f:
        f.write(json.dumps(result_))


def get_result(path):
    with open(path, "r") as f:
        result = f.read()
    return json.loads(result)


def write_json_result1(json_path, text_list):
    files = get_result(json_path)
    index = 0
    for item in files:
        for key in config_adapter.config.key_list:
            for value in files[item][key]:
                files[item][key][value]['T5_result'] = text_list[index]
                index += 1
    write_result(json_path, files)

def write_json_result2(json_path, text_list, val_key):
    files = get_result(json_path)
    for index, item in enumerate(files):
        item[val_key] = text_list[index]
    write_result(json_path, files)

training_args = Seq2SeqTrainingArguments(
    output_dir="./dataset_text_audio/pretrained_models/models/T5_out4",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8, 
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=450,
    save_steps=100,
    eval_steps=100,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    do_train=False,

)


#
@dataclass()
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_values = [{"input_values": feature["input_values"][0]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_values, max_length=True, padding=True,
                                                     return_tensors="pt")
        # Convert input features to half precision
        batch["input_values"] = batch["input_values"].half()
        label_features = [{"input_ids": feature["labels"][0]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    cache_path = r"./ASR/datasets/asr_test_228/dialog_cache_flag_real_eval"
    if os.path.exists(cache_path):
        try:
            shutil.rmtree(cache_path)
            print(f"Folder '{cache_path}' has been successfully deleted.")
        except Exception as e:
            print(f"An error occurred while deleting the folder: {e}")
    else:
        print(f"Folder '{cache_path}' does not exist.")

    eval_dataset = EvalAdapter()
    eval_dataset.set_datasets(ablation_flag_eval)
    raw_datasets = load_data_by_path("/root/code_project/speech_asr/dataset/libri_pre_16k_noised_dialog_eval.py",
                                     cache_path,
                                     False, True)
    # print(raw_datasets) ##['audio', 'text_upper', 'id']

    processor = SpeechT5Processor.from_pretrained(
        "./dataset_text_audio/pretrained_models/models/T5_out3/checkpoint-2700"
    )

    sampling_rate = processor.feature_extractor.sampling_rate
    raw_datasets["eval"] = raw_datasets["eval"].filter(lambda x: len(x["text_upper"].lower()) <= 448)
    raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=sampling_rate))
    print(len(raw_datasets["eval"]))

    def prepare_dataset(batch):
        # process audio
        sample = batch['audio']
        batch = processor(
            audio=sample["array"], sampling_rate=sample["sampling_rate"], text_target=batch["text_upper"].lower(),
            return_tensors="pt"
        )
        batch["input_length"] = len(sample["array"]) / sample["sampling_rate"]
        return batch

    common_voice = raw_datasets.map(
        prepare_dataset, remove_columns=raw_datasets.column_names["eval"], num_proc=32
    )

    print(common_voice["eval"])
    print("finish Wec")


    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("/root/code_project/speech_seq_to_seq/metrics/wer.py")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)
        pred_str_norm = [normalizer(pred) for pred in pred_str]
        label_str_norm = [normalizer(label) for label in label_str]
        pred_str_norm = [
            pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
        ]
        label_str_norm = [
            label_str_norm[i]
            for i in range(len(label_str_norm))
            if len(label_str_norm[i]) > 0
        ]
        wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)
        write_json_result2(eval_dataset.eval_dataset.val_path, pred_str, eval_dataset.eval_dataset.val_key)
        print({"wer_ortho": wer_ortho, "wer": wer})
        return {"wer_ortho": wer_ortho, "wer": wer}

    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #         print(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )
    # print(last_checkpoint)
    model = SpeechT5ForSpeechToText.from_pretrained(
        "./dataset_text_audio/pretrained_models/models/T5_out3/checkpoint-2700")

    from functools import partial

    model.config.use_cache = False

    model.generate = partial(
        model.generate, use_cache=True
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        eval_dataset=common_voice["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    trainer.evaluate()


if __name__ == '__main__':
    main()
