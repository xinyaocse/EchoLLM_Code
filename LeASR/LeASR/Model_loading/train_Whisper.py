import os
import shutil

from transformers import (AutoFeatureExtractor, WhisperProcessor)
from datasets import Audio
from transformers.trainer_utils import get_last_checkpoint

from dataset.fmcw_dataloader import load_data, DataCollatorSpeechSeq2SeqWithPadding, load_data_by_path
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments

normalizer = BasicTextNormalizer()
from transformers import Seq2SeqTrainer

import numpy as np
from typing import List



training_args = Seq2SeqTrainingArguments(
    output_dir="/root/public/dev8T/username/dataset_text_audio/pretrained_models/models/openAI_out4",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
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
    generation_max_length=225,
    save_steps=400,
    eval_steps=400,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    do_train=True,

)


@dataclass()
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Separate input features and labels as they require different padding methods
        # First return audio features as PyTorch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad the label sequences to maximum length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore in loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If bos token was added during tokenization, remove it as we'll add it later
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def main():
    # 1. Load and vectorize data for whisper
    # raw_datasets = load_data_by_path("/root/code_project/speech_asr/dataset/libri_pre_16k_noised_dialog.py",
    #                                  "/root/public/dev8T/username/ASR/datasets/asr_test_228/dialog_cache_noised_b",
    #                                  True, True)
    raw_datasets = load_data_by_path("/root/code_project/speech_asr/dataset/libri_pre_16k_noised_dialog_after.py",
                                     "/root/public/dev8T/username/ASR/datasets/asr_test_228/dialog_cache_real",
                                     True, True)
    # raw_datasets = load_data_by_path("/root/code_project/speech_asr/dataset/libri_pre_16k_noised_dialog_real.py",
    #                                  "/root/public/dev8T/username/ASR/datasets/asr_test_228/dialog_cache_real",
    #                                  True, True)
    # print(raw_datasets) ##['audio', 'text_upper', 'id']

    processor = WhisperProcessor.from_pretrained(
        "/root/public/dev8T/username/dataset_text_audio/pretrained_models/models/openAI_out1/checkpoint-800",
        language="Japanese",  # Explicitly specify language
        task="transcribe"  # Specify transcription task
    )

    sampling_rate = processor.feature_extractor.sampling_rate
    print(len(raw_datasets["train"]))
    print(len(raw_datasets["eval"]))
    # raw_datasets["train"] = raw_datasets["train"].select(range(10)) # Resampling
    # raw_datasets["eval"] = raw_datasets["eval"].select(range(1)) # Resampling
    raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=sampling_rate))

    def prepare_dataset(batch):
        sample = batch['audio']
        batch = processor(
            audio=sample["array"],
            sampling_rate=sample["sampling_rate"],
            # Remove lowercase conversion as it's not needed for Chinese
            text=batch["text_upper"]
        )
        batch["input_length"] = len(sample["array"]) / sample["sampling_rate"]
        return batch

    common_voice = raw_datasets.map(
        prepare_dataset, remove_columns=raw_datasets.column_names["train"], num_proc=32
    )

    print(common_voice["train"])
    print(common_voice["eval"])
    print("Finished Whisper processing")

    # Initialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # Replace -100 with pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        # Print per-sample WER (wer_ortho)
        print("\nIndividual Orthographic WER:")
        ortho_wer_samples = []
        for i, (p, l) in enumerate(zip(pred_str, label_str)):
            # Skip empty reference samples
            if not l.strip():
                continue
            # Calculate single sample WER
            wer = 100 * metric.compute(predictions=[p], references=[l])
            print('prediction:' + p + 'reference:' + l)
            ortho_wer_samples.append(wer1)
        print(sorted(ortho_wer_samples))

        return {
            "wer": sum(ortho_wer_samples)/len(ortho_wer_samples)
        }

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
    model = WhisperForConditionalGeneration.from_pretrained(
        "/root/public/dev8T/username/dataset_text_audio/pretrained_models/models/openAI_out1/checkpoint-800")
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="Japanese",
        task="transcribe"
    )
    model.config.forced_decoder_ids = forced_decoder_ids

    from functools import partial

    # Disable cache during training as it's incompatible with gradient checkpointing
    model.config.use_cache = False

    # Set language and task for generation and re-enable cache
    model.generate = partial(
        model.generate, use_cache=True
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    trainer.train()


if __name__ == '__main__':
    main()