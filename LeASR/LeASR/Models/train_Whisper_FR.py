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


def japanese_wer(ref: str, hyp: str) -> float:
    """
    Character-based Word Error Rate calculation (simplified version)

    :param ref: Reference text (ground truth)
    :param hyp: Recognized text
    :return: WER (0.0-1.0)
    """
    # Remove spaces (modify if punctuation needs to be preserved)
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")

    # Convert to character lists
    ref_chars = list(ref)
    hyp_chars = list(hyp)

    # Create dynamic programming matrix
    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
    for i in range(len(ref_chars) + 1):
        d[i, 0] = i
    for j in range(len(hyp_chars) + 1):
        d[0, j] = j

    # Calculate edit distance
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,  # Deletion
                d[i, j - 1] + 1,  # Insertion
                d[i - 1, j - 1] + cost  # Substitution
            )

    edits = d[len(ref_chars), len(hyp_chars)]
    return edits / len(ref_chars) if len(ref_chars) > 0 else 0.0

training_args = Seq2SeqTrainingArguments(
    output_dir="/root/public/dev8T/username/dataset_text_audio/pretrained_models/models/openAI_out_FR",
    # Output directory name on HF Hub
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,  # Halve batch size and double this parameter
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
        # Separate input features and labels which have different lengths
        # First return audio features as PyTorch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad labels to the max length
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
    # 1. Load and vectorize data for Whisper
    raw_datasets = load_data_by_path("/root/code_project/speech_asr/dataset/libri_pre_16k_noised_dialog_FR.py",
                                     "/root/public/dev8T/username/ASR/datasets/asr_test_228/dialog_cache_real",
                                     True, True)

    processor = WhisperProcessor.from_pretrained(
        "/root/public/dev8T/username/dataset_text_audio/pretrained_models/models/openAI_out1/checkpoint-800",
        language="French",  # Explicitly specify language
        task="transcribe"  # Specify transcription task
    )

    sampling_rate = processor.feature_extractor.sampling_rate

    # Filter datasets
    raw_datasets["train"] = raw_datasets["train"].filter(
        lambda example: len(example["text_upper"]) < 400,
        num_proc=4,
    )
    raw_datasets["eval"] = raw_datasets["eval"].filter(
        lambda example: len(example["text_upper"]) < 400,
        num_proc=4,
    )
    raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=sampling_rate))

    def prepare_dataset(batch):
        sample = batch['audio']
        batch = processor(
            audio=sample["array"],
            sampling_rate=sample["sampling_rate"],
            text=batch["text_upper"]
        )
        batch["input_length"] = len(sample["array"]) / sample["sampling_rate"]
        return batch

    common_voice = raw_datasets.map(
        prepare_dataset, remove_columns=raw_datasets.column_names["train"], num_proc=32
    )

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
        
        # Print individual WER for each sample
        print("\nIndividual Orthographic WER:")
        ortho_wer_samples = []
        for i, (p, l) in enumerate(zip(pred_str, label_str)):
            # Skip empty reference samples
            if not l.strip():
                continue
            # Calculate single sample WER
            wer1 = japanese_wer(p, l)
            print(f'Pred: {p} | Ref: {l}')
            ortho_wer_samples.append(wer1)
        print(sorted(ortho_wer_samples))

        return {
            "wer": sum(ortho_wer_samples)/len(ortho_wer_samples)
        }

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

    # Set language and task for generation, re-enable cache
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