import os
import shutil

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

        input_values = [{"input_values": feature["input_values"][0]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_values, max_length=True, padding=True,
                                                     return_tensors="pt")

        label_features = [{"input_ids": feature["labels"][0]} for feature in features]

        labels_batch = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    # Load dataset with specified path
    raw_datasets = load_data_by_path("/root/code_project/speech_asr/dataset/libri_pre_16k_noised_dialog.py",
                                     "./ASR/datasets/asr_test_228/dialog_cache_noised_b",
                                     True, True)

    # Initialize processor from pretrained checkpoint
    processor = SpeechT5Processor.from_pretrained(
        "./dataset_text_audio/pretrained_models/models/T5_out3/checkpoint-2700"
    )

    sampling_rate = processor.feature_extractor.sampling_rate

    # Filter data exceeding model length limit
    raw_datasets["train"] = raw_datasets["train"].filter(lambda x: len(x["text_upper"].lower()) <= 448)
    raw_datasets["eval"] = raw_datasets["eval"].filter(lambda x: len(x["text_upper"].lower()) <= 448)
    
    # Cast audio columns to Audio type
    raw_datasets = raw_datasets.cast_column("audio_pre", Audio(sampling_rate))
    raw_datasets = raw_datasets.cast_column("audio_after", Audio(sampling_rate))
    print(len(raw_datasets["train"]))
    print(len(raw_datasets["eval"]))

    def prepare_dataset(batch):
        # Process the pre-segment audio
        pre = processor(
            audio=batch["audio_pre"]["array"],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            truncation=True
        )
        # Process the post-segment audio
        after = processor(
            audio=batch["audio_after"]["array"],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            truncation=True
        )
        # Dynamically concatenate features
        combined_input = torch.cat([pre.input_values, after.input_values], dim=1)
        combined_mask = torch.cat([pre.attention_mask, after.attention_mask], dim=1)
        
        # Process text labels
        labels = processor(
            text_target=batch["text_upper"].lower(),
            return_tensors="pt"
        ).input_ids
        
        return {
            "input_values": combined_input,
            "attention_mask": combined_mask,
            "labels": labels,
            "input_length": combined_input.shape[1] / sampling_rate  # Total duration
        }

    # Process dataset
    common_voice = raw_datasets.map(
        prepare_dataset, remove_columns=raw_datasets.column_names["train"], num_proc=32
    )

    print(common_voice["train"])
    print(common_voice["eval"])
    print("Finish processing")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("/root/code_project/speech_seq_to_seq/metrics/wer.py")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

        # Normalize predictions and references
        pred_str_norm = [normalizer(pred) for pred in pred_str]
        label_str_norm = [normalizer(label) for label in label_str]

        # Filter empty references
        pred_str_norm = [
            pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
        ]
        label_str_norm = [
            label_str_norm[i]
            for i in range(len(label_str_norm))
            if len(label_str_norm[i]) > 0
        ]
        wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)
        return {"wer_ortho": wer_ortho, "wer": wer}

    # Initialize model from pretrained checkpoint
    model = SpeechT5ForSpeechToText.from_pretrained(
        "./dataset_text_audio/pretrained_models/models/T5_out3/checkpoint")

    from functools import partial

    model.config.use_cache = False

    # Configure model generation
    model.generate = partial(
        model.generate, use_cache=True
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()