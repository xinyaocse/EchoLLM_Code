# EchoLLM: LLM-Augmented Acoustic Eavesdropping Attack on Bone Conduction Headphones with mmWave Radar

This repository contains the implementation of EchoLLM, including vibration signal processing, contextual speech recognition with large language models, and associated datasets for evaluation.

- The *Datasets* folder includes representative subsets of all datasets used in the experiments.
- The *Vibration_Signal* directory contains the implementation of 8kHz vibration signal extraction and enhancement using the IWR1443+DCA1000EVM mmWave radar platform.
- The *LeASR* directory provides context-aware inference modules for audio content, leveraging four large language models (LLMs).
  
---

## Datasets

The *Datasets* folder includes curated subsets of the full datasets used in our experiments, as the complete datasets are too large to be fully released. These subsets are randomly sampled to preserve the diversity of experimental conditions. The folder is organized into two subdirectories:

- TrainSet: Contains training data for the CNN-based voice activity detector.
  - *CNN_TrainTest*: Training and testing sets for the CNN-based voice activity detector.
- TestSet: Contains evaluation datasets used in the various experimental settings described in the paper. These include:
  - *LeASR_RealWorld*: Real-world data collected for the LeASR module.
  - *Controlled_Experiments*: Datasets used in controlled environment experiments.
  - *Ablation_Study*: Datasets used in ablation studies.
  - *Attack_Robustness*: Samples to evaluate robustness under varying attack conditions.
  - *User_Diversity*: Data collected from users with diverse demographic and behavioral profiles.
  - *Multilingual*: Audio data in multiple languages for multilingual inference robustness.
  - *EchoLLM_Comparison*: Evaluation data for comparing EchoLLM against baseline acoustic eavesdropping attacks.
  - *Headphone_Comparison*: Datasets for comparing bone conduction headphones with other headphone types.
  - *Numerical_Inference*: Audio segments designed for inferring numerical data such as phone numbers or passcodes.
  - *Sensitive_Info*: Audio segments used for inferring sensitive personal or contextual information.

---

## Vibration_Signal: Vibration Signal Extraction & Vibration Signal Enhancement

This module implements signal extraction and enhancement based on mmWave radar sensing. The system uses the IWR1443+DCA1000EVM platform, configured per TI guidelines. It runs in an environment with mmWave Studio 2.1.1 and MATLAB Runtime Engine v8.5.1.

### Initialization

The *Initialization_script* folder contains the necessary ​​Lua script​ and ​​initialization files​​ for establishing connection with mmWave Studio. To perform initialization:

- ​​Launch mmWave Studio​​ and select the appropriate ​​serial port​​.
- Execute the initialization script until a ​​"SUCCESS" message​​ is displayed.

​Note:​​ Ensure the radar configuration file path is correctly set within the script, and customize the provided Lua script as needed to adjust radar parameters according to your specific requirements.

### Verification

The *Verification_script* folder offers methods to verify correct radar configuration and operation. Specifically, bone conduction headphones emit a linearly frequency-modulated signal, which the mmWave radar samples and records using Lua scripts provided in this folder.

### Measurement

The *Measurement_script* folder contains three scripts:

- *adc_dataCaptureTest_audio.lua*: Main radar configuration script.
- *adc_dataCapture_model.mlx*: Synchronizes mmWave data acquisition with batch audio playback.
- *final_data_process.mlx*: Provides a complete signal-processing pipeline, including:
    - Target range bin identification (Section 4.2.1: Victim vs. Other Objects).
    - SNR-based optimal range bin selection (Section 4.2.2: Victim vs. Headphone).
    - Phase variation signal extraction (Section 4.2.3: Headphone Phase Estimation).
    - Circle-fitting for noise suppression and clarity enhancement (Section 4.3.1: Background Reflection Reduction).
    - Motion calibration to remove artifacts (Section 4.3.2: Motion Calibration).
    - Exporting the final signal as *.wav* files.

Note: For unknown-length recordings, use a high radar frame rate and apply the CNN-based voice activity detector (EchoVAD) to identify speech segments.

### CNN-based Voice Activity Detector (EchoVAD)

- *CNN_wav.py*: Trains a CNN-based voice activity detector (EchoVAD) using 10,000 labeled spectrograms to classify speech vs. non-speech.
- *Speech_time_marker.py*: Applies the trained EchoVAD to segment *.wav* files, using 50 *ms* windows and 10 *ms* strides. Outputs estimated speech start and end times (used in Section 4.2.3).

---
## LeASR

### Demo

A lightweight ASR demo is available in the *Demo* folder. To run it, prepare an audio file and a fine-tuned model. Contextual audio segments can be concatenated to improve inference continuity.

### LeASR Directory Structure

The *LeASR* directory is organized into four main folders:
- *Model_loading*: Contains scripts for building, training, and evaluating the core speech recognition models.
   - *model_create.py*: Builds and saves pre-trained models (e.g., HuBART, HuBART-L, Whisper, SpeechT5).
   - Training & evaluation scripts:
      - *train_HuBART.py & eval_HuBART.py*: Training and evaluation of the HuBART model.
      - *train_HuBART_L.py & eval_HuBART_L.py*: Training and evaluation of the HuBART-L model.
      - *train_Whiper.py & eval_Whiper.py*: Training and evaluation of the Whiper model. Since Whisper is a multilingual model, language-specific decoder prompts must be configured. For example:
      ```python
          forced_decoder_ids = processor.get_decoder_prompt_ids(
              language="English",
              task="transcribe"
          )
      ```
      - *train_SpeechT5.py & eval_SpeechT5.py*: Training and evaluation of the SpeechT5 model.
- *LLM_models*: Provides four large language models (LLMs).
- *Data_loading*: Contains scripts managing data loading for LeASR by referencing external dataset configurations from the separate *Datasets/TrainSet* directory.
    - Training dataset loaders:
        - *libri_pre_16k_noised_Synthetic.py*: Loads synthetic English datasets.
        - *libri_pre_16k_noised_Real.py*: Loads real-world English datasets.
        - *libri_pre_16k_noised_CN.py*: Loads Chinese datasets.
        - *libri_pre_16k_noised_FR.py*: Loads French datasets.
        - *libri_pre_16k_noised_JP.py*: Loads Japanese datasets.
    - Experimental evaluation dataset loaders:
        - *exp_config.py*: Specifies the dataset paths and configurations for various experimental scenarios, including controlled experiments, robustness evaluations, ablation studies, multilingual testing, and sensitive information inference.
        - *libri_pre_16k_noised_eval_Exp_data.py*: Implements the data loading logic for experimental evaluations by calling and parsing the corresponding dataset configurations defined in *exp_config.py*.
- *Metrics*: Provides scripts for evaluating LeASR system performance.
    - *wer.py*: Computes the Word Error Rate (WER) metric during training.
    - *Cal_exp_wer.py*: Computes the specialized WER_B metric for detailed evaluation during experimental analyses.

#### Local Deployment Instructions
- First, prepare your dataset and a corresponding JSON configuration file. Several dataset loading templates are available in the *data_loading* folder for direct use or customization. You can also implement your own data loading method if necessary. An example JSON configuration is shown below:
```json
    [{
        "previous_text": "Did anyone get hurt?",
        "current_text": "Two people were injured.",
        "audio_pre": "04968-06_Raw_0.wav",
        "audio_after": "04968-07_Raw_0.wav",
        "id": "04968-07",
        "delay_time": "0.36",
        "transcript": "Did anyone get hurt? Two people were injured."
        ...(Experiment-specific data)
    },...]
```
- Second, update the paths for the pre-trained model (downloadable from Hugging Face) and the data loader script within your selected model script. Optionally, evaluation results can be exported back into a JSON file to facilitate WER_B metric computation.

Note: For combined models such as HuBERT+BART, training scripts are located within the *model* folder. You can configure training parameters as exemplified below:
```python
--dataset_name="librispeech_asr"
--model_name_or_path="/root/public/....(your path)"
--dataset_config_name="clean"
--train_split_name="train"
--eval_split_name="validation"
--output_dir="/root/public/....(your path)"
--preprocessing_num_workers="16"
--length_column_name="input_length"
--overwrite_output_dir=true
--num_train_epochs="5"
--per_device_train_batch_size="4"
--per_device_eval_batch_size="4"
--gradient_accumulation_steps="4"
--learning_rate="3e-4"
--warmup_steps="400"
--evaluation_strategy="steps"
--text_column_name="text"
--save_steps="400"
--eval_steps="400"
--logging_steps="10"
--save_total_limit="1"
--freeze_feature_encoder=false
--gradient_checkpointing=true
--fp16=true
--group_by_length=true
--predict_with_generate=true
--generation_max_length="50"
--generation_num_beams="1"
--do_train=true
--do_eval=true
--do_lower_case=true
```


