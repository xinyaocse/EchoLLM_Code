# EchoLLM: LLM-Augmented Acoustic Eavesdropping Attack on Bone Conduction Headphones with mmWave Radar

- The *Datasets* folder includes representative subsets of all datasets used in the experiments.
- The *Vibration_Signal* directory contains the implementation of 8kHz vibration signal extraction and signal enhancement using the IWR1443+DCA1000EVM mmWave radar platform.
- The *LeASR* directory provides context-aware inference modules for audio content, leveraging four different large language models (LLMs).
  
---

## Datasets

The Datasets folder contains representative subsets of all datasets used in this study, as the original datasets are too large to be fully released. These subsets are randomly selected and curated to ensure coverage of key experimental scenarios. The folder is organized into the following subdirectories:

- *CNN_TrainTest*: Training and testing data for the CNN-based classification model.
- *LeASR_RealWorld*: Real-world data collected for the LeASR module.
- *Controlled_Experiments*: Data used in controlled environment experiments.
- *Ablation_Study*: Datasets used in ablation studies.
- *Attack_Robustness*: Samples for evaluating robustness under varying attack conditions.
- *User_Diversity*: Data collected from users with diverse demographic and behavioral profiles.
- *Multilingual*: Audio data in multiple languages to test multilingual inference robustness.
- *EchoLLM_Comparison*: Evaluation data for comparing EchoLLM with other acoustic eavesdropping methods.
- *Headphone_Comparison*: Datasets for comparing bone conduction headphones with other types.
- *Numerical_Inference*: Audio clips designed for inferring numerical data such as phone numbers or passcodes.
- *Sensitive_Info*: Samples used for inferring sensitive personal or contextual information.

---

## Vibration_Signal: Vibration Signal Extraction & Vibration Signal Enhancement

The millimeter-wave radar system (IWR1443+DCA1000EVM) is configured and connected according to Texas Instruments' official guidelines. The provided code is designed to run in an environment equipped with mmWave-Studio 2.1.1 and MATLAB Runtime Engine v8.5.1. This documentation details the extraction and enhancement of vibration signals via mmWave radar to ensure accurate data acquisition and signal analysis through coordinated hardware and software tools.

### Initialization

The *Initialization_script* folder contains the necessary ​​Lua script​ and ​​initialization files​​ for establishing connection with mmWave Studio. To perform initialization:

- ​​Launch mmWave Studio​​ and select the appropriate ​​serial port​​.
- Execute the initialization script until a ​​"SUCCESS" message​​ is displayed.

​Note:​​ Ensure the radar configuration file path is correctly set within the script, and customize the provided Lua script as needed to adjust radar parameters according to your specific requirements.

### Verification

The *Verification_script* folder offers methods to verify correct radar configuration and operation. Specifically, bone conduction headphones emit a linearly frequency-modulated signal, which the mmWave radar samples and records using Lua scripts provided in this folder.

### Measurement

The *Measurement_script* folder contains five scripts, with *adc_dataCaptureTest_audio.lua* serving as the main radar configuration script. The remaining four scripts have the following functionalities:

- *adc_dataCapture_model.mlx*: Synchronizes mmWave data acquisition with batch audio playback via bone conduction headphones.
- *muti_loc_exp_test_mti_beamform.mlx*: Implements optimal range bin selection based on signal-to-noise ratio (SNR). (Corresponds to Section 4.2.2: Victim vs. Headphone)
- *fmcw_process_to_audio_local_circle.mlx*: Applies a circle-fitting algorithm to suppress noise and enhance signal clarity. (Corresponds to Section 4.3.1: Background Reflection Reduction)
- *final_data_process.mlx*: Provides a complete signal-processing pipeline, including:
    - Identifying the target range bin. (Corresponding to Section 4.2.1: Victim vs. Other Objects).
    - Extracting phase variation signals. (Corresponding to Section 4.2.3: Headphone Phase Estimation).
    - Eliminating background noise and head motion artifacts. (Corresponding to Section 4.3.2: Motion Calibration).
    - Exporting the processed signal as *.wav* files.

Note: When processing audio recordings of unknown length (i.e., recordings outside training or testing scenarios), configure a high radar frame rate and apply the CNN-based voice activity detector described below to accurately identify actual voice segments.


### CNN-based Voice Activity Detector (EchoVAD)

- *CNN_wav.py*: Implements the training pipeline for the proposed convolutional neural network (CNN)-based voice activity detector (EchoVAD). The model is trained on 10,000 manually labeled spectrogram images to distinguish between speech and non-speech segments, enabling precise detection of audio presence in bone conduction signals.
- *Speech_time_marker.py*: Applies the trained EchoVAD model to a given *.wav* file, segmenting it with a 50 *ms* window and a 10 *ms* sliding step to identify the estimated start and end times of speech segments. This process corresponds to the Headphone Phase Estimation step described in Section 4.2.3.

---
## LeASR

### Demo

If you simply want to try out LeASR, a lightweight automatic speech recognition (ASR) demo is provided in the **Demo** folder. To run the demo, you only need to prepare an audio file and a fine-tuned model. For converience, contextual audio segments can be concatenated into a single input to streamline the recognition process.

### detail of LeASR

#### Scripts for LeASR fine-tuning, training, and evaluation
- creat_model: Modifies the specified model name (e.g., HuBART, HuBART-L, Whisper, SpeechT5) and generates the corresponding pre-trained model at the designated file path.
- dataset: Contains scripts for dataset configure. During training and testing, the data loading logic for different datasets can be modified via this code. Users need to adjust paths and related metadata according to their local dataset setup.
- metrics: Provides code for computing the Word Error Rate (WER). Specifically, *wer.py* is used during training, while *Cal_exp_wer.py* is used to compute the evaluation-phase metric WER_B.
- others: Includes training and testing scripts for various encoder-decoder configurations. Multiple similar versions of the same model may exist to accommodate different dataset formats and experimental settings. Users should ensure they select the appropriate version that matches their dataset format.


#### Local Deployment Instructions
- First, prepare your dataset and the corresponding JSON configuration file. Several dataset reading templates are provided in the *dataset* folder, which you can either use directly or customize as needed. You may also implement your own data loading methods if necessary. An example JSON configuration file is shown below:
```json
    [{
        "previous_text": "Did anyone get hurt?",
        "current_text": "Two people were injured.",
        "target_audio": "/root/public/....",
        "audio_pre": "04968-06_Raw_0.wav",
        "audio_after": "04968-07_Raw_0.wav",
        "id": "04968-07",
        "delay_time": "0.36",
        "transcript": "Did anyone get hurt? Two people were injured."
        ...(Experiment-specific data)
    },...]
```
- Second, update the path to the pre-trained model and the data loader script in the selected model. (The pre-trained model can be downloaded from Hugging Face.) Optionally, you may write the evaluation results back to a JSON file for computing the WER_B metric.
- Note: For combined models such as HuBERT+BART, the corresponding training code is provided in the *model* folder. The following table outlines how to configure the training parameters.
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

### Data loading

- *dataset/libri_pre_16k_noised_dialog_.py*: Provides loading of complete datasets (training sets, testing sets, and validation sets) for specific languages, ​​supporting training and evaluation for different models:
  - *dataset/libri_pre_16k_noised_Synthetic.py*: Loading English synthetic datasets.
  - *dataset/libri_pre_16k_noised_Real.py*: Loading English real datasets.
  - *dataset/libri_pre_16k_noised_CN.py*: Loading Chinese datasets.
  - *dataset/libri_pre_16k_noised_FR.py*: Loading French datasets.
  - *dataset/libri_pre_16k_noised_JP.py*: Loading Japanese datasets.
  - *dataset/libri_pre_16k_noised_eval_Exp_data.py*: Call *exp_config.py*.
- *dataset/exp_config.py*: The code controls data loading for all experimental including:
  - Controlled Experiments datasets loading.
  - Ablation Study datasets loading.
  - Attack Robustness datasets loading.
  - User Diversity Study datasets loading.
  - Multilingual Robustness datasets loading.
  - EchoLLM vs. Other Eavesdropping Attacks datasets loading.
  - Bone Conduction vs. Other Headphones datasets loading.
  - Inferring Numerical Data datasets loading.
  - Inferring Sensitive Information datasets loading.


