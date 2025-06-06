# EchoLLM: LLM-Augmented Acoustic Eavesdropping Attack on Bone Conduction Headphones with mmWave Radar

- The code for 8kHz vibration signal extraction and signal enhancement based on IWR1443+DCA1000EVM platform is available in the *Vibration_Signal* directory.
- The *CNN-based_Classification_Model* directory contains a method for estimating audio delay using a convolutional neural network.
- The *LeASR* directory includes the implementation for context-aware inference of audio content leveraging four different large language models (LLMs).

---

## Vibration Signal Extraction & Vibration Signal Enhancement

The millimeter-wave radar system (IWR1443+DCA1000EVM) is configured and connected in accordance with TI's official guidelines. The provided code is designed to run in an environment equipped with mmWave-Studio 2.1.1 and MATLAB Runtime Engine v8.5.1. This documentation details the procedures for extracting and enhancing vibration signals using mmWave radar, utilizing the specified hardware and software tools to enable accurate data acquisition and signal analysis.

### Initialization

The *Initialization_script* folder contains the necessary ​​Lua script​ and ​​initialization file required​​ for establishing a connection with mmWave Studio. To proceed:

- ​​Launch mmWave Studio​​ and select the correct ​​serial port​​.
- Run the initialization script until ​​"SUCCESS"​​ is displayed.

​Note:​​ Ensure that the path to the radar configuration file is correctly specified in the script.

You can ​​customize the Lua script​​ to modify radar configuration parameters according to your specific requirements.

### Verification

The *Verification_script* folder provides a method to verify the successful configuration and operational status of the radar. Specifically, the bone conduction headphones emit a linearly frequency-modulated signal, which is sampled and recorded by the mmWave radar using the Lua scripts in this folder.

### Measurement

The *Measurement_script* folder contains five files. Among them, *a​​dc_dataCaptureTest_audio.lua​​* serves as the radar configuration file. The remaining four scripts are used for the following purposes:

- *adc_dataCapture_model.mlx*: Synchronizes mmWave data acquisition with batch audio playback via bone conduction headphones.
- *muti_loc_exp_test_mti_beamform.mlx*: Implements signal-to-noise (SNR)-based optimal range bin selection.
- *fmcw_process_to_audio_local_circle.mlx*: Applies a circle-fitting algorithm to suppress noise and enhance signal clarity.
- *final_data_process_923.mlx*: provides a complete processing pipeline, including:
    - Identifying the target range bin
    - Extracting phase variation signals
    - Eliminating background noise and head motion artifacts
    - Exporting the processed signal as a .wav files

Note: When dealing with unknown-length audio recordings (i.e., outside of the training or testing phases), it is recommended to configure a high radar frame rate in the profile and apply the CNN-based method introduced in the paper to detect the actual voice segments.

---

## CNN-based Classification Model

### CNN_wav.py

The *CNN_wav.py* script trains a convolutional neural network (CNN) to detect the presence of voice in audio signals. Specifically, 10,000 spectrogram images are manually labeled and used as the training dataset to enable voice activity classification.

### Speech_time_marker.py

The *Speech_time_marker.py* script takes a .wav file as input and segments it using a 50*ms* window size with a 10*ms* sliding step. It outputs the estimated start and end times of detected speech segments.

---
## How to use the LeASR

### Demo

If you simply want to try out LeASR, a lightweight automatic speech recognition (ASR) demo is provided in the **Demo** folder. To run the demo, you only need to prepare an audio file and a fine-tuned model. For converience, contextual audio segments can be concatenated into a single input to streamline the recognition process.

### LeASR

#### Scripts for LeASR fine-tuning, training, and evaluation
- creat_model: Selects the specified model name (e.g., HuBART, HuBART-L, Whisper, SpeechT5) and generates the corresponding pre-trained model at the designated file path.
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
        "delay_time": "0.3647305929570286",
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
--train_split_name="train.100"
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

### Notes

- For more information on how the dataset is loaded, refer to the beginning of the *libri_pre_16k_noised_dialog_.py* file and the *exp_config.py* file in the *dataset* folder.
  
    - The former defines the data loading process used for training, evaluation, and testing across the full language dataset.
    - The latter provides configuration settings for ablation experiments.
    
```python
# Dataset path configuration in libri_pre_16k_noised_dialog_JP.py
META_DATA_TRAIN_PATH = r'/root/public/dev8T/username/dataset_text_audio/ASR_test_JP_train.json'
META_DATA_TEST_PATH = r'/root/public/dev8T/username/dataset_text_audio/ASR_test_JP_test.json'
META_DATA_VAL_PATH = r'/root/public/dev8T/username/dataset_text_audio/ASR_eval_JP_eval.json'
```
For different data loading scenarios, the data loading logic in the evaluation script may need to be modified accordingly.

```python
# Data loading under different volumes, angles and motions in exp_config.py
exp_v6_config = ExpConfig({"Volume": ['50', '60', '70', '80', '90', '100'], "Angle": ["15", '30', '45', '60', '75'],
                           "Headphone": ["type2"], "distance_v80": ['40', '60', '80', '100'],
                           "Motion": ['static', "FB", "LR", "UD"]},
                          "/root/public/dev8T/username/ASR/exp/exp_v6_result.json",
                          "/root/public/dev8T/username/ASR/exp/exp_v6_pre_dataset/")
exp_v6_config.set_path_dict({"Volume": r"/root/public/dev8T/username/ASR/exp/exp_v6/volume/",
                             "Angle": r"/root/public/dev8T/username/ASR/exp/exp_v6/Angle/",
                             "Headphone": "/root/public/dev8T/username/ASR/exp/exp_v6/headphone/",
                             "Distance_v80": r"/root/public/dev8T/username/ASR/exp/exp_v6/distance_v80/",
                             "Motion": r"/root/public/dev8T/username/ASR/exp/exp_v6/Motion/"})
...
config_adapter = ConfigAdapter()
config_adapter.set_config(exp_v6_config)
```

- The required code environment is specified in *LeASR/LeASR/requirements.txt*.
