# EchoLLM: LLM-Augmented Acoustic Eavesdropping Attack on Bone Conduction Headphones with mmWave Radar

- The code for 8kHz vibration signal sampling and audio reconstruction based on IWR1443+DCA1000EVM platform is available in the Vibration_Signal directory.
- The CNN directory contains a method for estimating audio delay using a convolutional neural network.
- The LeASR directory includes the implementation for context-aware inference of audio content leveraging four different large language models (LLMs).

---

## Vibration Signal Extraction & Vibration Signal Enhancement

The millimeter-wave radar system (IWR1443+DCA1000EVM) is configured and connected in accordance with TI's official guidelines. The provided code is designed to run in an environment equipped with mmWave-Studio 2.1.1 and MATLAB Runtime Engine v8.5.1. This documentation details the procedures for extracting and enhancing vibration signals using mmWave radar, utilizing the specified hardware and software tools to enable accurate data acquisition and signal analysis.

### Initialization

The Initialization_script folder contains the necessary ​​Lua scripts​​ and ​​initialization files required​​ for establishing a connection with mmWave Studio. To proceed:

- ​​Launch mmWave Studio​​ and select the correct ​​serial port​​.
- Run the initialization scripts until ​​"SUCCESS"​​ is displayed.
- ​Note:​​ Ensure that the path to the radar configuration file is correctly specified in the script.

You can ​​customize the Lua script​​s to modify radar configuration parameters according to your specific requirements.

### Verification

The Verification_script folder provides a method to verify the successful configuration and operational status of the radar. Specifically, the bone conduction headphones emits a linearly frequency-modulated signal, which is sampled and recorded by the mmWave radar using the Lua scripts in this folder.

### Measurement

In the measurement folder, there are five files. Among them, *a​​dc_dataCaptureTest_audio.lua​​*a is the configuration file for radar parameters. The remaining four code files are used for:

- *adc_dataCapture_model.mlx*: Control mmWave data acquisition synchronously while the bone conduction headphone plays audio in batches.
- *muti_loc_exp_test_mti_beamform.mlx*: Implements an SNR-based decision mechanism for determining the optimal range bin.
- *fmcw_process_to_audio_local_circle.mlx*: Applies a circle-fitting algorithm for denoision.
- *final_data_process_923.mlx*: Provides a streamlined method to determine the range bin of the mmWave file, extract the corresponding phase change data, remove background noise, and remove head movements. Finally, save the result to a WAV file.


If it is used for unknown audio length (non-training and testing phases), the radar frame rate of the profile is set to a large and the CNN method in the paper is used to identify the voice time period.

---

## CNN-based Classification Model

### CNN_wav.py

In this script, we artificially label 10,000 spectral images and train them to determine whether there is a voice.
### Time_split.py

In this script, a .wav file is entered, split by a 50ms size and 10ms sliding window, and the start and end times of the sound are output.

---

## How to use the LeASR
---
### demo
If you just want to experience the use of LeASR, a simple ASR demo is provided in the **demo** file. You only need to prepare the audio and the fine-tuned model to complete the recognition process. And to simplify the steps, the contextual audio can be directly spliced for input.

### LeASR
#### This folder contains some of the source code for our fine-tuning training and evaluation
* creat_model: The hubert model was combined with BART's decoder (or BART-Large's) to generate a pre-trained model.
* dataset: The code in this folder configures the data for the experiment. Specifically, during the training or testing phase, the reads of different datasets are altered by changing that configuration code. Changes need to be made based on the address and related information of the local data.
* metrics: The calculation method of WER is provided for model training. Among them, wer.py is used in the model training phase, and Cal_exp_wer.py is used to calculate WER_B in the evaluation phase.
* others: Training and testing code for different encoder-decoder combinations. For the same model, there may be multiple codes with high similarity (because there are multiple dataset forms and test contents), if necessary, pay attention to match your own dataset form.

#### How to deploy locall?
- First, you need to prepare the dataset and JSON file for configuration. We provide multiple reading templates in the *dataset* folder. You can also rewrite the methods yourself. An example of our JSON file is as follows:
```json
    [{
        "previous_text": "Did anyone get hurt? ",
        "current_text": "Two people were injured. ",
        "target_audio": "/root/public/....",
        "audio_pre": "04968-06_Raw_0.wav",
        "audio_after": "04968-07_Raw_0.wav",
        "id": "04968-07",
        "delay_time": "0.3647305929570286",
        "transcript":"Did anyone get hurt? Two people were injured."
        ...(Experiment-specific data)
    },...]
```
- Second, change the position of the pre-trained model and the script position of the data loader in the model you selected (The pre-trained model can be obtained at Huggingface) and optionally write the solution evaluation results back to JSON for WER_B calculations.
- Note that for a combined model like HuBERT+BART, there is generated code in the *model* folder. The following table describes how to set the training parameters.
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

- For more information about how to load the dataset, see the file at the beginning of the *libri_pre_16k_noised_dialog_.py* and the *exp_config.py* file in the dataset folder. The former is used for training, evaluation and testing of the language as a whole. The latter is the configuration of the ablation experiment.
```python
# Dataset path settings in libri_pre_16k_noised_dialog_JP.py
META_DATA_TRAIN_PATH = r'/root/public/dev8T/username/dataset_text_audio/ASR_test_JP_train.json'
META_DATA_TEST_PATH = r'/root/public/dev8T/username/dataset_text_audio/ASR_test_JP_test.json'
META_DATA_VAL_PATH = r'/root/public/dev8T/username/dataset_text_audio/ASR_eval_JP_eval.json'
```
For different data loading scenarios, the loading code of the eval file needs to be changed
```python
# Data loading at different volumes, angles and Motion in exp_config
exp_v6_config = ExpConfig({"volume": ['50', '60', '70', '80', '90', '100'], "Angle": ["15", '30', '45', '60', '75'],
                           "headphone": ["type2"], "distance_v80": ['40', '60', '80', '100'],
                           "Motion": ['static', "FB", "LR", "UD"]},
                          "/root/public/dev8T/username/ASR/exp/exp_v6_result.json",
                          "/root/public/dev8T/username/ASR/exp/exp_v6_pre_dataset/")
exp_v6_config.set_path_dict({"volume": r"/root/public/dev8T/username/ASR/exp/exp_v6/volume/",
                             "Angle": r"/root/public/dev8T/username/ASR/exp/exp_v6/Angle/",
                             "headphone": "/root/public/dev8T/username/ASR/exp/exp_v6/headphone/",
                             "distance_v80": r"/root/public/dev8T/username/ASR/exp/exp_v4/distance_v80/",
                             "Motion": r"/root/public/dev8T/username/ASR/exp/exp_v6/Motion/"})
...
config_adapter = ConfigAdapter()
config_adapter.set_config(exp_v6_config)
```

- The code environment is detailed in *LeASR/LeASR/requirements.txt*.
