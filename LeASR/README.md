## How to use the LeASR
---
#### demo
> If you just want to experience the use of LeASR, a simple ASR demo is provided in the **demo** file. You only need to prepare the audio and the fine-tuned model to complete the recognition process.  

#### LeASR
##### This folder contains some of the source code for our fine-tuning training and evaluation
* creat_model: The hubert model was combined with BART's decoder to generate a pre-trained model
* dataset: Perhaps this part needs to meet its own dataset form. Since we have a variety of datasets, some of the code content is similar, just for reference.
* metrics: WER
* others: Training and testing code for different encoder-decoder combinations. For the same model, there may be multiple codes with high similarity (because there are multiple dataset forms and test contents), if necessary, pay attention to match your own dataset form.

#### The pre-trained model can be obtained at Huggingface