# EchoLLM

#### If you also want to implement millimeter-wave radar eavesdropping bone conduction headphones and LLM to use context-assisted inference, please supplement the intermediate files and follow the following process:
* You should have a millimeter-wave radar code to get the detection and positioning: the relevant parameters can be referred to the signal processing folder **Vibration_Signal**, you need to determine the location distance of the victim and save the bin file.
* Use the **bin2wav.mlx** file in **Vibration_Signal** to extract the headset vibration signal. Note: Be sure to change the distance. You can also edit the signal processing functions yourself.
* You'll now need to stitch the radar-processed audio with the audio above. If you want to focus more on what follows, it is recommended to insert a split speech signal such as "BOS"
* Once you have collected enough data, you can refer to the readme in the **LeASR** folder for ASR or fine-tuning tasks.
