from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer, Wav2Vec2Processor

# checkpoints to leverage
from model.CustomSeqModel import CustomSpeechEncoderDecoderModel

encoder_id = "./dataset_text_audio/pretrained_models/models/hubert-base-ls960/"
decoder_id = "./dataset_text_audio/pretrained_models/models/bart-large/"

# load and save speech-encoder-decoder model
# set some hyper-parameters for training and evaluation
model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_id, decoder_id,
                                                                  encoder_feat_proj_dropout=0.0,
                                                                  encoder_layerdrop=0.0, max_length=200)
model.config.decoder_start_token_id = model.decoder.config.bos_token_id
model.config.eos_token_id = model.decoder.config.eos_token_id
model.config.pad_token_id = model.decoder.config.pad_token_id
model.save_pretrained("./dataset_text_audio/pretrained_models/models/mode_hu-bart-large/")

# load and save processor
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
tokenizer = AutoTokenizer.from_pretrained(decoder_id)
processor = Wav2Vec2Processor(feature_extractor, tokenizer)
processor.save_pretrained("./dataset_text_audio/pretrained_models/models/mode_hu-bart-large/")