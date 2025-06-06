import json
import re
import string
import pandas as pd
import datasets
import os
import logging
import torchaudio

# Dataset path settings
META_DATA_TRAIN_PATH = r'/root/public/dev8T/username/dataset_text_audio/ASR_train_FR.json'
META_DATA_TEST_PATH = r'/root/public/dev8T/username/dataset_text_audio/ASR_test_FR.json'
META_DATA_VAL_PATH = r'/root/public/dev8T/username/dataset_text_audio/ASR_test_FR.json'

def lowercase_and_remove_punctuation(text):
    """
    Convert all characters in the string to lowercase and remove all punctuation.
    
    :param text: Input string to be processed
    :return: Processed string in lowercase without punctuation
    """
    # Convert to lowercase
    text_lower = str(text).lower()
    
    # Create translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    
    # Remove punctuation
    text_no_punctuation = text_lower.translate(translator)
    
    return text_no_punctuation

def filter_extra_spaces(sentence):
    # Use regex to remove leading/trailing spaces and extra spaces between words
    filtered_sentence = re.sub(r'\s+', ' ', sentence.strip())
    return filtered_sentence

# Define dataset features and their types
_FEATURES = datasets.Features(
    {
        "audio_pre": datasets.Audio(sampling_rate=16000),
        "audio_after": datasets.Audio(sampling_rate=16000),
        "text_upper": datasets.Value("string"),
        "id": datasets.Value("string")
    },
)

def get_result(path):
    with open(path, "r") as f:
        result = f.read()
    return json.loads(result)

# Contraction replacement patterns
replacement_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'i\'m', 'i am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would')]

class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s

replacer = RegexpReplacer()

class LibriNoised8k(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default", version=datasets.Version("0.0.1"))]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description="None",
            features=_FEATURES,
            supervised_keys=None,
            homepage="None",
            license="None",
            citation="None",
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files": get_result(META_DATA_TRAIN_PATH),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "files": get_result(META_DATA_TEST_PATH)
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "files": get_result(META_DATA_TEST_PATH)
                },
            ),
        ]

    def _generate_examples(self, files):
        for id_, item in enumerate(files):
            text_upper = filter_extra_spaces(lowercase_and_remove_punctuation(item['transcript']))
            
            try:
                audio_pre_path = item["audio_pre"]
                audio_after_path = item["audio_after"]
                data_name = os.path.basename(audio_after_path)
                audio_pre = {"path": audio_pre_path}
                audio_after = {"path": audio_after_path}
                yield id_, {"audio_pre": audio_pre, "audio_after": audio_after, "text_upper": text_upper, "id": data_name}
            except:
                continue