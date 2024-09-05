import json
import re
import string
import sys

import pandas as pd
import datasets
import os
import logging
sys.path.append('/root/code_project/audio/asr_pre_dataset/exp/')
from exp_config import config_adapter

import torchaudio

from dataset.exp_config import ExpConfig

META_DATA_TRAIN_PATH = config_adapter.config.json_path
META_DATA_TEST_PATH = config_adapter.config.json_path
META_DATA_VAL_PATH = config_adapter.config.json_path


VALUE_LIST = ExpConfig.VALUE_LIST


def lowercase_and_remove_punctuation(text):
    """
    Convert all characters in the string to uppercase and remove all punctuation.

    :param text: The string to be processed.
    :return: A new string that is the uppercase version of the input string without punctuation.
    """
    text_upper = str(text).lower()
    translator = str.maketrans('', '', string.punctuation)
    text_no_punctuation = text_upper.translate(translator)

    return text_no_punctuation


def filter_extra_spaces(sentence):
    filtered_sentence = re.sub(r'\s+', ' ', sentence.strip())

    return filtered_sentence



_FEATURES = datasets.Features(
    {
        "audio": datasets.Audio(sampling_rate=16000),
        # "text_pre": datasets.Value("string"),
        "text_upper": datasets.Value("string"),
        "id": datasets.Value("string")
    },
)


def get_result(path):
    with open(path, "r") as f:
        result = f.read()
    return json.loads(result)


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
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "files": get_result(META_DATA_TRAIN_PATH),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "files": get_result(META_DATA_TEST_PATH)
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "files": get_result(META_DATA_VAL_PATH)
                },
            ),
        ]

    def _generate_examples(self, files):
        # metadata = pd.read_json(metadata_path, lines=True)
        index = 0
        for id_, item in enumerate(files):
            for key in  config_adapter.config.key_list:
                for value in files[item][key]:
                    # print("aa", files[item][key][value])
                    audio_path = files[item][key][value]['noisy_audio_add_flag']
                    text_upper = filter_extra_spaces(lowercase_and_remove_punctuation(replacer.replace(files[item]['transcript_flag'].lower())))
                    data_name = os.path.basename(audio_path)
                    # audio, sr = torchaudio.load(audio_path)
                    audio = {"path": audio_path}
                    # trans = {"audio": audio, "text_pre": text_pre, "text_upper": text_upper}
                    yield index, {"audio": audio, "text_upper": text_upper, "id": data_name}
                    index += 1
