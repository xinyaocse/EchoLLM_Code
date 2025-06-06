import json
import re
import string

from jiwer import wer

from audio.asr_pre_dataset.exp.exp_config import config_adapter

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


def pre_process_text(text: str):
    return filter_extra_spaces(lowercase_and_remove_punctuation(text))


def write_result(path, result_):
    with open(path, "w") as f:
        f.write(json.dumps(result_))


def get_result(path):
    with open(path, "r") as f:
        result = f.read()
    return json.loads(result)


def cal_asr(deg_word, ref_word_):
    error_rate = wer(ref_word_, deg_word)
    print(error_rate)
    return error_rate


# def get_wer(reference, hypothesis):
#     ref_words = nltk.word_tokenize(reference.lower())
#     hyp_words = nltk.word_tokenize(hypothesis.lower())
#     edit_distance = nltk.edit_distance(ref_words, hyp_words)
#     wer = edit_distance / len(ref_words),
#     return wer

def uppercase_and_remove_punctuation(text):
    """
    Convert all characters in the string to uppercase and remove all punctuation.

    :param text: The string to be processed.
    :return: A new string that is the uppercase version of the input string without punctuation.
    """
    text_upper = text.upper()

    translator = str.maketrans('', '', string.punctuation)
    
    text_no_punctuation = text_upper.translate(translator)

    return text_no_punctuation

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



result_list = get_result(config_adapter.config.json_path)
ab_a_wer_total = 0
b_wer_total = 0
model_wer_total = 0
count = 0
asr_key_list = ["asr_result", "asr_result_large"]
for name in result_list:
    for key in config_adapter.config.key_list:
        for value in result_list[name][key]:
            for asr_key in asr_key_list:
                if asr_key in result_list[name][key][value]:
                    asr_result = result_list[name][key][value][asr_key]
                    if len(str(asr_result).split("bos")) == 1:
                        asr_result_b = str(asr_result)
                        print(asr_result)
                    else:
                        asr_result_b = str(asr_result).split("bos")[1].strip()
                    asr_result_label = filter_extra_spaces(
                        lowercase_and_remove_punctuation(replacer.replace(result_list[name]['transcript_flag'].lower())))
                    asr_result_b_label = filter_extra_spaces(
                        lowercase_and_remove_punctuation(replacer.replace(result_list[name]['current_text'].lower())))
                    result_list[name][key][value][asr_key + "_wer"] = cal_asr(asr_result, asr_result_label)
                    result_list[name][key][value][asr_key + "_b_wer"] = cal_asr(asr_result_b, asr_result_b_label)

write_result(config_adapter.config.json_path, result_list)
