import os

from lhotse import CutSet

cut_set = CutSet.from_jsonl("./ASR/datasets/libriheavy-master/libriheavy_cuts_medium_add_path.jsonl.gz")
SAVE_PATH = "./ASR/datasets/libriheavy-master/libriheavy_cuts_medium_filter_audio.jsonl.gz"
before_path = "./ASR/datasets/libriheavy-master/"

# for cut in cut_set:
#     for path_index in range(len(cut.recording.sources)):
#         cut.recording.sources[path_index].source = os.path.join(before_path, cut.recording.sources[path_index].source)

# max_iter = 100
# for i in range(max_iter):
cut_set = cut_set.filter(lambda c: c.supervisions[0].duration < 20)
print(len(cut_set))
cut_set.to_file(SAVE_PATH)
# import torchaudio
#
# path = r"./ASR/datasets/libriheavy_test/clean8k/00000.wav"
# data, rate = torchaudio.load(path)
#
# print(rate)
#
# print(data.size(1) / 8000)