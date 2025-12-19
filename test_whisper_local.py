import argparse
import functools
import time

import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from data_utils import to_simple

#from utils.utils import print_arguments, add_arguments

def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in vars(args).items():
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")

def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def str_none(val):
    if val == 'None':
        return None
    else:
        return val

def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = strtobool if type == bool else type
    type = str_none if type == str else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' Default: %(default)s.',
                           **kwargs)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("audio_path", type=str, default="./zyh.wav",              help="Path to the audio file for prediction")
# add_arg("model_path", type=str, default="./distil-whisper-small-zh/model",  help="Path to the merged model or model name on HuggingFace")
add_arg("model_path", type=str, default="./distil-whisper-small-zh/model",  help="Path to the merged model or model name on HuggingFace")
add_arg("language",   type=str, default="Chinese",                       help="Set language, can be full name or abbreviation. If None, the model will predict multilingual")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="Model task")
add_arg("device",     type=str, default="cuda",                          help="Set device")
add_arg("local_files_only", type=bool, default=True,  help="Whether to only load model locally without attempting to download")
args = parser.parse_args()
print_arguments(args)
# Get Whisper feature extractor, encoder and decoder
processor = WhisperProcessor.from_pretrained(args.model_path,
                                             language=args.language,
                                             task=args.task,
                                             device_map="auto",
                                             local_files_only=args.local_files_only)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)

# Load model
model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                        device_map="auto",
                                                        local_files_only=args.local_files_only)
if args.device=="cuda":
  model = model.half()
model.eval()

# Load audio
sample, sr = librosa.load(args.audio_path, sr=16000)
duration = sample.shape[-1]/sr
assert duration < 30, f"This program only supports audio inference for audio less than 30 seconds. Current audio is {duration} seconds. Please use another inference program!"
# Preprocess audio
input_features = processor(sample, sampling_rate=sr, return_tensors="pt", do_normalize=True).input_features
input_features = input_features.to(args.device)
if args.device == "cuda":
  input_features = input_features.half()
# Start recognition
begin_time = time.time()
# forced_decoder_ids = torch.
# print(input_features.device)
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=256)
# Decode results
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
end_time = time.time()
print(f"Recognition result: {transcription}")
print(f"Recognition result (Simplified): {to_simple(transcription)}")
print(f'Time elapsed: {end_time-begin_time} s. ')
