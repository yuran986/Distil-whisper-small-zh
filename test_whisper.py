from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import data_utils

# 这里的两个路径改成模型路径
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", low_cpu_mem_usage=True)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

model.to("cuda")

common_voice = load_dataset("mozilla-foundation/common_voice_16_1", "zh-CN", split="validation", streaming=True)
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

i = 0
for sample in iter(common_voice):
    # 此处设置了模型只识别前21个音频
    if i == 21: break
    inputs = processor(sample["audio"]["array"], sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    generated_ids = model.generate(input_features.to("cuda"), max_new_tokens=128)
    pred_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    pred_text = data_utils.to_simple(pred_text)
    print("Pred text:", pred_text)
    i += 1
