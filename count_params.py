from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio

model1 = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base", low_cpu_mem_usage=True)
model2 = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", low_cpu_mem_usage=True)

model_size1 = sum(t.numel() for t in model1.parameters())
model_size2 = sum(t.numel() for t in model2.parameters())
print('model_size1: ', model_size1)
print('model_size2: ', model_size2)
# print('reduce: ', (model_size1 - model_size2) / model_size1)