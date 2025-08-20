# pip install accelerate

import time

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi


torch_device = "hpu"
adapt_transformers_to_gaudi()

model_id = "google/gemma-3-4b-it"
# model_id = "models/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map=torch_device).eval()

processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

start_time = time.perf_counter()
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
elapsed_time = time.perf_counter() - start_time

print(f"Generation Elapsed Time: {elapsed_time} seconds")


decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

# **Overall Impression:** The image is a close-up shot of a vibrant garden scene,
# focusing on a cluster of pink cosmos flowers and a busy bumblebee.
# It has a slightly soft, natural feel, likely captured in daylight.
