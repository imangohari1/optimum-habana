# pip install accelerate

import argparse
import os
import time

import torch
from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser(
        description="A script demonstrating argparse usage."
    )

    # Add model
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="name or local path the gemma3 model to use (e.g., google/gemma-3-4b-it).",
    )
    # Add --device argument
    parser.add_argument(
        "--device",
        type=str,
        default="hpu",
        help='Specify the device to use (e.g., "hpu", "gpu").',
    )

    # Add --iteration argument
    parser.add_argument(
        "--iteration",
        type=int,
        default=5,
        help="Specify the number of iterations to run.",
    )

    # Add --use-graphs argument (boolean flag)
    parser.add_argument(
        "--use-hpu-graphs",
        action="store_true",
        help="Enable or disable the use of HPU graphs (hpu specific).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    # Add --iteration argument
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=100,
        help="number of max new tokens to generate.",
    )
    # Add --num_images argument
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="number of images in prompt.",
    )

    args = parser.parse_args()

    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")

    ## load the proper conditional generation

    model_id = args.model
    # model_id = "google/gemma-3-4b-it"
    # model_id = "models/gemma-3-4b-it"
    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16

    kwargs = {}
    if args.device == "hpu":
        try:
            from transformers.models.gemma3.modeling_gemma3 import (
                Gemma3ForConditionalGeneration,
            )

            from optimum.habana.transformers.modeling_utils import (
                adapt_transformers_to_gaudi,
            )
            from optimum.habana.transformers.models.gemma3.modeling_gemma3 import (
                GaudiGemma3ForConditionalGeneration,
            )
        except Exception as error:
            print(f"failed on loading OH Gemma3: {error}")
            exit

        torch_device = "hpu"
        adapt_transformers_to_gaudi()
        model = GaudiGemma3ForConditionalGeneration.from_pretrained(
            # model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=torch_device,
        ).eval()

        ## generation keys
        kwargs["lazy_mode"] = False
        if os.getenv("PT_HPU_LAZY_MODE") == "1":
            kwargs["lazy_mode"] = True
        if kwargs["lazy_mode"]:
            kwargs["hpu_graphs"] = args.use_hpu_graphs
        # kwargs = {}
    else:
        try:
            from transformers.models.gemma3.modeling_gemma3 import (
                Gemma3ForConditionalGeneration,
            )
        except Exception as error:
            print(f"failed on loading HF Gemma3: {error}")
            exit
        torch_device = "cuda"
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=torch_device
        ).eval()

    processor = AutoProcessor.from_pretrained(model_id)

    content_1x_img = [
        {
            "type": "image",
            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
        },
        {"type": "text", "text": "Describe these image in detail."},
    ]
    content_2x_img = [
        {
            "type": "image",
            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
        },
        {"type": "image", "image": "https://llava-vl.github.io/static/images/view.jpg"},
        {"type": "text", "text": "Describe these image in detail."},
    ]
    content_3x_img = [
        {
            "type": "image",
            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
        },
        {"type": "image", "image": "https://llava-vl.github.io/static/images/view.jpg"},
        {
            "type": "image",
            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/mosque.jpg",
        },
        {"type": "text", "text": "Describe these image in detail."},
    ]
    content_4x_img = [
        {
            "type": "image",
            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
        },
        {"type": "image", "image": "https://llava-vl.github.io/static/images/view.jpg"},
        {
            "type": "image",
            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/mosque.jpg",
        },
        {
            "type": "image",
            "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png",
        },
        {"type": "text", "text": "Describe these image in detail."},
    ]

    if args.num_images == 1:
        content = content_1x_img
    elif args.num_images == 2:
        content = content_2x_img
    elif args.num_images == 3:
        content = content_3x_img
    elif args.num_images == 4:
        content = content_4x_img
    else:
        print(f"WAR: unsupported num_images {args.num_images}. running with 1x")
        content = content_1x_img

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": content,
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

    for iter in range(0, args.iteration):
        start_time = time.perf_counter()
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=args.max_new_token,
                do_sample=False,
                **kwargs,
            )
            generation = generation[0][input_len:]
        elapsed_time = time.perf_counter() - start_time

        print(f"iteration {iter}: Generation Elapsed Time: {elapsed_time} seconds")

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)

    # **Overall Impression:** The image is a close-up shot of a vibrant garden scene,
    # focusing on a cluster of pink cosmos flowers and a busy bumblebee.
    # It has a slightly soft, natural feel, likely captured in daylight.


if __name__ == "__main__":
    main()
