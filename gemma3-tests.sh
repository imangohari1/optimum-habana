## pytests
PT_HPU_LAZY_MODE=1 RUN_SLOW=true python -m pytest --device gaudi2 tests/transformers/tests/models/gemma2 -s -v

## clm
PT_HPU_LAZY_MODE=1 python examples/language-modeling/run_clm.py --model_name_or_path google/gemma-3-4b-it --gaudi_config_name Habana/gpt2 --dataset_name wikitext --do_train --output_dir /tmp/tmp3vtnetji --overwrite_output_dir --learning_rate 0.0002 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_train_epochs 2 --use_habana --throughput_warmup_steps 3 --save_strategy no --use_lazy_mode --do_eval --dataset_config_name wikitext-2-raw-v1 --use_hpu_graphs_for_inference

## 4k prompt
bash swa-4k-prompt-test.sh

## text gen
PT_HPU_LAZY_MODE=1 python examples/text-generation/run_generation.py --model_name_or_path google/gemma-3-4b-it --use_hpu_graphs --use_kv_cache --max_new_tokens 100 --do_sample --prompt "Here is my prompt" --sdp_on_bf16

PT_HPU_LAZY_MODE=0 python examples/text-generation/run_generation.py --model_name_or_path google/gemma-3-4b-it  --use_kv_cache --max_new_tokens 100 --do_sample --prompt "Here is my prompt" --sdp_on_bf16

PT_HPU_LAZY_MODE=1 python examples/text-generation/run_generation.py --model_name_or_path google/gemma-2-2b --use_hpu_graphs --use_kv_cache --max_new_tokens 100 --do_sample --prompt "Here is my prompt" --sdp_on_bf16

## ci 

PT_HPU_LAZY_MODE=1  RUN_SLOW=true python -m pytest tests/test_text_generation_example.py::test_text_generation_bf16_1x[google/gemma-3-4b-it-1-False-True] -s -v --token $HFToken
PT_HPU_LAZY_MODE=1  RUN_SLOW=true python -m pytest tests/test_text_generation_example.py::test_text_generation_bf16_1x[google/gemma-2-9b-1-False-True] -s -v --token $HFToken
PT_HPU_LAZY_MODE=1  RUN_SLOW=true python -m pytest tests/test_text_generation_example.py::test_text_generation_bf16_1x -s -v --token $HFToken

## accuracy 

pip install -r examples/text-generation/requirements_lm_eval.txt
PT_HPU_LAZY_MODE=1 python examples/text-generation/run_lm_eval.py --model_name_or_path google/gemma-2-2b  --use_hpu_graphs --use_kv_cache --bf16 --batch_size=1  --max_new_tokens 8192 --tasks piqa -o tmp.json #gemma-2-9b-eval-max_new_token_8192_after_sliding_window.json
