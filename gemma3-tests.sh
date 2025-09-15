## multimodal
logfile=gemma3-multimodal-lazy-eager-with-swa-pixvalue-rmsnorm-rotary-fixes.$(date -u +%Y%m%d%H%M).$(hostname).log;
ScriptPath=./simple-gemma3-inference-hf.py
for i in $(echo "google/gemma-3-4b-it google/gemma-3-12b-it google/gemma-3-27b-it"); do
	# i.e. PT_HPU_LAZY_MODE=1 python simple-gemma3-inference-hf.py --use-hpu-graphs --bf16
	echo $i
	cmd="PT_HPU_LAZY_MODE=1 python $ScriptPath --use-hpu-graphs --bf16 --model $i"
	echo $cmd >>$logfile && eval $cmd 2>&1 | tee -a $logfile
	cmd="PT_HPU_LAZY_MODE=0 python $ScriptPath --bf16 --model $i"
	echo $cmd >>$logfile && eval $cmd 2>&1 | tee -a $logfile
done
cmd="PT_HPU_LAZY_MODE=1 python $ScriptPath --use-hpu-graphs --bf16 --model google/gemma-3-4b-it --max-new-token 1024 --num-images 4"
echo $cmd >>$logfile && eval $cmd 2>&1 | tee -a $logfile

## text gen
logfile=gemma3-4b-text-lazy-eager-with-swa-pixvalue-rmsnorm-rotary-fixes.$(date -u +%Y%m%d%H%M).$(hostname).log;
PT_HPU_LAZY_MODE=1 python examples/text-generation/run_generation.py --model_name_or_path google/gemma-3-4b-it --use_hpu_graphs --use_kv_cache --max_new_tokens 100 --do_sample --prompt "DeepSpeed is a machine learning framework" --sdp_on_bf16 2>&1 | tee -a $logfile
PT_HPU_LAZY_MODE=1 python examples/text-generation/run_generation.py --model_name_or_path google/gemma-3-4b-it --use_hpu_graphs --max_new_tokens 100 --do_sample --prompt "DeepSpeed is a machine learning framework" --sdp_on_bf16 2>&1 | tee -a $logfile
PT_HPU_LAZY_MODE=0 python examples/text-generation/run_generation.py --model_name_or_path google/gemma-3-4b-it  --use_kv_cache --max_new_tokens 100 --do_sample --prompt "DeepSpeed is a machine learning framework" --sdp_on_bf16 2>&1 | tee -a $logfile
PT_HPU_LAZY_MODE=0 python examples/text-generation/run_generation.py --model_name_or_path google/gemma-3-4b-it  --max_new_tokens 100 --do_sample --prompt "DeepSpeed is a machine learning framework" --sdp_on_bf16 2>&1 | tee -a $logfile
PT_HPU_LAZY_MODE=1 python examples/text-generation/run_generation.py --model_name_or_path google/gemma-3-4b-it --use_hpu_graphs --use_kv_cache --max_new_tokens 512 --do_sample --prompt "DeepSpeed is a machine learning framework" --sdp_on_bf16 2>&1 | tee -a $logfile

## ci
logfile=gemma3-ci-lazy-with-swa-pixvalue-rmsnorm-rotary-fixes.$(date -u +%Y%m%d%H%M).$(hostname).log;
PT_HPU_LAZY_MODE=1  RUN_SLOW=true python -m pytest tests/test_text_generation_example.py::test_text_generation_bf16_1x -s -v --token $HFToken 2>&1 | tee -a $logfile

## accuracy 
pip install -r examples/text-generation/requirements_lm_eval.txt
logdir=eval-with-swa-pixvalue-rmsnorm-rotary-fixes.$(date -u +%Y%m%d%H%M).$(hostname)
mkdir $logdir -p
PT_HPU_LAZY_MODE=1 python examples/text-generation/run_lm_eval.py --model_name_or_path google/gemma-3-4b-it  --use_hpu_graphs --use_kv_cache --bf16 --batch_size=1  --max_new_tokens 8192 --tasks piqa -o ${logdir}/gemma-3-4b-it-max_new_token_8192.json
PT_HPU_LAZY_MODE=1 python examples/text-generation/run_lm_eval.py --model_name_or_path google/gemma-3-4b-it  --use_hpu_graphs --use_kv_cache --bf16 --batch_size=1  --max_new_tokens 128 --tasks piqa -o ${logdir}/gemma-3-4b-it-max_new_token_128.json
PT_HPU_LAZY_MODE=1 python examples/text-generation/run_lm_eval.py --model_name_or_path google/gemma-3-27b-it  --use_hpu_graphs --use_kv_cache --bf16 --batch_size=1  --max_new_tokens 8192 --tasks piqa -o ${logdir}/gemma-3-27b-it-max_new_token_8192.json
PT_HPU_LAZY_MODE=1 python examples/text-generation/run_lm_eval.py --model_name_or_path google/gemma-3-27b-it  --use_hpu_graphs --use_kv_cache --bf16 --batch_size=1  --max_new_tokens 128 --tasks piqa -o  ${logdir}/gemma-3-27b-it-max_new_token_128.json

## clm
logfile=gemma3-4b-clm-lazy-with-swa-pixvalue-rmsnorm-rotary-fixes.$(date -u +%Y%m%d%H%M).$(hostname).log;
PT_HPU_LAZY_MODE=1 python examples/language-modeling/run_clm.py --model_name_or_path google/gemma-3-4b-it --gaudi_config_name Habana/gpt2 --dataset_name wikitext --do_train --output_dir /tmp/tmp3vtnetji --overwrite_output_dir --learning_rate 0.0002 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_train_epochs 2 --use_habana --throughput_warmup_steps 3 --save_strategy no --use_lazy_mode --do_eval --dataset_config_name wikitext-2-raw-v1 --use_hpu_graphs_for_inference 2>&1 | tee -a $logfile

exit 

## 4k prompt
logfile=gemma3-4b-4kprompt-g2-20250912-with-swa-pixelfix-rsmnorm.log;
bash swa-4k-prompt-test.sh 2>&1 | tee -a $logfile
