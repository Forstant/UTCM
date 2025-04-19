user_name=cht
purpose='推理'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DS_SKIP_CUDA_CHECK=1

saves_path=("cht_train0_5.0e-5_0.3_01-06-07-11-14")

# for path in ${saves_path[@]}; do
#   sed -i "s|^\(adapter_name_or_path: \).*$|\1outputs/saves/Qwen2.5-72B-Instruct/sft/${path}|" "configs/inference/qwen2.5-72B.yaml"
#   nohup llamafactory-cli api "configs/inference/qwen2.5-72B.yaml" >api.txt &
#   TASK_PID=$(ps -ef | grep qwen2.5-72B.yaml | grep -v grep | awk '{print $2}')
#   python test_api.py --path "$path" --max_size 100
#   kill "$TASK_PID"
# done

python ../LLaMA-Factory/scripts/vllm_infer.py \
  --model_name_or_path Qwen/Qwen2.5-72B-Instruct \
  --adapter_name_or_path outputs/saves/Qwen2.5-72B-Instruct/sft/cht_train0_5.0e-5_0.3_01-06-07-11-14 \
  --template qwen \
  --dataset_dir /GLOBALFS/sysu_wangp_1/AITCM/AITCM/datasets/sft \
  --dataset sft_test_data \
  --max_new_tokens 256 \
  --max_samples 100 \
  --save_name outputs/saves/Qwen2.5-72B-Instruct/predict_sft/cht_train0_5.0e-5_0.3_01-06-07-11-14/generated_predictions3.jsonl
