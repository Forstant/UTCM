# lyx
user_name=lyx
purpose='增量预训练'
export CUDA_VISIBLE_DEVICES=0
model_name=Mistral-7B-Instruct-v0.3
method=full_pt
current_time=$(date +"%m-%d-%H-%M-%S")
config=/GLOBALFS/sysu_wangp_1/AITCM/AITCM/configs/pt/${model_name}_pt.yaml

# lr_list=(5.0e-5 1.0e-5 5.0e-6 1.0e-6 5.0e-7)
lr_list=(5.0e-6)
# warmup_ratio_list=(0.3 0.25 0.2 0.15 0.1 0.05 0.0)
warmup_ratio_list=(0.15)
max_samples=100000.0
mkdir -p "/GLOBALFS/sysu_wangp_1/AITCM/AITCM/outputs/logs/${model_name}/${method}"
for lr in ${lr_list[@]}; do
    for warmup_ratio in ${warmup_ratio_list[@]}; do
        current_time=$(date +"%m-%d-%H-%M-%S")
        log_file="./outputs/logs/${model_name}/${method}/${current_time}.log"
        touch "$log_file"
        mkdir -p "outputs/saves/${model_name}/${method}/${current_time}"
        sed -i "s|^\(output_dir: \).*$|\1outputs/saves/${model_name}/${method}/${current_time}|" ${config}
        sed -i "s|^\(learning_rate: \).*$|\1$lr|" ${config}
        sed -i "s|^\(warmup_ratio: \).*$|\1$warmup_ratio|" ${config}
        sed -i "s|^\(max_samples: \).*$|\1$max_samples|" ${config}  # 更改sample数
        echo "用户：$user_name" >> "$log_file"
        echo "目的：$purpose" >> "$log_file"
        cat ${config} >> "$log_file"
        llamafactory-cli train ${config} 2>&1 | tee -a "$log_file"
	# llamafactory-cli train ${config}
    done
done
