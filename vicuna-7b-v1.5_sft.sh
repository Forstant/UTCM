# zyk
user_name=zyk
purpose='指令微调'

export CUDA_VISIBLE_DEVICES=0
model_name=vicuna-7b-v1.5

method=sft
current_time=$(date +"%m-%d-%H-%M-%S")

# lr_list=(5.0e-5 1.0e-5 5.0e-6)
lr_list=('1.0e-4 1.0e-5 1.0e-6 5.0e-4')
# warmup_ratio_list=(0.3 0.25 0.2 0.15 0.1 0.05 0.0)
warmup_ratio_list=('0.05 0.15 0.2 0.25 0.30 0.35')


# 循环遍历lr和warmup_ratio
for lr in ${lr_list[@]}; do
    for warmup_ratio in ${warmup_ratio_list[@]}; do
        # 将current_time再次赋值为当前时间
        current_time=$(date +"%m-%d-%H-%M-%S")
        # 创建日志文件并重定向输出
        log_file="./outputs/logs/${model_name}/${method}/${current_time}.log"
        touch "$log_file"

        # 更改'configs/sft/Meta-Llama-3-8B-Instruct_sft.yaml'文件中的output_dir
        # 先在原output_dir后面创建一个文件夹，名字为当前时间
        mkdir -p "outputs/saves/${model_name}/${method}/${current_time}"
        # 在最后面加上当前时间
        sed -i "s|^\(output_dir: \).*$|\1outputs/saves/${model_name}/${method}/${current_time}|" 'configs/sft/vicuna-7b-v1.5_sft.yaml'
        # 将lr和warmup_ratio改为当前的值
        sed -i "s|^\(learning_rate: \).*$|\1$lr|" 'configs/sft/vicuna-7b-v1.5_sft.yaml'
        sed -i "s|^\(warmup_ratio: \).*$|\1$warmup_ratio|" 'configs/sft/vicuna-7b-v1.5_sft.yaml'
        # 首先在文件中写入名字和目的
        echo "用户：$user_name" > "$log_file"
        echo "目的：$purpose" >> "$log_file"
        # 将配置文件的内容输出到日志文件中
        cat 'configs/sft/vicuna-7b-v1.5_sft.yaml' >> "$log_file"
        llamafactory-cli train 'configs/sft/vicuna-7b-v1.5_sft.yaml' &>> "$log_file"

        # 生成测试结果
        mkdir -p "outputs/saves/${model_name}/predict/${current_time}"
        predict_log_file="./outputs/logs/${model_name}/predict/${current_time}.log"
        touch "$predict_log_file"

        sed -i "s|^\(adapter_name_or_path: \).*$|\1outputs/saves/${model_name}/${method}/${current_time}|" 'configs/predict/vicuna-7b-v1.5_lora_predict.yaml'
        
        sed -i "s|^\(output_dir: \).*$|\1outputs/saves/${model_name}/predict/${current_time}|" 'configs/predict/vicuna-7b-v1.5_lora_predict.yaml'
        cat 'configs/predict/vicuna-7b-v1.5_lora_predict.yaml' >> "$predict_log_file"
        llamafactory-cli train 'configs/predict/vicuna-7b-v1.5_lora_predict.yaml' &>> "$predict_log_file"
    done
done
