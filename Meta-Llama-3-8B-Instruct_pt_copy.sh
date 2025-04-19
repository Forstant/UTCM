# pzb
user_name=pzb
purpose='增量预训练'
export CUDA_VISIBLE_DEVICES=0
model_name=Meta-Llama-3-8B-Instruct
method=full_pt
current_time=$(date +"%m-%d-%H-%M-%S")

lr_list=(1.0e-6)
# lr_list=(1.0e-5 1.0e-6 1.0e-7)
warmup_ratio_list=(0.15)
# warmup_ratio_list=(0.2 0.15 0.1)
# 循环遍历lr和warmup_ratio
for lr in ${lr_list[@]}; do
    for warmup_ratio in ${warmup_ratio_list[@]}; do
        # 将current_time再次赋值为当前时间
        current_time=$(date +"%m-%d-%H-%M-%S")
        # 创建日志文件并重定向输出
        log_file="./outputs/logs/${model_name}/${method}/${current_time}.log"
        touch "$log_file"
        # # 输出当前路径
        # echo "当前路径：$(pwd)"

        # 更改'configs/pt/Meta-Llama-3-8B-Instruct_pt.yaml'文件中的output_dir
        # 先在原output_dir后面创建一个文件夹，名字为当前时间
        mkdir -p "outputs/saves/${model_name}/${method}/${current_time}"
        # 在最后面加上当前时间
        sed -i "s|^\(output_dir: \).*$|\1outputs/saves/${model_name}/${method}/${current_time}|" 'configs/pt/Meta-Llama-3-8B-Instruct_pt.yaml'
        # 将lr和warmup_ratio改为当前的值
        sed -i "s|^\(learning_rate: \).*$|\1$lr|" 'configs/pt/Meta-Llama-3-8B-Instruct_pt.yaml'
        sed -i "s|^\(warmup_ratio: \).*$|\1$warmup_ratio|" 'configs/pt/Meta-Llama-3-8B-Instruct_pt.yaml'
        # 首先在文件中写入名字和目的
        echo "用户：$user_name" >> "$log_file"
        echo "目的：$purpose" >> "$log_file"
        # 将配置文件的内容输出到日志文件中
        cat 'configs/pt/Meta-Llama-3-8B-Instruct_pt.yaml' >> "$log_file"
        llamafactory-cli train 'configs/pt/Meta-Llama-3-8B-Instruct_pt.yaml' 2>&1 | tee "$log_file"
	# llamafactory-cli train 'configs/pt/Meta-Llama-3-8B-Instruct_pt.yaml'
    done
done
