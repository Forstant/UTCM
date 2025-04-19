# zyk
user_name=zyk
purpose='增量预训练'
export CUDA_VISIBLE_DEVICES=0
model_name=Qwen2.5-7B-Instruct
method=full_pt
current_time=$(date +"%m-%d-%H-%M-%S")

# 创建日志文件并重定向输出
log_file="./outputs/logs/${model_name}/${method}/${current_time}.log"
touch "$log_file"


mkdir -p "outputs/saves/${model_name}/${method}/${current_time}"
# 在最后面加上当前时间
sed -i "s|^\(output_dir: \).*$|\1outputs/saves/${model_name}/${method}/${current_time}|" 'configs/pt/Qwen2.5-7B-Instruct_pt.yaml'
# 首先在文件中写入名字和目的
echo "用户：$user_name" > "$log_file"
echo "目的：$purpose" > "$log_file"
# 将配置文件的内容输出到日志文件中
cat 'configs/pt/Qwen2.5-7B-Instruct_pt.yaml' > "$log_file"
llamafactory-cli train 'configs/pt/Qwen2.5-7B-Instruct_pt.yaml' &>> "$log_file"
