# lyx
user_name=lyx
purpose='增量预训练后的推理测试'
export CUDA_VISIBLE_DEVICES=0
model_name=DeepSeek-V2-Lite
method=predict
current_time=$(date +"%m-%d-%H-%M-%S")
# 先在原output_dir后面创建一个文件夹，名字为当前时间
mkdir -p "outputs/saves/${model_name}/${method}/${current_time}"
# 创建日志文件并重定向输出
log_file="./outputs/logs/${model_name}/${method}/${current_time}.log"
touch "$log_file"

# 更改'configs/predict/gemma-2-9b-it_predict.yaml'文件中的output_dir
# 在最后面加上当前时间
sed -i "s|^\(output_dir: \).*$|\1outputs/saves/${model_name}/${method}/${current_time}|" 'configs/predict/DeepSeek-V2-Lite_predict.yaml'
# 首先在文件中写入名字和目的
echo "用户：$user_name" >> "$log_file"
echo "目的：$purpose" >> "$log_file"
# 将配置文件的内容输出到日志文件中
cat 'configs/predict/DeepSeek-V2-Lite_predict.yaml' >> "$log_file"
llamafactory-cli train 'configs/predict/DeepSeek-V2-Lite_predict.yaml' &>> "$log_file"
