user_name=zyk_test_best
purpose='测试集拆分'

export CUDA_VISIBLE_DEVICES=0
model_name=Qwen2.5-7B-Instruct
method=sft





predict_config_name_1=Qwen2.5-7B-Instruct_lora_predict_best_test_1.yaml
predict_config_1="configs/predict/${predict_config_name_1}"

predict_config_name_2=Qwen2.5-7B-Instruct_lora_predict_best_test_2.yaml
predict_config_2="configs/predict/${predict_config_name_2}"

predict_config_name_3=Qwen2.5-7B-Instruct_lora_predict_best_test_3.yaml
predict_config_3="configs/predict/${predict_config_name_3}"



current_time=$(date +"%m-%d-%H-%M-%S")
save_path="outputs/saves/Qwen2.5-7B-Instruct/sft/lsy_1.0e-5_0.15_01-03-04-08-01"


## 生成测试结果1
predict_save_path_1="outputs/saves/${model_name}/predict_sft/${user_name}_1.0e-5_0.15_${current_time}_test_1"
predict_log_file_1="./outputs/logs/${model_name}/predict_sft/${user_name}_1.0e-5_0.15_${current_time}_test_1.log"
mkdir -p "${predict_save_path_1}"
touch "$predict_log_file_1"

# 声明最终的 yaml 文件路径变量
predict_yaml_file_1="${predict_save_path_1}/${predict_config_name_1}"

# 复制“原始配置文件”到“新配置文件”里，并修改相应变量
cp "${predict_config_1}" "${predict_yaml_file_1}"

# 修改yaml文件里的 adapter权重文件的路径、输出路径
sed -i "s|^\(adapter_name_or_path: \).*$|\1${save_path}|" "${predict_yaml_file_1}"
sed -i "s|^\(output_dir: \).*$|\1${predict_save_path_1}|" "${predict_yaml_file_1}"

cat "${predict_yaml_file_1}" >>"$predict_log_file_1" # 把yaml文件写到log文件的开头
llamafactory-cli train "${predict_yaml_file_1}" &>>"$predict_log_file_1" # 第一个参数指定预测阶段的yaml文件，第二参数：将结果写进predict_log_file


## 生成测试结果2
predict_save_path_2="outputs/saves/${model_name}/predict_sft/${user_name}_1.0e-5_0.15_${current_time}_test_2"
predict_log_file_2="./outputs/logs/${model_name}/predict_sft/${user_name}_1.0e-5_0.15_${current_time}_test_2.log"
mkdir -p "${predict_save_path_2}"
touch "$predict_log_file_2"

# 声明最终的 yaml 文件路径变量
predict_yaml_file_2="${predict_save_path_2}/${predict_config_name_2}"

# 复制“原始配置文件”到“新配置文件”里，并修改相应变量
cp "${predict_config_2}" "${predict_yaml_file_2}"

# 修改yaml文件里的 adapter权重文件的路径、输出路径
sed -i "s|^\(adapter_name_or_path: \).*$|\1${save_path}|" "${predict_yaml_file_2}"
sed -i "s|^\(output_dir: \).*$|\1${predict_save_path_2}|" "${predict_yaml_file_2}"

cat "${predict_yaml_file_2}" >>"$predict_log_file_2" # 把yaml文件写到log文件的开头
llamafactory-cli train "${predict_yaml_file_2}" &>>"$predict_log_file_2" # 第一个参数指定预测阶段的yaml文件，第二参数：将结果写进predict_log_file


## 生成测试结果3
predict_save_path_3="outputs/saves/${model_name}/predict_sft/${user_name}_1.0e-5_0.15_${current_time}_test_3"
predict_log_file_3="./outputs/logs/${model_name}/predict_sft/${user_name}_1.0e-5_0.15_${current_time}_test_3.log"
mkdir -p "${predict_save_path_3}"
touch "$predict_log_file_3"

# 声明最终的 yaml 文件路径变量
predict_yaml_file_3="${predict_save_path_3}/${predict_config_name_3}"

# 复制“原始配置文件”到“新配置文件”里，并修改相应变量
cp "${predict_config_3}" "${predict_yaml_file_3}"

# 修改yaml文件里的 adapter权重文件的路径、输出路径
sed -i "s|^\(adapter_name_or_path: \).*$|\1${save_path}|" "${predict_yaml_file_3}"
sed -i "s|^\(output_dir: \).*$|\1${predict_save_path_3}|" "${predict_yaml_file_3}"

cat "${predict_yaml_file_3}" >>"$predict_log_file_3" # 把yaml文件写到log文件的开头
llamafactory-cli train "${predict_yaml_file_3}" &>>"$predict_log_file_3" # 第一个参数指定预测阶段的yaml文件，第二参数：将结果写进predict_log_file
