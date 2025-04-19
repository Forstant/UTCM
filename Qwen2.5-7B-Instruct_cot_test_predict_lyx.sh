user_name=lyx_0325_cot_test_predict
purpose='经过mixed的数据在test_cot_data_annotated上进行预测'

export CUDA_VISIBLE_DEVICES=0
model_name=Qwen2.5-7B-Instruct

current_time=$(date +"%m-%d-%H-%M-%S")

predict_config_name_1=Qwen2.5-7B-Instruct_cot_test_predict.yaml
predict_config_1="configs/predict/${predict_config_name_1}"

## 生成测试结果1
predict_save_path_1="outputs/saves/${model_name}/predict_sft/${user_name}_${current_time}"
predict_log_file_1="./outputs/logs/${model_name}/predict_sft/${user_name}_${current_time}.log"
mkdir -p "${predict_save_path_1}"
touch "$predict_log_file_1"

echo "用户：$user_name" >>"$predict_log_file_1"
echo "目的：$purpose" >>"$predict_log_file_1"

# 声明最终的 yaml 文件路径变量
predict_yaml_file_1="${predict_save_path_1}/${predict_config_name_1}"

# 复制“原始配置文件”到“新配置文件”里，并修改相应变量
cp "${predict_config_1}" "${predict_yaml_file_1}"

# 修改yaml文件里的输出路径
sed -i "s|^\(output_dir: \).*$|\1${predict_save_path_1}|" "${predict_yaml_file_1}"

cat "${predict_yaml_file_1}" >>"$predict_log_file_1"                     # 把yaml文件写到log文件的开头
llamafactory-cli train "${predict_yaml_file_1}" &>>"$predict_log_file_1" # 第一个参数指定预测阶段的yaml文件，第二参数：将结果写进predict_log_file
