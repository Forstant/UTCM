user_name=zyk_0402_gam_test_fm_pt
purpose='使用基座模型、增量后模型对广安门数据进行测试'

export CUDA_VISIBLE_DEVICES=0
model_name=Qwen2.5-7B-Instruct

current_time=$(date +"%m-%d-%H-%M-%S")


predict_config_name_1=gam_fm_predict.yaml
predict_config_1="configs/predict/${predict_config_name_1}"

predict_config_name_2=gam_pt_predict.yaml
predict_config_2="configs/predict/${predict_config_name_2}"




## 生成测试结果1
predict_save_path_1="outputs/saves/${model_name}/predict_sft/${user_name}_1.0e-5_0.15_${current_time}_test_gam_fm"
predict_log_file_1="./outputs/logs/${model_name}/predict_sft/${user_name}_1.0e-5_0.15_${current_time}_test_gam_fm.log"
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

cat "${predict_yaml_file_1}" >>"$predict_log_file_1" # 把yaml文件写到log文件的开头
llamafactory-cli train "${predict_yaml_file_1}" &>>"$predict_log_file_1" # 第一个参数指定预测阶段的yaml文件，第二参数：将结果写进predict_log_file


## 生成测试结果2
predict_save_path_2="outputs/saves/${model_name}/predict_sft/${user_name}_1.0e-5_0.15_${current_time}_test_gam_pt"
predict_log_file_2="./outputs/logs/${model_name}/predict_sft/${user_name}_1.0e-5_0.15_${current_time}_test_gam_pt.log"
mkdir -p "${predict_save_path_2}"
touch "$predict_log_file_2"

echo "用户：$user_name" >>"$predict_log_file_2"
echo "目的：$purpose" >>"$predict_log_file_2"

# 声明最终的 yaml 文件路径变量
predict_yaml_file_2="${predict_save_path_2}/${predict_config_name_2}"

# 复制“原始配置文件”到“新配置文件”里，并修改相应变量
cp "${predict_config_2}" "${predict_yaml_file_2}"

# 修改yaml文件里的输出路径
sed -i "s|^\(output_dir: \).*$|\1${predict_save_path_2}|" "${predict_yaml_file_2}"

cat "${predict_yaml_file_2}" >>"$predict_log_file_2" # 把yaml文件写到log文件的开头
llamafactory-cli train "${predict_yaml_file_2}" &>>"$predict_log_file_2" # 第一个参数指定预测阶段的yaml文件，第二参数：将结果写进predict_log_file

