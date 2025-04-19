user_name=lsy_train_ts_task_0_doc_0
purpose='ts_train_task_0_doc_0的数据的sft'
export CUDA_VISIBLE_DEVICES=0
model_name=Qwen2.5-7B-Instruct
method=sft
model_config_name=Qwen2.5-7B-Instruct_ts_train_task_0_doc_0.yaml
model_config="configs/sft/${model_config_name}"
# 预测配置
predict_config_name_oriset=Qwen2.5-7B-Instruct_lora_predict_ts_train_task_0_doc_0_sft_test.yaml
predict_config_name_cotset=Qwen2.5-7B-Instruct_lora_predict_ts_train_task_0_doc_0_test_cot.yaml
predict_config_oriset="configs/predict/${predict_config_name_oriset}"
predict_config_cotset="configs/predict/${predict_config_name_cotset}"
# lr_list=(5.0e-5 1.0e-5 5.0e-6 1.0e-6 5.0e-7)
lr_list=(5.0e-5)
# warmup_ratio_list=(0.3 0.25 0.2 0.15 0.1 0.05)
warmup_ratio_list=(0.15)
# 循环遍历lr和warmup_ratio
for lr in ${lr_list[@]}; do
    for warmup_ratio in ${warmup_ratio_list[@]}; do
        # 将current_time再次赋值为当前时间
        current_time=$(date +"%m-%d-%H-%M-%S")
        # 创建日志文件并重定向输出
        log_file="./outputs/logs/${model_name}/${method}/${user_name}_${lr}_${warmup_ratio}_${current_time}.log"
        touch "$log_file"
        # # 输出当前路径
        # echo "当前路径：$(pwd)"

        # 复制模型配置文件并更改相应的参数
        # 先在原output_dir后面创建一个文件夹，名字为当前时间
        save_path="outputs/saves/${model_name}/${method}/${user_name}_${lr}_${warmup_ratio}_${current_time}"
        mkdir -p "${save_path}"
        # 复制配置文件
        cp ${model_config} "${save_path}/${model_config_name}"
        # 在最后面加上当前时间
        sed -i "s|^\(output_dir: \).*$|\1${save_path}|" "${save_path}/${model_config_name}"
        # 将lr和warmup_ratio改为当前的值
        sed -i "s|^\(learning_rate: \).*$|\1$lr|" "${save_path}/${model_config_name}"
        sed -i "s|^\(warmup_ratio: \).*$|\1$warmup_ratio|" "${save_path}/${model_config_name}"
        # 首先在文件中写入名字和目的
        echo "用户：$user_name" >>"$log_file"
        echo "目的：$purpose" >>"$log_file"
        # 将配置文件的内容输出到日志文件中
        cat "${save_path}/${model_config_name}" >>"$log_file"
        llamafactory-cli train "${save_path}/${model_config_name}" &>>"$log_file"

        # 生成测试结果
        predict_save_path_oriset="outputs/saves/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_oriset"
        mkdir -p "${predict_save_path_oriset}"
        predict_log_file_oriset="./outputs/logs/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_oriset.log"
        touch "$predict_log_file_oriset"

        # 复制配置文件
        cp ${predict_config_oriset} "${predict_save_path_oriset}/${predict_config_name_oriset}"

        sed -i "s|^\(adapter_name_or_path: \).*$|\1${save_path}|" "${predict_save_path_oriset}/${predict_config_name_oriset}"

        sed -i "s|^\(output_dir: \).*$|\1${predict_save_path_oriset}|" "${predict_save_path_oriset}/${predict_config_name_oriset}"
        cat "${predict_save_path_oriset}/${predict_config_name_oriset}" >>"$predict_log_file_oriset"
        llamafactory-cli train "${predict_save_path_oriset}/${predict_config_name_oriset}" &>>"$predict_log_file_oriset"

        # # 生成测试结果
        # predict_save_path_cotset="outputs/saves/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_cotset"
        # mkdir -p "${predict_save_path_cotset}"
        # predict_log_file_cotset="./outputs/logs/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_cotset.log"
        # touch "$predict_log_file_cotset"

        # # 复制配置文件
        # cp ${predict_config_cotset} "${predict_save_path_cotset}/${predict_config_name_cotset}"

        # sed -i "s|^\(adapter_name_or_path: \).*$|\1${save_path}|" "${predict_save_path_cotset}/${predict_config_name_cotset}"

        # sed -i "s|^\(output_dir: \).*$|\1${predict_save_path_cotset}|" "${predict_save_path_cotset}/${predict_config_name_cotset}"
        # cat "${predict_save_path_cotset}/${predict_config_name_cotset}" >>"$predict_log_file_cotset"
        # llamafactory-cli train "${predict_save_path_cotset}/${predict_config_name_cotset}" &>>"$predict_log_file_cotset"
    done
done
