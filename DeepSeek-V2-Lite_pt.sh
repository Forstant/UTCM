#!/usr/bin/bash

user_name=pzb
purpose='增量预训练'
export CUDA_VISIBLE_DEVICES=0
model_name=DeepSeek-V2-Lite
method=full_pt
predict_config_name=DeepSeek-V2-Lite_predict.yaml
lr_list=(5.0e-5)
warmup_ratio_list=(0.4 0.45)

model_config_name=${model_name}_pt.yaml
model_config="configs/pt/${model_config_name}"
predict_config="configs/predict/${predict_config_name}"

for lr in "${lr_list[@]}"; do
    for warmup_ratio in "${warmup_ratio_list[@]}"; do

        current_time=$(date +"%m-%d-%H-%M-%S")

        log_file="./outputs/logs/${model_name}/${method}/${user_name}_${lr}_${warmup_ratio}_${current_time}.log"
        touch "$log_file"

        save_path="outputs/saves/${model_name}/${method}/${user_name}_${lr}_${warmup_ratio}_${current_time}"
        mkdir -p "${save_path}"
        cp ${model_config} "${save_path}/${model_config_name}"

        sed -i "s|^\(output_dir: \).*$|\1${save_path}|" "${save_path}/${model_config_name}"
        sed -i "s|^\(learning_rate: \).*$|\1$lr|" "${save_path}/${model_config_name}"
        sed -i "s|^\(warmup_ratio: \).*$|\1$warmup_ratio|" "${save_path}/${model_config_name}"

        echo "用户：$user_name" >>"$log_file"
        echo "目的：$purpose" >>"$log_file"

        cat "${save_path}/${model_config_name}" >>"$log_file"
        llamafactory-cli train "${save_path}/${model_config_name}" &>>"$log_file"

        ## 测试
        # 生成测试结果
        predict_save_path="./outputs/saves/${model_name}/predict/${user_name}_${lr}_${warmup_ratio}_${current_time}"
        mkdir -p "${predict_save_path}"
        predict_log_file="./outputs/logs/${model_name}/predict/${user_name}_${lr}_${warmup_ratio}_${current_time}.log"
        touch "$predict_log_file"

        # 复制配置文件
        cp ${predict_config} "${predict_save_path}/${predict_config_name}"

        sed -i "s|^\(model_name_or_path: \).*$|\1${save_path}|" "${predict_save_path}/${predict_config_name}"

        sed -i "s|^\(output_dir: \).*$|\1${predict_save_path}|" "${predict_save_path}/${predict_config_name}"
        cat "${predict_save_path}/${predict_config_name}" >>"$predict_log_file"
        llamafactory-cli train "${predict_save_path}/${predict_config_name}" &>>"$predict_log_file"
    done
done
