user_name=zyk_sft_doc0_abs
purpose='对于doc0的sft和stf_test的测试'

export CUDA_VISIBLE_DEVICES=0
model_name=Qwen2.5-7B-Instruct
method=sft

model_config_name=Qwen2.5-7B-Instruct_sft_doc_0_abs_0.yaml
predict_config_name_oriset=Qwen2.5-7B-Instruct_sft_doc_0_abs_0_predict_test.yaml


model_config="configs/sft/${model_config_name}"
predict_config_oriset="configs/predict/${predict_config_name_oriset}"


lr_list=(5.0e-5 1.0e-5 5.0e-6 1.0e-6)
warmup_ratio_list=(0.3 0.25 0.2 0.15 0.1 0.05)


# 循环遍历lr和warmup_ratio
for lr in ${lr_list[@]}; do
    for warmup_ratio in ${warmup_ratio_list[@]}; do

    # 添加条件判断，如果lr和warmup_ratio为某些值则跳过当前循环
    if [[ "$lr" == "5.0e-5" && "$warmup_ratio" == "0.05" ]]; then
        continue
    fi

     # 0. 构建初始化参数
        # 将current_time再次赋值为当前时间
        current_time=$(date +"%m-%d-%H-%M-%S")
        # 创建save文件夹
        save_path="outputs/saves/${model_name}/${method}/${user_name}_${lr}_${warmup_ratio}_${current_time}"
        mkdir -p "${save_path}"

        train_yaml_file="${save_path}/${model_config_name}"
        #复制“训练yaml” 
        cp ${model_config} "${train_yaml_file}"
    
    # 1.构建日志文件
        # 1.1 创建日志文件并重定向输出
        log_file="./outputs/logs/${model_name}/${method}/${user_name}_${lr}_${warmup_ratio}_${current_time}.log"
        touch "$log_file"
        # 1.2 在文件中写入名字和目的
        echo "用户：$user_name" >>"$log_file"
        echo "目的：$purpose" >>"$log_file"
        # 1.3 将配置文件的内容输出到日志文件中
        cat "${train_yaml_file}" >>"$log_file"

    # 2. 修改“训练yaml” 配置文件中的内容       
        # 修改输出路径                       
        sed -i "s|^\(output_dir: \).*$|\1${save_path}|" "${train_yaml_file}" 
        # 将lr和warmup_ratio改为当前的值
        sed -i "s|^\(learning_rate: \).*$|\1$lr|" "${train_yaml_file}"
        sed -i "s|^\(warmup_ratio: \).*$|\1$warmup_ratio|" "${train_yaml_file}"
    

    # 3. 训练模型
        llamafactory-cli train "${train_yaml_file}" &>>"$log_file"


    # 4. 生成原始数据的测试结果
        # 4.1 创建保存路径和日志文件
        predict_save_path_oriset="./outputs/saves/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_oriset"
        predict_log_file_oriset="./outputs/logs/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_oriset.log"
        mkdir -p "${predict_save_path_oriset}"
        touch "$predict_log_file_oriset"
        predict_yaml_file="${predict_save_path_oriset}/${predict_config_name_oriset}"
        # 4.2 复制配置文件到指定路径下
        cp ${predict_config_oriset} "${predict_yaml_file}"
        # 4.3 修改“模型文件“和”微调后的适配器权重路径”以及“输出文件夹” （这里模型文件直接修改在yaml中，此处不做修改）
        sed -i "s|^\(adapter_name_or_path: \).*$|\1${save_path}|" "${predict_yaml_file}"
        sed -i "s|^\(output_dir: \).*$|\1${predict_save_path_oriset}|" "${predict_yaml_file}"
        # 4.4 将配置文件的内容输出到日志文件中
        echo "用户：$user_name" >>"$predict_log_file_oriset"
        echo "目的：$purpose" >>"$predict_log_file_oriset"
        cat "${predict_yaml_file}" >>"$predict_log_file_oriset"
        # 4.5 调用llamafactory-cli进行预测
        llamafactory-cli train "${predict_yaml_file}" &>>"$predict_log_file_oriset"
    done
done
