# 直接对qwen2.5-7b用广安门数据微调和测试,测试包括广安门测试集、原始测试集、cot测试集

user_name=zyk_0401_gam_directly_sft_predict
purpose='GAM数据的直接微调以及预测'

export CUDA_VISIBLE_DEVICES=0
model_name=Qwen2.5-7B-Instruct
method=sft

model_config_name=Qwen2.5-7B-Instruct_gam_direct_sft.yaml
predict_config_name_test=GAM-direct-sft-predict-qwen2.5-7b.yaml
predict_config_name_oriset=GAM-direct-oritest-qwen2.5-7b.yaml
# predict_config_name_cotset=GAM-direct-cottest-qwen2.5-7b.yaml

model_config="configs/sft/${model_config_name}"
predict_config_test="configs/predict/${predict_config_name_test}"
predict_config_oriset="configs/predict/${predict_config_name_oriset}"
# predict_config_cotset="configs/predict/${predict_config_name_cotset}"

lr_list=(5.0e-5)
warmup_ratio_list=(0.3 0.25 0.2 0.15 0.1 0.05)

# 循环遍历lr和warmup_ratio
for lr in ${lr_list[@]}; do
    for warmup_ratio in ${warmup_ratio_list[@]}; do
        # 0. 构建初始化参数
        # 将current_time再次赋值为当前时间
        current_time=$(date +"%m-%d-%H-%M-%S")
        # 创建save文件夹
        save_path="outputs/saves/${model_name}/${method}/${user_name}_${lr}_${warmup_ratio}_${current_time}"
        mkdir -p "${save_path}"

        # 1. 先修改“训练yaml”
        train_yaml_file="${save_path}/${model_config_name}"

        cp ${model_config} "${train_yaml_file}"                              # 复制配置文件，并修改配置文件中的内容
        sed -i "s|^\(output_dir: \).*$|\1${save_path}|" "${train_yaml_file}" # 在最后面加上当前时间
        # 将lr和warmup_ratio改为当前的值
        sed -i "s|^\(learning_rate: \).*$|\1$lr|" "${train_yaml_file}"
        sed -i "s|^\(warmup_ratio: \).*$|\1$warmup_ratio|" "${train_yaml_file}"

        # 2.构建日志文件
        # 2.1 创建日志文件并重定向输出
        log_file="./outputs/logs/${model_name}/${method}/${user_name}_${lr}_${warmup_ratio}_${current_time}.log"
        touch "$log_file"
        # 2.2 在文件中写入名字和目的
        echo "用户：$user_name" >>"$log_file"
        echo "目的：$purpose" >>"$log_file"
        # 2.3 将配置文件的内容输出到日志文件中
        cat "${save_path}/${model_config_name}" >>"$log_file"

        # 3. 训练模型
        llamafactory-cli train "${save_path}/${model_config_name}" &>>"$log_file"

        # 4. 生成广安门数据的测试结果
        # 4.1 创建保存路径和日志文件
        predict_save_path_test="outputs/saves/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_gam_test"
        predict_log_file_test="./outputs/logs/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_gam_test.log"
        mkdir -p "${predict_save_path_test}"
        touch "$predict_log_file_test"
        # 4.2 复制配置文件到指定路径下
        predict_yaml_test="${predict_save_path_test}/${predict_config_name_test}"
        cp ${predict_config_test} "${predict_yaml_test}"
        # 4.3 修改“模型文件“和”微调后的适配器权重路径”以及“输出文件夹” （这里模型文件直接修改在yaml中，此处不做修改）
        sed -i "s|^\(adapter_name_or_path: \).*$|\1${save_path}|" "${predict_yaml_test}"
        sed -i "s|^\(output_dir: \).*$|\1${predict_save_path_test}|" "${predict_yaml_test}"
        # 4.4 将配置文件的内容输出到日志文件中
        cat "${predict_yaml_test}" >>"$predict_log_file_test"
        # 4.5 调用llamafactory-cli进行预测
        llamafactory-cli train "${predict_yaml_test}" &>>"$predict_log_file_test"

        # 5. 生成原始10w条数据的测试结果
        # 5.1 创建保存路径和日志文件
        predict_save_path_oriset="outputs/saves/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_oriset"
        predict_log_file_oriset="./outputs/logs/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_oriset.log"
        mkdir -p "${predict_save_path_oriset}"
        touch "$predict_log_file_oriset"
        # 5.2 复制配置文件到指定路径下
        predict_yaml_oritest="${predict_save_path_oriset}/${predict_config_name_oriset}"
        cp ${predict_config_oriset} "${predict_yaml_oritest}"
        # 5.3 修改“模型文件“和”微调后的适配器权重路径”以及“输出文件夹” （这里模型文件直接修改在yaml中，此处不做修改）
        sed -i "s|^\(adapter_name_or_path: \).*$|\1${save_path}|" "${predict_yaml_oritest}"
        sed -i "s|^\(output_dir: \).*$|\1${predict_save_path_oriset}|" "${predict_yaml_oritest}"
        # 5.4 将配置文件的内容输出到日志文件中
        cat "${predict_yaml_oritest}" >>"$predict_log_file_oriset"
        # 5.5 调用llamafactory-cli进行预测
        llamafactory-cli train "${predict_yaml_oritest}" &>>"$predict_log_file_oriset"


    done
done

lr_list=(1.0e-5 5.0e-6 1.0e-6)
warmup_ratio_list=(0.05)

# 循环遍历lr和warmup_ratio
for lr in ${lr_list[@]}; do
    for warmup_ratio in ${warmup_ratio_list[@]}; do
        # 0. 构建初始化参数
        # 将current_time再次赋值为当前时间
        current_time=$(date +"%m-%d-%H-%M-%S")
        # 创建save文件夹
        save_path="outputs/saves/${model_name}/${method}/${user_name}_${lr}_${warmup_ratio}_${current_time}"
        mkdir -p "${save_path}"

        # 1. 先修改“训练yaml”
        train_yaml_file="${save_path}/${model_config_name}"

        cp ${model_config} "${train_yaml_file}"                              # 复制配置文件，并修改配置文件中的内容
        sed -i "s|^\(output_dir: \).*$|\1${save_path}|" "${train_yaml_file}" # 在最后面加上当前时间
        # 将lr和warmup_ratio改为当前的值
        sed -i "s|^\(learning_rate: \).*$|\1$lr|" "${train_yaml_file}"
        sed -i "s|^\(warmup_ratio: \).*$|\1$warmup_ratio|" "${train_yaml_file}"

        # 2.构建日志文件
        # 2.1 创建日志文件并重定向输出
        log_file="./outputs/logs/${model_name}/${method}/${user_name}_${lr}_${warmup_ratio}_${current_time}.log"
        touch "$log_file"
        # 2.2 在文件中写入名字和目的
        echo "用户：$user_name" >>"$log_file"
        echo "目的：$purpose" >>"$log_file"
        # 2.3 将配置文件的内容输出到日志文件中
        cat "${save_path}/${model_config_name}" >>"$log_file"

        # 3. 训练模型
        llamafactory-cli train "${save_path}/${model_config_name}" &>>"$log_file"

        # 4. 生成广安门数据的测试结果
        # 4.1 创建保存路径和日志文件
        predict_save_path_test="outputs/saves/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_gam_test"
        predict_log_file_test="./outputs/logs/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_gam_test.log"
        mkdir -p "${predict_save_path_test}"
        touch "$predict_log_file_test"
        # 4.2 复制配置文件到指定路径下
        predict_yaml_test="${predict_save_path_test}/${predict_config_name_test}"
        cp ${predict_config_test} "${predict_yaml_test}"
        # 4.3 修改“模型文件“和”微调后的适配器权重路径”以及“输出文件夹” （这里模型文件直接修改在yaml中，此处不做修改）
        sed -i "s|^\(adapter_name_or_path: \).*$|\1${save_path}|" "${predict_yaml_test}"
        sed -i "s|^\(output_dir: \).*$|\1${predict_save_path_test}|" "${predict_yaml_test}"
        # 4.4 将配置文件的内容输出到日志文件中
        cat "${predict_yaml_test}" >>"$predict_log_file_test"
        # 4.5 调用llamafactory-cli进行预测
        llamafactory-cli train "${predict_yaml_test}" &>>"$predict_log_file_test"

        # 5. 生成原始10w条数据的测试结果
        # 5.1 创建保存路径和日志文件
        predict_save_path_oriset="outputs/saves/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_oriset"
        predict_log_file_oriset="./outputs/logs/${model_name}/predict_sft/${user_name}_${lr}_${warmup_ratio}_${current_time}_oriset.log"
        mkdir -p "${predict_save_path_oriset}"
        touch "$predict_log_file_oriset"
        # 5.2 复制配置文件到指定路径下
        predict_yaml_oritest="${predict_save_path_oriset}/${predict_config_name_oriset}"
        cp ${predict_config_oriset} "${predict_yaml_oritest}"
        # 5.3 修改“模型文件“和”微调后的适配器权重路径”以及“输出文件夹” （这里模型文件直接修改在yaml中，此处不做修改）
        sed -i "s|^\(adapter_name_or_path: \).*$|\1${save_path}|" "${predict_yaml_oritest}"
        sed -i "s|^\(output_dir: \).*$|\1${predict_save_path_oriset}|" "${predict_yaml_oritest}"
        # 5.4 将配置文件的内容输出到日志文件中
        cat "${predict_yaml_oritest}" >>"$predict_log_file_oriset"
        # 5.5 调用llamafactory-cli进行预测
        llamafactory-cli train "${predict_yaml_oritest}" &>>"$predict_log_file_oriset"


    done
done
