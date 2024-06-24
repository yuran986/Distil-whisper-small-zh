### 蒸馏阶段
1. 运行命令```pip install -e .```配置环境
2. 运行命令```accelerate config```配置Accelerate
3. 我们使用的common_voice数据集需要登录hugging face，通过以下命令登录
    ```
    git config --global credential.helper store
    huggingface-cli login
    ```
4. 提取伪标签：执行```run_pseudo_labelling.sh```脚本，注意修改其中的model_name_or_path、dataset_name等参数
5. 模型初始化：运行以下命令创建并初始化学生模型：
    ```
    python create_student_model.py \
    --teacher_checkpoint "openai/whisper-small" \
    --encoder_layers 12 \
    --decoder_layers 2 \
    --save_dir "./distil-small-init"
    ```
    初始化的学生模型输出路径为```./distil-small-init```
6. 模型训练：执行run_distillation.sh脚本进行蒸馏训练，训练完毕的模型保存路径为```./model```，在本项目训练中冻结了encoder而只对decoder进行训练（通过设置freeze_encoder参数）
7. 模型评估：执行```run_eval_sf.sh```脚本进行模型评估，评估标准包括CER（字符错误率）和rtf（反实时因子）

此外，```test_whisper.py```文件可以在网络数据集上使用指定模型进行推理（即语音识别），```test_whisper_local.py```文件可以在本地wav音频文件上使用指定模型进行推理（即语音识别），```count_params.py```文件可以用于计算模型的参数量