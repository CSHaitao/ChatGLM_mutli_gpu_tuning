
# ChatGLM_LoRA

使用deepspeed+trainer+lora微调ChatGLM.简单高效。

## Data Structure

> {"instruction": "", "input": "工作时把右小指挤段又接上的，XXXX左右的工资，在说我们感觉太不公平了，刚手术做了XX个小时", "answer": "这种情况，是属于工伤的。按照法律规定，需要先行认定工伤，认定工伤后，再进行劳动能力鉴定，然后再具体索赔。涉及赔偿项目比较多，一般包括医疗费、护理费、伙食费、停工留薪期内的工资，构成伤残的，还可以主张一次性伤残补助金、一次性伤残就业补助金、一次性工伤医疗补助金等，每一项赔偿的标准不一样，需要根据实际情况来具体计算。由于工伤案件，涉及程序较多、法律关系复杂，建议委托专业律师代理处理。我们经常代理工伤赔偿案件，如果愿意，我们可以协助处理。详情可以电联，或者面谈。"}

instruction_data.json中提供了数据样例

## Train

1.修改train.sh中模型和数据路径
2.运行sh train.sh
```bash
    CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT --num_gpus=3 finetune.py \
        --train_path ./instrution_data.json \
        --max_len 768 \
        --max_input_len 512 \
        --model_name_or_path ./chatGLM-6B \
        --tokenizer_name ./chatGLM-6B \
        --lora_rank 8 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 10 \
        --save_steps 900 \
        --learning_rate 1e-5 \
        --fp16 \
        --remove_unused_columns false \
        --logging_steps 50 \
        --output_dir /output \
        --deepspeed /ds_config.json \
```
单卡运行可以改为num_gpus==1

## Requirements

```
python=3.9
transformers==4.28.1
tqdm==4.64.1
datasets==2.8.0
torch==2.0.0
pytorch==1.12.1
deepspeed==0.9.1
peft==0.2.0 
```

## Todo

- [ ] 增加模型并行和多卡inference

- [ ] 增加更多finetune方式

- [ ] 开源微调数据和微调模型

