

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,7'
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
import sys
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser,default_data_collator
from transformers import AutoTokenizer, AutoModel,AutoConfig
import torch
import torch.nn as nn
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import logging
from modeling_chatglm import ChatGLMForConditionalGeneration
from data import InstrutionDataset,InstrutionCollator
from arguments import ModelArguments, DataTrainingArguments, FinetuneArguments as TrainingArguments
from transformers.trainer_utils import is_main_process
import transformers
from datasets import load_dataset
from tokenization_chatglm import ChatGLMTokenizer


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss




def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")




def main():
    writer = SummaryWriter()


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()



    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection
 
    tokenizer = ChatGLMTokenizer.from_pretrained(model_args.tokenizer_name, trust_remote_code=True)

  
    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half()
        model.transformer.prefix_encoder.float()
    else:
        # Finetune
        model = model.float()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    print_trainable_parameters(model)
    
    ## data
    train_data = InstrutionDataset(
        data_path = data_args.train_path, prefix=prefix)
    
    data_collator = InstrutionCollator(
        tokenizer=tokenizer,
        max_len = training_args.max_len,
        max_input_len=training_args.max_input_len
    )

    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
        save_prefixencoder=model_args.pre_seq_len is not None
    )
    

    trainer.train()
    writer.close()

    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
