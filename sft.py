from unsloth import FastLanguageModel 
import torch
from trl import SFTTrainer
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import jsonlines
import wandb

max_seq_length = 4096
dtype = torch.bfloat16

list=[]
with jsonlines.open('datas/good_data.jsonl') as reader:
    for obj in reader:
        
        input1=obj["type"]
        input2=obj["keyword"]
        
        target=obj["full_paragraph"]
        
        script=f"""<human>: Make full_paragraph whose type is '{input1}' and whose keyword is '{input2}'.\n<bot>: {target} """
        list.append(script)
        
dict={}
dict["text"]=list
dataset = Dataset.from_dict(dict)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",    
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False, 
    loftq_config = None, 
)

wandb.login()
project_name = "SFT" 
wandb.init(project=project_name, name = "1")


training_args = SFTConfig(
    output_dir="sft_checkpoint",
    num_train_epochs=30000,
    save_steps=60,
    save_total_limit=10,
    bf16=True,
    per_gpu_train_batch_size=50,
    max_seq_length=max_seq_length)

trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field = "text")

trainer.train()


