from unsloth import FastLanguageModel
import torch
import json


with open("datas/bad_data.json","r") as label:
    label2=json.load(label)
    
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "dpo_checkpoint", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 4096,
    dtype = torch.bfloat16,
    load_in_4bit =True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

template ="""Make full_paragraph whose type is '{}' and whose keyword is '{}'."""

    
for data in label2:

    
    inputs = tokenizer(
    [
        template.format(
            data["problem_type"], 
            data["keyword"]
        )
    ], return_tensors = "pt").to("cuda")
    
    
    for i in range(10):
        try:
            outputs = model.generate(**inputs, max_new_tokens=1024, use_cache = True)
            outputs2=tokenizer.batch_decode(outputs)[0]
            
        except:
            print("fail")
            continue

        