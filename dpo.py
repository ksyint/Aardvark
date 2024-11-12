from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from datasets import Dataset, DatasetDict
import wandb
from transformers import TrainingArguments
from trl import DPOTrainer,DPOConfig
import jsonlines
import json
from datasets import Dataset


"""


앞으로 개선 방향 

1. 더 큰 모델을 쓰도록한다. 최소 LLAMA 3.1 70B Lora
2. 메모리 문제가 날경우 AWQ 양자화 시도도 해본다. (4,8 비트 등등)
3. Prompt Engineering : problem type 에 대한 설명을 더 자세하게 한다. 
4. DPO 강화학습 : Reject 된 Full Para 예시를 늘린다 (dpo 는 accept, reject twin 페어로 학습)
5. 예를 들면 full para 를 한국어로 번역하고 다시 영어로 번역을 해본다 
6. 이외에도 few shot 으로 full para 만드는 예시를 보여주고 (bad data) 그 예시에 맞게 gpt 가 bad data를 만들게 한다 
7. 굳이 Lora가 아닌 Full parameter tuning 도 해본다, 단 아쉽게도 8B 이내로만 가능할것이다. 
8. 확실한건 보통 Lora 나 dpo train 시에 unsloth framework 가 메모리가 덜든다.
9. LLM 답변 평가방법 :https://huggingface.co/spaces/allenai/reward-bench 를 참고한다 

이외 참고자료 

https://huggingface.co/docs/transformers/main_classes/quantization 
https://github.com/vllm-project/vllm
https://github.com/unslothai/unsloth
https://www.kaggle.com/code/hemanthkumar21/meta-llama3-8b-fine-tuning-unsloth

llama3 checkpoint 

"unsloth/llama-3-8b-bnb-4bit"    
"unsloth/llama-3-8b-Instruct-bnb-4bit"
"unsloth/llama-3-70b-bnb-4bit"

이제 코드를 돌리자 단 sft 부터 하자

"""

max_seq_length = 4096
dtype = torch.bfloat16
load_in_4bit = True 

list=[]
list2=[]
list3=[]
list4=[]
num=0

with jsonlines.open('datas/good_data.jsonl') as reader:
    
        for obj in reader:
            
            input1=obj["type"]
            input2=obj["keyword"]
            target=obj["full_paragraph"]
            
            
            with open('datas/bad_data.json',"r") as reader2:
                reader2=json.load(reader2)
                
                for obj2 in reader2:
                    
                    
                    input1_2=obj2['problem_type']
                    input2_2=obj2["keyword"]
                    target_2=obj2["full_paragraph"]
                    
                    if input2==input2_2:
                        
                        num+=1
                        dict={}
                        
                        dict["system"]="You are an AI assistant. Provide a full paragraph based on the problem type and the keyword."
                        dict["prompt"]=f"Make full paragraph based on the problem type '{input1_2}' and the keyword '{input2_2}'."
                        dict["chosen"]=target
                        dict["rejected"]=target_2
                        if target is not None and target_2 is not None:
                        
                            list.append(str(dict["system"]))
                            list2.append(str(dict["prompt"]))
                            list3.append(str(dict["chosen"]))
                            list4.append(str(dict["rejected"]))
                        
                    
                        
                    else:
                        pass
                    
                    
train_data={}     
train_data["system"]=list
train_data["prompt"]=list2
train_data["chosen"]=list3
train_data["rejected"]=list4

train_dataset = Dataset.from_dict(train_data)

dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': train_dataset
})


"""

이제 모델을 로딩하자


"""


                        
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name ="sft_checkpoint",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

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
project_name = "DPO" 
wandb.init(project=project_name, name = "1")


"""

이제 훈련 시작하자


"""

training_args = DPOConfig(
    output_dir="dpo_checkpoint",
    beta=0.1,
    num_train_epochs=30000,
    save_steps=50,
    bf16=True,
    max_length = max_seq_length,
    max_prompt_length = max_seq_length//2,
    per_gpu_train_batch_size=20,
    save_total_limit=10)


dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset_dict["train"],
    tokenizer=tokenizer
)

dpo_trainer.train()
