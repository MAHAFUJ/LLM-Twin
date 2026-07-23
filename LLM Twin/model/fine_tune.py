import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def fine_tune_twin():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        load_in_4bit=True, 
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files="data/processed/instruction_pairs.json", split="train")

    def format_instruction(sample):
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI developer, researcher, and Tech Lead at Outlier Research. 
Your guiding philosophy is: A Learner, AI Developer, Researcher.
You hold a BSc in Computer Science and Engineering and focus on machine learning and autonomous vehicles.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{sample['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{sample['response']}<|eot_id|>"""

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=format_instruction,
        args=TrainingArguments(
            output_dir="./twin_model",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=10
        ),
    )
    
    trainer.train()
    trainer.model.save_pretrained("./twin_model/final")

if __name__ == "__main__":
    pass