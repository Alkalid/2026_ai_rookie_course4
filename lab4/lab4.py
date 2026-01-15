# lab4_sft_training.py
from datasets import DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def formatting_samples(example, tokenizer):
    """
    將一筆 example["messages"] 轉成一段訓練用文字。
    這裡簡化：整段對話都拿來做 LM 訓練。
    """
    # TODO 1: 使用 tokenizer.apply_chat_template，tokenize=False, add_generation_prompt=False
    text = ""
    return {"text": text}

def main():
    # TODO 2: 載入 Lab3 產生的 train / val JSONL
    ds = DatasetDict(
        {
            "train": load_dataset("json", data_files="workdir/train.jsonl")["train"],
            "val": load_dataset("json", data_files="workdir/val.jsonl")["train"],
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)

    # TODO 3: 對資料集做 map，轉成 {"text": "..."} 格式
    ds_proc = ds.map(
        lambda ex: formatting_samples(ex, tokenizer),
        remove_columns=ds["train"].column_names,
    )

    # TODO 4: 準備 4-bit 量化設定（BitsAndBytesConfig）
    bnb_cfg = BitsAndBytesConfig(
        # load_in_4bit=True,
        # bnb_4bit_quant_type="nf4",
        # ...
    )

    # TODO 5: 載入 base model，啟用 gradient checkpointing
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_cfg,
        torch_dtype=...,
        device_map="auto",
    )
    base_model.config.use_cache = False
    _ = base_model.gradient_checkpointing_enable()

    # TODO 6: 設定 LoRA 參數（r, alpha, dropout, target_modules）
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            # 視模型結構填入 q_proj, k_proj, v_proj ... 等
        ],
    )
    peft_model = get_peft_model(base_model, lora_cfg)

    # TODO 7: 建立 SFTConfig（max_steps, batch_size, packing 等）
    sft_cfg = SFTConfig(
        output_dir="workdir/adapter",
        max_steps=200,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        learning_rate=1e-4,
        warmup_steps=50,
        max_seq_length=1024,
        packing=True,
        bf16=True,
        report_to=[],
    )

    # TODO 8: 建立 SFTTrainer 並呼叫 train()
    trainer = SFTTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=ds_proc["train"],
        eval_dataset=ds_proc["val"],
        args=sft_cfg,
        dataset_text_field="text",
    )

    trainer.train()

    # TODO 9: 儲存 LoRA 權重與 tokenizer
    trainer.save_model("workdir/adapter")
    tokenizer.save_pretrained("workdir/adapter")
    print("訓練完成，LoRA 權重已儲存。")

if __name__ == "__main__":
    main()