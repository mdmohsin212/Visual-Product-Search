import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from visual_product_search.logger import logging
import time
from visual_product_search.exception import ExceptionHandle
import sys


def train(model, dataloader, device, epochs=10, lr=1e-4):
    try:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        model = get_peft_model(model, lora_config)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        scaler = GradScaler()

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            start_epoch = time.time()
            logging.info(f"--------- Epoch {epoch+1}/{epochs} ---------")

            for batch_idx, batch in enumerate(dataloader, start=1):
                imgs = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                optimizer.zero_grad()
                with autocast(device_type="cuda", dtype=torch.float16):
                    img_embeds = model.get_image_features(pixel_values=imgs)
                    txt_embeds = model.get_text_features(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    img_embeds = nn.functional.normalize(img_embeds, dim=-1)
                    txt_embeds = nn.functional.normalize(txt_embeds, dim=-1)

                    logits = img_embeds @ txt_embeds.T * 100
                    labels = torch.arange(len(logits), device=device)
                    loss_i2t = loss_fn(logits, labels)
                    loss_t2i = loss_fn(logits.T, labels)
                    loss = (loss_i2t + loss_t2i) / 2

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

                if batch_idx % 50 == 0 or batch_idx == len(dataloader):
                    logging.info(
                        f"[Epoch {epoch+1} Batch {batch_idx}/{len(dataloader)}] "
                        f"Batch Loss: {loss.item():.4f}"
                    )

            avg_loss = total_loss / len(dataloader)
            epoch_time = time.time() - start_epoch
            logging.info(
                f"Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time/60:.2f} min"
            )
            
        return model

    except Exception as e:
        logging.critical("Training loop crashed")
        raise ExceptionHandle(e, sys)