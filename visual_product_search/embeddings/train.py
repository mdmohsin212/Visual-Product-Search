import torch
from torch.nn import CrossEntropyLoss
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from visual_product_search.logger import logging
from visual_product_search.exception import ExceptionHandle
import sys

def train(model, dataloader, device, epochs=5, lr=1e-5, grad_accum_steps=1, num_warmup_step=50):
    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(device="cuda")
    total_steps = epochs * len(dataloader)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_step,
        num_training_steps=total_steps
    )
    
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for step, batch in enumerate(dataloader):
                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    pixel_values = batch['pixel_values'].to(device)
                    
                    optimizer.zero_grad()
                    
                    with autocast(device_type="cuda"):
                        outputs = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        pixel_values=pixel_values)
                        
                        img_embd = outputs.image_embeds
                        text_embd = outputs.text_embeds
                        
                        img_embd = img_embd / img_embd.norm(p=2, dim=-1, keepdim=True)
                        text_embd = text_embd / text_embd.norm(p=2, dim=-1, keepdim=True)
                        
                        logits_per_image = img_embd @ text_embd.T
                        labels = torch.arange(len(img_embd)).to(device)

                        loss_i = CrossEntropyLoss()(logits_per_image, labels)
                        loss_t = CrossEntropyLoss()(logits_per_image.T, labels)
                        
                        loss = (loss_i + loss_t) / 2
                    
                    loss = loss / grad_accum_steps
                    scaler.scale(loss).backward()
                    
                    if (step + 1) % grad_accum_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                        
                    total_loss += loss.item() * grad_accum_steps 
                    
                    if step % 10 == 0:
                        logging.info(f"Epoch {epoch+1}/{epochs}, Step {step}, Loss: {loss.item():.4f}")
                
                except Exception as e:
                    logging.error(f"Failure at step {step} in epoch {epoch+1}")
                    raise ExceptionHandle(e, sys)
            
            
            avg_loss = total_loss / len(dataloader)
            logging.info(f" Epoch {epoch + 1} / {epochs} finished | Avg Loss : {avg_loss:.4f}")
        
        model.eval()
        return model
            
    except Exception as e:
        logging.critical("Training loop crashed")
        raise ExceptionHandle(e, sys)
    
    
    
    
# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0.0
#     start_epoch = time.time()
#     print(f"\n======== Epoch {epoch+1}/{EPOCHS} ========")
    
#     for batch_idx, (imgs, captions) in enumerate(dataloader, start=1):
#         batch_start = time.time()
#         imgs = imgs.to(DEVICE)
#         inputs_txt = processor(text=list(captions), return_tensors="pt", padding=True, truncation=True).to(DEVICE)

#         optimizer.zero_grad()
#         with autocast(device_type="cuda", dtype=torch.float16):
#             inputs_img = processor(images=imgs, return_tensors="pt").to(DEVICE)
#             img_embeds = model.get_image_features(**inputs_img)
#             txt_embeds = model.get_text_features(**inputs_txt)

#             img_embeds = nn.functional.normalize(img_embeds, dim=-1)
#             txt_embeds = nn.functional.normalize(txt_embeds, dim=-1)

#             logits = img_embeds @ txt_embeds.T * 100
#             labels = torch.arange(len(logits), device=DEVICE)
#             loss_i2t = loss_fn(logits, labels)
#             loss_t2i = loss_fn(logits.T, labels)
#             loss = (loss_i2t + loss_t2i) / 2

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         total_loss += loss.item()

#         if batch_idx % 50 == 0 or batch_idx == len(dataloader):
#             elapsed = time.time() - batch_start
#             print(f"[Epoch {epoch+1} Batch {batch_idx}/{len(dataloader)}] "
#                   f"Batch Loss: {loss.item():.4f}, "
#                   f"Elapsed: {elapsed:.2f}s, "
#                   f"GPU Memory Used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

#     avg_loss = total_loss / len(dataloader)
#     epoch_time = time.time() - start_epoch
#     print(f"Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f} | Time: {epoch_time/60:.2f} min")