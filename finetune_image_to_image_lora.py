import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from datasets import load_dataset, load_from_disk
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
import accelerate
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. Load dataset
# dataset = load_dataset("BryanW/HumanEdit", split="train")
# dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset = load_from_disk("split_dataset")['train']

# 2. Configure model and LoRA
model_id = "runwayml/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

# 3. Setup LoRA
lora_config = LoraConfig(
    r=32,  
    lora_alpha=64,
    target_modules=[
        "attn1.to_q", "attn1.to_k", "attn1.to_v",  # Self-attention
        "attn2.to_q", "attn2.to_k", "attn2.to_v",   # Cross-attention
        "ff.net.2" 
    ],
    lora_dropout=0.2,
    bias="lora_only",
    modules_to_save=["conv_in","conv_out"]  
)

unet = get_peft_model(unet, lora_config)
unet.enable_gradient_checkpointing()

preprocess_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 4. Data preprocessing
def preprocess(examples):
    original_images = [image.convert("RGB") for image in examples["INPUT_IMG"]]
    edited_images = [image.convert("RGB") for image in examples["OUTPUT_IMG"]]
    prompts = examples["EDITING_INSTRUCTION"]

    original_processed = torch.stack([
        preprocess_transform(img) for img in original_images
    ])

    edited_processed = torch.stack([
        preprocess_transform(img) for img in edited_images
    ])

    tokenized = tokenizer(
        prompts,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )

    return {
        "original_pixel_values": original_processed,
        "edited_pixel_values": edited_processed,
        "input_ids": tokenized.input_ids
    }

dataset.set_transform(preprocess)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 5. Training setup
num_epochs = 10
optimizer = torch.optim.AdamW(
    list(unet.parameters()),
    lr=5e-5
)
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_dataloader))

# 6. Training loop
device = "cuda"
unet.to(device)
text_encoder.to(device)
vae.to(device)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Convert images to latent space
        with torch.no_grad():
            original_latents = vae.encode(batch["original_pixel_values"].to(device)).latent_dist.sample()
            edited_latents = vae.encode(batch["edited_pixel_values"].to(device)).latent_dist.sample()
            original_latents = original_latents * 0.18215
            edited_latents = edited_latents * 0.18215

        # Sample noise
        noise = torch.randn_like(edited_latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (edited_latents.shape[0],), device=device).long()

        # Add noise
        noisy_latents = noise_scheduler.add_noise(edited_latents, noise, timesteps)

        # Get text embeddings
        encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]

        # Predict noise residual
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs={"original_latents": original_latents}
        ).sample

        # Compute loss
        # Combined loss function
        diffusion_loss = F.mse_loss(noise_pred, noise)
        preservation_loss = F.l1_loss(noisy_latents, original_latents)
        loss = diffusion_loss + 0.3 * preservation_loss  # Weighted sum
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        unet.save_pretrained("humanedit_lora_unet")
        # text_encoder.save_pretrained("humanedit_lora_text_encoder")
        # vae.save_pretrained("humanedit_lora_vae")

    print(f"Epoch {epoch} Loss: {loss.item()}")

# 7. Save LoRA weights
unet.save_pretrained("humanedit_lora_unet")
# text_encoder.save_pretrained("humanedit_lora_text_encoder")