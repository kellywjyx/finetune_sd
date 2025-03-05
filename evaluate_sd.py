from diffusers import StableDiffusionImg2ImgPipeline
import torch
from peft import PeftModel, PeftConfig
from transformers import CLIPModel, CLIPProcessor, Blip2Processor, Blip2ForConditionalGeneration
import lpips
from PIL import Image
import numpy as np
import json
from safetensors.torch import load_file
import torch
from datasets import load_dataset, load_from_disk

# =====================
# 1. Setup Metrics
# =====================

def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# CLIP metric
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# LPIPS metric
lpips_model = lpips.LPIPS(net='alex').to("cuda")

# BLIP-2 for edit success rate
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    torch_dtype=torch.float16
).to("cuda")

def calculate_metrics(original_img, edited_img, instruction):
    """Calculate all metrics for a single sample"""
    metrics = {}

    # Convert PIL images to tensors
    original_tensor = lpips.im2tensor(np.array(original_img)).to("cuda")
    edited_tensor = lpips.im2tensor(np.array(edited_img)).to("cuda")

    # 1. CLIP Score
    with torch.inference_mode():
        clip_inputs = clip_processor(
            text=[instruction],
            images=edited_img,
            return_tensors="pt",
            padding=True
        ).to("cuda")
        clip_score = clip_model(**clip_inputs).logits_per_image.item()
    metrics['clip_score'] = clip_score

    clear_gpu_memory()

    # 2. LPIPS
    metrics['lpips'] = lpips_model(original_tensor, edited_tensor).item()

    # 3. Edit Success Rate
    inputs = blip_processor(images=edited_img, return_tensors="pt").to("cuda", torch.float16)
    generated_ids = blip_model.generate(**inputs, max_new_tokens=20)
    caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    clear_gpu_memory()

    metrics['esr'] = int(any(word in caption.lower() for word in instruction.lower().split()))
    return metrics

# =====================
# 2. Model Setup
# =====================

# Original model
original_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Enable memory optimizations
original_pipe.enable_model_cpu_offload()
original_pipe.enable_xformers_memory_efficient_attention()
original_pipe.enable_attention_slicing()
original_pipe.unet.enable_gradient_checkpointing()
original_pipe.text_encoder.gradient_checkpointing_enable()

# Load LoRA weights
def load_lora_weights(model, lora_path):
    """Load LoRA weights from config.json and model.safetensors"""
    # Load config
    with open(f"{lora_path}/config.json", "r") as f:
        lora_config = json.load(f)
    
    # Load weights
    lora_weights = load_file(f"{lora_path}/model.safetensors")
    
    # Apply weights to model
    for key, value in lora_weights.items():
        # Convert key to match model parameter names
        if "lora_A" in key:
            target_key = key.replace("lora_A", "lora_A")
        elif "lora_B" in key:
            target_key = key.replace("lora_B", "lora_B")
        else:
            continue
        
        # Update model parameters
        with torch.no_grad():
            model.get_parameter(target_key).copy_(value)

# =====================
# 3. Evaluation Loop
# =====================
def evaluate_models(pipe, test_samples, num_samples=50):
    """Evaluate models on test samples"""
    original_scores = {'clip_score': [], 'lpips': [], 'esr': []}
    i = 0
    for sample in test_samples:
        if i >= num_samples:
            break
        input_image = sample["INPUT_IMG"].convert("RGB")
        instruction = sample["EDITING_INSTRUCTION"]

        input_image = input_image.resize((512, 512), Image.LANCZOS)

        # Generate outputs
        original_output = pipe(
            prompt=instruction,
            image=input_image,
            strength=0.8,
            guidance_scale=7.5,
            num_inference_steps=25,
            height=512,  # Reduce height
            width=512    # Reduce width
        ).images[0]

        # Calculate metrics
        orig_metrics = calculate_metrics(input_image, original_output, instruction)

        # Aggregate scores
        for k in original_scores:
            original_scores[k].append(orig_metrics[k])
    i += 1
    # Calculate averages
    return {k: np.mean(v) for k, v in original_scores.items()}

# =====================
# 4. Run Evaluation
# =====================
if __name__ == "__main__":

    # Load test data
    # data = load_dataset("BryanW/HumanEdit")
    # data = data['train'].train_test_split(test_size=0.1, seed=42)
    # data.save_to_disk("split_dataset")

    dataset = load_from_disk("split_dataset")
    # train_data = dataset["train"]
    test_data = dataset["test"]
    print(test_data)

    # Run evaluation
    results = evaluate_models(original_pipe, test_data, num_samples=10)

    clear_gpu_memory()

    # LoRA-adapted model
    lora_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    lora_pipe.enable_attention_slicing()
    lora_pipe.enable_xformers_memory_efficient_attention()

    # Load UNet LoRA
    adapter_config = PeftConfig.from_pretrained("humanedit_lora_unet")
    lora_pipe.unet = PeftModel.from_pretrained(lora_pipe.unet, "humanedit_lora_unet", config=adapter_config)
    # load_lora_weights(lora_pipe.text_encoder, "humanedit_lora_text_encoder")

    lora_results = evaluate_models(lora_pipe, test_data, num_samples=10)

    # Print results
    print("\nEvaluation Results:")
    print(f"{'Metric':<15} | {'Original Model':<15} | {'LoRA Model':<15}")
    print("-" * 45)
    for metric in ['clip_score', 'lpips', 'esr']:
        orig_val = results[metric]
        lora_val = lora_results[metric]
        print(f"{metric:<15} | {orig_val:^15.3f} | {lora_val:^15.3f}")

    # Additional statistics
    print("\nInterpretation:")
    print(f"- CLIP Score: Higher is better (max ~35)")
    print(f"- LPIPS: Lower is better (0=identical, 1=completely different)")
    print(f"- ESR: Edit Success Rate (0-1 scale)")
