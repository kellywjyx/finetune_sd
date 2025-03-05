# Fine-tuning Stable Diffusion Model using LoRA adapters

## Current Evaluation Results

| Metric       | Original Model | LoRA Model |
|-------------|---------------|------------|
| CLIP Score  | 26.895        | 26.889     |
| LPIPS       | 0.590         | 0.582      |
| ESR         | 0.724         | 0.748      |

## Interpretation
- **CLIP Score**: Higher is better (max ~35)
- **LPIPS**: Lower is better (0 = identical, 1 = completely different)
- **ESR**: Edit Success Rate (0-1 scale)
