# Fine-tuning Stable Diffusion Model using LoRA adapters

# Current Evaluation Results

| Metric       | Original Model | LoRA Model |
|-------------|---------------|------------|
| CLIP Score  | 26.995        | 25.596     |
| LPIPS       | 0.594         | 0.752      |
| ESR         | 0.748         | 0.599      |

## Interpretation
- **CLIP Score**: Higher is better (max ~35)
- **LPIPS**: Lower is better (0 = identical, 1 = completely different)
- **ESR**: Edit Success Rate (0-1 scale)
