import torch
import torch.nn.functional as F
import gc

def measure_vram(model, inputs, targets):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    model.zero_grad()
    loss = model(inputs, targets)
    loss.backward()

    vram_used = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
    print(f"VRAM Used: {vram_used:.2f} MB")

# Initialize inputs
batch_size, vocab_size, embed_size = 1024, 50000, 768
x = torch.randn(batch_size, embed_size, device="cuda")
W = torch.randn(embed_size, vocab_size, device="cuda")
targets = torch.randint(0, vocab_size, (batch_size,), device="cuda")

# Test memory-efficient model
from memory_efficient_ce import MemoryEfficientCrossEntropy

class CustomModel(torch.nn.Module):
    def forward(self, x, targets):
        return MemoryEfficientCrossEntropy.apply(x, W, targets)

model = CustomModel().cuda()
measure_vram(model, x, targets)

# Clear memory
gc.collect()
torch.cuda.empty_cache()
 
