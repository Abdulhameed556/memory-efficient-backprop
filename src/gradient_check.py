import torch
import torch.nn.functional as F
from memory_efficient_ce import MemoryEfficientCrossEntropy

# Initialize tensors
batch_size, vocab_size, embed_size = 1024, 50000, 768
x = torch.randn(batch_size, embed_size, device="cuda", requires_grad=True)
W = torch.randn(embed_size, vocab_size, device="cuda", requires_grad=True)
targets = torch.randint(0, vocab_size, (batch_size,), device="cuda")

# Standard cross-entropy
standard_loss = F.cross_entropy(x @ W, targets)
standard_loss.backward()
grad_W_standard = W.grad.clone()

# Reset gradients
W.grad.zero_()

# Memory-efficient cross-entropy
efficient_loss = MemoryEfficientCrossEntropy.apply(x, W, targets)
efficient_loss.backward()
grad_W_efficient = W.grad.clone()

# Check if gradients are equivalent
print("Gradient equivalence:", torch.allclose(grad_W_standard, grad_W_efficient, atol=1e-6))
 
