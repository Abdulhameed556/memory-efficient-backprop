import torch
import torch.nn.functional as F

class MemoryEfficientCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, targets):
        """
        Custom forward pass without storing logits (z = x @ W)
        """
        logits = x @ W  # Compute logits (not stored)
        loss = F.cross_entropy(logits, targets)  # Compute loss

        # Store only necessary values for backprop
        ctx.save_for_backward(x, W, targets)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Custom backward pass to compute gradients without storing logits.
        """
        x, W, targets = ctx.saved_tensors  # Retrieve saved tensors

        # Compute logits again (since we didn't store them)
        logits = x @ W
        probs = F.softmax(logits, dim=-1)  # Softmax probabilities

        # Compute gradient of cross-entropy loss
        probs[range(len(targets)), targets] -= 1  # One-hot adjustment
        probs /= len(targets)  # Normalize gradient
        grad_x = probs @ W.T  # Gradient w.r.t x
        grad_W = x.T @ probs  # Gradient w.r.t W

        return grad_x * grad_output, grad_W * grad_output, None 
