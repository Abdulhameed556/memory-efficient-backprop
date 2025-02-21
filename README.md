# Memory-Efficient Backpropagation for Cross-Entropy Loss  

## 🚀 Overview  
This repository contains a **memory-efficient implementation** of backpropagation for cross-entropy loss in PyTorch.  
The goal is to **reduce VRAM usage by 50%** while maintaining correct gradient computation.  

## ✅ Features  
- 📉 **50% VRAM Reduction** using custom `torch.autograd.Function`  
- 🏎 **Avoids OOM Errors** for large vocab sizes  
- 🔄 **Dynamic Chunk Sizes** for flexibility  
- 📊 **Gradient Equivalence with `torch.allclose()`**  
- 🏋️ **Tested on LLaMA-1B Model**  

 
