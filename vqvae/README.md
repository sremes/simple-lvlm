# VQ-VAE as image tokenizer

We will use VQ-VAE as an image tokenizer for a vision-language transformer model. The VQ-VAE model is trained on a image-only dataset without labels or captions. 
It learns to represent images as a sequence of discrete visual tokens, comparable to how a language model represents text as a sequence of discrete word tokens.

Original paper: "Neural Discrete Representation Learning" - https://arxiv.org/abs/1711.00937
Recent review with improvements: "Straightening Out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks" - https://proceedings.mlr.press/v202/huh23a/huh23a.pdf

Some other implementations:
https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
https://github.com/lucidrains/parti-pytorch/blob/main/parti_pytorch/vit_vqgan.py
https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/vae.py
https://github.com/minyoungg/vqtorch
