# simple-lvlm

This code implements a very simple large vision-language model.
- Image is processed by a CLIP-like vision model
- Image tokens are aligned to the same embedding space as the LLM
  - An attention layer applied to learnable queries with image tokens as keys and value
- Image tokens and instruction prompt are catenated
