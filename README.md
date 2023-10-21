# simple-lvlm

This code implements a very simple large vision-language model.
- Image is processed by a CLIP-like vision model
- Image tokens are aligned to the same embedding space as the LLM
  - An attention layer applied to learnable queries with image tokens as keys and value
- Image tokens and instruction prompt are catenated


## ROCm Docker

Fill in Huggingface token into `docker.env`. Then build the docker image and source the convenient alias `rocm` from `env.sh` to run commands inside the container.
```bash
docker build -f Dockerfile .
docker tag "image_id_from_above" rocm-transformers
source env.sh
rocm python model.py  # runs python inside the rocm container
```
