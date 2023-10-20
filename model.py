from utils import get_device

import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from peft import get_peft_model, IA3Config, TaskType


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class LVLM(torch.nn.Module):
    def __init__(self, language_model: str, vision_model: str, num_queries: int = 32) -> None:
        # initialize module
        super().__init__()
        self.device = "cpu" # get_device()
        
        # base models
        self.llm = AutoModelForCausalLM.from_pretrained(language_model, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model).to(self.device)
        for param in self.vision_model.parameters():
            param.requires_grad_(False)
        self.vision_model.eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model)

        # add ia3 adapter
        ia3_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "lm_head"],
            feedforward_modules=["lm_head"]
        )
        self.llm = get_peft_model(self.llm, ia3_config)
        print(self.llm)
        print("trainable params:", list(n for n, _ in filter(lambda x: x[1].requires_grad, self.llm.named_parameters())))
        
        # adapter between vision and llm
        embed_dim = self.llm.config.hidden_size
        self.vision_adapter = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=1,
            bias=False,
            batch_first=True,
            device=self.device,
        )
        self.vision_linear = torch.nn.Sequential(
            torch.nn.Linear(self.vision_model.config.hidden_size, embed_dim, device=self.device),
            torch.nn.LayerNorm(embed_dim, device=self.device),
        )
        self.num_queries = num_queries
        self.queries = torch.nn.Parameter(0.1 * torch.randn((1, num_queries, embed_dim), device=self.device))

        # position embeddings
        grid_size = self.vision_model.config.image_size // self.vision_model.config.patch_size
        self.pos_embed = torch.nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True)).float(),
            requires_grad=False,
        ).unsqueeze(dim=0).to(self.device)

    def get_image_tokens(self, image: torch.Tensor) -> torch.Tensor:
        # get image features and tokens for llm
        image_features = self.vision_linear(self.vision_model(image)[0])
        # print("self.queries:", self.queries.shape)
        # print("image_features:", image_features.shape)
        # print("self.pos_embed:", self.pos_embed.shape)
        image_tokens = self.vision_adapter(
            query=self.queries.repeat((image.shape[0], 1, 1)), 
            key=image_features + self.pos_embed, 
            value=image_features
        )[0]
        # print("image_tokens:", image_tokens.shape)
        return image_tokens

    def forward(self, image: torch.Tensor, text_input: list[str], text_output: list[str]) -> torch.Tensor:
        # get image tokens for llm
        image_embeds = self.get_image_tokens(image)
        # tokenize and embed texts
        input_tokens = self.tokenizer(text=[f"\nQuestion: {inp}\nAnswer:" for inp in text_input], padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        input_lengths = torch.sum(input_tokens.attention_mask, dim=1)
        full_tokens = self.tokenizer(text=[f"\nQuestion: {inp}\nAnswer: {out}" for inp, out in zip(text_input, text_output)], padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        # full_embeds = self.llm.model.model.embed_tokens(full_tokens.input_ids)
        full_embeds = self.llm.get_input_embeddings()(full_tokens.input_ids)
        # cat image tokens and text tokens
        inputs_embeds = torch.cat([image_embeds, full_embeds], dim=1)
        inputs_attention_mask = torch.cat([
            torch.ones(image_embeds.shape[:-1], device=self.device),
            full_tokens.attention_mask,
        ], dim=1)
        # create labels, ignoring all but tokens corresponding to `text_output`
        batch_size, num_tokens = full_tokens.input_ids.shape[:2]
        mask_input_tokens = torch.arange(num_tokens, device=self.device).reshape(-1, num_tokens).repeat((batch_size, 1)) < input_lengths.unsqueeze(1)
        labels = torch.cat([
            -100 * torch.ones(image_embeds.shape[:-1], device=self.device, dtype=torch.long),  # ignore image tokens in loss
            torch.where(mask_input_tokens, -100, full_tokens.input_ids),  # ignore tokens from `text_input` in loss
        ], dim=1)
        labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)  # ignore all padding
        # print("labels:", labels, labels.shape)

        output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs_attention_mask,
            labels=labels,
        )
        return output.loss


if __name__ == "__main__":
    # create inputs
    from PIL import Image
    import tqdm
    image = Image.open("cat.jpg")
    text_input = ["what is in this image?", "what does a cat do?"]
    text_output = ["cat", "cat says meow!"]
    # create model
    model = LVLM(language_model="stabilityai/stablelm-3b-4e1t", vision_model="openai/clip-vit-large-patch14")
    # model = LVLM(language_model="facebook/opt-1.3b", vision_model="openai/clip-vit-base-patch32")
    image = model.image_processor([image, image], return_tensors="pt")["pixel_values"].to(model.device)
    print("image:", image.shape)
    output = model(image, text_input, text_output)
    print("loss:", output)

    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=1e-4)
    for i in (bar := tqdm.tqdm(range(10))):
        optimizer.zero_grad()
        loss = model(image, text_input, text_output)
        loss.backward()
        optimizer.step()

        bar.set_description(f"Training progress")
        bar.set_postfix({"iter": i, "loss": loss.cpu().item()})
