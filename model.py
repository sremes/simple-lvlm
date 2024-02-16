from utils import get_device, get_2d_sincos_pos_embed

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPImageProcessor,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

from typing import Optional
import os
from pathlib import Path


class Captioner(torch.nn.Module):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.device, self.dtype = device, dtype

    def forward(
        self,
        image_features: torch.Tensor,
        texts: list[str],
        embeddings: torch.nn.Module,
    ) -> torch.Tensor:
        # get targets from caption/target text
        text_tokens = self.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        targets = embeddings(text_tokens.input_ids)
        # compute matching loss between image tokens and text tokens
        # by matching each image token to most similar text token
        costs = torch.cdist(image_features, targets, p=2)
        smallest_distances = torch.min(costs, dim=-1).values
        return smallest_distances.mean()


class LVLM(torch.nn.Module):
    def __init__(
        self,
        language_model: str,
        vision_model: str,
        num_queries: int = 32,
        device: Optional[torch.device] = None,
    ) -> None:
        # initialize module
        super().__init__()
        self.device = get_device() if not device else device

        # load the LLM
        if self.device.type == "cuda":
            # "cuda" supports quantization, use it!
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                language_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
                quantization_config=quantization_config,
                # attn_implementation="flash_attention_2",
            )
        else:
            # load without quantization that requires "cuda" device
            self.llm = AutoModelForCausalLM.from_pretrained(
                language_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
            ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            language_model, token=os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # add PEFT adapter
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "lm_head"],
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        self.llm = get_peft_model(self.llm, peft_config)

        # load the CLIP vision model
        if self.device.type == "cuda":
            # "cuda" supports quantization, use it!
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.vision_model = CLIPVisionModel.from_pretrained(
                vision_model,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
            )
        else:
            self.vision_model = CLIPVisionModel.from_pretrained(
                vision_model,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        for param in self.vision_model.parameters():
            param.requires_grad_(False)
        self.vision_model.eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model)

        # create adapter between vision and llm models
        embed_dim = self.llm.config.hidden_size
        self.vision_adapter = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=16,
            bias=False,
            batch_first=True,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.post_vision_norm = torch.nn.LayerNorm(
            embed_dim, eps=1e-4, dtype=torch.bfloat16, device=self.device
        )
        self.vision_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                embed_dim, 2 * embed_dim, device=self.device, dtype=torch.bfloat16
            ),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(
                2 * embed_dim, embed_dim, device=self.device, dtype=torch.bfloat16
            ),
            torch.nn.Dropout(0.1),
        )
        self.vision_mlp_norm = torch.nn.LayerNorm(
            embed_dim, eps=1e-4, dtype=torch.bfloat16, device=self.device
        )
        self.vision_linear = torch.nn.Sequential(
            torch.nn.Linear(
                self.vision_model.config.hidden_size,
                embed_dim,
                device=self.device,
                dtype=torch.bfloat16,
            ),
            torch.nn.Dropout(0.1),
            torch.nn.LayerNorm(embed_dim, device=self.device, dtype=torch.bfloat16),
        )
        self.num_queries = num_queries
        self.queries = torch.nn.Parameter(
            0.1
            * torch.randn(
                (1, num_queries, embed_dim), device=self.device, dtype=torch.bfloat16
            )
        )

        # captioning-like loss
        #self.captioner = Captioner(
        #    self.tokenizer, device=self.device, dtype=torch.bfloat16
        #)

    def get_image_tokens(self, image: torch.Tensor) -> torch.Tensor:
        # get image features and tokens for llm
        image_features = self.vision_linear(self.vision_model(image)[0])
        queries = self.queries.repeat((image.shape[0], 1, 1))
        image_tokens = (
            queries
            + self.vision_adapter(
                query=queries,
                key=image_features,  # + self.pos_embed,
                value=image_features,
            )[0]
        )
        image_tokens = self.post_vision_norm(image_tokens)
        image_tokens = image_tokens + self.vision_mlp(image_tokens)
        return self.vision_mlp_norm(image_tokens)

    def forward(
        self, image: torch.Tensor, text_input: list[str], text_output: list[str]
    ) -> torch.Tensor:
        # get image tokens for llm
        image_embeds = self.get_image_tokens(image)
        # tokenize and embed texts
        start_tokens = self.tokenizer(
            text=["<|user|>\n"] * len(text_input), return_tensors="pt"
        ).to(self.device)
        start_embeds = self.llm.get_input_embeddings()(start_tokens.input_ids)
        input_tokens = self.tokenizer(
            text=[f"\n\n{inp}<|endoftext|>\n<|assistant|>\n" for inp in text_input],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        input_lengths = torch.sum(input_tokens.attention_mask, dim=1)
        full_tokens = self.tokenizer(
            text=[
                f"\n\n{inp}<|endoftext|>\n<|assistant|>\n{out}<|endoftext|>"
                for inp, out in zip(text_input, text_output)
            ],
            padding=True,
            truncation=True,
            max_length=128 + 32,
            return_tensors="pt",
        ).to(self.device)
        full_embeds = self.llm.get_input_embeddings()(full_tokens.input_ids)
        # cat image tokens and text tokens
        inputs_embeds = torch.cat([start_embeds, image_embeds, full_embeds], dim=1)
        inputs_attention_mask = torch.cat(
            [
                start_tokens.attention_mask,
                torch.ones(image_embeds.shape[:-1], device=self.device),
                full_tokens.attention_mask,
            ],
            dim=1,
        )
        # create labels, ignoring all but tokens corresponding to `text_output`
        batch_size, num_tokens = full_tokens.input_ids.shape[:2]
        mask_output_tokens = torch.arange(num_tokens, device=self.device).reshape(
            -1, num_tokens
        ).repeat((batch_size, 1)) >= input_lengths.unsqueeze(1)
        labels = torch.cat(
            [
                -100
                * torch.ones(
                    start_embeds.shape[:-1], device=self.device, dtype=torch.long
                ),  # ignore start tokens
                -100
                * torch.ones(
                    image_embeds.shape[:-1], device=self.device, dtype=torch.long
                ),  # ignore image tokens in loss
                torch.where(
                    (mask_output_tokens * full_tokens.attention_mask).bool(),
                    full_tokens.input_ids,
                    -100,
                ),  # ignore tokens from `text_input` in loss, and tokens in padded part
            ],
            dim=1,
        )

        output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs_attention_mask,
            labels=labels,
        )

        #cap_loss = self.captioner(
        #    image_embeds, text_output, self.llm.get_base_model().get_input_embeddings()
        #)

        return output.loss  # + 0.1 * cap_loss

    def generate(self, image: torch.Tensor, question: str, **kwargs) -> str:
        """Generate answer with given image and question."""
        # get image tokens for llm
        image_embeds = self.get_image_tokens(image)

        # tokenize and embed texts
        start_tokens = self.tokenizer(
            text=["<|user|>\n"], return_tensors="pt"
        ).to(self.device)
        start_embeds = self.llm.get_input_embeddings()(start_tokens.input_ids)

        prompt_tokens = self.tokenizer(
            text=[f"\n\n{question}<|endoftext|>\n<|assistant|>\n"],
            return_tensors="pt",
        ).to(self.device)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_tokens.input_ids)

        # cat image tokens and text tokens
        inputs_embeds = torch.cat([start_embeds, image_embeds, prompt_embeds], dim=1)

        # call the llm generate implementation
        output = self.llm.generate(
            inputs_embeds=inputs_embeds,
            **kwargs,
        )[0]
        return self.tokenizer.decode(output, skip_special_tokens=False)


if __name__ == "__main__":
    # create inputs
    from PIL import Image
    import tqdm

    image = Image.open("cat.jpg")
    text_input = ["what is in this image?", "what does a cat do?"]
    text_output = ["cat", "cat says meow!"]
    # create model
    model = LVLM(
        language_model="stabilityai/stablelm-3b-4e1t",
        vision_model="openai/clip-vit-large-patch14",
    )
    print(model.llm)
    print(
        "trainable params:",
        list(
            n for n, _ in filter(lambda x: x[1].requires_grad, model.named_parameters())
        ),
    )
    image = model.image_processor([image, image], return_tensors="pt")[
        "pixel_values"
    ].to(model.device)
    print("image:", image.shape)
    output = model(image, text_input, text_output)
    print("loss:", output)
