import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from transformers import GPT2LMHeadModel


from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np


def load_data(path: str) -> tuple[np.array, np.array]:
    data = pd.read_csv(f"{path}/labels.csv", delimiter="|", skipinitialspace=True)
    return data["image_name"].tolist(), data["comment"].tolist()


class FlickrDataset(Dataset):
    def __init__(
        self,
        data=None,
        root_dir=None,
        labels_file=None,
        tokenizer_model="gpt2",
        eos_token="<|endoftext|>",
        transform=None,
        max_length=256,
    ):
        """
        Args:
            data (tuple): tuple of (images, captions) arrays. Used if `labels_file` is None.
            root_dir (str or Path): directory containing the images.
            labels_file (str or Path): CSV file with `image_name` and `comment` columns.
            tokenizer_model (str): tokenizer model to use.
            eos_token (str): end-of-sequence token for captions.
            transform (callable): transformations to apply to the images.
            max_length (int): maximum tokenized caption length.
        """
        self.transform = transform or transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.bos_token = self.tokenizer.eos_token
        self.eos_token = eos_token
        self.max_length = max_length

        if labels_file is not None:
            self.data_mode = "csv"
            self.root_dir = Path(root_dir)
            self.data = pd.read_csv(labels_file)
        elif data is not None:
            self.data_mode = "arrays"
            self.images = data[0]
            self.captions = data[1]
            self.data_folder = root_dir
        else:
            raise ValueError("Either `data` or `labels_file` must be provided.")

    def __len__(self):
        if self.data_mode == "csv":
            return len(self.data)
        elif self.data_mode == "arrays":
            return len(self.images)

    def __getitem__(self, idx):
        if self.data_mode == "csv":
            image_path = self.root_dir / "images" / self.data.iloc[idx]["image_name"]
            caption = self.data.iloc[idx]["comment"]
        else:  # arrays
            image_path = f"{self.data_folder}/{self.images[idx]}"
            caption = self.captions[idx]

        # Process image
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)

        # Process caption
        caption = self.tokenizer.bos_token + caption + self.tokenizer.eos_token
        tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return image, input_ids, attention_mask


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert (
            self.embed_dim % self.n_heads == 0
        ), "embedding dimension by be divisible by number of heads"
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.scale = self.head_size**-0.5

        self.register_buffer(
            "mask", torch.tril(torch.ones(1, 1, self.seq_len, self.seq_len))
        )

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_size).transpose(1, 2)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)

        attention = (attention @ v).transpose(1, 2).contiguous().view(b, t, c)

        out = self.proj(attention)
        out = self.resid_dropout(out)

        return out


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert (
            self.embed_dim % self.n_heads == 0
        ), "embedding dimension by be divisible by number of heads"
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.k = nn.Linear(self.embed_dim, self.embed_dim)
        self.v = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale = self.head_size**-0.5

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, q, k, v):
        b, t, c = q.shape

        q = self.q(q).view(b, t, self.n_heads, self.head_size).transpose(1, 2)
        k = self.k(k).view(b, -1, self.n_heads, self.head_size).transpose(1, 2)
        v = self.v(v).view(b, -1, self.n_heads, self.head_size).transpose(1, 2)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)
        attention = (attention @ v).transpose(1, 2).contiguous().view(b, t, c)

        out = self.proj(attention)
        out = self.resid_dropout(out)

        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.mlp_dropout = config.mlp_dropout

        self.fc = nn.Linear(self.embed_dim, self.embed_dim * self.mlp_ratio)
        self.proj = nn.Linear(self.embed_dim * self.mlp_ratio, self.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.mlp_dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = CrossAttention(config)

    def forward(self, x, enc_out):
        x = x + self.attn(self.ln_1(x))
        x = x + self.cross_attn(self.ln_2(x), enc_out, enc_out)
        x = x + self.mlp(self.ln_3(x))
        return x


class CaptioningModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        vit = create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.patch_embed = vit.patch_embed
        num_patches = self.patch_embed.num_patches

        self.cls_token = vit.cls_token
        embed_len = num_patches + vit.num_prefix_tokens
        self.pos_embed = vit.pos_embed
        self.pos_drop = nn.Dropout(p=0.0)

        self.blocks = nn.ModuleList([vit.blocks[i] for i in range(config.depth)])

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.embed_dim),
                wpe=nn.Embedding(config.seq_len, config.embed_dim),
                drop=nn.Dropout(config.emb_dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.depth)]),
                ln_f=nn.LayerNorm(config.embed_dim),
            )
        )
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def _pos_embed(self, x):
        pos_embed = self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + pos_embed
        return self.pos_drop(x)

    def pretrained_layers_trainable(self, trainable=False):
        layers = [
            self.cls_token,
            self.patch_embed,
            self.pos_embed,
            self.blocks,
            self.transformer.wte,
            self.transformer.wpe,
            self.transformer.ln_f,
            self.lm_head,
        ]
        gpt_layers = [
            [
                self.transformer.h[i].ln_1,
                self.transformer.h[i].ln_2,
                self.transformer.h[i].attn,
                self.transformer.h[i].mlp,
            ]
            for i in range(self.config.depth)
        ]
        for l in gpt_layers:
            layers.extend(l)

        for layer in layers:
            if not isinstance(layer, nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = trainable
            else:
                layer.requires_grad = trainable

        total_frozen_params = sum(
            [p.numel() for p in self.parameters() if not p.requires_grad]
        )
        print(f"{total_frozen_params=}")

    def unfreeze_gpt_layers(
        self,
    ):
        gpt_layers = [
            [
                self.transformer.h[i].ln_1,
                self.transformer.h[i].ln_2,
                self.transformer.h[i].attn,
                self.transformer.h[i].mlp,
            ]
            for i in range(self.config.depth)
        ]
        flatten = []
        for l in gpt_layers:
            flatten.extend(l)

        for layer in flatten:
            if not isinstance(layer, nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = True
            else:
                layer.requires_grad = True

    @classmethod
    def from_pretrained(self, config):
        model = CaptioningModel(config)
        sd = model.state_dict()
        keys = sd.keys()
        ignore_matches = [
            "blocks.",
            "cross_attn.",
            "ln_3",
            "cls_token",
            "pos_embed",
            "patch_embed.",
            ".attn.mask",
        ]
        vit_keys = [
            key for key in keys if any(match in key for match in ignore_matches)
        ]
        gpt_keys = [key for key in keys if key not in vit_keys]

        gpt2_small = GPT2LMHeadModel.from_pretrained("gpt2")
        sd_hf = gpt2_small.state_dict()
        hf_keys = sd_hf.keys()
        hf_keys = [k for k in hf_keys if not k.endswith(".attn.masked_bias")]
        hf_keys = [k for k in hf_keys if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        for k in hf_keys:
            if any(match in k for match in ignore_matches):
                continue
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        model.load_state_dict(sd)

        return model

    def forward(self, image, input_ids, labels=None):
        image = self.patch_embed(image)
        image = self._pos_embed(image)

        token_embeddings = self.transformer.wte(input_ids)  # batch x seq_len
        pos_embs = torch.arange(0, input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.transformer.wpe(pos_embs)
        input_ids = self.transformer.drop(token_embeddings + positional_embeddings)

        for i in range(self.config.depth):
            image = self.blocks[i](image)
            input_ids = self.transformer.h[i](input_ids, image)

        input_ids = self.transformer.ln_f(input_ids)
        lm_logits = self.lm_head(input_ids)

        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return loss

        return lm_logits

    def generate(
        self, image, sequence, max_tokens=50, temperature=1.0, deterministic=False
    ):
        self.eval()
        with torch.no_grad():
            image = self.patch_embed(image)
            image = self._pos_embed(image)
            for i in range(self.config.depth):
                image = self.blocks[i](image)

            batch_size = sequence.size(0)
            device = sequence.device
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_tokens):
                token_embeddings = self.transformer.wte(sequence)
                pos_ids = (
                    torch.arange(0, sequence.size(1), device=device)
                    .unsqueeze(0)
                    .expand_as(sequence)
                )
                positional_embeddings = self.transformer.wpe(pos_ids)
                hidden_states = self.transformer.drop(
                    token_embeddings + positional_embeddings
                )

                for i in range(self.config.depth):
                    hidden_states = self.transformer.h[i](hidden_states, image)

                hidden_states = self.transformer.ln_f(hidden_states)
                logits = self.lm_head(hidden_states)[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                if deterministic:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)

                eos_mask = (next_token == self.config.eos_token_id).any(dim=1)
                finished = finished | eos_mask

                next_token = next_token.masked_fill(
                    eos_mask.unsqueeze(1), self.config.eos_token_id
                )
                sequence = torch.cat((sequence, next_token), dim=-1)

                if finished.all():
                    break

        return sequence
