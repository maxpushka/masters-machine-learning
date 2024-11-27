import matplotlib.pyplot as plt
import random
from model import CaptioningModel
from transformers import GPT2TokenizerFast
import torch
from torch.utils.data import DataLoader


def visualize_samples(dataloader, count, text):
    batch = next(iter(dataloader))
    images, input_ids, _ = batch
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    plt.figure(figsize=(16, 12))
    for i in range(min(count, len(images))):
        image = images[i].permute(1, 2, 0).numpy()
        caption = tokenizer.decode(input_ids[i], skip_special_tokens=True)

        plt.subplot(count, 2, 2 * i + 1)
        plt.text(0.5, 0.5, caption, fontsize=12, ha="center", va="center", wrap=True)
        plt.axis("off")

        plt.subplot(count, 2, 2 * i + 2)
        plt.imshow(image)
        plt.axis("off")

    plt.suptitle(text)
    plt.tight_layout()
    plt.savefig(f"results/{text}.png")


def plot_metrics(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/losses.png")


def plot_sample_predictions(
    model: CaptioningModel,
    tokenizer: GPT2TokenizerFast,
    device: str,
    data_loader: DataLoader,
    num_samples: int = 6,
):
    model.eval()
    samples = [
        data_loader.dataset[random.randint(0, len(data_loader.dataset) - 1)]
        for _ in range(num_samples)
    ]
    images, captions, _ = data_loader.collate_fn(samples)
    images = images.to(device)
    captions = captions.to(device)

    bos_tokens = torch.full(
        (num_samples, 1), tokenizer.bos_token_id, dtype=torch.long
    ).to(images.device)
    predictions = model.generate(images, sequence=bos_tokens)
    generated_captions = []
    for pred, target in zip(predictions, captions):
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        generated_captions.append(pred_text)

    cols = int(num_samples**0.5)
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(15, 30))
    for i, (image, target_caption, _) in enumerate(samples):
        plt.subplot(rows, cols, i + 1)
        plt.title(generated_captions[i].replace("<|endoftext|>", ""))
        plt.axis("off")
    plt.savefig("results/predictions.png")
    plt.show()
