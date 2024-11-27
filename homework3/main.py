import os
import click
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
import torch
from transformers import get_linear_schedule_with_warmup, GPT2TokenizerFast
from nltk.translate.bleu_score import sentence_bleu

from model import CaptioningModel, FlickrDataset, load_data
from utils import visualize_samples, plot_metrics, plot_sample_predictions


model_config = SimpleNamespace(
    vocab_size=50_257,
    embed_dim=768,
    num_heads=12,
    seq_len=1024,
    depth=12,
    attention_dropout=0.1,
    residual_dropout=0.1,
    mlp_ratio=4,
    mlp_dropout=0.1,
    emb_dropout=0.1,
)


def collate_fn(samples, padding_value=0):
    img_list, ids_list, masks_list = zip(*samples)
    image_batch = torch.stack(img_list)
    ids_batch = pad_sequence(ids_list, batch_first=True, padding_value=padding_value)
    masks_batch = pad_sequence(
        masks_list, batch_first=True, padding_value=padding_value
    )

    return image_batch, ids_batch, masks_batch


@click.command()
@click.option("--data_folder", type=str, default="./data")
@click.option("--bs", type=int, default=32)
@click.option("--device", type=str, default="cuda")
@click.option("--n_epochs", type=int, default=10)
@click.option("--lr", type=float, default=1e-4)
def main(data_folder, bs, device, n_epochs, lr):
    data = load_data(data_folder)
    dataset = FlickrDataset(data, f"{data_folder}/flickr30k_images")

    # Create datasets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn
    )

    # Visualize a few images and captions
    visualize_samples(train_dataloader, count=4, text="visualize_samples_train")
    visualize_samples(val_dataloader, count=4, text="visualize_samples_val")

    # Create model
    model = CaptioningModel(model_config).to(device)
    model.pretrained_layers_trainable(trainable=False)
    n_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    n_total_params = sum(p.numel() for p in model.parameters())
    print(
        f"trainable parameters: {n_trainable_params}, total parameters: {n_total_params}"
    )

    # Create tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model_config.eos_token_id = tokenizer.eos_token_id

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * n_epochs,
    )

    # Define training & validation loop
    train_losses = list()
    train_perplexities = list()

    # Define validation loop
    val_losses = list()
    val_perplexities = list()
    val_bleus = list()

    for epoch in range(n_epochs):
        try:
            # Training phase
            model.train()
            total_train_loss = 0
            total_train_perplexity = 0

            for _, (images, captions, _) in enumerate(train_dataloader):
                images, captions = images.to(device), captions.to(device)
                labels = captions.clone()
                optimizer.zero_grad()

                # Compute loss and backpropagate
                loss = model(images, captions, labels=labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()
                total_train_perplexity += torch.exp(loss).item()

            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_train_perplexity = total_train_perplexity / len(train_dataloader)
            train_losses.append(avg_train_loss)
            train_perplexities.append(avg_train_perplexity)

            print(
                f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Train Perplexity: {avg_train_perplexity:.4f}"
            )

            # Validation phase
            model.eval()
            total_val_loss = 0
            total_val_perplexity = 0
            bleu_scores = []

            with torch.no_grad():
                for _, (images, captions, _) in enumerate(val_dataloader):
                    images, captions = images.to(device), captions.to(device)
                    labels = captions.clone()
                    loss = model(images, captions, labels=captions)

                    total_val_loss += loss.item()
                    total_val_perplexity += torch.exp(loss).item()

                    # Generate predictions and calculate BLEU scores
                    bos_tokens = torch.full(
                        (captions.size(0), 1), tokenizer.bos_token_id, dtype=torch.long
                    ).to(device)
                    predictions = model.generate(
                        images, sequence=bos_tokens, max_tokens=50
                    )
                    for pred, target in zip(predictions, captions):
                        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
                        target_text = tokenizer.decode(target, skip_special_tokens=True)
                        bleu_scores.append(
                            sentence_bleu([target_text.split()], pred_text.split())
                        )

            avg_val_loss = total_val_loss / len(val_dataloader)
            avg_val_perplexity = total_val_perplexity / len(val_dataloader)
            avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

            val_losses.append(avg_val_loss)
            val_perplexities.append(avg_val_perplexity)
            val_bleus.append(avg_bleu_score)

            print(
                f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}, Val Perplexity: {avg_val_perplexity:.4f}, BLEU: {avg_bleu_score:.4f}"
            )

            # generate intermediate predictions
            if epoch % 5 == 0 or epoch == n_epochs - 1:
                sample_texts = []
                with torch.no_grad():
                    for _, (images, _, _) in enumerate(val_dataloader):
                        images = images.to(device)
                        bos_tokens = torch.full(
                            (5, 1), tokenizer.bos_token_id, dtype=torch.long
                        ).to(device)
                        predictions = model.generate(
                            images[:5], sequence=bos_tokens, max_tokens=50
                        )

                        sample_texts = [
                            tokenizer.decode(pred, skip_special_tokens=True)
                            for pred in predictions
                        ]
                        break  # only generate for the first batch

                print(f"Sample Predictions: {sample_texts}")

        except KeyboardInterrupt:
            print("Training interrupted.")
            break

    # Plot metrics
    plot_metrics(train_losses, val_losses)
    # Plot sample predictions
    plot_sample_predictions(model, tokenizer, device, val_dataloader)


if __name__ == "__main__":
    main()
