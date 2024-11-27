Image captioning with CLIP and GPT2

1. In PyTorch, write a dataset class for pairs of (image, caption). Add the
   token "<|endoftext|>" to the end of each caption, as required for the GPT-2
   model.

2. Write a collator function or class (link) to preprocess images and captions
   in an appropriate format.

3. Complete the model subclass in a 'model.py' file. Write the code, referring
   to relevant papers or code implementations. Additionally, implement a
   `generate` method that accepts
   `(image, sequence, max_tokens, 
temperature=1.0, deterministic=True/False)`
   as inputs.

4. Develop a training pipeline, and select a suitable loss function with
   justification. Track metrics during the training phase.

5. Plot the metric results and generate a few sample predictions.
