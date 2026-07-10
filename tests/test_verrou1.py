import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "I absolutely love this movie but the ending was terrible."

inputs = tokenizer(
    text,
    return_tensors="pt",
    padding="max_length",
    max_length=20,
    truncation=True
)

with torch.no_grad():
    outputs = model.embeddings(
        input_ids=inputs["input_ids"]
    )

embeddings = outputs

baseline = torch.zeros_like(embeddings)

mask_layer = TokenGumbelSoftmaxMask(
    seq_len=20
)

z, mask = mask_layer(
    embeddings,
    baseline,
    training=False,
    attention_mask=inputs["attention_mask"]
)

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("\nTokens et scores :")

for token, score in zip(tokens, mask[0]):
    print(
        f"{token:<15} {score.item():.4f}"
    )