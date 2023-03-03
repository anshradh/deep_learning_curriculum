"""Creates a dataset of shakespeare's works."""
import torch
import re
import requests

response = requests.get("https://www.gutenberg.org/files/100/100-0.txt")
response.encoding = "utf-8-sig"
content = response.text
for char in ["\n", "\r", "\d", "\t"]:
    content = content.replace(char, " ")

split_content = re.split(r"\b", content)
word2idx = dict()
idx2word = dict()
idx = 0
for word in split_content:
    if word not in word2idx:
        word2idx[word] = idx
        idx2word[idx] = word
        idx += 1


def tokenize(text: str):
    split = re.split(r"\b", text)
    return [word2idx[token] for token in split]


def detokenize(indices: torch.Tensor):
    if indices.ndim == 1:
        return "".join([idx2word[idx.item()] for idx in indices])
    elif indices.ndim == 2:
        return ["".join([idx2word[idx.item()] for idx in row]) for row in indices]


if __name__ == "__main__":
    indices = []
    for token in split_content:
        indices.append(word2idx[token])

    chunked_indices = torch.tensor(indices).split(970)
    chunked_indices = [chunk for chunk in chunked_indices if len(chunk) == 970]
    chunked_indices = torch.stack(chunked_indices)
    chunked_indices = chunked_indices[:-1]
    data = dict(
        train=chunked_indices[: chunked_indices.shape[0] // 2],
        val=chunked_indices[chunked_indices.shape[0] // 2 :],
    )
    torch.save(
        chunked_indices,
        "~/deep_learning_curriculum/exercises/transformers/data/shakespeare.pt",
    )
