import torch
import re

f = open("shakespeare.txt", "r")
content = f.read()
split_content = re.split(r"\b", content)
all_words = set(split_content)
word2idx = {word: idx for idx, word in enumerate(all_words)}
idx2word = {idx: word for idx, word in enumerate(all_words)}


def tokenize(text):
    split = re.split(r"\b", text)
    return [word2idx[token] for token in split]


def detokenize(indices):
    return "".join([idx2word[idx.item()] for idx in indices])


if __name__ == "__main__":
    indices = []
    for token in split_content:
        indices.append(word2idx[token])

    chunked_indices = torch.tensor(indices).split(970)
    chunked_indices = [chunk for chunk in chunked_indices if len(chunk) == 970]
    chunked_indices = torch.stack(chunked_indices)
    chunked_indices = chunked_indices[:-1]
    torch.save(chunked_indices, "shakespeare.pt")
