# Language Model
## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

### Train byte-level BPE (TinyStories, vocab 10k + `<|endoftext|>`)

With **`datasets/tinystories/train.csv`** in place (default), run from the repo root:

```sh
uv run python -m eecs148b_hw1.tokenizer.train
```

Each CSV row’s `text` field is one story; stories are concatenated with **`<|endoftext|>`** between them. Outputs:

- `out/tinystories_vocab.json`
- `out/tinystories_merges.txt`

For a plain **`.txt`** corpus instead, pass `--input path/to/file.txt`. If that file does not contain `<|endoftext|>`, blank-line-separated blocks are joined with the special token unless you pass `--no-inject-eot`. Override column name with `--text-column` for nonstandard CSVs.

### Sample documents and compression ratio

With `out/tinystories_vocab.json`, `out/tinystories_merges.txt`, and `datasets/tinystories/train.csv` in place:

```sh
uv run python -m eecs148b_hw1.experiments.tokenizer
```

This reservoir-samples 10 stories, encodes each with the trained tokenizer (prints leading token IDs), and prints the aggregate **UTF-8 bytes per token**. Use `--encode-splits` only if you also want full train/val `.npy` token files (large memory use).

