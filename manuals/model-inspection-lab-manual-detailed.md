# Lab Manual (Detailed Edition): Inspecting Transformer Models Layer by Layer

**Goal:** see, with your own eyes, every major piece of a transformer: tokens, the embedding table, per-token vectors, attention heads, pooling, the classification head, the decoder's language-model head, and LoRA adapters. All of it happens inside real models you can poke at.

**What is different in this edition:** every lab now has (1) a *Concepts* section with formal definitions, (2) code broken into small cells with a comment on nearly every line, (3) an explanation of what each output means, and (4) the same Record questions and answer key as the short edition (question numbers match).

**Models used**
- `microsoft/deberta-v3-base`: an **encoder** model, about 184 million parameters.
- `Qwen/Qwen2.5-0.5B`: a **decoder** model, about 0.5 billion parameters.

Both run on a laptop CPU.

**How each lab is laid out**
1. *Objective*: what you will be able to see by the end.
2. *Concepts*: formal definitions of the ideas the lab touches. Read these first.
3. *Procedure*: code cells, run in order. Run the cells in the order given, because later cells reuse variables from earlier ones.
4. *Reading the output*: what each printed value means.
5. *Record*: questions to answer in writing before you look at the answer key.

---

## Contents

- Part 0: Glossary (formal definitions)
- Part 0.5: Python and PyTorch syntax primer
- Lab 0: Setup
- Lab 1: The tokenizer
- Lab 2: Load the encoder and read its blueprint
- Lab 3: Parameters
- Lab 4: The embedding table
- Lab 5: The forward pass
- Lab 6: Attention heads
- Lab 7: Hooks
- Lab 8: The classification head
- Lab 9: The decoder side by side
- Lab 10: LoRA under the microscope
- Lab 11: Representation geometry
- Lab 12: Embedding models and retrieval (the RAG building block)
- Quick reference
- Answer key

**Variables carried between labs.** If you restart the kernel, re-run the cells that define these.

| Variable | Defined in | Used by |
|---|---|---|
| `name`, `tok`, `text`, `enc`, `toks` | Lab 1 | Labs 2 to 8, 10, 11 |
| `base` | Lab 2 | Labs 3 to 7, 11 |
| `clf`, `labels` | Lab 8 | Lab 8, 10 (`labels`) and 11 (`labels`) |
| `dname`, `dtok`, `dec` | Lab 9 | Lab 9 |
| `pm` | Lab 10 | Lab 10 |
| `ename`, `etok`, `emodel`, `embed`, `E` | Lab 12 | Lab 12 (also uses `texts`, `y`, `embed_all` from Lab 11) |

---

# Part 0: Glossary (formal definitions)

Read this once now and come back to it whenever a term is unclear. Terms are grouped by topic. Where a term has a precise mathematical meaning, it is given.

## Data and numbers

**Tensor.** A grid of numbers with any number of axes, all of one data type (for example 32-bit float), described by its *shape*. The number of axes sets what it is called: 0 axes is a single number (a *scalar*), 1 axis is a *vector*, 2 axes is a *matrix*, and 3 or more axes is usually just called a tensor. "Tensor" is also the general word that covers all of these. PyTorch's tensor class is `torch.Tensor`.

**Shape.** The tuple listing the size of each axis. The number of entries in the tuple is the number of axes. `(768,)` has 1 axis (a vector of 768 numbers), `(5, 768)` has 2 (a matrix: 5 rows of 768), and `(2, 5, 768)` has 3 (a 3-axis tensor holding 2 × 5 = 10 vectors of 768 numbers each, 7,680 numbers in all). So `(2, 5, 768)` is a tensor but *not* a vector; it is a stack of vectors. Throughout this manual, the three axes of most model outputs are written `(batch, tokens, hidden)`.

**Batch.** A group of inputs processed together in one call. Batch size 1 means one text at a time. The *batch dimension* is always the first dimension.

**Vector.** An ordered list of numbers: a tensor with exactly one axis, shape `(n,)`. In this manual, a vector is usually the 768-number (encoder) or 896-number (decoder) description the model keeps for one token at one point in the network. A model's output for a whole text is many such vectors stacked into a tensor with more axes.

**Dimension.** This word has two meanings, and which one is meant depends on what it describes:

- **Of a tensor: an axis.** "A 3-dimensional tensor" means a tensor with 3 axes, such as `(batch, tokens, hidden)`. Here the count of dimensions is the length of the shape tuple.
- **Of a vector: its length.** "A 768-dimensional vector" means a vector containing 768 numbers. Here the dimension is the size of its one axis.

The two combine without conflict: a `(2, 5, 768)` tensor is 3-dimensional (3 axes), and each of the vectors inside it is 768-dimensional (768 numbers).

### Picture: tensor, shape, vector, dimension

<!-- figure:tensor-shapes:start -->
```text
 scalar        vector          matrix             3-axis tensor
 shape ()      shape (4,)      shape (3, 4)       shape (2, 3, 4)
 0 axes        1 axis          2 axes             3 axes

 [ 7 ]         [# # # #]       [# # # #]  <- one vector     two such matrices
                               [. . . .]                     stacked: 6 vectors
                               [. . . .]                     of 4 numbers each
```
<!-- figure:tensor-shapes:end -->

How to read it (the `#` cells are one vector):

- **Tensor** is the whole family. Every picture above is a tensor; the name only stops being special-purpose once there are 3 or more axes.
- **Shape** is the label under each picture: one size per axis. `(3, 4)` means 3 rows and 4 columns.
- **Vector** is the 1-axis picture. Inside the matrix and the 3-axis tensor, every row is also a vector of 4 numbers.
- **Dimension, sense 1 (axes).** Count the axes: 0, 1, 2, 3. "A 3-dimensional tensor" has the shape `(2, 3, 4)`.
- **Dimension, sense 2 (length of a vector).** The 4 numbers in each row. "A 4-dimensional vector" has 4 entries.
- **In the model:** read the 3-axis shape as `(batch, tokens, hidden) = (2, 3, 4)`: 2 texts, 3 tokens each, 4 numbers per token. The real encoder uses `hidden = 768`.

**Parameter (also called a weight).** A number inside the model that is learned during training and stays fixed afterwards. "184M parameters" means about 184 million such numbers in total.

**Hyperparameter.** A setting that a person chooses *before* training and that is *not* learned by training. It fixes the model's structure (number of layers, vector width, number of heads) or how training runs (learning rate, batch size). Contrast with *parameter*, which training does learn. These values are stored in the model's *config*.

**Config.** The record of a model's hyperparameters (`hidden_size`, `num_hidden_layers`, and so on). Stored as a JSON file next to the weights.

## Text to numbers

**Token.** The unit of text the model works with. It can be a whole word, part of a word, a single character, or a special marker. Tokens are decided by the tokenizer, not by humans reading the text.

**Vocabulary.** The fixed list of every token the tokenizer knows, each paired with a unique integer **ID** (a *token ID*). The number of entries is the *vocabulary size*.

**Tokenizer.** The program that converts text into a list of token IDs (*encoding*) and back (*decoding*). Its vocabulary is fixed at training time and must match the model.

**Subword tokenization.** A scheme where common words are single tokens and rare words are split into smaller known pieces. This lets a fixed vocabulary cover any text. DeBERTa-v3 uses SentencePiece; Qwen uses byte-level BPE (byte-pair encoding).

**`▁` (the "lower one eighth block").** SentencePiece's visible marker for "a space came before this piece". `▁crashed` means " crashed" as a new word. A piece without it continues the previous word.

**Special tokens.** Reserved tokens that carry structural meaning rather than text. In DeBERTa: `[CLS]` (classification: placed first, and later used as the summary position), `[SEP]` (separator: placed last, or between two segments), `[PAD]` (padding: filler to make texts in a batch equal length), `[UNK]` (unknown).

**Attention mask.** A tensor of 1s and 0s, one entry per token. 1 means "real token, use it", 0 means "padding, ignore it".

**`input_ids`.** The tensor of token IDs. Shape `(batch, tokens)`.

## Inside the model

**Embedding table (embedding matrix).** A matrix of shape `(vocab_size, hidden_size)`. Row *i* is the starting vector for token ID *i*. Converting IDs to vectors is a pure row lookup: `vectors = E[input_ids]`. The table's values are learned parameters, trained together with the rest of the model. Not to be confused with an *embedding model* (see *Search and retrieval*), which is a whole model that turns a full text into one vector.

**Hidden state.** The vector the model holds for a token at a given depth. "Hidden" means *internal*: it is neither the raw input you give the model nor the final output you read from it. It says nothing about whether the vector contains context. The embedding stage's output (the input to the first layer) is the starting hidden state: it is static, with no context yet, but it still counts because it is internal and is what layer 1 transforms. Each layer then produces a new set of hidden states. Shape `(batch, tokens, hidden)`.

**Hidden size (`d_model`).** The dimension of a hidden state. 768 for `deberta-v3-base`, 896 for `Qwen2.5-0.5B`.

**Layer (transformer block).** One repeated unit of the network. Each layer takes hidden states in and gives hidden states of the same shape out. It contains an attention sub-block and an MLP sub-block, each wrapped in a residual connection and normalization. DeBERTa-v3-base has 12 layers; Qwen2.5-0.5B has 24.

**Linear layer (`nn.Linear`).** The operation `y = x Wᵀ + b`. The weight matrix `W` has shape `(out_features, in_features)`; `b` is a bias vector of length `out_features`. It maps a vector of length `in_features` to one of length `out_features`.

**Attention.** The mechanism by which every token builds a new vector as a weighted average of other tokens' vectors. For each token, the model computes a **query** (what am I looking for), and each token provides a **key** (what do I contain) and a **value** (what I contribute). The weight that token *i* gives token *j* is

```
weight(i, j) = softmax over j of ( query_i · key_j / sqrt(head_dim) )
```

and the new vector for token *i* is `Σ_j weight(i, j) × value_j`. The result for one sequence is a `tokens × tokens` matrix of weights whose rows each sum to 1.

**Projection.** A linear layer used to re-express a vector as a new vector (a new "view" of it). Every output number is a weighted sum of *all* the input numbers plus a bias: `y = x Wᵀ + b`. Row *k* of `W` is the recipe for output number *k*: how much of each input to take. The weights are learned. A projection can keep the length (768 → 768) or change it (768 → 3072).

**Q / K / V projections.** Three linear layers (`query_proj`, `key_proj`, `value_proj` in DeBERTa; `q_proj`, `k_proj`, `v_proj` in Qwen) that turn each hidden state into its query, key, and value vectors. A token needs three roles: as a *query* it says what it is looking for, as a *key* it says what it offers to be matched against, and as a *value* it carries the content passed along when it is attended to. Each role has its own learned recipes.

**Output projection (mixing).** The linear layer (`attention.output.dense`) applied after the attention heads' results are joined end to end into one 768-number vector. Each of its 768 outputs is a weighted sum of all 768 joined numbers, so it combines what the different heads found ("mixing"). Without it, each head's results would stay in its own 64 slots for the whole network and never combine with the other heads.

**Attention head.** One independent copy of the attention computation operating on a slice of each token's vector (for example 64 of its 768 numbers), found *inside every layer*. The slicing is across the numbers in each vector, not across the tokens: every head looks at all the tokens in the text. With `hidden_size = 768` and `num_attention_heads = 12`, each head works on a 768 / 12 = **64-dimensional** slice (*head dimension*). The 12 heads run in parallel and their outputs are concatenated and mixed by an output projection (`attention.output.dense`). Not to be confused with a *task head* (the output layer at the very end of the model).

**Multi-head attention.** Attention with several heads in parallel, so different heads can learn different relationships (for example, "previous token" or "the subject of this verb").

**Head.** On its own, the word "head" names one of two unrelated things in this manual:

- an **attention head**: one of the parallel copies of the attention computation inside each layer (12 per layer in DeBERTa-v3-base). There are many of them, repeated in every layer.
- a **task head** (a *classification head* or an *LM head*): the small output layer added once, on top of the whole backbone, that turns its vectors into the final prediction.

To tell which is meant: talk of layers, queries and keys, or "12 heads" means attention heads; talk of labels, classes, logits or vocabulary scores, or of "adding a head" to a model, means a task head. The two entries *Attention head* and *Task head* give the exact definitions.

**MLP (feed-forward block).** Two linear layers with an activation function between them. It operates on each token independently. It expands the vector (768 → 3072), applies a nonlinearity, then contracts it back (3072 → 768).

**Activation function.** A nonlinear function applied elementwise. GELU (Gaussian Error Linear Unit) is used in DeBERTa; SiLU is used in Qwen. Without one, stacking linear layers would collapse into a single linear layer.

**Residual connection.** Adding a block's input to its output: `out = x + block(x)`. It lets information and gradients bypass a block and makes deep networks trainable.

**Layer normalization (`LayerNorm`).** Rescales each token's vector to have zero mean and unit variance (then applies a learned scale and shift). It keeps the numbers well-sized from layer to layer. Qwen uses a variant called RMSNorm.

**Precision (dtype).** How many bits are used to store each number. *float32* (32-bit) keeps about 7 significant digits. *float16* (half precision) uses half the memory but keeps only about 3 significant digits, so results are rounded coarsely (near 1.0, the smallest step is about 0.001). Checkpoints are often stored in float16 to save space. Recent `transformers` versions load a model in its stored precision unless you convert it, for example with `.float()`. Exact comparisons such as `torch.allclose(..., atol=1e-5)` only make sense in float32.

**Softmax.** Converts a vector of arbitrary real numbers (*logits*) into probabilities: `softmax(z)_i = exp(z_i) / Σ_j exp(z_j)`. Outputs are positive and sum to 1.

**Logits.** The raw, unnormalized scores a model emits before softmax. Larger means more likely. Logits are not probabilities and can be negative.

**Temperature (T).** A divisor applied to logits before softmax: `softmax(z / T)`. T < 1 sharpens the distribution (more confident); T > 1 flattens it. It never changes which entry is largest.

## Model families and heads

**Encoder model.** A transformer where every token may attend to *every* token, both left and right (*bidirectional* attention). Good for understanding a whole text (classification, search). Example: DeBERTa, BERT.

The word "encode" is used at three different levels, so watch which one is meant:

- **Tokenizer encoding:** text → integer IDs (`tok(text)`, `tok.encode(...)`).
- **Embedding lookup:** IDs → one static vector per token. This is the first stage *inside* an encoder model. It has no attention and no context, so it is not itself an encoder in the transformer sense, even though it is learned (it is trained together with the layers above it, not as a separate model).
- **Encoder model:** the whole stack, which turns static vectors into context-aware ones.

**Decoder model.** A transformer with a *causal mask*: a token may attend only to itself and tokens to its left. Trained to predict the next token. Example: Qwen, GPT, Llama.

**Causal mask.** The rule that forbids attending to the future. In the attention-weight matrix, every entry above the diagonal is forced to 0, giving a lower-triangular matrix.

**Backbone (base model).** The stack of embeddings plus layers, with no task-specific output. Its output is one vector per token, not a prediction.

**Task head (also called an output head).** A small final component, added once on top of the whole backbone, that turns backbone output into the thing you want. A *classification head* maps one vector to class scores. An *LM head* (language-model head) maps each token's vector to a score for every vocabulary entry. Not to be confused with an *attention head*, which is a repeated part inside every layer.

**Pooling.** Reducing the per-token vectors `(tokens, hidden)` to a single vector `(hidden)` for the whole text. Two common choices: **CLS pooling** (take the vector at position 0) and **mean pooling** (average the vectors of the real tokens).

**GQA (grouped-query attention).** An attention variant in which several query heads share one key/value head, to save memory. The config shows `num_key_value_heads` smaller than `num_attention_heads`.

**Weight tying.** Reusing one matrix in two roles. In some decoders the LM head's weight matrix *is* the embedding table, used backwards.

## Training and adapters

**Pretraining.** Training a model on a huge generic corpus so it learns general language structure.

**Fine-tuning.** Continuing training on a smaller, task-specific dataset.

**Frozen parameter.** One that is excluded from training (`requires_grad = False`).

**LoRA (Low-Rank Adaptation).** A fine-tuning method that freezes the original weight `W` and learns a small correction instead:

```
W_effective = W + (alpha / r) × B × A
```

where `A` has shape `(r, in_features)`, `B` has shape `(out_features, r)`, and the **rank** `r` is much smaller than the layer's size. Only `A` and `B` are trained. `B` starts at all zeros so the correction starts at exactly zero.

**PEFT (Parameter-Efficient Fine-Tuning).** The Hugging Face library that implements LoRA and similar methods. It wraps an existing model.

## Search and retrieval

**Embedding model (text embedding).** A complete model that turns a *whole text* into *one* vector, built so that texts with similar meaning get nearby vectors. Usually an encoder plus pooling, trained with a contrastive objective. It is not the same as the *embedding table*: the table maps one token to one static vector, while an embedding model runs all the layers and returns a single context-aware vector for the whole text. Used for search, clustering and RAG.

**Contrastive training.** Training on pairs of texts that belong together (for example a question and a passage that answers it). The loss pulls the vectors of matching pairs closer and pushes non-matching ones apart; the other items in the same batch usually serve as the non-matches. This is what turns a plain encoder into a good embedding model.

**L2 normalization.** Rescaling a vector to length 1 by dividing by its length. After this, the dot product of two vectors equals their cosine similarity, and vector sizes no longer matter. Embedding models usually normalize their output, so searching becomes a single matrix multiply.

**RAG (retrieval-augmented generation).** Answering a question in two stages. First *retrieve* the most relevant pieces of your own text by comparing embeddings. Then paste them into the prompt of a generative language model (a decoder) so it answers using them. Documents are split into *chunks* (paragraph-sized pieces) before embedding. The embedding model and the generating model are two separate models.

**Vector database.** A store that keeps embedding vectors alongside their original text and quickly finds the vectors nearest to a query vector. At small scale it is just a matrix and a sort, as in Lab 12. Large systems use approximate nearest-neighbour indexes to stay fast with millions of vectors.

## Training vocabulary used in passing

**Training vs. inference.** *Training* adjusts parameters to reduce error on examples. *Inference* just runs the finished model to get an output. Every lab here is inference.

**Gradient.** For each parameter, a number saying which direction and how much to change it to reduce the error. Training computes gradients; inference does not need them (hence `set_grad_enabled(False)`).

**Dropout.** A training-time technique that randomly zeroes some values to prevent overfitting. In evaluation mode (`.eval()`) it is switched off.

**Checkpoint.** A saved set of a model's parameters (plus config). Downloading `deberta-v3-base` downloads a checkpoint.

**Hub.** The Hugging Face Hub, the website and file store where checkpoints and tokenizers are published under names like `microsoft/deberta-v3-base`.

**Overfitting.** When a model memorizes its training examples instead of learning patterns that carry over to new ones.

## Inspection tools

**Forward pass.** One run of the model from input to output.

**Forward hook.** A function you attach to a module; PyTorch calls it every time that module finishes computing, handing you the module's inputs and output. It lets you record internal values without editing the model.

**Cosine similarity.** A measure of how aligned two vectors are: `cos(a, b) = (a · b) / (‖a‖ ‖b‖)`. It is 1 for the same direction, 0 for unrelated (orthogonal), −1 for opposite. It ignores vector length.

**PCA (principal component analysis).** A method for squashing high-dimensional points down to 2 or 3 dimensions while keeping as much of the spread as possible, so you can plot them.

**Linear probe.** A simple classifier (here, logistic regression) trained on frozen vectors to test whether some information (such as priority) can be read out of them linearly. High accuracy means that information is present and easy to extract.

---

# Part 0.5: Python and PyTorch syntax primer

The labs use compact Python. This table spells out each construct so none of it is a mystery. Longer equivalents are also written out in comments inside the code.

| You will see | What it means | Long form |
|---|---|---|
| `a, b = f()` | *Tuple unpacking*: the function returns two values, assigned to `a` and `b`. | `r = f(); a = r[0]; b = r[1]` |
| `model(**enc)` | `**` unpacks a dict into keyword arguments. | `model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], ...)` |
| `x[0]`, `x[-1]` | Indexing. `-1` is the last item. | |
| `x[:, 0]` | `:` means "everything along this dimension". Here: all batch items, position 0. | |
| `x[1:]` | Slice: from index 1 to the end. | |
| `x[0, :8]` | Several dimensions at once: item 0, then the first 8 entries. | |
| `f"{v:.3f}"` | *f-string*: embeds a value in text. `:.3f` = 3 decimals, `:2d` = integer padded to width 2, `:15s` = string padded to width 15. | |
| `[g(x) for x in xs]` | *List comprehension*: build a list by applying `g` to each item. | `out = []`<br>`for x in xs:`<br>`    out.append(g(x))` |
| `[x for x in xs if cond(x)]` | Comprehension with a filter. | adds `if cond(x):` before `append` |
| `A if cond else B` | *Conditional expression*: one value or the other. | `if cond:`<br>`    v = A`<br>`else:`<br>`    v = B` |
| `lambda m, i, o: ...` | A tiny unnamed function. | `def f(m, i, o):`<br>`    return ...` |
| `{k: v for k, v in d.items()}` | *Dict comprehension*. | |
| `.item()` | Convert a 1-number tensor to a plain Python number. | |
| `.shape` | The tensor's shape. `tuple(t.shape)` prints it as `(3, 768)`. | |
| `.numel()` | Number of elements (product of the shape). | |
| `.sum(-1)` | Sum along the *last* dimension. `-1` always means last. | |
| `.mean(0)` | Average along dimension 0. | |
| `a @ b` | Matrix multiplication. | |
| `W.T` | Transpose (swap the two dimensions). | |
| `.unsqueeze(0)` | Insert a new size-1 dimension at position 0: `(768,)` becomes `(1, 768)`. | |
| `.clone()` | Make an independent copy, so later changes to the original don't affect it. | |
| `.eval()` | Put the model into evaluation mode (turns off dropout, so output is deterministic). | |
| `with torch.no_grad():` / `torch.set_grad_enabled(False)` | Stop PyTorch recording the data needed for training. Saves memory and time. | |

---

# Lab 0: Setup

**Objective:** start a notebook environment with every library installed and confirm it works.

## Concepts

- **Container.** A packaged, isolated Linux environment. The Jupyter image already has Python and PyTorch installed, so you don't have to configure your machine.
- **Volume mount.** A folder from your computer made visible inside the container. Without it, everything you save is lost when the container stops.
- **Kernel.** The Python process that runs your notebook cells. Variables live in the kernel's memory, so restarting the kernel clears them. Libraries installed during a session are only picked up after a restart.

## Procedure

**Step 1: start the container** (in a terminal on your machine, not in Jupyter).

```bash
# docker and podman take the same arguments here; use whichever you have.
#   -p 8888:8888        publish container port 8888 on your machine's port 8888
#                       (so http://localhost:8888 reaches Jupyter)
#   -v "$PWD":/home/jovyan/work:Z
#                       mount the current directory ($PWD) at /home/jovyan/work
#                       inside the container. The quotes protect paths that contain
#                       spaces. The :Z suffix tells SELinux (on Fedora/RHEL) to relabel
#                       the folder so the container may write to it. Docker on a
#                       non-SELinux system ignores this; leave it on for podman.
#   quay.io/jupyter/pytorch-notebook:2026-09-01
#                       the image name and a date tag (pin the tag so a rebuild of
#                       "latest" can't change your environment).
podman run -p 8888:8888 -v "$PWD":/home/jovyan/work:Z quay.io/jupyter/pytorch-notebook:2026-09-01
```

Open the URL it prints (it includes a one-time token). Put your notebook inside the `work` folder so it is saved on your machine.

*Troubleshooting:* if the notebook cannot save files in `work` under rootless podman, add `--userns=keep-id:uid=1000,gid=100` to the `podman run` line. That maps your user ID onto the container's `jovyan` user.

**Step 2: install libraries** (first notebook cell).

```python
# %pip is a Jupyter "magic" that installs into the *kernel's* Python environment
# (plain "!pip" might install into a different Python).
#   transformers : Hugging Face library with the models and tokenizers
#   peft         : LoRA and other adapter methods (Lab 10)
#   sentencepiece: the tokenizer engine DeBERTa-v3 needs (Lab 1)
#   accelerate   : helper for loading models efficiently
#   scikit-learn : PCA and logistic regression (Lab 11)
%pip install transformers peft sentencepiece accelerate scikit-learn
```

Restart the kernel (Kernel menu → Restart) so the new libraries are picked up.

**Step 3: imports and a sanity check.**

```python
import torch, transformers, peft              # three libraries, imported on one line
import torch.nn.functional as F               # F holds stateless functions, e.g. F.cosine_similarity
import matplotlib.pyplot as plt               # plotting

# By default PyTorch records every operation so it can compute gradients for training.
# We never train in this manual, so turn that off globally: less memory, faster.
torch.set_grad_enabled(False)

print(torch.__version__, transformers.__version__, peft.__version__)
```

## Reading the output

Three version strings and no errors. Disabling gradients is safe for every lab here because nothing trains. (Lab 10 only *builds* a trainable model; it never calls an optimizer.)

---

# Lab 1: The tokenizer (text → integers)

**Objective:** see what the model actually receives. It never sees words, only integer IDs.

## Concepts

Re-read the glossary entries for *token*, *vocabulary*, *subword tokenization*, *special tokens*, and *attention mask*. In short, the tokenizer is a lookup system: text is cut into pieces that exist in a fixed vocabulary, and each piece is replaced by its integer ID. The tokenizer and the model are a matched pair: the embedding table has one row per vocabulary ID, so ID 1234 must mean the same piece to both. Loading `AutoTokenizer.from_pretrained(name)` with the same `name` as the model guarantees they match.

`from_pretrained(name)` accepts either a Hugging Face Hub repo name (as here; it downloads once and caches) or a path to a local folder.

## Procedure

**Step 1: load the tokenizer.**

```python
from transformers import AutoTokenizer

# "microsoft/deberta-v3-base" is a Hub repo: <organization>/<model-name>.
name = "microsoft/deberta-v3-base"

# AutoTokenizer reads the repo's config, works out which tokenizer class this model
# uses (for DeBERTa-v3, a SentencePiece tokenizer), downloads its files (vocabulary,
# special-token settings), and returns the right object.
tok = AutoTokenizer.from_pretrained(name)
```

If this fails with a message about `protobuf` or `tiktoken`, run `%pip install protobuf`, restart the kernel, and retry.

**A warning you may see, and should ignore.** Some recent versions of `transformers` print a warning when loading this tokenizer, saying it has an "incorrect regex pattern" and suggesting `fix_mistral_regex=True`. For DeBERTa-v3 that warning is a false alarm. In a test, the default tokenizer gave the correct pieces (`['▁Prod', '▁checkout', '▁failing', ...]`), while adding the flag broke it, splitting the text into fragments and bare spaces (`['Pro', 'd', ' ', 'checkout', ...]`). Do **not** add the flag. If you see the warning, check that your `tok.tokenize(...)` output looks like the examples in this lab and carry on.

**Step 2: encode a sentence.**

```python
text = "Prod checkout failing for all customers after deploy"

# Calling the tokenizer object does the whole encoding job:
#   1. split the text into vocabulary pieces
#   2. add special tokens ([CLS] at the start, [SEP] at the end)
#   3. replace each piece with its integer ID
#   4. build the attention mask
# return_tensors="pt" asks for PyTorch tensors ("pt" = PyTorch; "tf", "np", "jax" also exist).
# Without it you would get plain Python lists.
enc = tok(text, return_tensors="pt")

# enc is a BatchEncoding, which behaves like a dict. Its keys are the named inputs
# the model expects.
print(enc.keys())
# expect: input_ids, token_type_ids, attention_mask

print(enc["input_ids"])
# a tensor of shape (1, number_of_tokens): batch of 1, then one integer per token
print(enc["attention_mask"])
# the same shape, all 1s (no padding in a single sentence)
print(enc["token_type_ids"])
# the same shape, all 0s. "Segment IDs": which of two sentences each token belongs to.
# With a single sentence every token is segment 0.
```

**Step 3: turn the IDs back into readable pieces.**

```python
# enc["input_ids"] has shape (1, T). The [0] picks the first (only) batch item,
# leaving shape (T,). convert_ids_to_tokens maps each ID back to its vocabulary string.
toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
print(toks)
# illustrative output (yours may split words differently):
# ['[CLS]', '▁Pro', 'd', '▁checkout', '▁failing', '▁for', '▁all', '▁customers', '▁after', '▁deploy', '[SEP]']

print("vocab size:", len(tok))
# len(tok) = number of entries in the vocabulary
```

**Step 4: tokenize without the extras.**

```python
# .tokenize() only splits the text into string pieces. It does NOT add [CLS]/[SEP],
# does NOT convert to IDs, does NOT build tensors or a mask. Use it when you only
# want to see how a word breaks apart. This sentence is chosen to be full of rare
# words (a misspelling-prone word, a mixed letters-and-digit name), so you will see
# splitting clearly.
print(tok.tokenize("unreproducible regression in OAuth2 callback"))
```

**Difference between Step 3 and Step 4:** Step 3 shows exactly what the model will receive (specials included, via the IDs). Step 4 shows only the splitting, with no extras. The sentence differs on purpose: the ticket sentence is the "real input", and the Step 4 sentence is a stress test of splitting.

## Reading the output

- `input_ids` holds the integers. `attention_mask` is 1 for real tokens and 0 for padding.
- `[CLS]` is added at the start and `[SEP]` at the end automatically.
- `▁` means "this piece starts a new word". A piece without it continues the previous word.
- Rare words break into several pieces. Every piece will later get its own vector.

**Gotcha: `▁` is not the underscore `_` on your keyboard.** The marker is the character U+2581 ("lower one eighth block"), which only looks like an underscore. A typed `_` is an ordinary character and will not match the marker. Copy `▁` from the tokenizer's output or write it as `"▁"`. For example, `tok.convert_tokens_to_ids("_oxygen")` fails, but `tok.convert_tokens_to_ids("▁oxygen")` works.

**Two vocabulary entries per word.** A word can exist in the vocabulary both with the marker and without it, as two different tokens with different IDs and different embedding rows:

- `▁oxygen` is the word starting a word.
- `oxygen` (no marker) is the same letters in the middle of a word. For instance, `tok.tokenize("deoxygenated")` gives `['▁de', 'oxygen', 'ated']`.

Either form can be missing from the vocabulary. `crashed` has the `▁crashed` form but no bare `crashed`.

**Missing entries.** `convert_tokens_to_ids` is an exact dictionary lookup. It never splits a word, and a string that is not in the vocabulary returns `tok.unk_token_id` (3 for this tokenizer). To see how a word is actually split inside real text, use `tok.tokenize(word)` instead.

## Record

1. How many tokens does the ticket sentence become, including the special tokens?
2. How many pieces does "unreproducible" split into?
3. What is the vocabulary size?

---

# Lab 2: Load the encoder and read its blueprint

**Objective:** see the module tree, the physical structure of the transformer, and read its hyperparameters.

## Concepts

- **Parameter vs. hyperparameter.** These two words are easy to confuse.
  - A **parameter** is a number the model *learns* during training (for example, one entry of a weight matrix). A model has hundreds of millions of them. You never set them by hand.
  - A **hyperparameter** is a number or choice that a person *decides before training* and that defines the model's shape or how it trains. Examples: how many layers to stack, how wide each hidden vector is, how many attention heads there are. Training never changes them.
  - Analogy: if the model is a building, the hyperparameters are the blueprint (number of floors, room sizes); the parameters are the furniture arrangement that gets worked out afterwards.
- **Config.** The file (`config.json`) where a model records its hyperparameters. `base.config` prints it. You need these values to understand the shapes you will see in later labs. The four to know:
  - `hidden_size`: how many numbers describe one token at each stage (the vector length).
  - `num_hidden_layers`: how many transformer layers are stacked.
  - `num_attention_heads`: how many attention heads run in parallel in each layer.
  - `intermediate_size`: the width of the MLP's expanded middle step.
- **Module.** In PyTorch, a building block (`nn.Module`) that can contain other modules. A whole model is a tree of modules. `print(model)` draws the tree.
- **`AutoModel`.** Loads the *backbone only*: embeddings plus layers, with no task head. The matching class for a task head is `AutoModelForSequenceClassification` (Lab 8) or `AutoModelForCausalLM` (Lab 9).
- **Eager vs fused attention.** *Eager* attention computes the full `tokens × tokens` weight matrix in plain PyTorch operations, so it can hand you that matrix. *Fused* kernels (SDPA, FlashAttention) compute the same result faster without ever materializing the matrix, so they cannot return it.

## Procedure

**Step 1: load the model.**

```python
from transformers import AutoModel

# Downloads the weights (a few hundred MB, cached after the first time) and builds the
# module tree.
#   attn_implementation="eager" : use the plain attention code so that Lab 6 can ask
#                                 for the attention-weight matrices.
#   .float()                    : convert every weight to 32-bit floats. This checkpoint
#                                 is stored in 16-bit (half) precision, and recent
#                                 versions of transformers load it as stored. 16-bit
#                                 numbers carry only about 3 significant digits, which
#                                 makes some of the exact checks in later labs fail.
#                                 Converting up loses nothing: the stored values are
#                                 already 16-bit.
#   .eval()                     : switch off dropout (random zeroing used only in
#                                 training). Without it, running the same input twice
#                                 would give slightly different outputs.
base = AutoModel.from_pretrained(name, attn_implementation="eager").float().eval()

# Check the precision the weights are actually in. Expect torch.float32.
print(next(base.parameters()).dtype)
```

**Step 2: read the hyperparameters.**

```python
print(base.config)
# A dictionary-like printout. Find these four values and write them down:
#   hidden_size            : width of every hidden-state vector
#   num_hidden_layers      : how many transformer layers are stacked
#   num_attention_heads    : heads per layer
#   intermediate_size      : width inside the MLP
```

**Step 3: list the top-level pieces.**

```python
# named_children() yields (name, module) pairs for the direct children only.
# type(m).__name__ gets the module's class name as a string.
for n, m in base.named_children():
    print(n, "→", type(m).__name__)
# expect: embeddings → DebertaV2Embeddings, encoder → DebertaV2Encoder
```

**Step 4: zoom into one layer.**

```python
# base.encoder.layer is a list of the 12 layers. [0] is the first one.
print(base.encoder.layer[0])
```

## Reading the output

In the printout of `layer[0]`, find:

- `attention.self.query_proj` / `key_proj` / `value_proj`: the Q/K/V projections (each a `Linear` 768 → 768). **LoRA attaches here** in Lab 10.
- `attention.output.dense`: mixes the outputs of all heads back together.
- `intermediate.dense` (768 → 3072) then `output.dense` (3072 → 768): the MLP, expanding, applying GELU, then contracting.
- `LayerNorm` modules: keep activations well-scaled.

The config printout may list `"dtype": "float16"`. That records how the checkpoint is *stored*. After `.float()` the weights in memory are float32, which is what the dtype check in Step 1 prints.

Notice there is **no head** on `base`. It ends with the encoder layers, so its output is a set of vectors, not predictions.

## Record

4. From the config: what are `hidden_size`, `num_hidden_layers`, `num_attention_heads` and `intermediate_size`?
5. What is the per-head dimension (`hidden_size / num_attention_heads`)?

---

# Lab 3: Parameters (where the numbers live)

**Objective:** connect module names to weight matrices and their shapes, and see where the parameter budget goes.

## Concepts

- **`named_parameters()`** yields `(name, tensor)` for every learned number in the model. The *name* is the module's path in the tree plus the parameter's own name (`weight` or `bias`), separated by dots, for example `encoder.layer.0.attention.self.query_proj.weight`.
- **Linear weight shape.** `nn.Linear(in, out)` stores `weight` with shape **`(out, in)`** and computes `y = x @ weight.T + bias`. The shape reads "out rows, in columns", which looks backwards at first. A `(3072, 768)` weight maps 768 inputs to 3072 outputs.
- **Parameter count of a Linear.** `out × in + out` (weights plus bias).

## Procedure

**Step 1: list parameter names and shapes.**

```python
# Printing all ~200 parameters would flood the screen. This loop prints a parameter if
# EITHER it is not inside any numbered layer (embeddings etc.) OR it is inside layer 0.
# Long form of the condition:
#     if ("layer." not in n) or n.startswith("encoder.layer.0."):
#         print(...)
# (The original compact form "A or B" is the same thing.)
for n, p in base.named_parameters():
    if "layer." not in n or n.startswith("encoder.layer.0."):
        # f"{n:65s}" pads the name to 65 characters so the shapes line up in a column.
        # tuple(p.shape) prints the shape like (768, 768).
        print(f"{n:65s} {tuple(p.shape)}")
```

**Step 2: count parameters.**

```python
# p.numel() = number of elements in tensor p. The expression inside sum(...) is a
# generator expression; long form:
#     total = 0
#     for p in base.parameters():
#         total += p.numel()
total = sum(p.numel() for p in base.parameters())

# Size of the embedding table alone: (vocab_size, 768) → vocab_size × 768 numbers.
emb   = base.embeddings.word_embeddings.weight.numel()

# Size of ONE transformer layer (all its parameters).
layer = sum(p.numel() for p in base.encoder.layer[0].parameters())

# Divide by 1e6 to print in millions. :.1f = one decimal, :.2f = two decimals.
print(f"total {total/1e6:.1f}M | embedding table {emb/1e6:.1f}M | one layer {layer/1e6:.2f}M")
```

## Reading the output

- `query_proj.weight` at `(768, 768)` is a linear map from 768 numbers to 768 numbers.
- `intermediate.dense.weight` at `(3072, 768)` maps 768 to 3072. Its bias has shape `(3072,)`.
- `word_embeddings.weight` at `(vocab, 768)` is the lookup table, one row per token.
- Compare the embedding table with the total. One surprise to look for: a large share of the "model" is just the dictionary.

## Record

6. What fraction of all the parameters is the embedding table?
7. How many parameters does one transformer layer have, and does that × 12 plus the embeddings roughly add up to the total?

---

# Lab 4: The embedding table (static vectors)

**Objective:** see step 1 of the forward pass, a plain table lookup where context does not matter yet.

## Concepts

The first thing a transformer does with `input_ids` is replace each ID by its row of the embedding table. This is context-free: the word "crashed" gets the *same* vector wherever it appears. Everything contextual happens in the layers that follow (Lab 5). Cosine similarity (see glossary) measures whether two vectors point the same way.

## Procedure

**Step 1: get one token's vector.**

```python
E = base.embeddings.word_embeddings.weight      # the table, shape (vocab_size, 768)

# Look up the ID of the single vocabulary piece "▁crashed" (with the word-start marker).
tid = tok.convert_tokens_to_ids("▁crashed")

# If the string is not a vocabulary entry, the tokenizer returns the unknown-token ID.
# assert stops the cell with a message if the condition is false:
#     if not (tid != tok.unk_token_id):
#         raise AssertionError("not a single token: pick another word")
assert tid != tok.unk_token_id, "not a single token: pick another word"

# E[tid] is ROW number tid: a vector of 768 numbers. [:8] shows the first 8 of them.
print(E[tid].shape, E[tid][:8])
```

**Step 2: find the closest vectors.**

```python
# Cosine similarity between this token's vector and EVERY row of E.
#   E[tid]            has shape (768,)
#   E[tid].unsqueeze(0) has shape (1, 768)  (a "table" with one row)
#   E                 has shape (vocab, 768)
# PyTorch broadcasts the single row against all rows; dim=-1 means "compare along the
# 768 numbers". The result `sims` has shape (vocab,): one similarity per token.
sims = F.cosine_similarity(E[tid].unsqueeze(0), E, dim=-1)

# .topk(10) returns the 10 largest values AND their positions (indices). The positions
# are token IDs. The first one will be the token itself (similarity 1.0).
top_ids = sims.topk(10).indices
print(tok.convert_ids_to_tokens(top_ids))
```

## Reading the output

- `E[tid]` is one row of 768 numbers. No single number means anything on its own. The meaning is in the pattern across all 768.
- The nearest neighbours are often spelling or morphology relatives (other forms of the word, capitalized variants), sometimes also genuine synonyms. At this stage the model has a dictionary, not understanding. Meaning-in-context is built by the layers.

## Record

8. List three of the nearest-neighbour tokens. Are they related by meaning, by spelling, or by both?

---

# Lab 5: The forward pass (one vector per token, per layer)

**Objective:** watch a ticket flow through the 12 layers, and prove that the vectors become contextual.

## Concepts

- **`output_hidden_states=True`** asks the model to return the hidden states after *every* stage, not only the last. You get a tuple of 13 tensors for a 12-layer model: index 0 is the embedding output (before any layer), and indices 1 to 12 are the outputs of layers 1 to 12.
- **Contextual vector.** A hidden state that reflects the surrounding tokens as well as the token itself. Layer 0 is not contextual (pure lookup). Later layers are.
- **How DeBERTa treats position.** Unlike BERT, DeBERTa-v3 does not add a position vector to the input. It injects relative position inside the attention computation. So at index 0 the vector for a given token is identical wherever the token appears.

## Procedure

**Step 1: check the shapes.**

```python
# base(**enc, output_hidden_states=True) is the same as:
#   base(input_ids=enc["input_ids"], token_type_ids=..., attention_mask=...,
#        output_hidden_states=True)
out = base(**enc, output_hidden_states=True)

# out.hidden_states is a tuple. len() = 1 (embeddings) + 12 (layers) = 13
print(len(out.hidden_states))

# Each entry has shape (batch, tokens, hidden) = (1, T, 768)
print(out.hidden_states[0].shape)

# last_hidden_state is the final layer's output. [-1] picks the last entry of the
# tuple. torch.equal checks every number is identical → expect True.
print(torch.equal(out.hidden_states[-1], out.last_hidden_state))
```

**Step 2: the contextuality experiment.** The word "crashed" appears in two sentences with different meanings. We compare its vector in the two sentences at each layer.

```python
a = "The service crashed on prod after the deploy"       # software sense
b = "The stock market crashed after the earnings report" # finance sense

# A helper function so we don't repeat ourselves. It returns BOTH the token strings and
# all 13 hidden-state tensors for one sentence.
def run(s):
    e = tok(s, return_tensors="pt")                      # encode the sentence
    t = tok.convert_ids_to_tokens(e["input_ids"][0])     # readable tokens
    hs = base(**e, output_hidden_states=True).hidden_states   # run the model
    return t, hs

# Tuple unpacking: run() returns (tokens, hidden_states); we keep them as
# ta/ha for sentence a, tb/hb for sentence b. The ";" simply puts two statements
# on one line.
ta, ha = run(a); tb, hb = run(b)

# Find the position of "▁crashed" in each token list.
ia, ib = ta.index("▁crashed"), tb.index("▁crashed")

for L in range(len(ha)):                       # L = 0, 1, ..., 12
    # ha[L]          : layer L's hidden states for sentence a, shape (1, T, 768)
    # ha[L][0, ia]   : batch item 0, token position ia → a vector of 768 numbers
    # cosine_similarity(..., dim=0): the vectors are 1-D, so compare along dim 0
    # .item() turns the resulting 1-number tensor into a normal Python float
    s = F.cosine_similarity(ha[L][0, ia], hb[L][0, ib], dim=0).item()
    print(f"layer {L:2d}  cos sim = {s:.3f}")
```

**Step 3 (optional but recommended): a baseline.** Raw cosine similarity can mislead in the upper layers, so measure what "similar" means there. Compare the same word in two contexts with two *different* words, one from each sentence.

```python
ja = ta.index("▁service")      # a different word, in sentence a
jb = tb.index("▁market")       # a different word, in sentence b
                               # (.index raises ValueError if a word is not a single token)

for L in range(len(ha)):
    same_word  = F.cosine_similarity(ha[L][0, ia], hb[L][0, ib], dim=0).item()  # crashed vs crashed
    diff_words = F.cosine_similarity(ha[L][0, ja], hb[L][0, jb], dim=0).item()  # service vs market
    print(f"layer {L:2d}  same word {same_word:.3f}   different words {diff_words:.3f}")
```

## Reading the output

- **Layer 0 ≈ 1.000.** Before any attention, "crashed" is just a table lookup, identical in both sentences. Mathematically a cosine cannot exceed 1, so a printed value slightly above 1 (such as 1.001) means a rounding artefact. The usual cause is that the model is running in 16-bit precision: `1.0009765625` is exactly 1 + 2⁻¹⁰, one rounding step above 1 in float16. You can confirm it with `torch.equal(ha[0][0, ia], hb[0][0, ib])` (True: the vectors are identical) and `next(base.parameters()).dtype`. With the `.float()` in Lab 2 you should see `torch.float32` and a value of 1.000.
- **The similarity drops sharply, then partly recovers.** The "crashed" vector absorbs "service / prod / deploy" in sentence a and "stock market / earnings" in sentence b, so the two vectors drift apart. The drop is steepest in the first layer or two and the minimum sits in the middle layers. This is the contextual vector made visible. Do not expect a steady fall: in the upper layers the similarity climbs back up (a run of this lab gave 1.00, a minimum of about 0.51 near layer 4, and about 0.90 at layer 12).
- **Why it comes back up (two things that likely contribute).**
  1. **Anisotropy (the main effect).** In the upper layers the vectors of *all* tokens share a large common component, so even unrelated tokens get a high cosine similarity. Most of the rebound is the whole space bunching up, not "crashed" meaning the same thing again. The Step 3 baseline shows this. In one run, "service" versus "market" rose from 0.12 at layer 0 to 0.76 at layer 12, while the gap between the same-word and different-word columns shrank from 0.89 to 0.14. The meaningful quantity is that gap: it says how much of a vector is still "which token am I" instead of "what surrounds me".
  2. **The pretraining task (a possible smaller contribution).** The final layers are shaped by what the model was trained to do. DeBERTa-v3 was pretrained to judge, for each token, whether it was replaced by a different one, which depends strongly on which token it is. That could pull vectors of the same token slightly back together near the top. The baseline suggests this is small next to the anisotropy effect.
- Models of this kind are often reported to be most context-dependent in the middle layers, and this is one reason people probe those layers for meaning (Lab 11).

## Record

9. Similarity at layer 0, layer 6 and layer 12?
10. Between which two layers does the similarity drop the most, and at which layer is it lowest? Does it keep falling to the top layer?

---

# Lab 6: Attention heads

**Objective:** see what "multi-head attention" physically is: per layer, a set of token-to-token weight matrices.

## Concepts

Re-read the glossary entries *attention*, *attention head*, and *softmax*. For one head in one layer, the output of the attention weights is a `tokens × tokens` matrix. **Row *i*** is token *i*'s distribution over all tokens (how it spreads its attention); **column *j*** is how much each token looks at token *j*. Each row is a softmax output, so each row sums to 1.

The full attention tensor for one layer has shape `(batch, heads, query_tokens, key_tokens)`.

**Attention sink.** A common pattern in which a head puts most of its weight on a special token such as `[CLS]` or `[SEP]`. It works as a "do nothing" default when the head has nothing useful to attend to.

### How to read the heatmap

The heatmap in Step 2 is the `tokens × tokens` matrix of one head, drawn as colours. Here is a small made-up example with numbers printed in the cells (the numbers are invented for teaching; they are not model output):

<!-- figure:attention-heatmap:start -->
```text
                     column = the token being looked AT
                  [CLS]   The  service crashed [SEP]   row total
  [CLS]            0.10  0.05    0.35    0.40   0.10      1.00
  The              0.10  0.30    0.45    0.10   0.05      1.00
  service          0.05  0.20    0.25    0.40   0.10      1.00
 >crashed<         0.05  0.10    0.60    0.15   0.10      1.00   <- "crashed" looks mostly at "service"
  [SEP]            0.30  0.05    0.15    0.30   0.20      1.00
  ^ row = the token doing the LOOKING
```
<!-- figure:attention-heatmap:end -->

The rules, in order:

1. **Rows are "who is looking", columns are "who is being looked at".** In code, `A[L][0, H][i, j]` is how much token *i* attends to token *j*. The row label on the left is the token doing the looking.
2. **Read one row at a time, left to right.** It answers "where does *this* token get its information from?" In the example, the row for "crashed" has 0.60 under "service", so most of the new vector for "crashed" is built from "service".
3. **Each row adds up to 1.** That is the softmax. A token has a fixed budget of attention (100%) to divide among all tokens, so a bright cell in a row means the other cells in that row get less.
4. **Read a column to see who is popular.** Reading *down* a column answers "which tokens read from *this* token?" A bright column means many tokens attend to it. You can't read a column as adding to 1; columns need not sum to anything.
5. **Colour is the number.** With the viridis colour map used in the lab, dark purple is close to 0 and bright yellow is large. Check the colour bar: matplotlib picks the scale separately for each plot, so the same colour can mean different numbers in two different plots. To compare plots fairly, add `vmin=0, vmax=1` to `plt.imshow`.
6. **Direction of information:** the weight in row *i*, column *j* says how much of token *j*'s *value* flows into token *i*'s new vector. Attention moves information from the column token to the row token.

Some patterns come up again and again:

<!-- figure:attention-patterns:start -->
```text
  Self              Previous token       Sink                 Broad
  bright diagonal   bright just below    one bright column    evenly dim,
                    the diagonal                              no standout
  # . . .           # . . .              . . . #              + + + +
  . # . .           # . . .              . . . #              + + + +
  . . # .           . # . .              . . . #              + + + +
  . . . #           . . # .              . . . #              + + + +
```
<!-- figure:attention-patterns:end -->

- **Self:** each token mostly looks at itself.
- **Previous (or next) token:** a stripe one step off the diagonal. This head tracks word order.
- **Sink:** almost all rows put their weight on one token, often `[CLS]` or `[SEP]`. It is a resting place for a head with nothing useful to attend to, not a statement that the sink token is important. If layer 0, head 0 shows one bright vertical stripe, this is what you are seeing.
- **Broad:** attention is spread thinly everywhere. This head is averaging over the whole sentence.
- **Encoder versus decoder:** in this encoder lab, every cell can be non-zero. In a decoder (Lab 9) the area above the diagonal is always empty, because tokens can't look at later ones.

This lab uses `enc` and `toks` from Lab 1 (the ticket sentence).

## Procedure

**Step 1: get the attention weights and check their shape.**

```python
# output_attentions=True works because we loaded with attn_implementation="eager" (Lab 2).
out = base(**enc, output_attentions=True)

A = out.attentions              # a tuple with one tensor per layer (12 entries)

# A[0] is layer 0: shape (batch, heads, query_tokens, key_tokens) = (1, 12, T, T)
print(len(A), A[0].shape)

# A[0][0, 0] = layer 0, batch item 0, head 0 → a (T, T) matrix.
# .sum(-1) sums along the last dimension (across each row). Expect all 1.0, because
# each row is a softmax.
print(A[0][0, 0].sum(-1))
```

**Step 2: draw one head as a picture.**

```python
L, H = 11, 0                    # choose layer 11 (the last), head 0. Change these freely.

plt.figure(figsize=(6, 5))                      # canvas size in inches
plt.imshow(A[L][0, H], cmap="viridis")          # draw the (T, T) matrix as a heat map
# Label both axes with the token strings. rotation=90 stands the x labels upright.
plt.xticks(range(len(toks)), toks, rotation=90)
plt.yticks(range(len(toks)), toks)
plt.title(f"layer {L}, head {H}  (row = token looking, column = token looked at)")
plt.colorbar()                                  # legend: colour → weight
plt.show()
```

**Step 3: what does `[CLS]` look at?**

```python
# A[11]              : layer 11 attention, shape (1, 12, T, T)
# A[11][0, :, 0, :]  : batch item 0, ALL heads, query position 0 (that is [CLS]),
#                      ALL key positions → shape (12, T): one row per head
# .mean(0)           : average over dimension 0 (the heads) → shape (T,)
cls_focus = A[11][0, :, 0, :].mean(0)

# Pair each token with its weight, then sort from largest to smallest.
#   zip(toks, cls_focus.tolist())  → pairs (token, weight)
#   key=lambda x: -x[1]            → sort by the negative of the weight (x[1]), i.e.
#                                    biggest first. Long form of the lambda:
#                                       def sort_key(x):
#                                           return -x[1]
for t, w in sorted(zip(toks, cls_focus.tolist()), key=lambda x: -x[1]):
    print(f"{t:15s} {w:.3f}")
```

## Reading the output

- Every row sums to 1: each token distributes 100% of its attention across all tokens.
- **Encoder:** the matrix is full, with weight both above *and* below the diagonal. Tokens look both left and right.
- Different heads in the same layer behave differently. Try `H = 0 … 11`: some attend to the previous or next token, some spread broadly, and some dump nearly all their weight on `[CLS]` or `[SEP]` (the attention sink).

## Record

11. Describe two heads in layer 11 with visibly different patterns.
12. Which content words does `[CLS]` attend to most?

---

# Lab 7: Hooks (tap any wire in the circuit)

> **Notice!!** This lab leans on **linear algebra**: dot products, matrix-times-vector, and matrix multiplication (the "Projections and mixing" section and Steps 4 to 6 use them heavily). If those feel unfamiliar, you do not need to push through. Work through the companion **[Linear Algebra for AI Models: an interactive study guide](linear-algebra-study-guide.md)** first, then come back. Its Parts 1 to 3 (vectors and the dot product, matrices and the transpose, multiplying matrices) cover what Lab 7 needs.

**Objective:** capture the output of any internal module during a forward pass. Think of it as an oscilloscope probe.

## Concepts

A **forward hook** is a function with the signature `hook(module, inputs, output)`. You attach it with `module.register_forward_hook(hook)`. Every time that module runs, PyTorch calls your function, and a module can run **more than once** in a single forward pass (Step 2 shows a real case). `inputs` is always a **tuple** of the positional arguments the module received; `output` is whatever the module returned. `register_forward_hook` returns a **handle**; calling `handle.remove()` detaches the hook. Always remove hooks when finished, or they keep firing on every later run.

**Closure.** A function defined inside another function that remembers the outer function's variables. Here `probe("L6_query")` returns a hook that remembers the key `"L6_query"`, so each hook stores its result under its own name.

### Projections and mixing (used in Steps 4 to 6)

A **projection** is a linear layer: each output number is a weighted sum of *all* the input numbers, plus a bias. A tiny example with 3 numbers in and 2 out (these numbers are invented for teaching):

<!-- figure:projection-by-hand:start -->
```text
      W (2x3)        x       b        y
    [ 1  0  2 ]    [ 1 ]   [ 0   ]   [ 7   ]
    [ 0  1 -1 ]  . [ 2 ] + [ 0.5 ] = [-0.5 ]
                   [ 3 ]

  output 0 = 1*1 + 0*2 + 2*3 + 0     = 7     (row 0 of W, matched against x)
  output 1 = 0*1 + 1*2 + (-1)*3 + .5 = -0.5
```
<!-- figure:projection-by-hand:end -->

Each row of `W` is a recipe for one output number. PyTorch stores and applies it as `x @ W.T + b` (the same numbers with rows and columns flipped), which is the form used in the code below.

In the model, `query_proj.weight` is 768 × 768. Row *k* is the recipe for query feature *k*, built from all 768 input numbers. The rows come in groups of 64: rows 0 to 63 produce head 0's query numbers, rows 64 to 127 produce head 1's, and so on. So each head gets its own 64 recipes applied to the *whole* token vector. This is why a head's 64 numbers are a different learned view of the full token, not a literal cut of it. `key_proj` and `value_proj` work the same way with their own recipes.

**Mixing.** After each head has done its attention, the 12 results (64 numbers each) are joined end to end into one 768-number vector. `attention.output.dense` is another projection, 768 × 768, so every output number is a weighted sum of all 768 joined numbers, from every head:

<!-- figure:heads-mixed:start -->
```text
  head 0 result | head 1 result | head 2 result      (each head: its own numbers)
        \_____________|______________/
                join end to end
                      |
        [ head 0 | head 1 | head 2 ]                 joined vector
                      |   x W_o  (a learned matrix, 768 x 768 in the real model)
                      v
        [ every output = a blend of ALL heads ]      output vector
```
<!-- figure:heads-mixed:end -->

## Procedure

**Step 1: define the recorder.**

```python
acts = {}                          # an empty dict that will collect captured outputs

def probe(key):
    # probe() builds and returns a hook function that remembers `key`.
    def hook(module, inputs, output):
        # Keep EVERY call, in order: a module can run more than once per forward pass.
        # setdefault(key, []) returns the list stored under key, creating an empty one first
        # if there is none. Long form:
        #     if key not in acts:
        #         acts[key] = []
        #     acts[key].append(output)
        acts.setdefault(key, []).append(output)
    return hook
```

**Step 2: attach hooks to two modules, run the model, detach.**

```python
handles = [
    # intermediate = the MLP's first half: Linear (768→3072) followed by GELU.
    # The hook captures its OUTPUT: each token's widened vector.
    base.encoder.layer[6].intermediate.register_forward_hook(probe("L6_mlp_expand")),
    # query_proj = the Q projection in layer 6's attention.
    base.encoder.layer[6].attention.self.query_proj.register_forward_hook(probe("L6_query")),
]

base(**enc)                        # one forward pass; the hooks fire during it

# Long form of the one-line loop below:
#     for h in handles:
#         h.remove()
for h in handles: h.remove()       # always remove hooks

# How many times did each module run, and what shape did each call return?
# [tuple(o.shape) for o in outs] is a list comprehension. Long form:
#     shapes = []
#     for o in outs:
#         shapes.append(tuple(o.shape))
for key, outs in acts.items():
    print(key, len(outs), "call(s):", [tuple(o.shape) for o in outs])
# expect (attention runs before the MLP, so the query hook is listed first):
#         L6_query      2 call(s): [(1, T, 768), (1, 512, 768)]
#         L6_mlp_expand 1 call(s): [(1, T, 3072)]
```

**Step 3 (optional): split the queries into heads yourself.**

```python
q = acts["L6_query"][0]                    # the FIRST call: the tokens' queries, (1, T, 768)
T = q.shape[1]                             # number of tokens
# view reshapes without copying: split the 768 numbers into 12 groups of 64.
# transpose(1, 2) swaps the token and head dimensions so heads come first.
q_heads = q.view(1, T, 12, 64).transpose(1, 2)
print(q_heads.shape)                       # (1, 12, T, 64)
```

**Step 4: reproduce a projection by hand.**

```python
layer = base.encoder.layer[0]                 # work inside the first layer
# Input to layer 0 = the embedding stage's output (index 0 of hidden_states). Shape (1, T, 768).
x = base(**enc, output_hidden_states=True).hidden_states[0]

proj = layer.attention.self.query_proj        # the Q projection (a Linear 768 -> 768)
W, b = proj.weight, proj.bias                 # W: (768, 768), one ROW per output number; b: (768,)

q_manual = x @ W.T + b                        # the projection written out: x times W-transpose plus bias
q_module = proj(x)                            # the same thing, through the module
print(W.shape, torch.allclose(q_manual, q_module, atol=1e-5))     # (768, 768) True

# One single output number as an explicit weighted sum of all 768 inputs:
#   t = token position 3, k = query feature number 5
#   x[0, t]   : that token's 768 input numbers
#   W[k]      : row k of W, the recipe for output number k (768 weights)
t, k = 3, 5
by_hand = (x[0, t] * W[k]).sum() + b[k]       # multiply matching numbers, add them up, add the bias
print(by_hand.item(), q_manual[0, t, k].item())    # the two agree to about 6 digits (tiny float differences from the order of addition)
```

**Step 5: one head's slice comes from 64 rows of the recipe book.**

```python
h = 2                                         # pick head 2
rows = W[64 * h : 64 * (h + 1)]               # rows 128..191 of W: head 2's 64 recipes, shape (64, 768)
q_head2 = x @ rows.T + b[64 * h : 64 * (h + 1)]    # apply only those 64 recipes, shape (1, T, 64)

# Compare with the matching 64-number slice of the full projection result.
print(torch.allclose(q_head2, q_manual[..., 64 * h : 64 * (h + 1)], atol=1e-5))   # True
```

**Step 6: reproduce the mixing, and prove it mixes.**

```python
cap = {}                                      # storage for the hook

def grab(module, inputs, output):
    cap["joined"] = inputs[0]                 # the 12 heads' results joined together, (1, T, 768)
    cap["out"] = output                       # what output.dense returned

hd = layer.attention.output.dense.register_forward_hook(grab)
base(**enc)                                   # one forward pass; the hook fires inside layer 0
hd.remove()                                   # always detach

Wo, bo = layer.attention.output.dense.weight, layer.attention.output.dense.bias
print(torch.allclose(cap["joined"] @ Wo.T + bo, cap["out"], atol=1e-5))   # True: mixing is one projection

# The mixing test: change ONLY head 3's 64 numbers, then count how many OUTPUT numbers change.
nudged = cap["joined"].clone()                # an independent copy
nudged[:, :, 3 * 64 : 4 * 64] += 1.0          # add 1.0 to the 64 numbers belonging to head 3
out2 = layer.attention.output.dense(nudged)   # run just the mixing layer on the changed input

# For each token: how many of the 768 outputs differ by more than a tiny amount?
changed = ((out2 - cap["out"]).abs() > 1e-6).sum(-1)
print(changed)                                # expect 768 for every token
```

## Reading the output

- `(1, T, 3072)`: inside the MLP, each token's vector is temporarily 4× wider (768 → 3072). The MLP runs once per forward pass, so there is one call.
- **`query_proj` runs twice.** The first call is `(1, T, 768)`: the queries for your T tokens. The second is `(1, 512, 768)`: DeBERTa also passes its **relative-position embeddings** (one vector per relative distance, 2 × 256 position buckets = 512 rows) through the *same* query projection, which is how it scores "content against position" (the config's `share_att_key: true`). A hook that kept only the last call would show the position embeddings by mistake. That is why Step 1 stores a list.
- The tokens' queries, `(1, T, 768)`, are split internally into 12 heads of 64 dimensions. The "heads" are just slices of this one projection.
- Hooks are how interpretability research, probing, and activation capture are done in practice.
- **Step 4:** the projection really is `x @ W.T + b` with nothing hidden. Each output number is row *k* of `W` matched against the token's 768 numbers, plus a bias. `W` has shape `(768, 768)`.
- **Step 5:** head 2's 64 numbers use only 64 of the 768 rows of `W`. Every row still takes all 768 input numbers, so each head sees the whole token through its own 64 recipes.
- **Step 6:** the mixing layer is also just a matrix multiply. Nudging only head 3's 64 numbers changes **all 768** outputs, not only 64. If the layer did no mixing (a pure pass-through), only head 3's own 64 slots would change. That is what "mixing" means: every output blends all the heads.

## Record

13. What would the per-head query shape be after the split?
13a. In Step 4, what is the shape of `W`, and which row of `W` builds query feature 5?
13b. In Step 6, how many of the 768 output numbers change when only head 3's 64 numbers are nudged? How many would change if there were no mixing?

---

# Lab 8: The classification head (pooling → 3 numbers)

**Objective:** confirm by hand that the head is pooling followed by one matrix multiply, and see what softmax and temperature do.

## Concepts

Re-read *backbone*, *head*, *pooling*, *logits*, *softmax*, *temperature*.

A sequence-classification model is: backbone → pooler → classifier.

- The **pooler** (here DeBERTa's `ContextPooler`) takes the hidden state at position 0 (`[CLS]`), applies a linear layer and an activation, and outputs one 768-vector for the whole text.
- The **classifier** is a single `Linear(768 → num_labels)` that turns that vector into one logit per class.

**Newly initialized weights.** When you load a base checkpoint with a classification head, the pooler and classifier do not exist in the checkpoint, so they are created with random numbers. The library warns you. Until trained, their output is meaningless.

**Expected value.** The probability-weighted average of numeric values: `E[x] = Σ p_i × x_i`. Here it turns three class probabilities into a single "expected priority" number.

## Procedure

**Step 1: build the classifier and list the new head's parameters.**

```python
from transformers import AutoModelForSequenceClassification

# Class index ↔ meaning. This is just a dictionary we define.
labels = {0: "low", 1: "medium", 2: "high"}

clf = AutoModelForSequenceClassification.from_pretrained(
    name,
    num_labels=3,                                          # three output classes
    id2label=labels,                                       # store human-readable names
    label2id={v: k for k, v in labels.items()},            # dict comprehension: invert it
    attn_implementation="eager",
).float().eval()    # .float(): 32-bit weights, same reason as in Lab 2 (the exact checks below need it)

# Print only the parameters that are NOT in the backbone (whose names start "deberta.").
# Long form of the condition:
#     if n.startswith("deberta.") == False:
for n, p in clf.named_parameters():
    if not n.startswith("deberta."):
        print(n, tuple(p.shape))
# expect pooler.dense.weight (768, 768), pooler.dense.bias, classifier.weight (3, 768),
# classifier.bias (3,)
```

Read the warning that appears when loading: it says `pooler.dense` and `classifier` were **newly initialized**. That is the fresh head, with random weights.

**Step 2: capture what goes into and out of the classifier, and recompute it by hand.**

```python
cap = {}                              # storage for hook results

# The hook's three arguments are (module, inputs, output).
#   inputs is a tuple, so inputs[0] is the first (only) input: the pooled vector.
# Long form of the compact lambda used in many tutorials:
#     def grab(module, inputs, output):
#         cap["x"] = inputs[0]        # the pooled vector fed to the classifier
#         cap["logits"] = output      # the classifier's output
def grab(module, inputs, output):
    cap["x"] = inputs[0]
    cap["logits"] = output

h = clf.classifier.register_forward_hook(grab)    # attach to the classifier only
out = clf(**enc)                                  # run; the hook captures x and logits
h.remove()                                        # detach

W, b = clf.classifier.weight, clf.classifier.bias        # shapes (3, 768) and (3,)
manual = cap["x"] @ W.T + b                              # y = x Wᵀ + b, by hand

# allclose(a, b, atol=1e-5): True when every number differs by at most 0.00001.
print(torch.allclose(manual, out.logits, atol=1e-5))     # expect True
```

**Step 3: show that the pooler uses only the `[CLS]` position.**

```python
# Run just the backbone to get the per-token vectors, shape (1, T, 768).
hidden = clf.deberta(**enc).last_hidden_state

# Feed them to the pooler: output should equal what the classifier received.
print(torch.allclose(clf.pooler(hidden), cap["x"], atol=1e-5))    # expect True

# Experiment: keep position 0 but replace every other position with random noise.
#   hidden.clone()               : an independent copy, so we don't damage `hidden`
#   scrambled[:, 1:]             : all batch items, positions 1..end
#   torch.randn_like(...)        : random numbers with the same shape
scrambled = hidden.clone()
scrambled[:, 1:] = torch.randn_like(scrambled[:, 1:])

# If the pooler only reads position 0, the output must be unchanged.
print(torch.allclose(clf.pooler(scrambled), cap["x"], atol=1e-5))   # expect True
```

**Step 4: logits → probabilities → decision, and the effect of temperature.**

```python
z = out.logits                      # raw scores, shape (1, 3)
p = z.softmax(-1)                   # probabilities along the last dim; they sum to 1
print("probs", p, "argmax", labels[p.argmax(-1).item()])
# argmax(-1) is the index of the biggest probability; .item() makes it a plain int
# which we use as a key into `labels`.

for T in [0.5, 1.0, 2.0]:
    print(f"T={T}", (z / T).softmax(-1))      # divide logits by T, then softmax

# Expected priority = Σ p_i × value_i, with values 0, 1, 2 for low, medium, high.
# torch.tensor([0., 1., 2.]) is the vector of values; multiplying elementwise and
# summing is the dot product. .item() → plain float.
print("expected priority (0=low, 2=high):", (p * torch.tensor([0., 1., 2.])).sum().item())
```

## Reading the output

- The manual `x @ W.T + b` reproduces the model's logits exactly. There is no hidden machinery in the head.
- The pooler output is unchanged when every token except `[CLS]` is scrambled, which proves it reads only the `[CLS]` vector.
- The probabilities are close to uniform and meaningless, because the head is random. That is exactly why it has to be trained.
- Temperature changes how *confident* the probabilities look, never which label wins.

## Record

14. Shapes of `classifier.weight` and `pooler.dense.weight`?
15. How many parameters is the whole head (pooler plus classifier)? Compare that with the backbone.
16. Does the argmax change with T? Does the confidence?

---

# Lab 9: The decoder side by side

**Objective:** see the same anatomy in a decoder, plus the two differences that matter: the causal mask and the LM head.

## Concepts

Re-read *decoder model*, *causal mask*, *LM head*, *GQA*, *weight tying*.

- **Next-token prediction.** A decoder is trained so that at each position it predicts the token that comes *next*. Its logits have one score for every vocabulary entry at every position: shape `(batch, tokens, vocab)`. For generating text, you use the last position's scores.
- **LM head.** A single `Linear(hidden → vocab_size)` with no bias. It plays the role of the encoder's classifier, but with `vocab_size` output rows instead of 3.
- **Label-token trick.** To get a classifier from a pretrained decoder with no new layer, write a prompt that ends right where the label would go (`... Priority:`) and compare the LM head's scores for the first token of each label word. The label words get a proper probability distribution among themselves after a 3-way softmax.
- **Qwen's layer parts.** `q_proj / k_proj / v_proj / o_proj` are the attention projections. The MLP has three linear layers: `gate_proj` and `up_proj` (expand) and `down_proj` (contract), combined as `down(SiLU(gate(x)) × up(x))`.

## Procedure

**Step 1: load the decoder and look at its anatomy.**

```python
from transformers import AutoModelForCausalLM

dname = "Qwen/Qwen2.5-0.5B"
dtok = AutoTokenizer.from_pretrained(dname)             # Qwen's own tokenizer, NOT DeBERTa's

# .float() converts the weights to 32-bit floats (the checkpoint may be stored in 16-bit),
# which is safest on a CPU.
dec = AutoModelForCausalLM.from_pretrained(dname, attn_implementation="eager").float().eval()

print(dec)                           # the module tree: compare with the encoder's

c = dec.config
print(c.hidden_size, c.num_hidden_layers, c.num_attention_heads,
      c.num_key_value_heads, c.vocab_size)
# hidden width, layers, query heads, KV heads, vocabulary size

print("lm_head:", tuple(dec.lm_head.weight.shape))      # (vocab_size, hidden_size)

# Are the LM head and the input embedding table the SAME memory?
#   c.tie_word_embeddings   : the config's flag for weight tying
#   .data_ptr()             : the memory address where the tensor's data begins.
#                             Equal addresses mean literally the same matrix.
print("tied to embeddings:", c.tie_word_embeddings,
      dec.lm_head.weight.data_ptr() == dec.model.embed_tokens.weight.data_ptr())
```

**Step 2: see the causal mask.**

```python
# A short text whose next word we will predict later.
denc = dtok("The service crashed on prod after the", return_tensors="pt")

# Ask for logits plus all attention matrices and hidden states in one pass.
dout = dec(**denc, output_attentions=True, output_hidden_states=True)

# Layer 0, batch item 0, head 0 → a (T, T) matrix.
A0 = dout.attentions[0][0, 0]

# torch.triu(M, diagonal=1) keeps only the entries strictly ABOVE the diagonal and zeroes
# the rest. If the causal mask works, everything above the diagonal is already 0,
# so the maximum is 0.
print("max weight above diagonal:", torch.triu(A0, diagonal=1).abs().max().item())

plt.imshow(A0, cmap="viridis"); plt.title("decoder layer 0 head 0"); plt.colorbar(); plt.show()
```

**Step 3: next-token prediction.**

```python
# One set of vocab-wide scores per input position.
print(dout.logits.shape)               # (1, T, vocab_size)

# [0, -1] = batch item 0, LAST position. This row scores every possible next token.
# softmax turns the scores into probabilities.
nxt = dout.logits[0, -1].softmax(-1)

top = nxt.topk(5)                      # top.values and top.indices

# Long form of the comprehension:
#     result = []
#     for v, i in zip(top.values, top.indices):
#         result.append((dtok.decode(i), round(v.item(), 3)))
#     print(result)
print([(dtok.decode(i), round(v.item(), 3)) for v, i in zip(top.values, top.indices)])
```

**Step 4: verify the LM head by hand.**

```python
last_h = dout.hidden_states[-1][0, -1]     # last layer's vector at the last position, (896,)
real   = dout.logits[0, -1]                # what the model reported, (vocab_size,)

# Depending on your transformers version, hidden_states[-1] may or may not already include
# the final normalization. Try both and see which one matches.
for label, vec in [("as returned", last_h), ("after final norm", dec.model.norm(last_h))]:
    print(label, torch.allclose(dec.lm_head(vec), real, atol=1e-3))
# At least one line should print True. The LM head is just one Linear layer.
# (With transformers 5.x, "as returned" is the one that prints True: hidden_states[-1]
# already includes the final norm.)
```

**Step 5: the label-token trick (a 3-way classifier with no new layer).**

```python
prompt = "Ticket: Prod checkout failing for all customers after deploy.\nPriority:"
penc = dtok(prompt, return_tensors="pt")

# Encode each label word WITH a leading space, since that is how it would appear after
# "Priority:". add_special_tokens=False avoids extra start/end tokens.
# Long form of the comprehension:
#     ids = []
#     for w in [" low", " medium", " high"]:
#         ids.append(dtok.encode(w, add_special_tokens=False))
ids = [dtok.encode(w, add_special_tokens=False) for w in [" low", " medium", " high"]]
print(ids)        # a list of 3 lists. Each inner list should have ONE id.

# Use the first (here only) token of each label word.
label_ids = [i[0] for i in ids]

# .logits[0, -1, label_ids] : batch 0, last position, and only the 3 chosen vocabulary
# entries (indexing with a list picks exactly those 3). Result shape (3,) → 3 rows of
# the LM head. Softmax over just those 3 gives a distribution among the labels.
z = dec(**penc).logits[0, -1, label_ids]
print(dict(zip(["low", "medium", "high"], z.softmax(-1).tolist())))
```

**Step 6: a real classification head on the decoder.**

```python
dclf = AutoModelForSequenceClassification.from_pretrained(dname, num_labels=3).float().eval()

# The classification model needs to know which token is padding so it can find the last
# real token. Qwen's tokenizer usually defines one already; this guard fills it in if not.
# Long form of the single-line "if" block:
#     if dclf.config.pad_token_id is None:
#         dclf.config.pad_token_id = dtok.pad_token_id
if dclf.config.pad_token_id is None:
    dclf.config.pad_token_id = dtok.pad_token_id

# `score` is the decoder's classification head: Linear(hidden → 3), newly initialized.
print(dclf.score)
```

## Reading the output

- The decoder attention matrix is **lower-triangular**: the maximum weight above the diagonal is 0. Each token sees only itself and tokens to its left. Compare with the full matrix from Lab 6.
- Logits exist at *every* position, and each one predicts the *next* token.
- The label-token trick yields three probabilities straight from the pretrained LM head. Even untrained, they may lean sensibly.
- The anatomy matches the encoder: embedding table, stacked layers, MLP. Differences: `lm_head` instead of a 3-row classifier; fewer KV heads than query heads (GQA); the head and the embedding table share weights.

## Record

17. Decoder `hidden_size`, layers, query heads and KV heads?
18. What are the top-5 next tokens after "...crashed on prod after the"?
19. What untrained probabilities does the label-token trick give for the outage ticket?

---

# Lab 10: LoRA under the microscope

**Objective:** see what LoRA physically adds, and verify that it starts as an exact no-op.

## Concepts

Re-read *LoRA* and *PEFT*. A full fine-tune updates every weight. LoRA instead leaves `W` (say 768 × 768 = 589,824 numbers) frozen and learns two thin matrices `A` (r × 768) and `B` (768 × r) with, here, r = 8. Together `A` and `B` hold only 8 × 768 + 768 × 8 = 12,288 numbers, about 2% of `W`. The layer then computes

```
y = x Wᵀ  +  (alpha / r) × x Aᵀ Bᵀ
```

- **`r` (rank):** how many "directions" the correction may use. Bigger = more capacity, more parameters.
- **`lora_alpha`:** a scaling constant; the correction is multiplied by `alpha / r` (here 16 / 8 = 2).
- **`target_modules`:** which layers get an adapter. Here the Q, K, V projections.
- **`modules_to_save`:** layers that are *not* given an adapter but are copied and trained fully. The newly initialized head must be trained (a random head cannot be fixed with a low-rank nudge), so its layers go here.
- **No-op at initialization:** `B` starts at zeros, so `(alpha/r)·B·A = 0` and the model's output is bit-for-bit what it was before wrapping.

## Procedure

**Step 1: build a fresh classifier and record its output *before* adding LoRA.**

```python
from peft import LoraConfig, get_peft_model

fresh = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3).float().eval()  # 32-bit, as in Lab 2

# .clone() keeps an independent copy. Without it, `before` might be silently changed if
# the underlying tensor were reused.
before = fresh(**enc).logits.clone()
```

**Step 2: describe the adapter.**

```python
cfg = LoraConfig(
    task_type="SEQ_CLS",                                      # sequence classification
    r=8,                                                      # rank
    lora_alpha=16,                                            # scaling numerator
    lora_dropout=0.0,                                         # no dropout, for a clean test
    target_modules=["query_proj", "key_proj", "value_proj"],  # attach to Q, K, V in every layer
    modules_to_save=["classifier", "pooler"],                 # train the new head layers fully
)
```

**Step 3: wrap the model.**

```python
pm = get_peft_model(fresh, cfg).eval()      # `fresh` is modified in place and wrapped
pm.print_trainable_parameters()             # prints trainable / total / percent
```

**Step 4: look inside one wrapped layer.**

```python
# pm.base_model.model is the original model inside PEFT's wrappers. Then walk down to
# layer 0's query projection.
q = pm.base_model.model.deberta.encoder.layer[0].attention.self.query_proj
print(q)    # a LoRA wrapper: the original frozen Linear plus lora_A and lora_B

# "default" is the name of the adapter (PEFT supports several at once).
print("A:", tuple(q.lora_A["default"].weight.shape),    # expect (8, 768)
      "B:", tuple(q.lora_B["default"].weight.shape))    # expect (768, 8)

# B should be all zeros right now, so its largest absolute value is 0.
print("B max at init:", q.lora_B["default"].weight.abs().max().item())

# The wrapped model's output must match the output recorded before wrapping.
print("output unchanged:", torch.allclose(before, pm(**enc).logits, atol=1e-5))
```

**Step 5: list exactly what is trainable.**

```python
# p.requires_grad is True for parameters that training would update.
# Long form of the comprehension:
#     trainable = []
#     for n, p in pm.named_parameters():
#         if p.requires_grad:
#             trainable.append(n)
trainable = [n for n, p in pm.named_parameters() if p.requires_grad]

print(len(trainable))        # how many trainable tensors
print(trainable[:6])         # first six names (LoRA matrices of the early layers)
print(trainable[-4:])        # last four names (the copied head layers)
```

## Reading the output

- `query_proj` is now a wrapper: the original frozen `Linear` plus `lora_A` (r × 768) and `lora_B` (768 × r). The effective weight is `W + (α/r)·B·A`, a rank-8 correction to a 768 × 768 matrix.
- `B` starts at **zero**, so the adapter adds nothing at first and the logits are unchanged. Training grows `B` away from zero.
- The trainable list contains LoRA `A`/`B` matrices plus full copies of `pooler` and `classifier`. Without `modules_to_save` the random head would never learn.
- The trainable percentage is tiny compared with the full model.

## Record

20. Trainable parameters, and the percentage of the total?
21. How many parameters does one rank-8 adapter on a 768×768 projection add, compared with the projection itself?

---

# Lab 11: Representation geometry (the payoff experiment)

**Objective:** see priority as *geometry*: where tickets land in vector space, and at which layer priority information appears.

## Concepts

- **Representation.** The vector the model produces for a whole text. We build one by pooling the per-token hidden states of some layer (Pooling, glossary).
- **Padding.** When several texts of different lengths are batched, shorter ones are filled with `[PAD]` tokens to the same length. The attention mask marks these so that **mean pooling** must exclude them: otherwise padding would pull the average.
- **Mean pooling (masked).** `mean = Σ_t (mask_t × h_t) / Σ_t mask_t`.
- **Linear probe & cross-validation.** We train a logistic regression on the vectors and measure accuracy on held-out data (*5-fold cross-validation*: split into 5 parts, train on 4, test on the 1 left out, rotate, average). High accuracy at layer *L* means priority information is linearly readable from layer *L*.

The toy data below is only enough to make the code run. **Replace it with a few hundred real labelled tickets** for meaningful results.

## Procedure

**Step 1: the data.**

```python
texts = [
  # low priority
  "Typo in settings page tooltip", "Update footer copyright year", "Button color slightly off in dark mode",
  "Add sort option to admin table", "Rename internal variable for clarity",
  # medium priority
  "Export to CSV slow for large reports", "Search results occasionally out of order",
  "Email notifications delayed by a few minutes", "Intermittent timeout on reports page",
  "Mobile layout breaks on small tablets",
  # high priority
  "Prod checkout failing for all customers", "Login down for every user after deploy",
  "Payment processing returning 500 errors", "Data loss when saving orders",
  "Security vulnerability exposing customer emails",
]

# One label per ticket: five 0s (low), five 1s (medium), five 2s (high).
# [0]*5 means "a list of five zeros"; + joins lists.
y = [0]*5 + [1]*5 + [2]*5
```

**Step 2: a function that turns texts into one vector per layer.**

```python
def embed_all(model, texts, pool="mean", bs=16):
    """Return a list with one array per layer, each of shape (number_of_tickets, 768)."""
    layers = None                                     # will hold the running result
    for i in range(0, len(texts), bs):                # process bs texts at a time
        batch = texts[i:i+bs]                         # slice of the list
        # padding=True       : pad the batch to its longest text
        # truncation=True, max_length=512 : cut anything longer
        e = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        hs = model(**e, output_hidden_states=True).hidden_states     # 13 tensors

        # mask: (batch, tokens) → (batch, tokens, 1) so it can multiply (batch, tokens, 768)
        m = e["attention_mask"].unsqueeze(-1).float()

        vecs = []
        for h in hs:                                  # h = one layer's (batch, tokens, 768)
            if pool == "cls":
                vecs.append(h[:, 0])                  # CLS pooling: position 0 → (batch, 768)
            else:
                # masked mean: sum real tokens' vectors, divide by the count of real tokens
                vecs.append((h * m).sum(1) / m.sum(1))

        if layers is None:
            layers = vecs                             # first batch: start the list
        else:
            # later batches: append new rows under the existing ones, layer by layer
            layers = [torch.cat([old, new]) for old, new in zip(layers, vecs)]

    return [v.numpy() for v in layers]                # convert to NumPy for scikit-learn

per_layer = embed_all(base, texts, pool="mean")
print(len(per_layer), per_layer[0].shape)            # 13 layers, each (15, 768)
```

**Step 3: PCA plot of the last layer.**

```python
from sklearn.decomposition import PCA

# per_layer[-1] = last layer's (15, 768) array. PCA(2) learns the two directions of
# greatest spread; fit_transform returns each ticket's (x, y) position, shape (15, 2).
Z = PCA(2).fit_transform(per_layer[-1])

# Plot each class in its own colour.
for lab, col in zip([0, 1, 2], ["tab:green", "tab:orange", "tab:red"]):
    # indices of tickets whose label is `lab`
    idx = [i for i, v in enumerate(y) if v == lab]
    pts = Z[idx]                                      # their 2-D positions
    plt.scatter(pts[:, 0], pts[:, 1], c=col, label=labels[lab])

plt.legend(); plt.title("last layer, mean-pooled, BASE model"); plt.show()
```

**Step 4: layer-wise linear probe.**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# StandardScaler rescales each of the 768 features to mean 0 and standard deviation 1
# (learned on the training folds only). Without it, a few huge "outlier" features in the
# top layers dominate the classifier (see Reading the output).
# make_pipeline(A, B) means "apply A, then B" as one model.
probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

for L, X in enumerate(per_layer):                     # L = layer number, X = its (15, 768) array
    # cv=5: five-fold cross-validation → five accuracy numbers → .mean() averages them
    acc = cross_val_score(probe, X, y, cv=5).mean()
    print(f"layer {L:2d}  probe accuracy {acc:.2f}")
```

**Step 5: repeat with your fine-tuned model.**

Load its encoder (for a PEFT model, call `merge_and_unload()` first to fold the adapters back into plain weights), pass it to `embed_all(...)`, and redraw the PCA plot.

## Reading the output

- **Base model:** some loose grouping by topic, but priority classes overlap. Pretraining never cared about priority.
- **Fine-tuned model:** three tighter, separable clusters in the last layer. This is "fine-tuning tilts the space toward your loss", made visible.
- **Probe curve, with the toy data:** read it loosely. With 15 tickets, each fold tests only 3 of them, so accuracy moves in coarse steps and changes noticeably with small differences. In one run (mean pooling, with the scaler) it was 0.80 at layer 0, peaked at 1.00 at layer 4, and fell to 0.53 at layer 12. Layer 0 is already well above chance (0.33) because single words such as "failing", "down" or "loss" are clues even with no context.
- **Why the scaler matters.** In the top layers a few features become huge: in one run the largest value at layer 12 was about 15, while a typical feature varied by only about 0.15. These are the same shared "outlier" directions behind the anisotropy you measured in Lab 5. Without standardising, the probe in that run fell to 0.27 at layer 12 (below chance); with it, 0.53. Try removing `StandardScaler()` to see the difference.
- **With real data** (a few hundred labelled tickets), the curve is far more reliable. The layer where accuracy climbs is where the network builds the information that predicts priority, which for models like this is often reported to be in the middle layers.
- Try `pool="cls"` against `pool="mean"`. On base DeBERTa-v3, mean pooling separates better: in one run CLS pooling scored between 0.33 and 0.67, well below mean pooling. `[CLS]` only becomes a good summary through fine-tuning.

## Record

22. Do the classes separate in the base model? In the fine-tuned model?
23. At which layer does the probe accuracy jump?
24. Which pooling works better for you, CLS or mean?

---

# Lab 12: Embedding models and retrieval (the RAG building block)

**Objective:** build, by hand, the kind of model used for semantic search and RAG: one vector per whole text. See what it does that the base encoder from Lab 11 does not.

## Concepts

Re-read the glossary group *Search and retrieval*, and the entries *pooling*, *cosine similarity* and *encoder model*.

- **Two meanings of "embedding".** Up to now, "embedding" meant the **embedding table**, the lookup at the start of a model (Lab 4). In search and RAG, an **embedding model** is a whole model that maps a *full text* to **one vector**. It is built from an encoder, plus pooling, plus special training.
- **What the training does.** Contrastive training uses pairs that belong together, such as a question and the passage that answers it. It pulls those vectors together and pushes unrelated ones apart. A base encoder like DeBERTa was never trained to do that, which is why its pooled vectors (Lab 11) are poor for search.
- **Normalizing.** The output is rescaled to length 1. Then the dot product of two vectors is exactly their cosine similarity, so comparing every text with every other is one matrix multiply: `E @ E.T`.
- **How RAG uses it.** Documents are split into chunks and embedded once. Each question is embedded with the **same** model, the nearest chunk vectors are found, and a separate language model writes the answer using those chunks:

<!-- figure:rag-flow:start -->
```text
  A. Index (once):     Documents -> Chunks -> [Embedding model] -> Vector database
                                                                          |
  B. Question (each):  Question -> [Embedding model, SAME one] -> Nearest vectors -> LLM -> Answer
```
<!-- figure:rag-flow:end -->

The model used here is `sentence-transformers/all-MiniLM-L6-v2`, a small public embedding model: 6 layers, 384 numbers per token, 12 heads of 32. The labs so far used the big base encoder, so this is a deliberately different, smaller model. This lab uses `texts`, `y`, `embed_all`, `base` and `tok` from Lab 11 for the comparison, so run Lab 11 first (at least the data and the `embed_all` function).

## Procedure

**Step 1: load the embedding model.**

```python
from transformers import AutoTokenizer, AutoModel

ename = "sentence-transformers/all-MiniLM-L6-v2"
etok = AutoTokenizer.from_pretrained(ename)      # this model's own tokenizer, not DeBERTa's
emodel = AutoModel.from_pretrained(ename).float().eval()    # .float() and .eval() as in Lab 2

c = emodel.config
print(c.hidden_size, c.num_hidden_layers, c.num_attention_heads)   # 384 6 12
```

**Step 2: write the embedding function.** This is pooling (Lab 11) plus normalizing.

```python
def embed(texts):
    # Tokenize a batch; padding=True pads to the longest text. max_length=256 is this
    # model's design limit: longer text is cut off.
    e = etok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    hid = emodel(**e).last_hidden_state                  # (n_texts, tokens, 384): one vector per token
    m = e["attention_mask"].unsqueeze(-1).float()        # (n_texts, tokens, 1): 1 = real token, 0 = padding
    pooled = (hid * m).sum(1) / m.sum(1)                 # masked mean over tokens -> (n_texts, 384)
    return F.normalize(pooled, dim=-1)                   # rescale each row to length 1
```

**Step 3: embed the 15 tickets from Lab 11 and check the result.**

```python
E = embed(texts)                              # one row per ticket
print(tuple(E.shape))                         # (15, 384)
print(E.norm(dim=-1)[:3])                     # each row has length 1.0
```

**Step 4 (optional): check against the library.** The `sentence-transformers` library packages exactly this recipe.

```python
%pip install sentence-transformers            # then restart the kernel if the import fails
from sentence_transformers import SentenceTransformer
st = SentenceTransformer(ename, device="cpu")
E_lib = st.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
print(torch.allclose(E, E_lib, atol=1e-4))    # True: the library does the same three steps
```

**Step 5: similarity of every ticket to every other.**

```python
S = E @ E.T                                   # (15, 15): row i, column j = similarity of ticket i and ticket j
print(S.diag()[:3])                           # each ticket vs itself: 1.0

plt.figure(figsize=(6, 5))
plt.imshow(S, cmap="viridis")
plt.colorbar()
plt.xlabel("ticket number"); plt.ylabel("ticket number")
plt.title("ticket-to-ticket similarity (MiniLM)")
plt.show()
```

**Step 6: a tiny search engine.**

```python
def search(query, M, embed_fn, k=3):
    q = embed_fn([query])                     # embed the question with the SAME model: shape (1, dim)
    scores = (M @ q.T).squeeze(1)             # similarity of every ticket to the question: shape (15,)
    top = scores.topk(k)                      # the k highest scores and their positions
    # Long form of the comprehension:
    #     out = []
    #     for v, i in zip(top.values, top.indices):
    #         out.append((texts[i], round(v.item(), 3)))
    return [(texts[i], round(v.item(), 3)) for v, i in zip(top.values, top.indices)]

for query in ["customers cannot pay for their orders", "typo on a web page", "reports take a long time to load"]:
    print("\nQUERY:", query)
    for r in search(query, E, embed):
        print("  ", r)
```

**Step 7: the same search with the base DeBERTa from Lab 11.**

```python
# Mean-pooled last layer of the base model, normalized, using the embed_all function from Lab 11.
def bembed(ts):
    return F.normalize(torch.tensor(embed_all(base, ts, pool="mean")[-1]), dim=-1)

Bn = bembed(texts)                            # (15, 768)

for query in ["customers cannot pay for their orders", "typo on a web page"]:
    print("\nQUERY:", query)
    for r in search(query, Bn, bembed):
        print("   deberta", r)
```

**Step 8: put numbers on the difference.**

```python
n = len(texts)
off = ~torch.eye(n, dtype=torch.bool)                  # True everywhere except the diagonal (a ticket vs itself)

for label, M in [("MiniLM", E), ("DeBERTa-base", Bn)]:
    S2 = M @ M.T
    vals = S2[off]                                     # similarity between two DIFFERENT tickets
    print(f"{label:13s} average {vals.mean():.3f}   spread (max-min) {vals.max() - vals.min():.3f}")

# Does a ticket's nearest neighbour have the same priority? (chance is about 29%)
yt = torch.tensor(y)
for label, M in [("MiniLM", E), ("DeBERTa-base", Bn)]:
    S2 = (M @ M.T).clone()
    S2.fill_diagonal_(-1)                              # a ticket must not match itself
    nn = S2.argmax(1)                                  # the nearest other ticket for each ticket
    print(label, "nearest neighbour has the same priority:", (yt[nn] == yt).float().mean().item())
```

## Reading the output

- **Step 3:** the shape is `(15, 384)` and every row has length 1. The vector length 384 is this model's hidden size.
- **Step 5:** the diagonal is 1 (each ticket matches itself). Unlike the attention heatmaps in Lab 6, this matrix is **symmetric** (ticket *i* against *j* equals *j* against *i*), and its rows do **not** sum to 1. Look for tickets about the same subject standing out as brighter off-diagonal cells.
- **Step 6:** the top results should make sense for each query. In one run, "customers cannot pay for their orders" returned "Prod checkout failing for all customers" (0.47), "Payment processing returning 500 errors" (0.42) and "Data loss when saving orders" (0.39), and "reports take a long time to load" returned the two report-related tickets first.
- **Step 7 and 8:** the base model's scores are all bunched near 0.9, so a good match barely stands out from a bad one. In one run the average similarity between two different tickets was about **0.07 for MiniLM and 0.92 for base DeBERTa**, and the spread was about 0.58 versus 0.21. This is the same effect you measured in Lab 5: in the upper layers of a base model, every vector shares a large common component (anisotropy), so raw cosine similarity of pooled base vectors is a weak search signal. Contrastive training removes it.
- **Nearest-neighbour priority:** in one run about 80% for MiniLM and about 33% for base DeBERTa (chance is about 29%). This toy set was written by hand, with similar-sounding tickets sharing a priority, so MiniLM benefits from topic. Treat it as an illustration, not a benchmark: swap in real tickets before drawing conclusions.
- The base model still *contains* the information (the linear probes in Lab 11 can read it). The difference is that its vectors aren't arranged so that nearby means similar. That arrangement is what contrastive training adds.

**Optional follow-ups**
- Try your own queries and tickets. Which queries does MiniLM get wrong?
- Embed a long paragraph (more than 256 word pieces) and check that the end is ignored.
- Compare `pool="cls"` with mean pooling by editing `embed`. MiniLM was trained with mean pooling, so CLS should be worse.

## Record

25. What are the shape of `E` and the length of each row? Why is the length 1?
26. For two normalized vectors, what does their dot product equal?
27. What are the top-3 tickets for "customers cannot pay for their orders"? Do they make sense?
28. Compare the average similarity between different tickets for MiniLM and base DeBERTa. Which model separates tickets better, and why?

---

# Quick reference

| Command | Shows you |
|---|---|
| `tok(text, return_tensors="pt")` | token IDs plus the attention mask |
| `tok.tokenize(text)` | just the string pieces (no specials, no IDs) |
| `tok.convert_ids_to_tokens(ids)` | the actual token pieces |
| `model.config` | the architecture's hyperparameters |
| `print(model)` | the module tree |
| `model.named_children()` / `named_modules()` | top-level / all submodules |
| `model.named_parameters()` | every weight matrix with its name and shape |
| `sum(p.numel() for p in model.parameters())` | parameter count |
| `model(**enc, output_hidden_states=True)` | per-token vectors after every layer |
| `model(**enc, output_attentions=True)` | the attention matrices (load with `attn_implementation="eager"`) |
| `module.register_forward_hook(fn)` | capture any internal output (call `.remove()` on the handle after) |
| `logits.softmax(-1)` / `.argmax(-1)` | probabilities / the winning label |
| `pm.print_trainable_parameters()` | what LoRA will actually train |

---

# Answer key

Exact numbers depend on library versions, so approximate values are marked ≈. **Try each question before reading.**

1. ≈10 to 11 tokens, including `[CLS]` and `[SEP]`. Count the list printed in Lab 1.
2. Usually 3 to 4 pieces.
3. ≈128,000. `len(tok)` is slightly below `config.vocab_size` (128,100), because the embedding table is padded beyond the last real token.
4. `hidden_size` 768, `num_hidden_layers` 12, `num_attention_heads` 12, `intermediate_size` 3072.
5. 768 / 12 = 64.
6. ≈53%: about 98M of the ≈184M parameters. Roughly half of this "base" model is the dictionary.
7. ≈7.1M per layer (Q, K, V, O: 4 × 768² ≈ 2.36M; MLP: 2 × 768 × 3072 ≈ 4.72M; plus small bias and norm terms). 12 × 7.1M ≈ 85M, plus ≈98M of embeddings, plus a little for relative-position embeddings and norms, gives ≈184M.
8. Varies. Expect mostly spelling or morphological relatives (other forms of the same word, capitalization variants), with a few meaning-related words.
9. ≈1.000 at layer 0, roughly 0.5 around layer 6, and roughly 0.9 at layer 12. It is a dip followed by a rebound, not a steady fall. Exact values vary with your sentences.
10. The biggest drop is at the very start (layer 0 to layer 1, for example ≈1.00 to ≈0.74). The lowest point is in the middle layers (≈0.5 around layers 4 to 6). It does not keep falling: it climbs back to ≈0.9 by layer 12. The baseline in Step 3 shows most of that climb is just all vectors becoming more similar: in one run the different-word similarity rose from ≈0.12 to ≈0.76, and the same-word advantage over it shrank from ≈0.89 at layer 0 to ≈0.14 at layer 12.
11. Varies. For example, one head attending mostly to `[SEP]` (an attention sink), another to the neighbouring tokens.
12. Varies, and the special tokens usually dominate. In one run `[CLS]` put 0.42 on itself and 0.29 on `[SEP]`, while content words got small shares (`▁Prod` 0.06, then `▁checkout`, `▁deploy`, `▁failing` at about 0.03 to 0.04 each).
13. `(1, 12, tokens, 64)`: batch, heads, tokens, head dimension.
13a. `(768, 768)`, and row 5 (the sixth row, counting from 0). Query feature 5 is that row of `W` matched against the token's 768 numbers, plus `b[5]`.
13b. All 768 change (the printout shows 768 for every token). With no mixing only 64 would change, the slots belonging to head 3.
14. `classifier.weight` is `(3, 768)`; `pooler.dense.weight` is `(768, 768)`.
15. ≈593k: pooler 768 × 768 + 768 = 590,592, plus classifier 3 × 768 + 3 = 2,307, total 592,899. Versus ≈184M for the backbone, so about 0.3%.
16. The argmax does not change; the confidence does. Lower T sharpens the distribution and higher T flattens it, but the order never changes.
17. ≈896 hidden size, 24 layers, 14 query heads, 2 KV heads. Check your config.
18. Varies. In one run: " last" (0.09), " " (0.06), " update" (0.05), " latest" (0.04), " release" (0.04). All plausible continuations.
19. Varies. In one run: low 0.31, medium 0.25, high 0.44, so it already leaned towards "high" for the outage ticket. The point is that it produces a distribution over exactly three options with zero training.
20. **≈1.04M trainable parameters, ≈0.56% of the total.** Breakdown: LoRA matrices 36 adapters × 12,288 = 442,368, plus the trainable copies of `pooler` and `classifier` ≈ 592,899. The total shown by PEFT (≈185M) includes the frozen originals of those head layers as well as the trainable copies. (The short edition of this manual said "roughly 1 to 2%"; 0.56% is the figure these numbers give.)
21. r × (768 + 768) = 8 × 1,536 = 12,288, versus 589,824 for the full matrix (≈2%).
22. Base: overlapping. Fine-tuned: should be clearly separated. If not, investigate your training run.
23. Depends on your data, and on the toy set it is noisy. In one run (mean pooling, with the scaler): 0.80 at layer 0, a peak of 1.00 at layer 4, and 0.53 at layer 12. With real data, the climb is often in the middle layers.
24. On the base model mean pooling wins clearly (in one run CLS stayed between 0.33 and 0.67). After fine-tuning the gap tends to close.
25. `(15, 384)`: 15 tickets, 384 numbers each. Every row is divided by its own length in `F.normalize`, so each row has length exactly 1.
26. Their cosine similarity. For unit-length vectors the formula `(a · b) / (‖a‖ ‖b‖)` reduces to `a · b`.
27. In one run: "Prod checkout failing for all customers" (≈0.47), "Payment processing returning 500 errors" (≈0.42), "Data loss when saving orders" (≈0.39). They should be about checkout, payments and orders.
28. In one run the average similarity between different tickets was ≈0.07 for MiniLM and ≈0.92 for base DeBERTa (spread ≈0.58 versus ≈0.21). MiniLM separates better because it was trained contrastively, so unrelated texts end up far apart. Base DeBERTa's upper-layer vectors share a large common direction (anisotropy, as in Lab 5), so everything looks similar.
