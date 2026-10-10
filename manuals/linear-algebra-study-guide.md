# Linear Algebra for AI Models: An Interactive Study Guide

**Goal:** understand, from first principles, the handful of linear algebra operations that do almost all of the work inside a neural network, and be able to do each one by hand.

**The one-sentence version:** a neural network stores what it has learned in matrices, represents every input as a vector, and spends nearly all of its compute multiplying the two together, with a simple nonlinear function in between.

**How to use this guide**

1. Skim *Part 0: Glossary* once. You don't need to memorize it: every glossary term in the text has a dotted underline, and every symbol in italic math type (such as {{W}} or {{x}}) shows its definition when you hover, focus or tap it.
2. Read Parts 1 to 8 in order. Each part has a *how to do it* recipe, a *why it is used* note, at least one picture, and practice problems.
3. Do every practice problem with pencil and paper **before** opening its solution.
4. Play with the four interactive widgets (matrix multiplication, transformations, dot product, softmax). Predict what will happen before you move a control.

**Conventions.** Vectors are lowercase italic ({{u}}, {{x}}), matrices are uppercase italic ({{W}}, {{M}}), and plain numbers (scalars) are lowercase Greek or italic. Counting starts at 1 in the math (row 1, column 1). Python and PyTorch count from 0, so math row 1 is Python index 0.

---

## Contents

- Part 0: Glossary (formal definitions)
- Part 0.5: Notation cheat sheet
- Part 1: Vectors and the dot product
- Part 2: Matrices and the transpose
- Part 3: Multiplying matrices
- Part 4: Matrices as transformations
- Part 5: A neural network layer
- Part 6: Attention
- Part 7: Derivatives, Jacobians and backpropagation
- Part 8: Rank, SVD and LoRA
- Practice set
- Quick reference

---

# Part 0: Glossary (formal definitions)

Terms are grouped by topic. Symbols come first because the rest of the guide uses them constantly. Use the filter box to search.

## Variables and symbols

### Data flowing through a model

**x (input vector).** The vector that goes *into* an operation or a layer. Shape `(n,)`: {{n}} numbers. Its {{j}}-th entry is written {{x}}<sub>j</sub>.

**y (output vector).** The vector that comes *out* of an operation, for example {{y}} = {{W}}{{x}}. Shape `(m,)`: {{m}} numbers.

**z (pre-activation, or scores).** The result of a layer's linear part before the nonlinearity: {{z}} = {{W}}{{x}} + {{b}}. When it is fed to softmax it is called the *logits* or *scores*.

**h (hidden vector).** A layer's output *after* the activation function: {{h}} = {{σ}}({{z}}). "Hidden" because it is an intermediate result, not the model's final output.

**X (batch matrix).** Many input vectors stacked as the **rows** of one matrix, one row per example. Shape `(batch, n)`.

**Z (batch pre-activations).** The batch version of {{z}}: one row of pre-activations per example, {{Z}} = {{X}}{{W}}{{ᵀ}} + {{b}}. Shape `(batch, m)`.

**t (target vector).** The answer the model should have produced, used by the loss to measure the error.

**T (number of tokens).** The length of a sequence: how many tokens (word pieces) the model is processing at once. Used for attention shapes such as `(T, d)`.

### Learned parameters

**W (weight matrix).** A layer's learned matrix of shape `(m, n)`: one **row** per output, one **column** per input. Entry {{W}}<sub>ij</sub> is how much input {{j}} contributes to output {{i}}.

**b (bias vector).** A layer's learned vector of shape `(m,)`, added after the matrix product so the output can be shifted even when the input is all zeros.

**A (LoRA down-projection).** In LoRA, the small learned matrix of shape `(r, d)` that squeezes a {{d}}-number input down to {{r}} numbers.

**B (LoRA up-projection).** In LoRA, the small learned matrix of shape `(d, r)` that expands {{r}} numbers back up to {{d}}. Initialized to zeros so that training starts from the unchanged model.

**Δ (change in).** "Delta", a change. {{Δ}}{{W}} is the change added to a weight matrix, so the new weights are {{W}} + {{Δ}}{{W}}.

### Generic objects used in the explanations

**u (generic vector).** A vector used in examples, usually the first of two.

**w (second generic vector).** A second vector used in examples. Lowercase {{w}} is a vector; uppercase {{W}} is the weight matrix. They are different things.

**M (generic matrix).** A matrix used in examples. Its entry in row {{i}}, column {{j}} is {{M}}<sub>ij</sub>.

**N (second generic matrix).** A second matrix used in examples, for products such as {{M}}{{N}}.

**I (identity matrix).** The square matrix with 1s on the diagonal and 0s everywhere else. Multiplying by it changes nothing: {{I}}{{x}} = {{x}}.

**m (number of rows).** The number of rows of a matrix, or the length of an output vector.

**n (number of columns).** The number of columns of a matrix, or the length of an input vector.

**p (columns of the second matrix).** In a product of an `(m, n)` matrix and an `(n, p)` matrix, {{p}} is the second matrix's column count, and the result is `(m, p)`.

**i (row index).** Which row (or which output). Runs from 1 to {{m}}.

**j (column index).** Which column (or which input). Runs from 1 to {{n}}.

**θ (angle).** "Theta", the angle between two vectors, in degrees or radians.

**λ (eigenvalue).** "Lambda", the scale factor in {{M}}{{v}} = {{λ}}{{v}}: how much an eigenvector is stretched.

### Attention

**q (query vector).** The vector a token uses to ask "what am I looking for?". Length {{d}}.

**k (key vector).** The vector a token offers to be matched against: "what do I contain?". Length {{d}}. With several tokens they are numbered {{k|k₁}}, {{k|k₂}}, and so on.

**v (value vector).** The vector a token hands over when it is attended to: "what do I pass on?". Also used in Part 4 for an eigenvector, where {{M}}{{v}} = {{λ}}{{v}}.

**Q (query matrix).** All the query vectors stacked as rows, shape `(T, d)`.

**K (key matrix).** All the key vectors stacked as rows, shape `(T, d)`.

**V (value matrix).** All the value vectors stacked as rows, shape `(T, d_v)`. In the singular value decomposition (Part 8), {{V}} instead names the matrix of right singular vectors in {{M}} = {{U}}{{Σ}}{{V}}{{ᵀ}}.

**d (vector length in attention).** The number of entries in each query and key vector (the *head dimension*). Attention scores are divided by {{√d}}.

**√d (square root of d).** The scaling factor in attention. Dividing by it keeps scores in a sensible range as {{d}} grows (Part 6 explains why).

**s (attention scores).** The vector of raw similarities between one query and every key: {{s}}<sub>i</sub> = {{q}}{{·}}{{k|k}}<sub>i</sub> / {{√d}}.

**α (attention weights).** "Alpha", the scores after softmax. All positive, adding up to 1. {{α}}<sub>i</sub> is the fraction of attention given to token {{i}}.

**o (attention output).** The weighted blend of value vectors: {{o}} = {{Σ}}<sub>i</sub> {{α}}<sub>i</sub> {{v}}<sub>i</sub>.

**τ (temperature).** "Tau", a positive number that the scores are divided by before softmax. Below 1 sharpens the distribution; above 1 flattens it.

### Learning

**L (loss).** A single number measuring how wrong the output is. Training tries to make it small.

**δ (upstream gradient).** "Delta", shorthand for the gradient of the loss with respect to a layer's output, {{δ}} = {{∂}}{{L}}/{{∂}}{{y}}. It is what flows backward into the layer during backpropagation.

**J (Jacobian matrix).** The matrix of all first partial derivatives of a vector function. For {{y}} = f({{x}}), {{J}}<sub>ij</sub> = {{∂}}{{y}}<sub>i</sub>/{{∂}}{{x}}<sub>j</sub>.

**η (learning rate).** "Eta", the small positive step size in gradient descent: {{W}} ← {{W}} − {{η}} {{∂}}{{L}}/{{∂}}{{W}}.

**σ (activation function).** "Sigma", the nonlinear function applied to every entry of {{z}} separately, such as ReLU. (The capital {{Σ}} is a different symbol.)

### Decompositions and sizes

**r (rank).** The rank of a matrix, or in LoRA the chosen small rank of the update {{B}}{{A}}.

**U (left singular vectors).** In the singular value decomposition {{M}} = {{U}}{{Σ}}{{V}}{{ᵀ}}, the orthogonal matrix applied last: a final rotation (or reflection).

**Σ (sum, or singular-value matrix).** Capital sigma has two jobs. As an operator, {{Σ}}<sub>j</sub> means "add up over every value of {{j}}". In the singular value decomposition it is the diagonal matrix of singular values (the stretch factors).

### Operators and notation

**ᵀ (transpose).** A superscript T: {{M}}{{ᵀ}} is {{M}} flipped across its diagonal, so rows become columns. ({{M}}{{ᵀ}})<sub>ij</sub> = {{M}}<sub>ji</sub>.

**· (dot).** Between two vectors, the dot product: {{u}}{{·}}{{w}} = {{Σ}}<sub>j</sub> {{u}}<sub>j</sub>{{w}}<sub>j</sub>. Between numbers, ordinary multiplication.

**⊙ (elementwise product).** Multiply two same-shaped vectors or matrices position by position. Also called the Hadamard product.

**‖u‖ (norm, or length).** The length of a vector: ‖{{u}}‖ = √({{u}}{{·}}{{u}}) = √({{u}}<sub>1</sub>² + {{u}}<sub>2</sub>² + …).

**∂ (partial derivative).** "Partial". {{∂}}{{L}}/{{∂}}{{x}}<sub>j</sub> is how fast {{L}} changes when only {{x}}<sub>j</sub> moves and every other input is held still.

**∇ (gradient).** "Nabla" or "del". {{∇}}<sub>x</sub>{{L}} is the vector of all partial derivatives of {{L}} with respect to the entries of {{x}}. Same shape as {{x}}.

**ℝ (real numbers).** The set of ordinary decimal numbers. ℝ<sup>n</sup> means "vectors of {{n}} real numbers", and ℝ<sup>m×n</sup> means "matrices with {{m}} rows and {{n}} columns".

**exp (exponential).** exp(a) = e<sup>a</sup>, where e ≈ 2.718. Always positive, and grows quickly. Softmax uses it.

## Objects

**Scalar.** A single number, such as 3 or −0.5. Shape `()`.

**Vector.** An ordered list of numbers, shape `(n,)`. Geometrically, an arrow from the origin to a point in {{n}}-dimensional space. In a model, one token's or one example's description.

**Column vector.** A vector written vertically, as a matrix with one column, shape `(n, 1)`. In math, "vector" means column vector unless stated otherwise, which is why a layer is written {{W}}{{x}}.

**Row vector.** A vector written horizontally, as a matrix with one row, shape `(1, n)`. The transpose of a column vector: {{x}}{{ᵀ}}.

**Matrix.** A rectangular grid of numbers with {{m}} rows and {{n}} columns, shape `(m, n)`. It can be read three ways: as a table of numbers, as a stack of row vectors, or as a function that turns vectors into other vectors (a linear map).

**Tensor.** A grid of numbers with any number of axes: 0 axes is a scalar, 1 a vector, 2 a matrix, 3 or more just "tensor". A batch of token vectors has shape `(batch, tokens, hidden)`.

**Shape.** The size of each axis, listed in order. A matrix with 2 rows and 3 columns has shape `(2, 3)`, also written 2 × 3 ("two by three"). Rows always come first.

**Dimension.** Two meanings. The dimension *of a vector* is its length (a 768-dimensional vector has 768 numbers). The dimension *of a tensor* can also mean its number of axes (a matrix is 2-dimensional).

**Entry (element).** One number inside a vector or matrix, located by its index or indices. {{M}}<sub>23</sub> is the entry in row 2, column 3.

**Square matrix.** A matrix with as many rows as columns, shape `(n, n)`. Only square matrices can have a determinant, an inverse or eigenvalues.

**Diagonal matrix.** A square matrix whose only nonzero entries are on the main diagonal (top-left to bottom-right). Multiplying by it scales each coordinate separately.

**Identity matrix.** The diagonal matrix with all 1s on the diagonal, written {{I}}. The "do nothing" matrix: {{I}}{{x}} = {{x}} and {{M}}{{I}} = {{M}}.

**Symmetric matrix.** A square matrix equal to its own transpose: {{M}} = {{M}}{{ᵀ}}. Its eigenvalues are always real and its eigenvectors can be chosen perpendicular.

## Vector operations

**Vector addition.** Add matching entries: [[1, 2]] + [[3, 5]] = [[4, 7]]. Both vectors must have the same length. Geometrically, put the arrows tip to tail.

**Scalar multiplication.** Multiply every entry by the same number: 3 × [[1, −2]] = [[3, −6]]. Geometrically, stretch (or flip, if negative) the arrow.

**Linear combination.** A weighted sum of vectors: a{{u}} + c{{w}} for numbers a and c. Matrix-vector products, attention outputs and LoRA updates are all linear combinations.

**Dot product (inner product).** Multiply matching entries and add the results: {{u}}{{·}}{{w}} = {{Σ}}<sub>j</sub> {{u}}<sub>j</sub>{{w}}<sub>j</sub>. The answer is one number. It equals ‖{{u}}‖ ‖{{w}}‖ cos {{θ}}, so it is large when two vectors are long and point the same way.

**Norm.** The length of a vector, ‖{{u}}‖ = √({{u}}{{·}}{{u}}), also called the L2 norm or Euclidean length. It is Pythagoras' theorem extended to any number of entries.

**Unit vector.** A vector of length 1. Any nonzero vector can be *normalized* to a unit vector by dividing it by its norm: {{u}} / ‖{{u}}‖.

**Cosine similarity.** The dot product of two vectors divided by both lengths: cos {{θ}} = {{u}}{{·}}{{w}} / (‖{{u}}‖ ‖{{w}}‖). Ranges from −1 (opposite) through 0 (perpendicular) to 1 (same direction). Ignores length, measures direction only.

**Orthogonal.** Perpendicular: two vectors are orthogonal when their dot product is 0. An *orthogonal matrix* has columns that are mutually perpendicular unit vectors; it only rotates or reflects, never stretches.

**Elementwise product (Hadamard product).** Multiply matching entries and keep them separate: [[1, 2, 3]] {{⊙}} [[4, −1, 2]] = [[4, −2, 6]]. Adding up the result gives the dot product.

**Outer product.** A column vector times a row vector, {{u}}{{w}}{{ᵀ}}. The result is a whole matrix with entry ({{i}}, {{j}}) = {{u}}<sub>i</sub>{{w}}<sub>j</sub>. Every outer product of nonzero vectors has rank 1.

## Matrix operations

**Transpose.** Flip a matrix across its main diagonal, so row {{i}} becomes column {{i}}: ({{M}}{{ᵀ}})<sub>ij</sub> = {{M}}<sub>ji</sub>. An `(m, n)` matrix becomes `(n, m)`. Rules: ({{M}}{{ᵀ}}){{ᵀ}} = {{M}}, and ({{M}}{{N}}){{ᵀ}} = {{N}}{{ᵀ}}{{M}}{{ᵀ}} (the order reverses).

**Matrix-vector product.** {{W}}{{x}}: output entry {{i}} is the dot product of row {{i}} of {{W}} with {{x}}. Equivalently, a linear combination of {{W}}'s columns, weighted by the entries of {{x}}. Needs {{W}}'s column count to equal {{x}}'s length.

**Matrix multiplication.** {{M}}{{N}}: entry ({{i}}, {{j}}) is row {{i}} of {{M}} dotted with column {{j}} of {{N}}. An `(m, n)` matrix times an `(n, p)` matrix gives `(m, p)`. Order matters: {{M}}{{N}} usually differs from {{N}}{{M}}. Doing {{M}}{{N}} to a vector means "apply {{N}} first, then {{M}}".

**Linear map (linear transformation).** A function f that respects addition and scaling: f({{u}} + {{w}}) = f({{u}}) + f({{w}}) and f(c{{u}}) = c f({{u}}). Every linear map between vector spaces of fixed size is multiplication by some matrix, and every matrix defines one.

**Basis vectors.** The standard basis vectors are the columns of the identity matrix: [[1, 0]] and [[0, 1]] in 2-D. Every vector is a linear combination of them, which is why a matrix is fully described by where it sends them: those images are its columns.

**Determinant.** A single number computed from a square matrix. For 2 × 2, det [[a, b; c, d]] = ad − bc. Geometrically, the factor by which the matrix scales area (or volume). Negative means the space is flipped; zero means it is squashed flat, and the matrix has no inverse.

**Inverse.** The matrix {{M}}<sup>−1</sup> that undoes {{M}}: {{M}}<sup>−1</sup>{{M}} = {{I}}. It exists only for square matrices with a nonzero determinant. For 2 × 2: [[a, b; c, d]]<sup>−1</sup> = (1 / (ad − bc)) [[d, −b; −c, a]].

**Rank.** The number of linearly independent columns of a matrix (equal to the number of independent rows). Geometrically, the dimension of the space the outputs can reach. A full-rank `(n, n)` matrix has rank {{n}}; an outer product has rank 1.

**Eigenvector.** A nonzero vector {{v}} whose direction a square matrix does not change: {{M}}{{v}} = {{λ}}{{v}}. The matrix only stretches it (or flips it, if {{λ}} is negative).

**Eigenvalue.** The stretch factor {{λ}} that goes with an eigenvector. Found by solving det({{M}} − {{λ}}{{I}}) = 0.

**Singular value decomposition (SVD).** A factorization of *any* matrix as {{M}} = {{U}}{{Σ}}{{V}}{{ᵀ}}: a rotation, then stretching along the axes by the *singular values* (the diagonal of {{Σ}}), then another rotation. Keeping only the {{r}} largest singular values gives the best possible rank-{{r}} approximation of {{M}}.

## Calculus for learning

**Derivative.** For a function of one number, the rate of change: how much the output moves per unit of input movement, at a given point. The derivative of a² is 2a.

**Partial derivative.** For a function of several numbers, the derivative with respect to one of them while the others are held fixed. Written with {{∂}}.

**Gradient.** The vector (or matrix) of all partial derivatives of a single number, usually the loss, with respect to the entries of a vector or matrix. It has the **same shape** as the thing you differentiate with respect to, and points in the direction that increases the loss fastest.

**Jacobian.** For a function from {{n}} numbers to {{m}} numbers, the `(m, n)` matrix of all partial derivatives: row {{i}} describes how output {{i}} changes, column {{j}} how every output responds to input {{j}}. For {{y}} = {{W}}{{x}} the Jacobian with respect to {{x}} is {{W}} itself.

**Chain rule.** The derivative of a composition is the product of the local derivatives: if {{y}} depends on {{x}} and {{L}} depends on {{y}}, then {{∂}}{{L}}/{{∂}}{{x}} = {{J}}{{ᵀ}} {{δ}}, where {{J}} is {{y}}'s Jacobian and {{δ}} = {{∂}}{{L}}/{{∂}}{{y}}.

**Loss.** A function that turns the model's output and the correct answer into one number measuring the error, such as the squared error ½‖{{y}} − {{t}}‖² or the cross-entropy.

**Forward pass.** Running the model on an input, start to finish, to compute its output and the loss.

**Backpropagation.** The algorithm that computes the gradient of the loss with respect to every parameter by applying the chain rule backward through the network, from the loss to the input, reusing each intermediate result.

**Gradient descent.** The training rule: move each parameter a small step against its gradient, {{W}} ← {{W}} − {{η}} {{∂}}{{L}}/{{∂}}{{W}}, and repeat.

**Learning rate.** The step size {{η}} in gradient descent. Too large and training overshoots and diverges; too small and it crawls.

## In models

**Parameter (also called a weight).** A number inside the model that training adjusts. The entries of every {{W}} and {{b}} are parameters.

**Bias.** The vector {{b}} added after a matrix product, letting a layer shift its outputs. Unrelated to "bias" in the statistical or social sense.

**Layer.** The repeating building block of a network. A linear (or dense, or fully connected) layer computes {{z}} = {{W}}{{x}} + {{b}}; usually an activation function follows.

**Activation function.** The nonlinear function {{σ}} applied to every entry after a linear layer. Without it, stacked layers would collapse into one matrix (Part 5).

**ReLU.** Rectified linear unit, the most common activation function: ReLU(a) = max(0, a). Keeps positive numbers, replaces negatives with 0. Its derivative is 1 for positive inputs and 0 for negative ones.

**Logits.** The raw, unnormalized scores a model produces before softmax. Any real numbers, positive or negative.

**Softmax.** Turns a vector of scores into probabilities: softmax({{z}})<sub>i</sub> = exp({{z}}<sub>i</sub>) / {{Σ}}<sub>j</sub> exp({{z}}<sub>j</sub>). Every output is positive, they add to 1, and the largest score gets the largest share.

**Temperature.** A divisor {{τ}} applied to the scores before softmax: softmax({{z}} / {{τ}}). Low temperature makes the result sharper (closer to all-or-nothing); high temperature makes it flatter.

**Embedding.** A learned vector that represents a discrete thing, such as a token. A model's embedding table is a matrix with one row per vocabulary entry; looking up a token's vector is selecting a row.

**Attention.** The operation that lets each token build its new vector as a weighted blend of other tokens' vectors, with weights from dot-product similarity: softmax({{Q}}{{K}}{{ᵀ}} / {{√d}}){{V}}.

**Query, key and value vectors.** Three vectors computed from each token by three learned matrices. The query says what the token is looking for, the key says what it offers to be matched on, and the value is what it passes on when chosen.

**LoRA (Low-Rank Adaptation).** A way to fine-tune a large model by freezing every original weight matrix {{W}} and learning only a low-rank update {{Δ}}{{W}} = {{B}}{{A}}, where {{B}} is `(d, r)` and {{A}} is `(r, d)` with small {{r}}. Trains a tiny fraction of the parameters.

---

# Part 0.5: Notation cheat sheet

How to read the notation aloud, and what to do when you see it.

| You see | Read it as | It means |
|---|---|---|
| {{x}}<sub>j</sub> | "x sub j" | entry {{j}} of vector {{x}} |
| {{W}}<sub>ij</sub> | "W i j" | the entry of {{W}} in row {{i}}, column {{j}} (row first, always) |
| ℝ<sup>n</sup> | "R n" | vectors of {{n}} real numbers |
| ℝ<sup>m×n</sup> | "R m by n" | matrices with {{m}} rows and {{n}} columns |
| {{M}}{{ᵀ}} | "M transpose" | {{M}} with rows and columns swapped |
| {{u}}{{·}}{{w}} | "u dot w" | the dot product, one number |
| {{u}}{{ᵀ}}{{w}} | "u transpose w" | the same dot product, written as a matrix product: a `(1, n)` row times an `(n, 1)` column gives a 1 × 1 result |
| {{u}}{{w}}{{ᵀ}} | "u w transpose" | the outer product: an `(m, 1)` column times a `(1, n)` row gives an `(m, n)` matrix |
| {{M}}{{N}} | "M N" | matrix product; apply {{N}} first, then {{M}} |
| {{u}} {{⊙}} {{w}} | "u Hadamard w" | elementwise product |
| ‖{{u}}‖ | "norm of u" | length of {{u}} |
| {{Σ}}<sub>j</sub> | "sum over j" | add up the expression for every value of {{j}} |
| {{∂}}{{L}}/{{∂}}{{W}} | "partial L partial W" | gradient of the loss with respect to {{W}}; same shape as {{W}} |
| {{M}}<sup>−1</sup> | "M inverse" | the matrix that undoes {{M}} |
| ← | "becomes" | update in place (used in training rules) |
| `W @ x`, `W.T`, `a * b` | PyTorch | matrix product, transpose, elementwise product |

---

# Part 1: Vectors and the dot product

**Objective:** add, scale and dot vectors by hand, and know what the dot product measures.

## What a vector is

A vector is an ordered list of numbers. Order matters: [[3, 2]] and [[2, 3]] are different vectors. You can read it two ways:

- **As data.** A model describes every token with a vector of, say, 768 numbers. Each entry is one learned "feature".
- **As an arrow.** In 2-D, [[3, 2]] is the arrow from the origin that goes 3 along the first axis and 2 along the second. In 768-D you can't draw it, but every geometric idea (length, angle, perpendicular) still works exactly the same way.

<!-- figure:vector-arrow:start -->
```text
   2 |      * u = [3, 2]
     |    /
     |  /
     |/________ 3
```
<!-- figure:vector-arrow:end -->

## Addition and scaling

<div class="recipe">

**How to add two vectors and scale a vector**

1. Check that the two vectors have the same length. If not, they can't be added.
2. Add position by position: [[1, 2, 3]] + [[4, −1, 2]] = [[1 + 4, 2 + (−1), 3 + 2]] = [[5, 1, 5]].
3. To scale by a number, multiply every entry by it: 2 × [[1, 2, 3]] = [[2, 4, 6]].

</div>

## The dot product

<div class="recipe">

**How to compute a dot product** {{u}}{{·}}{{w}}

1. Check that both vectors have the same length {{n}}.
2. Multiply matching positions: {{u}}<sub>1</sub>{{w}}<sub>1</sub>, {{u}}<sub>2</sub>{{w}}<sub>2</sub>, …, {{u}}<sub>n</sub>{{w}}<sub>n</sub>.
3. Add all the products. The answer is a single number, not a vector.

</div>

<div class="eq"><span class="lbl">Definition</span>{{u}} {{·}} {{w}} = {{Σ}}<sub>j</sub> {{u}}<sub>j</sub> {{w}}<sub>j</sub> = {{u}}<sub>1</sub>{{w}}<sub>1</sub> + {{u}}<sub>2</sub>{{w}}<sub>2</sub> + … + {{u}}<sub>n</sub>{{w}}<sub>n</sub></div>

Worked example with {{u}} = [[1, 2, 3]] and {{w}} = [[4, −1, 2]]:

<!-- figure:dot-product:start -->
```text
u      1     2     3
w      4    -1     2
u*w    4  + -2  +  6   =  8
```
<!-- figure:dot-product:end -->

**What the number means.** The same dot product has a geometric formula:

<div class="eq"><span class="lbl">Geometric form</span>{{u}} {{·}} {{w}} = ‖{{u}}‖ ‖{{w}}‖ cos {{θ}}</div>

where {{θ}} is the angle between the arrows. Since cos 0° = 1, cos 90° = 0 and cos 180° = −1:

- positive dot product: the vectors point the same general way (angle under 90°);
- zero: they are perpendicular (orthogonal), sharing nothing;
- negative: they point apart (angle over 90°).

## Length and cosine similarity

<div class="recipe">

**How to find a vector's length (norm)**

1. Dot the vector with itself (square each entry and add): for [[1, 2, 3]], 1 + 4 + 9 = 14.
2. Take the square root: {{‖u‖}} = √14 ≈ 3.742.

**How to find the cosine similarity and angle**

1. Compute the dot product {{u}}{{·}}{{w}}.
2. Divide by both lengths: cos {{θ}} = {{u}}{{·}}{{w}} / (‖{{u}}‖ ‖{{w}}‖).
3. For the angle, take the inverse cosine: {{θ}} = arccos(cos {{θ}}).

</div>

For our example, ‖{{w}}‖ = √(16 + 1 + 4) = √21 ≈ 4.583, so cos {{θ}} = 8 / (3.742 × 4.583) ≈ 0.467 and {{θ}} ≈ 62.2°.

<!-- widget:dot -->

<div class="why">

The dot product is the model's similarity test. Attention compares a query to every key with a dot product (Part 6), retrieval systems rank documents by cosine similarity between embedding vectors, and every output of a layer is a dot product between a row of weights and the input (Part 3). It is cheap, it is differentiable, and it says "how much do these two descriptions agree?" in one number.

</div>

## Practice

<div class="problem">

**1.1** Let {{u}} = [[2, −1, 0]] and {{w}} = [[1, 3, 5]]. Compute {{u}} + {{w}}, 3{{u}}, {{u}}{{·}}{{w}}, and ‖{{u}}‖.

<details class="practice"><summary>Show solution</summary><div>

- {{u}} + {{w}} = [[3, 2, 5]].
- 3{{u}} = [[6, −3, 0]].
- {{u}}{{·}}{{w}} = 2·1 + (−1)·3 + 0·5 = 2 − 3 + 0 = −1. Slightly negative: the vectors point a little apart.
- ‖{{u}}‖ = √(4 + 1 + 0) = √5 ≈ 2.236.

</div></details>
</div>

<div class="problem">

**1.2** What is the cosine similarity of [[1, 0]] and [[1, 1]]? What angle is that?

<details class="practice"><summary>Show solution</summary><div>

Dot product = 1·1 + 0·1 = 1. Lengths: 1 and √2. Cosine = 1/√2 ≈ 0.707, so the angle is 45°. Draw it: one arrow along the axis, one along the diagonal.

</div></details>
</div>

<div class="problem">

**1.3** Find a nonzero vector orthogonal to [[3, 4]].

<details class="practice"><summary>Show solution</summary><div>

You need 3a + 4c = 0. Swap the entries and negate one: [[4, −3]] gives 12 − 12 = 0. (Any multiple, such as [[−4, 3]], also works.)

</div></details>
</div>

---

# Part 2: Matrices and the transpose

**Objective:** read a matrix's shape and entries, and transpose it by hand.

## Reading a matrix

A matrix with {{m}} rows and {{n}} columns has shape `(m, n)`, said "m by n". Rows are horizontal, columns are vertical, and an entry's address is always (row, column):

<div class="eq">{{M}} = [[1, 2, 3; 4, 5, 6]] &nbsp; has shape (2, 3), &nbsp; {{M}}<sub>12</sub> = 2, &nbsp; {{M}}<sub>21</sub> = 4</div>

Three useful ways to see the same matrix:

1. **A table of numbers**, for example a layer's weights.
2. **A stack of row vectors.** {{M}} above is the rows [[1, 2, 3]] and [[4, 5, 6]]. A batch of data {{X}} is stored this way: one example per row.
3. **A function on vectors.** Multiplying by {{M}} turns a 3-number vector into a 2-number vector (Parts 3 and 4).

## The transpose

<div class="recipe">

**How to transpose a matrix**

1. Write the new matrix with the shape reversed: `(m, n)` becomes `(n, m)`.
2. Take row 1 of the original and write it, in order, as column 1 of the new matrix.
3. Repeat for every row: row {{i}} becomes column {{i}}.
4. Check: the entry at ({{i}}, {{j}}) of {{M}}{{ᵀ}} is the entry at ({{j}}, {{i}}) of {{M}}. The diagonal entries ({{i}} = {{j}}) don't move.

</div>

<div class="eq">[[1, 2, 3; 4, 5, 6]]{{ᵀ}} = [[1, 4; 2, 5; 3, 6]]</div>

<!-- figure:transpose:start -->
```text
[1 2 3]         [1 4]
[4 5 6]   ->    [2 5]
                [3 6]
```
<!-- figure:transpose:end -->

**Transposing vectors.** A column vector {{x}} of shape `(n, 1)` transposes to a row vector {{x}}{{ᵀ}} of shape `(1, n)`. This is why the dot product is often written {{u}}{{ᵀ}}{{w}}: a `(1, n)` row times an `(n, 1)` column is a 1 × 1 matrix, which is just the number {{u}}{{·}}{{w}}.

**Two rules to remember.**

- Transposing twice gets you back: ({{M}}{{ᵀ}}){{ᵀ}} = {{M}}.
- The transpose of a product reverses the order: ({{M}}{{N}}){{ᵀ}} = {{N}}{{ᵀ}}{{M}}{{ᵀ}}. (It has to: the shapes would not match otherwise.)

<div class="why">

The transpose shows up wherever the "direction" of data has to be flipped. Attention computes {{Q}}{{K}}{{ᵀ}} so that every query row meets every key row in a dot product. PyTorch stores a linear layer's weight as `(out, in)` and computes `x @ W.T` because the data rows come in as `(batch, in)`. And in backpropagation, gradients travel backward through a layer by multiplying by {{W}}{{ᵀ}} (Part 7).

</div>

## Practice

<div class="problem">

**2.1** Transpose [[7, 0; −2, 5; 1, 1]]. What are the shapes before and after?

<details class="practice"><summary>Show solution</summary><div>

Before `(3, 2)`, after `(2, 3)`: [[7, −2, 1; 0, 5, 1]]. Column 1 of the answer is row 1 of the original, read top to bottom: 7, 0. Check one entry: original row 2, column 1 is −2; answer row 1, column 2 is −2.

</div></details>
</div>

<div class="problem">

**2.2** Is [[2, 1; 1, 3]] symmetric? Is [[0, 1; −1, 0]]?

<details class="practice"><summary>Show solution</summary><div>

The first is symmetric: its transpose is itself (the off-diagonal entries match). The second is not: its transpose is [[0, −1; 1, 0]], which is the negative of the original. (Such a matrix is called *skew-symmetric*; this one rotates the plane 90°.)

</div></details>
</div>

---

# Part 3: Multiplying matrices

**Objective:** compute matrix-vector and matrix-matrix products by hand, check shapes first, and understand the two ways to read a product.

## Matrix times vector

<div class="recipe">

**How to compute** {{y}} = {{W}}{{x}}

1. Check shapes: {{W}} is `(m, n)` and {{x}} must have length {{n}}. The result {{y}} has length {{m}}.
2. For each row {{i}} of {{W}}: dot that row with {{x}}. That number is {{y}}<sub>i</sub>.
3. Stack the {{m}} answers into {{y}}.

</div>

<div class="eq"><span class="lbl">Entry formula</span>{{y}}<sub>i</sub> = {{Σ}}<sub>j</sub> {{W}}<sub>ij</sub> {{x}}<sub>j</sub></div>

Example: {{W}} = [[1, 0, 2; 0, 1, −1]], {{x}} = [[1; 2; 3]].

- Row 1: 1·1 + 0·2 + 2·3 = 7.
- Row 2: 0·1 + 1·2 + (−1)·3 = −1.

So {{W}}{{x}} = [[7; −1]]. There is a second, equally correct way to read it:

<!-- figure:matvec-views:start -->
```text
Row view:     y_i = (row i of W) . x
Column view:  y   = x_1 (col 1) + x_2 (col 2) + x_3 (col 3)
```
<!-- figure:matvec-views:end -->

**The column view** says {{W}}{{x}} is a linear combination of {{W}}'s columns, with the entries of {{x}} as the weights. It is the key to Part 4: the columns of a matrix are where it sends the basis vectors, and every output is built from them.

## Matrix times matrix

<div class="recipe">

**How to compute** {{M}}{{N}}

1. Write the shapes side by side: `(m, n) · (n, p)`. The inner two numbers must be equal, or the product does not exist. The result is `(m, p)`, the outer two.
2. For each output position ({{i}}, {{j}}): take row {{i}} of {{M}} and column {{j}} of {{N}}, and dot them.
3. Work row by row to stay organized. There are {{m}} × {{p}} entries, each costing {{n}} multiplications.

</div>

<div class="eq"><span class="lbl">Entry formula</span>({{M}}{{N}})<sub>ij</sub> = {{Σ}}<sub>k</sub> {{M}}<sub>ik</sub> {{N}}<sub>kj</sub></div>

<!-- figure:matmul-shapes:start -->
```text
(m x n) . (n x p) = (m x p)
entry (i, j) = row i of M . column j of N
```
<!-- figure:matmul-shapes:end -->

Worked example: {{M}} = [[1, 2; 3, 4]], {{N}} = [[5, 6; 7, 8]].

- (1, 1): row [[1, 2]] · column [[5; 7]] = 5 + 14 = 19
- (1, 2): row [[1, 2]] · column [[6; 8]] = 6 + 16 = 22
- (2, 1): row [[3, 4]] · column [[5; 7]] = 15 + 28 = 43
- (2, 2): row [[3, 4]] · column [[6; 8]] = 18 + 32 = 50

So {{M}}{{N}} = [[19, 22; 43, 50]]. Now try {{N}}{{M}}: you get [[23, 34; 31, 46]]. **Order matters.** Matrix multiplication is associative (({{M}}{{N}}){{x}} = {{M}}({{N}}{{x}})) but not commutative.

<!-- widget:matmul -->

## Two special products

**Outer product.** A column times a row, {{u}}{{w}}{{ᵀ}}. The shapes `(m, 1) · (1, n)` make the inner size 1, so each entry is a single multiplication: entry ({{i}}, {{j}}) = {{u}}<sub>i</sub>{{w}}<sub>j</sub>.

<!-- figure:outer-product:start -->
```text
[1]                  [3  4  5]
[2] . [3 4 5]   =    [6  8 10]
```
<!-- figure:outer-product:end -->

Every row is a multiple of the same row, so an outer product has rank 1. Weight gradients have exactly this form (Part 7), and LoRA is a sum of a few of them (Part 8).

**Elementwise (Hadamard) product.** Not a matrix product at all: multiply same-shaped arrays position by position. [[1, 2, 3]] {{⊙}} [[4, −1, 2]] = [[4, −2, 6]]. In PyTorch, `a * b` is elementwise and `a @ b` is the matrix product; mixing them up is one of the most common bugs.

<div class="why">

Matrix multiplication is the workhorse of every model. A whole batch of inputs through a layer is one product, {{X}}{{W}}{{ᵀ}}, and attention for every token at once is two more. It is used so heavily because it is the most general linear operation (Part 4), it composes (one product can stand for two layers' worth of linear work), and its {{m}} × {{p}} independent dot products map perfectly onto graphics processing units (GPUs), which are built to do thousands of multiply-adds in parallel.

</div>

## Practice

<div class="problem">

**3.1** Compute [[2, −1; 0, 3]] · [[1, 4; 2, 0]].

<details class="practice"><summary>Show solution</summary><div>

- (1, 1): 2·1 + (−1)·2 = 0
- (1, 2): 2·4 + (−1)·0 = 8
- (2, 1): 0·1 + 3·2 = 6
- (2, 2): 0·4 + 3·0 = 0

Result: [[0, 8; 6, 0]].

</div></details>
</div>

<div class="problem">

**3.2** {{Q}} and {{K}} both have shape `(5, 64)`. What are the shapes of {{Q}}{{K}}{{ᵀ}} and {{K}}{{ᵀ}}{{Q}}? Does {{Q}}{{K}} exist?

<details class="practice"><summary>Show solution</summary><div>

- {{Q}}{{K}}{{ᵀ}}: `(5, 64) · (64, 5)` = `(5, 5)`, one score for every pair of tokens. This is attention's score matrix.
- {{K}}{{ᵀ}}{{Q}}: `(64, 5) · (5, 64)` = `(64, 64)`, a feature-by-feature matrix.
- {{Q}}{{K}}: `(5, 64) · (5, 64)`, inner sizes 64 and 5 don't match, so it doesn't exist.

</div></details>
</div>

<div class="problem">

**3.3** Compute the outer product [[1; −2]] [[3, 0, 1]].

<details class="practice"><summary>Show solution</summary><div>

[[3, 0, 1; −6, 0, −2]]. Row 2 is −2 times row 1, so the rank is 1.

</div></details>
</div>

---

# Part 4: Matrices as transformations

**Objective:** see a matrix as a function that moves space, and use that picture to understand determinant, inverse, rank and eigenvectors.

## Columns are where the axes land

Multiply any 2 × 2 matrix by the basis vector [[1; 0]] and you get its first column; by [[0; 1]] and you get its second column. Because every vector is a combination of the basis vectors (and the map is linear), **knowing where the basis vectors go tells you where everything goes**. The whole grid of the plane moves along with them, staying straight, parallel and evenly spaced.

<!-- figure:transform-square:start -->
```text
before: unit square        after M = [[2, 1], [0, 1]]: a parallelogram
column 1 = image of [1, 0]; column 2 = image of [0, 1]
```
<!-- figure:transform-square:end -->

<!-- widget:transform -->

Try the presets above. *Scale* stretches everything; *rotate 90°* turns it; *shear* slides the top sideways; *reflect* flips it; *collapse* flattens the plane onto a line.

## Determinant: the area factor

<div class="recipe">

**How to compute a 2 × 2 determinant**

1. For [[a, b; c, d]], multiply down the main diagonal: ad.
2. Multiply up the other diagonal: bc.
3. Subtract: det = ad − bc.

</div>

The result is how much the matrix scales area. det [[2, 1; 0, 1]] = 2·1 − 1·0 = 2, so the unit square becomes a parallelogram of area 2. A negative determinant means the plane is flipped over; zero means it is squashed onto a line (or a point).

## Inverse: undoing the matrix

If the determinant is not zero, nothing was squashed, so the transformation can be undone:

<div class="eq">[[a, b; c, d]]<sup>−1</sup> = <sup>1</sup>&frasl;<sub>ad − bc</sub> [[d, −b; −c, a]]</div>

<div class="recipe">

**How to invert a 2 × 2 matrix**

1. Compute det = ad − bc. If it is 0, stop: there is no inverse.
2. Swap a and d. Negate b and c.
3. Divide every entry by the determinant.
4. Check: multiplying the original by your answer must give {{I}}.

</div>

Example: [[2, 1; 1, 1]] has determinant 1, so its inverse is [[1, −1; −1, 2]]. Check: [[2, 1; 1, 1]] · [[1, −1; −1, 2]] = [[2 − 1, −2 + 2; 1 − 1, −1 + 2]] = [[1, 0; 0, 1]]. ✓

## Rank: how many directions survive

Rank is the number of independent directions the outputs can reach. A 2 × 2 matrix with nonzero determinant has rank 2 (it covers the plane). [[1, 2; 0.5, 1]] (the *collapse* preset) has columns [[1; 0.5]] and [[2; 1]], and the second is twice the first: everything lands on one line, so the rank is 1 and the determinant is 0.

## Eigenvectors: directions that don't turn

Most vectors change direction when multiplied by {{M}}. An **eigenvector** is one that doesn't: it only gets stretched by its **eigenvalue** {{λ}}.

<div class="eq">{{M}}{{v}} = {{λ}}{{v}}</div>

<div class="recipe">

**How to find eigenvalues and eigenvectors of a 2 × 2 matrix**

1. Subtract {{λ}} from each diagonal entry: {{M}} − {{λ}}{{I}}.
2. Set its determinant to zero and solve the quadratic for {{λ}}.
3. For each {{λ}}, find a nonzero {{v}} with ({{M}} − {{λ}}{{I}}){{v}} = 0. (Both rows of that matrix are now multiples of each other, so one row is enough.)
4. Check by multiplying: {{M}}{{v}} must equal {{λ}}{{v}}.

</div>

Example: {{M}} = [[2, 1; 1, 2]] (the *symmetric* preset).

1. det [[2 − λ, 1; 1, 2 − λ]] = (2 − λ)² − 1 = 0, so 2 − λ = ±1, giving {{λ}} = 3 and {{λ}} = 1.
2. For {{λ}} = 3: [[−1, 1; 1, −1]]{{v}} = 0 gives {{v}} = [[1; 1]]. Check: [[2, 1; 1, 2]][[1; 1]] = [[3; 3]] = 3 × [[1; 1]]. ✓
3. For {{λ}} = 1: [[1, 1; 1, 1]]{{v}} = 0 gives {{v}} = [[1; −1]]. Check: [[2 − 1; 1 − 2]] = [[1; −1]]. ✓

In the widget, the dashed green lines are these two directions. Points on them slide along the line; everything else turns.

<div class="why">

Seeing matrices as transformations is what makes the rest of AI linear algebra intuitive. A layer *moves* its input into a new space where the next step is easier. The determinant and rank tell you whether information is lost (a squashed direction can never be recovered). Eigenvectors and singular vectors (Part 8) pick out the directions a matrix cares about most, which is the idea behind principal component analysis (PCA), low-rank compression, and LoRA.

</div>

## Practice

<div class="problem">

**4.1** Compute the determinant of [[3, 1; 6, 2]]. What is its rank, and why?

<details class="practice"><summary>Show solution</summary><div>

det = 3·2 − 1·6 = 0. The rank is 1: the first column [[3; 6]] is 3 times the second column [[1; 2]], so all outputs lie on one line and the matrix has no inverse.

</div></details>
</div>

<div class="problem">

**4.2** Invert [[4, 7; 2, 6]] and check your answer.

<details class="practice"><summary>Show solution</summary><div>

det = 24 − 14 = 10. Swap and negate: [[6, −7; −2, 4]]. Divide by 10: [[0.6, −0.7; −0.2, 0.4]].
Check row 1 of the original against column 1: 4·0.6 + 7·(−0.2) = 2.4 − 1.4 = 1. Row 1 against column 2: 4·(−0.7) + 7·0.4 = 0. The rest also give {{I}}.

</div></details>
</div>

<div class="problem">

**4.3** Find the eigenvalues and eigenvectors of [[4, 1; 2, 3]].

<details class="practice"><summary>Show solution</summary><div>

det [[4 − λ, 1; 2, 3 − λ]] = (4 − λ)(3 − λ) − 2 = λ² − 7λ + 10 = (λ − 5)(λ − 2), so {{λ}} = 5 and {{λ}} = 2.

- {{λ}} = 5: [[−1, 1; 2, −2]]{{v}} = 0 gives {{v}} = [[1; 1]]. Check: [[4 + 1; 2 + 3]] = [[5; 5]]. ✓
- {{λ}} = 2: [[2, 1; 2, 1]]{{v}} = 0 gives {{v}} = [[1; −2]]. Check: [[4 − 2; 2 − 6]] = [[2; −4]] = 2 × [[1; −2]]. ✓

</div></details>
</div>

---

# Part 5: A neural network layer

**Objective:** compute a layer's output by hand, count its parameters, and understand why it needs a nonlinearity.

## The layer equation

<div class="eq"><span class="lbl">One layer</span>{{z}} = {{W}}{{x}} + {{b}}, &nbsp;&nbsp; {{h}} = {{σ}}({{z}})</div>

<!-- figure:layer:start -->
```text
x  ->  z = Wx + b  ->  h = sigma(z)
```
<!-- figure:layer:end -->

- {{x}}: the input, {{n}} numbers.
- {{W}}: the weights, shape `(m, n)`. Row {{i}} is the "pattern" output {{i}} looks for: {{z}}<sub>i</sub> is large when {{x}} lines up with that row (a dot product, Part 1).
- {{b}}: the bias, {{m}} numbers, a per-output offset.
- {{σ}}: the activation function, applied to each entry separately. ReLU is the usual choice: negative numbers become 0, positive ones pass through.

<div class="recipe">

**How to run a layer by hand**

1. Compute {{W}}{{x}} (Part 3).
2. Add {{b}} entry by entry to get {{z}}.
3. Apply {{σ}} to each entry of {{z}} to get {{h}}.

</div>

Example: {{W}} = [[1, 0, 2; 0, 1, −1]], {{x}} = [[1; 2; 3]], {{b}} = [[0; 0.5]], {{σ}} = ReLU.

1. {{W}}{{x}} = [[7; −1]] (from Part 3).
2. {{z}} = [[7 + 0; −1 + 0.5]] = [[7; −0.5]].
3. {{h}} = [[max(0, 7); max(0, −0.5)]] = [[7; 0]].

**Parameter count.** {{W}} has {{m}} × {{n}} entries and {{b}} has {{m}}, so this layer has 2 × 3 + 2 = 8 parameters. A real 768 → 3,072 layer has 768 × 3,072 + 3,072 ≈ 2.36 million.

## Why the nonlinearity is essential

Stack two layers without an activation:

<div class="eq">{{W}}<sub>2</sub>({{W}}<sub>1</sub>{{x}}) = ({{W}}<sub>2</sub>{{W}}<sub>1</sub>){{x}}</div>

The product {{W}}<sub>2</sub>{{W}}<sub>1</sub> is just another matrix, so a hundred linear layers can do no more than one. A linear map can only stretch, rotate, shear and flatten space (Part 4); it can never bend it. The activation function bends it, and alternating "linear, bend, linear, bend" is what lets a network approximate curved decision boundaries.

## The batch form

In practice many inputs are processed at once. Stack them as the rows of {{X}}, shape `(batch, n)`:

<div class="eq">{{Z}} = {{X}}{{W}}{{ᵀ}} + {{b}}</div>

The shapes are `(batch, n) · (n, m) = (batch, m)`: one output row per input row, and {{b}} is added to every row ("broadcast"). This is exactly what PyTorch's `nn.Linear` computes, which is why its weight is stored as `(out_features, in_features)` and used transposed.

```python
import torch
W = torch.tensor([[1., 0., 2.], [0., 1., -1.]])   # (2, 3)
b = torch.tensor([0., 0.5])                       # (2,)
X = torch.tensor([[1., 2., 3.], [0., 1., 0.]])    # (batch=2, 3)
Z = X @ W.T + b                                   # (2, 2)
H = torch.relu(Z)
print(Z)   # tensor([[ 7.0, -0.5], [ 0.0,  1.5]])
print(H)   # tensor([[ 7.0,  0.0], [ 0.0,  1.5]])
```

<div class="why">

The linear part {{W}}{{x}} + {{b}} is the most general way to combine inputs with learned weights that stays cheap and differentiable, and the activation function is the cheapest possible way to add the bending that linear maps lack. Every block in a transformer (projections, attention outputs, the multilayer perceptron (MLP) inside each layer, the final classifier) is built from this pattern.

</div>

## Practice

<div class="problem">

**5.1** {{W}} = [[1, −1; 2, 0; 0, 3]], {{x}} = [[2; 1]], {{b}} = [[0; −5; 1]]. Compute {{z}} and {{h}} = ReLU({{z}}). How many parameters does this layer have?

<details class="practice"><summary>Show solution</summary><div>

{{W}}{{x}} = [[2 − 1; 4 + 0; 0 + 3]] = [[1; 4; 3]]. {{z}} = [[1; −1; 4]]. {{h}} = [[1; 0; 4]]. Parameters: 3 × 2 + 3 = 9.

</div></details>
</div>

<div class="problem">

**5.2** Two linear layers with no activation: {{W}}<sub>1</sub> = [[1, 1; 0, 1]], {{W}}<sub>2</sub> = [[2, 0; 1, 1]]. Find the single matrix that does the same job.

<details class="practice"><summary>Show solution</summary><div>

{{W}}<sub>1</sub> is applied first, so the combined matrix is {{W}}<sub>2</sub>{{W}}<sub>1</sub> = [[2·1 + 0·0, 2·1 + 0·1; 1·1 + 1·0, 1·1 + 1·1]] = [[2, 2; 1, 2]].

</div></details>
</div>

---

# Part 6: Attention

**Objective:** compute single-query attention by hand, and understand each step of softmax({{Q}}{{K}}{{ᵀ}} / {{√d}}){{V}}.

## The idea

Each token builds its new vector by looking at the other tokens and taking a weighted blend of what they offer. Three learned matrices turn each token's vector into three new vectors:

- a **query** {{q}}: what this token is looking for;
- a **key** {{k}}: what this token can be found by;
- a **value** {{v}}: what this token hands over if chosen.

## The four steps for one query

<div class="recipe">

**How to compute attention for one query**

1. **Score.** Dot the query with every key and divide by {{√d}}: {{s}}<sub>i</sub> = {{q}}{{·}}{{k|k}}<sub>i</sub> / {{√d}}.
2. **Softmax.** Turn the scores into weights that are positive and add to 1: {{α}} = softmax({{s}}).
3. **Blend.** Multiply each value vector by its weight and add: {{o}} = {{Σ}}<sub>i</sub> {{α}}<sub>i</sub>{{v}}<sub>i</sub>.
4. **Sanity check.** The weights should add to 1, and the output should lie "between" the value vectors, closest to the ones with the biggest weights.

</div>

<!-- figure:attention:start -->
```text
q, keys -> scores q.k/sqrt(d) -> softmax -> weights -> sum of alpha_i v_i
```
<!-- figure:attention:end -->

## Softmax, step by step

<div class="eq"><span class="lbl">Softmax</span>softmax({{z}})<sub>i</sub> = exp({{z}}<sub>i</sub>) / {{Σ}}<sub>j</sub> exp({{z}}<sub>j</sub>)</div>

<div class="recipe">

**How to compute softmax by hand**

1. (Optional, for stability) subtract the largest score from every score. The answer doesn't change, and the exponentials can't overflow.
2. Exponentiate each score: exp(a) = e<sup>a</sup> with e ≈ 2.718. Useful values: e<sup>0</sup> = 1, e<sup>1</sup> ≈ 2.718, e<sup>2</sup> ≈ 7.389, e<sup>−1</sup> ≈ 0.368.
3. Add them up.
4. Divide each exponential by the sum.

</div>

Why exponentiate? It makes every weight positive even for negative scores, it keeps the order of the scores, and it exaggerates differences: a score 2 higher gets about 7.4 times the weight.

<!-- widget:softmax -->

## Why divide by √d

If the {{d}} entries of {{q}} and {{k}} are roughly independent with mean 0 and variance 1, then {{q}}{{·}}{{k}} is a sum of {{d}} such products, so its variance is about {{d}} and its typical size is about {{√d}}. With {{d}} = 64 the raw scores would routinely be around ±8, which pushes softmax to put almost all its weight on one token, and makes its gradients nearly zero, so learning stalls. Dividing by {{√d}} brings the scores back to a typical size of about 1, whatever {{d}} is.

## The matrix form

Doing all {{T}} queries at once just stacks them as rows:

<div class="eq"><span class="lbl">Scaled dot-product attention</span>Attention({{Q}}, {{K}}, {{V}}) = softmax( {{Q}}{{K}}{{ᵀ}} / {{√d}} ) {{V}}</div>

| Step | Shape | What it holds |
|---|---|---|
| {{Q}}, {{K}} | `(T, d)` | one query and one key per token, as rows |
| {{Q}}{{K}}{{ᵀ}} | `(T, d) · (d, T) = (T, T)` | row {{i}}: token {{i}}'s score against every token |
| softmax, per row | `(T, T)` | each row now adds to 1 (the attention map) |
| {{V}} | `(T, d_v)` | one value vector per token |
| result | `(T, T) · (T, d_v) = (T, d_v)` | row {{i}}: token {{i}}'s blended output |

<div class="why">

Attention is how a transformer mixes information *between* tokens (the MLP layers only work on one token at a time). It uses the dot product because that is the cheapest learnable similarity test, softmax because it turns similarities into a smooth, differentiable "choice", and matrix products because they compute all {{T}} × {{T}} comparisons in one GPU-friendly operation.

</div>

## Practice

<div class="problem">

**6.1 (worked problem).** Query {{q}} = [[1, 0]]. Keys {{k|k₁}} = [[1, 0]], {{k|k₂}} = [[0, 1]]. Values {{v|v₁}} = [[2, 0]], {{v|v₂}} = [[0, 4]]. {{d}} = 2. Find the output.

<details class="practice"><summary>Show solution</summary><div>

1. Scores: {{q}}{{·}}{{k|k₁}} = 1, {{q}}{{·}}{{k|k₂}} = 0. Divide by √2 ≈ 1.414: {{s}} = [[0.707, 0]].
2. Exponentiate: e<sup>0.707</sup> ≈ 2.028, e<sup>0</sup> = 1. Sum ≈ 3.028.
3. Weights: {{α}} ≈ [[2.028 / 3.028, 1 / 3.028]] = [[0.670, 0.330]]. They add to 1. ✓
4. Blend: 0.670 × [[2, 0]] + 0.330 × [[0, 4]] = [[1.34, 1.32]].

The query matches key 1, so value 1 gets two thirds of the weight; but the scores are close, so value 2 still contributes a third.

</div></details>
</div>

<div class="problem">

**6.2** Compute softmax([[2, 1, 0]]). Then compute it again at temperature {{τ}} = 0.5 and say what changed.

<details class="practice"><summary>Show solution</summary><div>

- Exponentials: e<sup>2</sup> ≈ 7.389, e<sup>1</sup> ≈ 2.718, e<sup>0</sup> = 1. Sum ≈ 11.107. Weights ≈ [[0.665, 0.245, 0.090]].
- At {{τ}} = 0.5, divide the scores by 0.5 first: [[4, 2, 0]]. Exponentials ≈ 54.60, 7.389, 1; sum ≈ 62.99. Weights ≈ [[0.867, 0.117, 0.016]].

The order is the same, but the low temperature made the distribution much sharper. Check it with the widget.

</div></details>
</div>

<div class="problem">

**6.3** In 6.1, change the query to {{q}} = [[0, 3]]. Without computing exactly, which value dominates, and roughly how strongly?

<details class="practice"><summary>Show solution</summary><div>

Scores: 0 / 1.414 = 0 and 3 / 1.414 ≈ 2.12. Weights ≈ [[1 / (1 + 8.33), 8.33 / 9.33]] ≈ [[0.107, 0.893]], since e<sup>2.12</sup> ≈ 8.33. Value 2 dominates: output ≈ 0.107 × [[2, 0]] + 0.893 × [[0, 4]] ≈ [[0.21, 3.57]].

</div></details>
</div>

---

# Part 7: Derivatives, Jacobians and backpropagation

**Objective:** know what a gradient and a Jacobian are, derive the gradients of a linear layer, and do one training step by hand.

## From derivatives to gradients

A **derivative** answers "if I nudge the input a little, how much does the output move?". For f(a) = a², the derivative is 2a: at a = 3, a nudge of 0.01 moves the output by about 0.06.

With many inputs, nudge one at a time. That is a **partial derivative**, {{∂}}f / {{∂}}{{x}}<sub>j</sub>: the rate of change when only {{x}}<sub>j</sub> moves. Collect all of them for a function that returns **one number** (like the loss) and you get the **gradient**:

<div class="eq">{{∇}}<sub>x</sub>{{L}} = [[∂L/∂x₁, ∂L/∂x₂, …, ∂L/∂xₙ]]</div>

The gradient has the same shape as {{x}}, and points in the direction that increases {{L}} fastest. So stepping the opposite way decreases the loss: that is all gradient descent is.

## The Jacobian

When the function returns **many numbers** (like a layer, {{x}} ↦ {{y}}), every output has its own gradient. Stack them as rows and you get the **Jacobian**:

<div class="eq"><span class="lbl">Jacobian of y = f(x), shape (m, n)</span>{{J}}<sub>ij</sub> = {{∂}}{{y}}<sub>i</sub> / {{∂}}{{x}}<sub>j</sub></div>

<!-- figure:jacobian:start -->
```text
        x1        x2        x3
y1  dy1/dx1   dy1/dx2   dy1/dx3
y2  dy2/dx1   dy2/dx2   dy2/dx3
```
<!-- figure:jacobian:end -->

<div class="recipe">

**How to build a Jacobian by hand**

1. Write each output {{y}}<sub>i</sub> as a formula in the inputs.
2. For each output (one row) and each input (one column), take the partial derivative: treat every other input as a constant.
3. Plug in the point you care about.

</div>

Example: {{y}} = [[x₁x₂; x₁ + 3x₂]]. Row 1: {{∂}}(x₁x₂)/{{∂}}x₁ = x₂ and {{∂}}(x₁x₂)/{{∂}}x₂ = x₁. Row 2: 1 and 3. So {{J}} = [[x₂, x₁; 1, 3]], and at (2, 1) it is [[1, 2; 1, 3]].

**For a linear map the Jacobian is the matrix itself.** If {{y}} = {{W}}{{x}}, then {{y}}<sub>i</sub> = {{Σ}}<sub>j</sub> {{W}}<sub>ij</sub>{{x}}<sub>j</sub>, and differentiating by {{x}}<sub>j</sub> leaves {{W}}<sub>ij</sub>. So {{J}} = {{W}}: a linear layer's "sensitivity" is the same everywhere.

**For an elementwise function the Jacobian is diagonal.** Output {{i}} of ReLU depends only on input {{i}}, so all off-diagonal entries are 0 and the diagonal holds 1 (where the input was positive) or 0 (where it was not). Multiplying by it is the same as an elementwise product with that 0/1 mask.

## The chain rule, in matrix form

A network is a chain: {{x}} → {{y}} → … → {{L}}. The chain rule says derivatives multiply along the chain. Going **backward**, if you already know {{δ}} = {{∂}}{{L}}/{{∂}}{{y}} (a vector shaped like {{y}}), then

<div class="eq"><span class="lbl">Chain rule, backward</span>{{∂}}{{L}}/{{∂}}{{x}} = {{J}}{{ᵀ}} {{δ}}</div>

The transpose is there because {{J}} maps nudges of {{x}} forward to nudges of {{y}}, while gradients travel the other way. This "vector times Jacobian" is the one operation backpropagation repeats at every step. It never builds a whole Jacobian for a big layer; it just multiplies by the right matrix.

## The three gradients of a linear layer

For {{y}} = {{W}}{{x}} with {{δ}} = {{∂}}{{L}}/{{∂}}{{y}} known:

<div class="eq">{{∂}}{{L}}/{{∂}}{{x}} = {{W}}{{ᵀ}}{{δ}} &nbsp;&nbsp;&nbsp; {{∂}}{{L}}/{{∂}}{{W}} = {{δ}}{{x}}{{ᵀ}} &nbsp;&nbsp;&nbsp; {{∂}}{{L}}/{{∂}}{{b}} = {{δ}} &nbsp;(if {{y}} = {{W}}{{x}} + {{b}})</div>

**Where the weight gradient comes from.** {{W}}<sub>ij</sub> only appears in output {{y}}<sub>i</sub>, multiplied by {{x}}<sub>j</sub>. So nudging {{W}}<sub>ij</sub> changes {{y}}<sub>i</sub> at rate {{x}}<sub>j</sub>, which changes {{L}} at rate {{δ}}<sub>i</sub>. Chain them: {{∂}}{{L}}/{{∂}}{{W}}<sub>ij</sub> = {{δ}}<sub>i</sub>{{x}}<sub>j</sub>. A grid of {{δ}}<sub>i</sub>{{x}}<sub>j</sub> products is exactly the outer product {{δ}}{{x}}{{ᵀ}} (Part 3). Check shapes: `(m, 1) · (1, n) = (m, n)`, the same as {{W}}. ✓

<!-- figure:backprop:start -->
```text
x --> [y = Wx] --> [L = 1/2 ||y - t||^2]       forward
x <--  W^T d   <--  d = y - t                  backward;  dL/dW = d x^T
```
<!-- figure:backprop:end -->

**The squared-error loss.** For {{L}} = ½‖{{y}} − {{t}}‖² = ½ {{Σ}}<sub>i</sub> ({{y}}<sub>i</sub> − {{t}}<sub>i</sub>)², the derivative of each term is ({{y}}<sub>i</sub> − {{t}}<sub>i</sub>) (the ½ cancels the 2), so {{δ}} = {{y}} − {{t}}: the error itself.

<div class="why">

Training means adjusting millions of parameters to reduce one number, the loss. The gradient says which way to move each parameter, and backpropagation computes all of them for roughly the cost of one extra forward pass, because the chain rule lets each layer reuse the gradient handed back by the layer after it. And because each step is "multiply by a transpose" or "take an outer product", the backward pass is made of the same matrix products as the forward pass, on the same hardware.

</div>

## Practice

<div class="problem">

**7.1 (worked problem).** {{W}} = [[1, 2; 3, 4]], {{x}} = [[1; −1]], {{t}} = [[0; 1]], {{L}} = ½‖{{W}}{{x}} − {{t}}‖². Find {{y}}, {{δ}}, {{∂}}{{L}}/{{∂}}{{W}} and {{∂}}{{L}}/{{∂}}{{x}}.

<details class="practice"><summary>Show solution</summary><div>

1. Forward: {{y}} = [[1 − 2; 3 − 4]] = [[−1; −1]]. Loss = ½((−1)² + (−2)²) = 2.5.
2. Output gradient: {{δ}} = {{y}} − {{t}} = [[−1; −2]].
3. Weight gradient (outer product): {{δ}}{{x}}{{ᵀ}} = [[−1; −2]] [[1, −1]] = [[−1, 1; −2, 2]].
4. Input gradient: {{W}}{{ᵀ}}{{δ}} = [[1, 3; 2, 4]] [[−1; −2]] = [[−1 − 6; −2 − 8]] = [[−7; −10]].

</div></details>
</div>

<div class="problem">

**7.2** Continue 7.1: take one gradient-descent step with {{η}} = 0.1 and check that the loss went down.

<details class="practice"><summary>Show solution</summary><div>

{{W}} ← {{W}} − 0.1 × [[−1, 1; −2, 2]] = [[1.1, 1.9; 3.2, 3.8]].
New {{y}} = [[1.1 − 1.9; 3.2 − 3.8]] = [[−0.8; −0.6]]. New loss = ½(0.64 + 2.56) = 1.6, down from 2.5. ✓

</div></details>
</div>

<div class="problem">

**7.3** Find the Jacobian of f({{x}}) = [[x₁²; x₁x₂; 5x₂]] at {{x}} = (3, −1). What shape is it?

<details class="practice"><summary>Show solution</summary><div>

Rows are outputs, columns are inputs: [[2x₁, 0; x₂, x₁; 0, 5]]. At (3, −1): [[6, 0; −1, 3; 0, 5]]. Shape `(3, 2)`: 3 outputs, 2 inputs.

</div></details>
</div>

<div class="problem">

**7.4** A ReLU layer had pre-activations {{z}} = [[3; −2; 0.5]], and the gradient arriving from above is {{∂}}{{L}}/{{∂}}{{h}} = [[0.4; 0.7; −1]]. What is {{∂}}{{L}}/{{∂}}{{z}}?

<details class="practice"><summary>Show solution</summary><div>

The ReLU Jacobian is diagonal with 1 where {{z}} > 0: mask [[1; 0; 1]]. Elementwise: [[0.4; 0; −1]]. The middle entry was negative, so ReLU output 0 there and no gradient flows back through it.

</div></details>
</div>

---

# Part 8: Rank, SVD and LoRA

**Objective:** connect rank and the singular value decomposition to low-rank fine-tuning, and count LoRA's parameters.

## Rank, revisited

The rank of a matrix is the number of independent directions in its output. Two quick ways to spot a low rank:

- one row or column is a multiple (or combination) of others: [[1, 2; 2, 4]] has rank 1, because row 2 is twice row 1;
- the matrix is an outer product, or a sum of a few: [[1, 2; 2, 4]] = [[1; 2]] [[1, 2]].

**Any matrix of rank {{r}} can be written as a sum of {{r}} outer products**, and that is the bridge to LoRA.

## The singular value decomposition

<div class="eq">{{M}} = {{U}} {{Σ}} {{V}}{{ᵀ}}</div>

Every matrix, square or not, factors into a rotation ({{V}}{{ᵀ}}), a stretch along the axes by the singular values (the diagonal of {{Σ}}, sorted largest first), and another rotation ({{U}}). The number of nonzero singular values is the rank.

<!-- figure:svd:start -->
```text
circle --(V^T: rotate)--> circle --(Sigma: stretch)--> ellipse --(U: rotate)--> ellipse
```
<!-- figure:svd:end -->

Written as a sum of outer products, {{M}} = σ₁ u₁v₁{{ᵀ}} + σ₂ u₂v₂{{ᵀ}} + …, where u<sub>i</sub> and v<sub>i</sub> are the columns of {{U}} and {{V}} and σ<sub>i</sub> the singular values. Keeping only the first {{r}} terms gives the **best possible rank-{{r}} approximation** of {{M}} (the Eckart–Young theorem). If the singular values fall off quickly, a small {{r}} keeps almost everything.

<div class="recipe">

**How to find singular values of a small matrix by hand**

1. Compute {{M}}{{ᵀ}}{{M}} (always square and symmetric).
2. Find its eigenvalues (Part 4). They are never negative.
3. The singular values are their square roots.

</div>

Example: {{M}} = [[1, 1; 0, 0]]. {{M}}{{ᵀ}}{{M}} = [[1, 1; 1, 1]], whose eigenvalues are 2 and 0. Singular values: √2 and 0, so the rank is 1.

## LoRA: fine-tuning with a low-rank update

Fine-tuning changes a weight matrix by some {{Δ}}{{W}}. The LoRA (Low-Rank Adaptation) bet is that the useful change has low rank, so instead of learning all of {{Δ}}{{W}} it learns two thin matrices:

<div class="eq"><span class="lbl">LoRA</span>{{W}}<sub>new</sub> = {{W}} + {{Δ}}{{W}} = {{W}} + {{B}}{{A}}, &nbsp; {{B}} ∈ ℝ<sup>d×r</sup>, {{A}} ∈ ℝ<sup>r×d</sup></div>

<!-- figure:lora:start -->
```text
[ d x d ]   =   [d x r] . [r x d]
d^2 numbers     2 d r numbers
```
<!-- figure:lora:end -->

- {{W}} is frozen: it never changes.
- {{A}} squeezes the input down to {{r}} numbers; {{B}} expands it back to {{d}}.
- {{B}} starts at all zeros, so {{B}}{{A}} = 0 and training begins from exactly the original model.
- In practice the update is also multiplied by a constant scaling factor; it doesn't change the shapes or the counting.
- At inference, {{B}}{{A}} can be added into {{W}} once, so the adapted model is exactly as fast as the original.

<div class="why">

Rank is the right measure of "how big a change" because a rank-{{r}} update can only move the outputs along {{r}} directions. Full fine-tuning of a `(d, d)` matrix trains {{d}}² numbers; LoRA trains 2{{d}}{{r}}, often under 1% of the parameters, with much less memory for the optimizer, and you can store one tiny adapter per task while sharing the big frozen model.

</div>

## Practice

<div class="problem">

**8.1 (worked problem).** A weight matrix is 4096 × 4096. Compare its parameter count with a LoRA update of rank {{r}} = 8.

<details class="practice"><summary>Show solution</summary><div>

Full: 4096² = 16,777,216. LoRA: {{B}} has 4096 × 8 = 32,768 and {{A}} has 8 × 4096 = 32,768, total 65,536. Ratio: 65,536 / 16,777,216 ≈ 0.39%.

</div></details>
</div>

<div class="problem">

**8.2** Write [[2, 6; 1, 3]] as an outer product. What is its rank? What are its singular values?

<details class="practice"><summary>Show solution</summary><div>

Every row is a multiple of [[1, 3]]: [[2, 6; 1, 3]] = [[2; 1]] [[1, 3]]. Rank 1.
{{M}}{{ᵀ}}{{M}} = [[2·2 + 1·1, 2·6 + 1·3; 6·2 + 3·1, 6·6 + 3·3]] = [[5, 15; 15, 45]]. Its trace is 50 and determinant 225 − 225 = 0, so the eigenvalues are 50 and 0. Singular values: √50 ≈ 7.07 and 0.
(Shortcut: for an outer product {{u}}{{w}}{{ᵀ}}, the single nonzero singular value is ‖{{u}}‖ ‖{{w}}‖ = √5 × √10 = √50.)

</div></details>
</div>

<div class="problem">

**8.3** A model has hidden size {{d}} = 768. You put rank-8 LoRA adapters on the query and value projections (both 768 × 768) in each of 12 layers. How many trainable parameters is that?

<details class="practice"><summary>Show solution</summary><div>

Per adapter: 2 × 768 × 8 = 12,288. Adapters: 2 per layer × 12 layers = 24. Total: 24 × 12,288 = 294,912, against 589,824 for just one full 768 × 768 matrix.

</div></details>
</div>

---

# Practice set

Mixed review. Do them in any order, on paper, then check.

<div class="problem">

**P.1** Shapes only: {{X}} is `(32, 768)`, {{W}} is `(3072, 768)`. What is the shape of {{X}}{{W}}{{ᵀ}}? Of {{W}}{{X}}{{ᵀ}}? Which one matches the batch convention?

<details class="practice"><summary>Show solution</summary><div>

{{X}}{{W}}{{ᵀ}}: `(32, 768) · (768, 3072) = (32, 3072)`, one output row per example (the batch convention). {{W}}{{X}}{{ᵀ}}: `(3072, 768) · (768, 32) = (3072, 32)`, the same numbers transposed (by the reversal rule, ({{X}}{{W}}{{ᵀ}}){{ᵀ}} = {{W}}{{X}}{{ᵀ}}).

</div></details>
</div>

<div class="problem">

**P.2** Two embedding vectors are [[2, 0, 1]] and [[1, 1, −2]]. What is their cosine similarity, and what does it mean?

<details class="practice"><summary>Show solution</summary><div>

Dot product = 2 + 0 − 2 = 0, so the cosine similarity is 0: the vectors are orthogonal and, as far as this measure goes, unrelated.

</div></details>
</div>

<div class="problem">

**P.3** Compute [[1, 2, 0; 0, 1, 1]] · [[1, 0; 2, 1; 0, 3]], then the product in the other order. Do the shapes differ?

<details class="practice"><summary>Show solution</summary><div>

First order, `(2, 3) · (3, 2) = (2, 2)`: [[1 + 4 + 0, 0 + 2 + 0; 0 + 2 + 0, 0 + 1 + 3]] = [[5, 2; 2, 4]].
Other order, `(3, 2) · (2, 3) = (3, 3)`: [[1, 2, 0; 2, 5, 1; 0, 3, 3]].
Different shapes, so clearly not equal. Try it in the matrix multiplication explorer with the swap button.

</div></details>
</div>

<div class="problem">

**P.4** Attention with three keys. {{q}} = [[1, 1]], {{k|k₁}} = [[2, 0]], {{k|k₂}} = [[0, 2]], {{k|k₃}} = [[−1, −1]], values {{v|v₁}} = [[1, 0]], {{v|v₂}} = [[0, 1]], {{v|v₃}} = [[5, 5]], {{d}} = 2. Compute the weights and the output.

<details class="practice"><summary>Show solution</summary><div>

Dot products: 2, 2, −2. Divided by √2: {{s}} ≈ [[1.414, 1.414, −1.414]]. Exponentials: e<sup>1.414</sup> ≈ 4.113 (twice) and e<sup>−1.414</sup> ≈ 0.243. Sum ≈ 8.469. Weights ≈ [[0.486, 0.486, 0.029]].
Output ≈ 0.486 × [[1, 0]] + 0.486 × [[0, 1]] + 0.029 × [[5, 5]] ≈ [[0.63, 0.63]]. Key 3 points away from the query, so its large value barely leaks in.

</div></details>
</div>

<div class="problem">

**P.5** A layer {{z}} = {{W}}{{x}} + {{b}} with {{W}} = [[2, −1]] (shape `(1, 2)`), {{x}} = [[1; 3]], {{b}} = [[1]], target {{t}} = 2 and loss ½({{z}} − {{t}})². Find {{∂}}{{L}}/{{∂}}{{W}}, {{∂}}{{L}}/{{∂}}{{b}} and {{∂}}{{L}}/{{∂}}{{x}}.

<details class="practice"><summary>Show solution</summary><div>

{{z}} = 2 − 3 + 1 = 0. {{δ}} = {{z}} − {{t}} = −2.
{{∂}}{{L}}/{{∂}}{{W}} = {{δ}}{{x}}{{ᵀ}} = −2 × [[1, 3]] = [[−2, −6]].
{{∂}}{{L}}/{{∂}}{{b}} = {{δ}} = −2.
{{∂}}{{L}}/{{∂}}{{x}} = {{W}}{{ᵀ}}{{δ}} = [[2; −1]] × (−2) = [[−4; 2]].

</div></details>
</div>

<div class="problem">

**P.6** Without computing, decide whether each matrix has an inverse: [[1, 2; 2, 4]], [[0, 1; 1, 0]], [[3, 0; 0, 0]].

<details class="practice"><summary>Show solution</summary><div>

- [[1, 2; 2, 4]]: row 2 = 2 × row 1, determinant 0, no inverse.
- [[0, 1; 1, 0]]: swaps the two coordinates; determinant −1, invertible (it is its own inverse).
- [[3, 0; 0, 0]]: squashes the second axis to 0, determinant 0, no inverse.

</div></details>
</div>

---

# Quick reference

| Operation | Rule | Shapes |
|---|---|---|
| Dot product | multiply matching entries, add | `(n,)` and `(n,)` → number |
| Norm | √(sum of squares) | `(n,)` → number |
| Cosine similarity | {{u}}{{·}}{{w}} / (‖{{u}}‖ ‖{{w}}‖) | → number in [−1, 1] |
| Transpose | row {{i}} → column {{i}} | `(m, n)` → `(n, m)` |
| Matrix-vector | each output = row · vector | `(m, n)`, `(n,)` → `(m,)` |
| Matrix product | entry ({{i}}, {{j}}) = row {{i}} · column {{j}} | `(m, n)`, `(n, p)` → `(m, p)` |
| Outer product | entry ({{i}}, {{j}}) = {{u}}<sub>i</sub>{{w}}<sub>j</sub> | `(m,)`, `(n,)` → `(m, n)`, rank 1 |
| Elementwise | multiply position by position | same shape in and out |
| 2 × 2 determinant | ad − bc | → number |
| 2 × 2 inverse | swap a, d; negate b, c; divide by det | `(2, 2)` → `(2, 2)` |
| Eigenvalues | solve det({{M}} − {{λ}}{{I}}) = 0 | `(n, n)` → {{n}} numbers |
| Softmax | exp each, divide by the sum | `(n,)` → `(n,)`, adds to 1 |
| Layer | {{σ}}({{W}}{{x}} + {{b}}) | `(n,)` → `(m,)` |
| Attention | softmax({{Q}}{{K}}{{ᵀ}} / {{√d}}){{V}} | `(T, d)` → `(T, d_v)` |
| Linear-layer gradients | {{W}}{{ᵀ}}{{δ}}, {{δ}}{{x}}{{ᵀ}}, {{δ}} | match {{x}}, {{W}}, {{b}} |
| Gradient descent | {{W}} ← {{W}} − {{η}} {{∂}}{{L}}/{{∂}}{{W}} | same shape as {{W}} |
| LoRA | {{W}} + {{B}}{{A}} | 2{{d}}{{r}} trainable instead of {{d}}² |
