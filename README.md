# PyTorch Notes

- **Autograd**: automatic differentiation and gradient computation
- **Tensors**: tensor creation, manipulation, operations, and device behavior

---

## PyTorch Autograd

Autograd is PyTorch's automatic differentiation system. It records operations on tensors when `requires_grad=True` and computes gradients automatically during backpropagation.

### Key concepts

- `requires_grad=True`: enables gradient tracking for a tensor.
- `backward()`: computes gradients of a scalar output with respect to all tensors that require gradients.
- `tensor.grad`: stores the computed gradient after backpropagation.
- `zero_()`: clears existing gradients before a new backward pass.
- `torch.no_grad()`, `detach()`, `requires_grad_(False)`: disable gradient tracking for inference or when gradients are not needed.

### Notes from the autograd notebook

- Basic scalar example:
  - `x = torch.tensor(3.0, requires_grad=True)`
  - `y = x ** 2`
  - `y.backward()` computes `dy/dx = 2 * x`.
  - `x.grad` stores the gradient value.

- Chain rule example with sine:
  - `x = torch.tensor(4.0, requires_grad=True)`
  - `y = x ** 2`
  - `z = torch.sin(y)`
  - `z.backward()` computes gradients through the chain `d(sin(y))/dy * dy/dx`.

- Manual binary cross-entropy loss and autograd comparison:
  - Implemented a scalar binary cross-entropy loss function with clipping to avoid `log(0)`.
  - Computed gradients manually using the chain rule:
    - `dL/dy_pred`
    - `dy_pred/dz`
    - `dz/dw` and `dz/db`
  - Compared manual calculations with PyTorch autograd by enabling `requires_grad=True` on `w` and `b`, computing the loss, calling `loss.backward()`, and reading `w.grad` and `b.grad`.

- Gradient reset and no-grad behavior:
  - Demonstrated that gradients accumulate by default and must be reset if reused.
  - Showed how `x.grad.zero_()` clears a tensor's gradients.
  - Demonstrated disabling tracking with `requires_grad_(False)`, `x.detach()`, and the `with torch.no_grad():` context.

## PyTorch Tensors

Tensors are the core data structure in PyTorch. They are similar to NumPy arrays but can run on GPUs and support automatic differentiation.

### Tensor creation

Common tensor constructors shown in the notes:

- `torch.empty(shape)`
- `torch.zeros(shape)`
- `torch.ones(shape)`
- `torch.rand(shape)`
- `torch.arange(start, end, step)`
- `torch.linspace(start, end, steps)`
- `torch.eye(size)`
- `torch.full(shape, fill_value)`
- `torch.tensor(data)`
- `torch.manual_seed(seed)` for reproducible random tensors

### Shapes and shape-related functions

- `tensor.shape` reports tensor dimensions.
- `torch.empty_like(tensor)`, `torch.zeros_like(tensor)`, `torch.ones_like(tensor)` create same-shaped tensors with different contents.
- `torch.rand_like(tensor, dtype=...)` creates a new tensor matching shape with a different dtype.

### Data types

- Tensor dtypes are accessible via `tensor.dtype`.
- You can specify dtypes when creating tensors, for example `dtype=torch.int32` or `dtype=torch.float64`.

### Mathematical operations

Elementwise arithmetic operations:

- addition: `tensor + scalar`
- subtraction: `tensor - scalar`
- multiplication: `tensor * scalar`
- division: `tensor / scalar`
- integer division: `(tensor * 100) // 3`
- modulo: `tensor % scalar`
- power: `tensor ** 2`

Elementwise operations between tensors of the same shape:

- `a + b`, `a - b`, `a * b`, `a / b`, `a % b`, `a ** b`

Additional elementwise math functions:

- `torch.abs(tensor)`
- `torch.neg(tensor)`
- `torch.round(tensor)`
- `torch.ceil(tensor)`
- `torch.floor(tensor)`
- `torch.clamp(tensor, min=..., max=...)`

### Reduction operations

Reduction functions compute aggregated values over tensor elements:

- `torch.sum(tensor)`
- `torch.sum(tensor, dim=0)` or `dim=1`
- `torch.mean(tensor)`
- `torch.mean(tensor, dim=0)`
- `torch.median(tensor)`
- `torch.min(tensor)` / `torch.max(tensor)`
- `torch.prod(tensor)`
- `torch.std(tensor)`
- `torch.var(tensor)`
- `torch.argmax(tensor)`
- `torch.argmin(tensor)`

### Matrix operations

Core linear algebra operations:

- matrix multiplication: `torch.matmul(f, g)`
- dot product: `torch.dot(vector1, vector2)`
- transpose: `torch.transpose(f, 0, 1)`
- determinant: `torch.det(h)`
- inverse: `torch.inverse(h)`

### Comparison operations

Elementwise comparisons include:

- `tensor > other`
- `tensor < other`
- `tensor == other`
- `tensor != other`

### Special functions

PyTorch provides common activation and math functions:

- `torch.log(tensor)`
- `torch.exp(tensor)`
- `torch.sqrt(tensor)`
- `torch.sigmoid(tensor)`
- `torch.softmax(tensor, dim=...)`
- `torch.relu(tensor)`

### In-place operations and cloning

- In-place operators end with `_`, such as `tensor.add_(other)` and `tensor.relu_()`.
- In-place operations modify the tensor directly.
- Use `tensor.clone()` to make a copy before modifying when you need to preserve the original tensor.
- Assigning `a = n` makes `a` a reference to `n`; use `n.clone()` to create a separate tensor.

### GPU tensor operations

The notebook covers GPU support and device movement:

- `torch.cuda.is_available()` checks if a CUDA GPU is available.
- `torch.device('cuda')` creates a CUDA device object.
- `torch.rand((2, 3), device=device)` creates a tensor directly on GPU.
- `tensor.to(device)` moves an existing tensor to GPU.
- GPU operations can be much faster for large tensor math, especially matrix multiplication.
- Use `torch.cuda.synchronize()` to ensure GPU work is finished before timing.

### Reshaping tensors

Common tensor shape transforms:

- `tensor.reshape(...)`
- `tensor.flatten()`
- `tensor.permute(...)`
- `tensor.unsqueeze(dim)`
- `tensor.squeeze(dim)`

### NumPy interoperability

PyTorch tensors and NumPy arrays can convert back and forth:

- `tensor.numpy()` converts a CPU tensor to a NumPy array.
- `torch.from_numpy(ndarray)` converts a NumPy array into a PyTorch tensor.

---

## Files in this repository

- `pytorch_autograd.ipynb`: contains examples and explanations for gradient tracking, backward passes, manual loss derivatives, and disabling gradients.
- `Tensors.ipynb`: contains tensor creation, dtype and shape examples, math operations, reductions, matrix operations, GPU support, reshaping, and NumPy interoperability.
