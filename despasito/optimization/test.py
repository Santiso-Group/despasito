import jax
import jax.numpy as np

def tanh(x):
	y = np.exp(-2.0 * x)
	return (1.0 - y) / (1.0 + y)

grad_tanh = jax.jit(jax.grad(tanh))
print(grad_tanh(1.0))
