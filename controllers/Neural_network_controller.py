import jax
import jax.numpy as jnp
from controllers.Base_controller import BaseController

class NeuralNetworkController(BaseController):
    """
    A Neural Network based PID controller.
    
    Inputs:
        Vector [error, integral_error, derivative_error]
    
    Architecture:
        Input Layer (3 neurons) -> Hidden Layers -> Output Layer (1 neuron)
    
    Output:
        Control signal U
    """

    def __init__(self, layer_sizes, activation_function, seed=42):
        self.layer_sizes = layer_sizes
        self.activation_function_name = activation_function.lower()
        self.key = jax.random.PRNGKey(seed)
        self.initial_params = self._init_network_params()

    def _init_network_params(self):
        """
        Helper-function to initialize weights and biases for all layers.

        Returns: A list of tuples: [(w1, b1), (w2, b2), ...].
        """

        # The inputlayer can only have three nodes: P, I, D
        input_dim = 3
        params = []

        # One key for each layer
        keys = jax.random.split(self.key, len(self.layer_sizes) + 1)

        current_input_dim = input_dim
        for i, size in enumerate(self.layer_sizes):
            w_key, b_key = jax.random.split(keys[i])

            # Scale the weights based on size to avoid explotion in values
            scale = jnp.sqrt(2.0 / current_input_dim)
            weight = jax.random.normal(w_key, (current_input_dim, size)) * scale
            bias = jnp.zeros(size)

            params.append((weight, bias))
            current_input_dim = size

            # Outputlayer
            w_key_out, _ = jax.random.split(keys[-1])
            scale_out = jnp.sqrt(2.0 / current_input_dim)
        
            weight_final = jax.random.normal(w_key_out, (current_input_dim, 1)) * scale_out
            bias_final = jnp.zeros(1)
            
            params.append((weight_final, bias_final))
            
            return params
    
    def reset(self):
        """Reset internal state (not strictly used in JAX scan, but required by Base)."""
        pass

    def get_params(self):
        """Returns the initial random weights and biases."""
        return self.initial_params

    def update_params(self, new_params):
        """Updates the internal parameters (used if running outside main loop)."""
        self.initial_params = new_params
    
    def get_initial_state(self):
        # State vector: [integral_sum, prev_error]
        return jnp.array([0.0, 0.0])
    
    def _activation(self, x):
        """Applies the configured activation function."""
        if self.activation_function_name == "relu":
            return jnp.maximum(0, x)
        elif self.activation_function_name == "sigmoid":
            return 1 / (1 + jnp.exp(-x))
        elif self.activation_function_name == "tanh":
            return jnp.tanh(x)
        else:
            return x # Linear / No activation fallback
        
    def update(self, params, state, error, dt):
        """
        Forward pass.
        
        Args:
            params: List of (weight, bias)-tuples.
            state: [integral_sum, prev_error]
            error: float
            dt: float
        """
        integral_sum, prev_error = state[0], state[1]

        new_integral_sum = integral_sum + (error * dt)
        d_error = (error - prev_error) / dt
        
        input_vector = jnp.array([error, new_integral_sum, d_error])
        
        # Forward Pass
        activation = input_vector
        
        # Loop through every layer except the last
        for w, b in params[:-1]:
            z = jnp.dot(activation, w) + b
            activation = self._activation(z)
        
        # Outputlayer
        w_final, b_final = params[-1]
        u_raw = jnp.dot(activation, w_final) + b_final
        u_val = u_raw[0]

        # --- ENDRING START: Fartsgrense (Output Scaling) ---
        
        # 1. Bruk Tanh for å tvinge verdien mellom -1.0 og 1.0
        # 2. Gang med 0.1 for å si at maks endring er 10% per steg.
        u_constrained = jnp.tanh(u_val) * 0.1
        
        # --- ENDRING SLUTT ---

        new_state = jnp.array([new_integral_sum, error])
        
        return u_constrained, new_state
