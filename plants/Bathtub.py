import jax.numpy as jnp
from plants.Base_plant import BasePlant

class BathtubPlant(BasePlant):
    """
    The system simulates the water level height (H) in a bathtub with a constant
    cross-sectional area (A) and a drain with cross-sectional area (C).
    """
    def __init__(self, area, drain_c, initial_height, target_height):
        self.area = area      # Cross-sectional area of the bathtub (A)
        self.drain_c = drain_c # Cross-sectional area of the drain (C)
        self.H0 = initial_height
        self.target = target_height
        self.g = 9.8          # Gravitational constant

    def get_target(self):
        return self.target
    
    def reset(self):
        """Resets the internal state (used if running outside JAX scan)."""
        self.current_h = self.initial_height
        return self.current_h

    def get_initial_state(self):
        """Returns the initial state (height) as a JAX array."""
        return jnp.array(self.H0)
    
    def get_output(self, state):
        """The water height is the same as the state"""
        return state

    def update(self, state, control_signal, disturbance):
        """
        Calculates the state transition for one timestep.
        
        Physics dynamics:
        1. Velocity out: V = sqrt(2 * g * H)
        2. Flow rate out: Q = V * C
        3. Volume change: dB = U + D - Q
        4. Height change: dH = dB / A

        Args:
            state (float): Current water height (H).
            control_signal (float): Inflow controlled by the agent (U).
            disturbance (float): Random noise/inflow (D).

        Returns:
            new_H (float): The updated water height.
        """
        H = state
        U = control_signal
        D = disturbance
        
        # Enforce non-negative height constraint. 
        # Prevents numerical instability (NaN) when taking the square root.
        H_safe = jnp.maximum(H, 0.0)
        
        # Calculate velocity
        V = jnp.sqrt(2 * self.g * H_safe)
        
        # Calculate the flow rate of exiting water
        Q = V * self.drain_c
        
        # Calculate change in volume (Inflow - Outflow)
        dB = U + D - Q
        
        # Calculate change in height based on volume change
        dH = dB / self.area
        
        # Update state
        new_H = H + dH
        return new_H