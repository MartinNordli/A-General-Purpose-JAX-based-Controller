import jax.numpy as jnp
from plants.Base_plant import BasePlant

class BathtubPlant(BasePlant):
    def __init__(self, area, drain_c, initial_height, target_height):
        self.area = area
        self.drain_c = drain_c
        self.initial_height = initial_height
        self.g = 9.8
        self.current_h = initial_height
        self.target = target_height
    
    def reset(self):
        self.current_h = self.initial_height
        return self.current_h
    
    def get_target(self):
        """
        The goal is to keep the waterlevel stable at the initial height.
        """
        return self.target
    
    def update(self, control_signal, disturbance):
        g = self.g
        H = self.current_h
        H_safe = jnp.maximum(H, 0.0) # To avoid negative values for H and thus a negative root.
        A = self.area
        C = self.drain_c
        V = jnp.sqrt(2 * g * H_safe)
        Q = V * C
        U = control_signal
        D = disturbance

        # The change in volume
        dB = U + D - Q

        # The change in height
        dH = dB / A

        # Update the state
        self.current_h = H + dH

        return self.current_h
    