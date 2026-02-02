import jax.numpy as jnp
from plants.Base_plant import BasePlant

"""
The system simulates the water level height (H) in a bathtub with a constant
cross-sectional area (A) and a drain with cross-sectional area (C).
"""
class BathtubPlant(BasePlant):
    
    def __init__(self, area, drain_c, initial_height, target_height):
        """Initializes the bathtub plant.

        :param area: Cross-sectional area of the bathtub (A).
        :param drain_c: Cross-sectional area of the drain (C).
        :param initial_height: Initial height.
        :param target_height: Target height.
        """
        self.area = area
        self.drain_c = drain_c
        self.H0 = initial_height
        self.target = target_height
        self.g = 9.8

    def get_target(self):
        """Returns the target height."""
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
        """
        # Hent høyde. Hvis state er et array (som [1.0]), henter vi verdien.
        H = state[0] if isinstance(state, jnp.ndarray) and state.ndim > 0 else state
        
        # --- FIX 1: SKALERING & SIKRING AV INPUT ---
        # 1. jnp.abs: Vi sikrer at vi aldri fyller med "negativt vann".
        # 2. * 100.0: Samme triks som dronen. Nettverket gir små tall (0-1), 
        #    men badekaret trenger mye vann for å fylles. 
        #    Dette fungerer som en "forsterker" på kranen.
        U = jnp.abs(control_signal) * 100.0 
        
        D = disturbance
        dt = 0.1 # --- FIX 2: Tidssteg (gjør simuleringen mye mer stabil)
        
        # Sikring: Høyde kan ikke være negativ i fysikk-formelen
        H_safe = jnp.maximum(H, 0.0)
        
        # Physics dynamics:
        # Velocity out: V = sqrt(2 * g * H)
        V = jnp.sqrt(2 * self.g * H_safe)
        
        # Flow rate out: Q = V * C
        Q = V * self.drain_c
        
        # Volume change: dB = Inflow (U+D) - Outflow (Q)
        dB = U + D - Q
        
        # Height change: dH = dB / A
        dH = dB / self.area
        
        # Update state med tidssteg
        new_H = H + dH * dt
        
        # SIKRING: Høyden kan aldri være lavere enn 0 (tomt kar).
        new_H = jnp.maximum(new_H, 0.0)
        
        # --- ENDRING HER ---
        # FJERNET jnp.atleast_1d()
        # Vi returnerer bare new_H direkte. 
        # Da holder tallene seg "flate" (skalarer), og kontrolleren krasjer ikke.
        return new_H