import jax.numpy as jnp
from plants.Base_plant import BasePlant
from config import Config

class CournotPlant(BasePlant):

    def __init__(self, p_max, q1_start, q2_start, cm, target_profit):
        self.p_max = p_max
        self.cm = cm
        self.target_profit = target_profit
        self.initial_q = jnp.array([q1_start, q2_start])

    def reset(self):
        return self.initial_q
    
    def get_initial_state(self):
        return self.initial_q
    
    def get_output(self, state):
        q1, q2 = state
        q_total = q1 + q2
        price = self.p_max - q_total
        profit = q1 * (price - self.cm)
        return profit

    def update(self, state, control_signal, disturbance):
        q1, q2 = state
        U = control_signal
        D = disturbance

        q1_new = jnp.clip(q1 + U, 0.0, 1.0)
        q2_new = jnp.clip(q2 + D, 0.0, 1.0) # Konkurrenten endres av støy (D)
        
        return jnp.stack([q1_new, q2_new])


    def get_target(self):
        return self.target_profit