import jax.numpy as jnp
from plants.Base_plant import BasePlant
from config import Config

"""
The system simulates a Cournot competition between two firms.
"""
class CournotPlant(BasePlant):

    def __init__(self, p_max, q1_start, q2_start, cm, target_profit):
        """Initializes the cournot plant.

        :param p_max: Maximum price.
        :param q1_start: Initial quantity for firm 1.
        :param q2_start: Initial quantity for firm 2.
        :param cm: Marginal cost.
        :param target_profit: Target profit.
        """
        self.p_max = p_max
        self.cm = cm
        self.target_profit = target_profit
        self.initial_q = jnp.array([q1_start, q2_start]) # Initial quantities

    def reset(self):
        """Resets the internal state (used if running outside JAX scan)."""
        return self.initial_q
    
    def get_initial_state(self):
        """Returns the initial state of the plant."""
        return self.initial_q
    
    def get_output(self, state):
        """Calculates the profit of the plant.

        :param state: Current state of the plant.

        :returns: Output of the plant.
        """
        q1, q2 = state
        q_total = q1 + q2
        price = self.p_max - q_total
        profit = q1 * (price - self.cm)
        return profit

    def update(self, state, control_signal, disturbance):
        """Updates the state of the plant.

        :param state: Current state of the plant.
        :param control_signal: Control signal from the controller.
        :param disturbance: Disturbance from the environment.

        :returns: New state of the plant.
        """
        q1, q2 = state
        U = control_signal
        D = disturbance

        q1_new = jnp.clip(q1 + U, 0.0, 1.0)
        q2_new = jnp.clip(q2 + D, 0.0, 1.0) # Konkurrenten endres av støy (D)
        
        return jnp.stack([q1_new, q2_new])


    def get_target(self):
        """Returns the target profit of the plant."""
        return self.target_profit