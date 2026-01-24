import jax.numpy as jnp
from controllers.Base_controller import BaseController

class ClassicController(BaseController):
    def __init__(self, kp, ki, kd):
        self.initial_params = jnp.array([kp, ki, kd])

    def get_initial_state(self):
        """
        Returns the starting state of the controller.
        """
        return jnp.array([0.0, 0.0])

    def reset(self):
        self.integral_sum = 0.0
        self.prev_error = 0.0
        self.is_first_step = True

    def update(self, params, state, error, dt):
        """
        Args:
            params: Array [kp, ki, kd]
            state: Array [integral_sum, prev_error]
            error: Current error
            dt: Timestep
        Returns:
            u: Control signal
            new_state: Updated [integral_sum, error]
        """
        kp, ki, kd = params[0], params[1], params[2]
        integral_sum, prev_error = state[0], state[1]

        # 1. Integral del
        new_integral_sum = integral_sum + (error * dt)

        # 2. Derivert del
        # Vi antar at prev_error er 0 ved start, så derivatet blir riktig nok.
        d_error = (error - prev_error) / dt

        # 3. Beregn U
        u = (kp * error) + (ki * new_integral_sum) + (kd * d_error)

        # Returner U og den nye tilstanden som skal brukes i neste steg
        new_state = jnp.array([new_integral_sum, error])
        
        return u, new_state
    
    def get_params(self):
        return self.params
    
    def update_params(self, new_params):
        self.params = new_params
