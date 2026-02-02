import jax.numpy as jnp
from controllers.Base_controller import BaseController

"""A classic PID controller."""
class ClassicController(BaseController):
    
    def __init__(self, kp, ki, kd):
        """
        Initializes the controller with the given parameters.

        :param kp: Proportional gain
        :param ki: Integral gain
        :param kd: Derivative gain
        """
        self.initial_params = jnp.array([kp, ki, kd])

    def get_initial_state(self):
        """Returns the starting state of the controller."""
        return jnp.array([0.0, 0.0])

    def reset(self):
        """Resets the internal state of the controller."""
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

        # Integral part
        new_integral_sum = integral_sum + (error * dt)

        # Derivative part
        # We assume that prev_error is 0 at start, so the derivative is correct enough.
        d_error = (error - prev_error) / dt

        # Calculate U
        u = (kp * error) + (ki * new_integral_sum) + (kd * d_error)

        # Return U and the new state to be used in the next step
        new_state = jnp.array([new_integral_sum, error])
        
        return u, new_state
    
    def get_params(self):
        """Returns the initial parameters of the controller."""
        return self.initial_params
    
    def update_params(self, new_params):
        """Updates the parameters of the controller."""
        self.initial_params = new_params
