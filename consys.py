import jax
import jax.numpy as jnp
from config import Config

class Consys:
    """
    Control System wrapper.
    
    This class integrates a Plant and a Controller into a single feedback loop.
    It handles the execution of simulation epochs using JAX to ensure the 
    entire process is differentiable.
    """
    def __init__(self, plant, controller):
        self.plant = plant
        self.controller = controller

    def run_epoch(self, params, disturbance_array):
        """
        Executes a full simulation epoch.
        
        Using `jax.lax.scan` allows for efficient compilation and Backpropagation 
        Through Time (BPTT) over the entire simulation history.

        Args:
            params: The trainable parameters of the controller (e.g., PID gains or Neural Net weights).
            disturbance_array: An array of disturbance values (D) for every timestep in the epoch.

        Returns:
            mse: The Mean Squared Error (MSE) for the entire epoch.
        """
        target = self.plant.get_target()
        
        # Retrieve initial functional states from the components
        init_plant_state = self.plant.get_initial_state()
        init_controller_state = self.controller.get_initial_state()
        
        # Initialize the carry tuple (the state that propagates through time)
        init_carry = (init_plant_state, init_controller_state)

        # Define the single-step transition function for jax.lax.scan
        def step_fn(carry, disturbance_t):
            plant_state, controller_state = carry
            
            # Calculate Error (E = Target - Output)
            current_y = plant_state 
            error = target - current_y
            
            # Computes the control signal (U) and the new internal state of the controller.
            # Fixed timestep dt=1.0 is assumed for simplicity.
            u, new_controller_state = self.controller.update(params, controller_state, error, dt=1.0)
            
            # Evolves the plant state based on the control signal (U) and disturbance (D).
            new_plant_state = self.plant.update(plant_state, u, disturbance_t)
            
            # Pack the new state (carry) and the metric to be tracked (Squared Error)
            new_carry = (new_plant_state, new_controller_state)
            step_output = error ** 2
            
            return new_carry, step_output

        # Execute the simulation loop efficiently
        final_carry, squared_errors = jax.lax.scan(step_fn, init_carry, disturbance_array)
        
        # Compute the Mean Squared Error (MSE) for the epoch
        mse = jnp.mean(squared_errors)
        
        return mse