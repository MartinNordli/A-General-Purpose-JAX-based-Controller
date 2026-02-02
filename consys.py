import jax
import jax.numpy as jnp
from config import Config

"""
This class integrates a Plant and a Controller into a single feedback loop.
    
It handles the execution of simulation epochs using JAX to ensure the 
entire process is differentiable.
"""
class Consys:

    def __init__(self, plant, controller):
        """Initializes the control system.

        :param plant: The plant to use.
        :param controller: The controller to use.
        """
        self.plant = plant
        self.controller = controller

    def run_epoch(self, params, disturbance_array):
        """
        Executes a full simulation epoch.
        
        Using `jax.lax.scan` allows for efficient compilation and Backpropagation 
        Through Time (BPTT) over the entire simulation history.

        :param params: The trainable parameters of the controller (e.g., PID gains or Neural Net weights).
        :param disturbance_array: An array of disturbance values (D) for every timestep in the epoch.

        :returns: The Mean Squared Error (MSE) for the entire epoch.
        """
        target = self.plant.get_target()
        
        # Retrieve initial functional states from the plant and controller
        init_plant_state = self.plant.get_initial_state()
        init_controller_state = self.controller.get_initial_state()
        
        # Initialize the carry tuple (the state that propagates through time)
        init_carry = (init_plant_state, init_controller_state)

        # Define the single-step transition function for jax.lax.scan
        def step_fn(carry, disturbance_t):
            plant_state, controller_state = carry
            
            # Calculate Error (E = Target - Output)
            current_y = self.plant.get_output(plant_state)
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
    
    def run_simulation(self, params, disturbance_array):
        """
        Runs a simulation without calculating gradients to get data for plotting.

        :param params: The trainable parameters of the controller (e.g., PID gains or Neural Net weights).
        :param disturbance_array: An array of disturbance values (D) for every timestep in the epoch.

        :returns: The history of states (State History).
        """
        target = self.plant.get_target()
        init_carry = (self.plant.get_initial_state(), self.controller.get_initial_state())

        def step_fn(carry, disturbance_t):
            """
            Defines the single-step transition function for jax.lax.scan.

            :param carry: The current state of the plant and controller.
            :param disturbance_t: The disturbance value for the current timestep.

            :returns: The new state of the plant and controller.
            """
            plant_state, controller_state = carry
            
            current_y = self.plant.get_output(plant_state)
            error = target - current_y
            u, new_controller_state = self.controller.update(params, controller_state, error, dt=1.0)
            new_plant_state = self.plant.update(plant_state, u, disturbance_t)
            
            new_carry = (new_plant_state, new_controller_state)
    
            return new_carry, plant_state

        final_carry, history = jax.lax.scan(step_fn, init_carry, disturbance_array)
        
        return history