import jax
import jax.numpy as jnp
import numpy as np
from config import Config
from consys import Consys
from utils.visualization import plot_results, plot_system_response

# Import specific plants and controllers
from plants.Bathtub import BathtubPlant
from plants.Cournot import CournotPlant
from plants.Drone import DronePlant
from controllers.Classic_controller import ClassicController
from controllers.Neural_network_controller import NeuralNetworkController

def main():
    """
    Main execution loop for the JAX-based Control System.
    Selects plant and controller based on configuration, runs the training loop,
    and visualizes the results.
    """
    
    # ==========================================
    # 1. Initialization and Configuration
    # ==========================================
    
    # --- Plant Selection ---
    if Config.PLANT_TO_RUN == "Bathtub":
        plant = BathtubPlant(
            area=Config.BATHTUB_AREA,
            drain_c=Config.BATHTUB_DRAIN_C, 
            initial_height=Config.BATHTUB_H0,
            target_height=Config.BATHTUB_TARGET
        )
    elif Config.PLANT_TO_RUN == "Cournot":
        plant = CournotPlant(
            p_max=Config.COURNOT_P_MAX,
            cm=Config.COURNOT_CM,
            target_profit=Config.COURNOT_TARGET_PROFIT,
            q1_start=Config.COURNOT_Q1_START,
            q2_start=Config.COURNOT_Q2_START
        )
    elif Config.PLANT_TO_RUN == "Drone":
        plant = DronePlant(
            mass=Config.DRONE_MASS,
            init_height=Config.DRONE_INITIAL_HEIGHT,
            target_height=Config.DRONE_TARGET
        )
    else:
        raise ValueError(f"Invalid plant selected in Config: {Config.PLANT_TO_RUN}")

    # --- Controller Selection ---
    if Config.CONTROLLER_TO_USE == "Classic":
        controller = ClassicController(
            kp=Config.PID_KP_START,
            ki=Config.PID_KI_START,
            kd=Config.PID_KD_START
        )
        # For the classic controller, we initialize parameters from the config/class
        current_params = controller.initial_params
        
    elif Config.CONTROLLER_TO_USE == "NeuralNet":
        controller = NeuralNetworkController(
            layer_sizes=Config.NN_LAYER_SIZES,
            activation_function=Config.NN_ACTIVATION
        )
        current_params = controller.initial_params
    else:
        raise ValueError(f"Invalid controller selected in Config: {Config.CONTROLLER_TO_USE}")

    # Initialize the Control System (CONSYS)
    system = Consys(plant, controller)
    
    # Storage for visualization
    mse_history = []
    param_history = []
    
    # Define the gradient function using JAX
    # jax.jit compiles the function for performance. 
    # value_and_grad returns both the loss (MSE) and the gradients relative to argnums=0 (params).
    grad_fn = jax.jit(jax.value_and_grad(system.run_epoch, argnums=0))

    print(f"Starting training for Plant: {Config.PLANT_TO_RUN} using Controller: {Config.CONTROLLER_TO_USE}...")

    # ==========================================
    # 2. Training Loop
    # ==========================================
    for epoch in range(Config.NUM_EPOCHS):
        # Generate random disturbances for the entire epoch
        # Using a uniform distribution based on the config settings
        disturbances = np.random.uniform(
            low=-Config.DISTURBANCE, 
            high=Config.DISTURBANCE, 
            size=(Config.TIMESTEPS_PER_EPOCH,)
        )
        disturbances = jnp.array(disturbances) # Convert to JAX array

        # Execute simulation and compute gradients
        loss, grads = grad_fn(current_params, disturbances)
        
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

        current_params = jax.tree_util.tree_map(
            lambda p, g: p - Config.LEARNING_RATE * g,
            current_params,
            grads
        )

        if Config.CONTROLLER_TO_USE == "Classic":
             current_params = jax.tree_util.tree_map(lambda p: jnp.maximum(p, 0.0), current_params)
        
        # Record history
        mse_history.append(loss)
        param_history.append(current_params)
        
        # Periodic logging
        if epoch % 10 == 0:
            # Formatting parameters for cleaner output
            param_str = str(current_params) if hasattr(current_params, '__len__') and len(current_params) < 5 else "NeuralNet Weights"
            print(f"Epoch {epoch}: MSE = {loss:.5f}, Params = {param_str}")

    # ==========================================
    # 3. Visualization
    # ==========================================
    print("\nTraining complete. Generating plots...")
    
    # Only pass parameter history if using the Classic controller (as visualized in the assignment)
    if Config.CONTROLLER_TO_USE == "Classic":
        plot_results(mse_history, param_history)
    else:
        plot_results(mse_history)
    
    # ==========================================
    # 4. Final Simulation Check
    # ==========================================
    print("\nRunning verification simulation...")
    
    test_disturbances = np.random.uniform(
        low=-Config.DISTURBANCE, 
        high=Config.DISTURBANCE, 
        size=(Config.TIMESTEPS_PER_EPOCH,)
    )
    test_disturbances = jnp.array(test_disturbances)

    # Run a simulation using the best parameters (current).
    final_history = system.run_simulation(current_params, test_disturbances)
    
    target_val = plant.get_target() if Config.PLANT_TO_RUN == "Bathtub" else None
    plot_system_response(final_history, plant_name=Config.PLANT_TO_RUN, target=target_val)

if __name__ == "__main__":
    main()