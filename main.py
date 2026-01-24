import jax
import jax.numpy as jnp
import numpy as np
from config import Config
from consys import Consys
from utils.visualization import plot_results

# Import specific plants and controllers
from plants.Bathtub import BathtubPlant
from plants.Cournot import CournotPlant
from controllers.Classic_controller import ClassicController
# from controllers.Neural_network_controller import NeuralNetworkController  # Uncomment when implemented

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
            target_profit=Config.COURNOT_TARGET_PROFIT
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
        # Placeholder for Neural Network initialization
        # controller = NeuralNetworkController(...) 
        # current_params = controller.init_weights(...) 
        raise NotImplementedError("Neural Network Controller is not yet fully implemented in main.py")
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
        
        # Update Controller Parameters (Gradient Descent)
        # w_new = w_old - learning_rate * gradient
        current_params = current_params - (Config.LEARNING_RATE * grads)
        
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

if __name__ == "__main__":
    main()