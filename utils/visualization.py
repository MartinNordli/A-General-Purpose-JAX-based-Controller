import matplotlib.pyplot as plt
import numpy as np

def plot_results(mse_history, param_history=None):
    """
    Tegner grafene som kreves av oppgaven.
    
    Args:
        mse_history: Liste med MSE-verdier per epoke.
        param_history: (Valgfri) Liste/Array med PID-parametere per epoke.
                       Form: (antall_epoker, 3) der kolonnene er kp, ki, kd.
    """
    # Lag en figur med 1 eller 2 subplots avhengig av om vi har parametere
    num_plots = 2 if param_history is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(12, 5))
    
    # Hvis vi bare har ett plot, pakk det inn i en liste for konsistens
    if num_plots == 1:
        axes = [axes]

    # --- Plot 1: Læringsprogresjon (MSE) ---
    ax1 = axes[0]
    ax1.plot(mse_history, label='MSE', color='black')
    ax1.set_title('Learning Progression')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.grid(True)
    
    # --- Plot 2: Kontrollparametere (Kun for PID) ---
    if param_history is not None:
        ax2 = axes[1]
        # Konverter til numpy array hvis det er en JAX array
        params = np.array(param_history)
        
        ax2.plot(params[:, 0], label='Kp')
        ax2.plot(params[:, 1], label='Ki')
        ax2.plot(params[:, 2], label='Kd')
        
        ax2.set_title('Control Parameters Evolution')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Parameter Value')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.show() # Eller plt.savefig('result.png')