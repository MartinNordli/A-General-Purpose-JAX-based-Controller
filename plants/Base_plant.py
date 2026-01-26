import jax.numpy as jnp
from abc import ABC, abstractmethod

class BasePlant(ABC):
    """
    This acts as a 'blueprint' for the other plants.
    The controllers should be designed with respect to these methods.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        """
        Should reset the plant to a starting state.
        Returns: Start-state (Y).
        """
        pass

    @abstractmethod
    def get_initial_state(self):
        pass

    @abstractmethod
    def get_output(self, state):
        """
        Beregner den observerbare verdien (Y) fra tilstanden.
        For Badekar: Returnerer H.
        For Cournot: Returnerer Profit.
        """
        pass

    @abstractmethod
    def update(self, control_signal, disturbance):
        """
        Runs one timestep.
        
        :param control_signal: Input from the controller.
        :param disturbance: Noise.
        """
        pass

    @abstractmethod
    def get_target(self):
        """
        Returns the target (T) of the plant.
        """
        pass