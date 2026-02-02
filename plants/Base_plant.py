import jax.numpy as jnp
from abc import ABC, abstractmethod

"""
This acts as a 'blueprint' for the other plants.
The controllers should be designed with respect to these methods.
"""
class BasePlant(ABC):
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initializes the plant.

        :param kwargs: Keyword arguments.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Should reset the plant to a starting state.

        :returns: Start-state (Y).
        """
        pass

    @abstractmethod
    def get_initial_state(self):
        """Returns the initial state of the plant."""
        pass

    @abstractmethod
    def get_output(self, state):
        """
        Calculates the observable value (Y) from the state.

        :param state: Current state of the plant.

        :returns: Observable value (Y).
        """
        pass

    @abstractmethod
    def update(self, state, control_signal, disturbance):
        """
        Runs one timestep.

        :param state: Current state of the plant.
        :param control_signal: Input from the controller.
        :param disturbance: Noise.

        :returns: New state of the plant.
        """
        pass

    @abstractmethod
    def get_target(self):
        """Returns the target (T) of the plant."""
        pass