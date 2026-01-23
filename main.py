from config import Config
from plants.Bathtub import BathtubPlant

def run():
    active_plant_name = "Bathtub"

    if active_plant_name == "Bathtub":
        plant = BathtubPlant(
            area = Config.BATHTUB_AREA,
            drain_c = Config.BATHTUB_DRAIN_C,
            initial_height = Config.BATHTUB_H0,
            target_height = Config.BATHTUB_TARGET
        )
    # elif active_plant_name == "Cournot"...

    print(f"Plant initialized: {active_plant_name}")
    print(f"Initial State: {plant.reset()}")

    new_state = plant.update(control_signal=0.5, disturbance=0.1)
    print(f"State after 1 step: {new_state}")


if __name__ == "__main__":
    run() 