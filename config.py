class Config:
    PLANT_TO_RUN = "Bathtub"
    # CONTROLLER_TO_USE = "Classic"

    CONTROL_SIGNAL = 0.5
    DISTURBANCE = 0.1

    LEARNING_RATE = 0.01
    NUM_EPOCHS = 100
    TIMESTEPS_PER_EPOCH = 100

    # --- Bathtub ---
    BATHTUB_AREA = 10.0
    BATHTUB_DRAIN_C = 0.1
    BATHTUB_H0 = 5.0

    # --- Cournot ---
    COURNOT_P_MAX = 10.0
    COURNOT_CM = 0.1

    # --- Targets ---
    BATHTUB_TARGET = 5.0
    COURNOT_TARGET_PROFIT = 2.5