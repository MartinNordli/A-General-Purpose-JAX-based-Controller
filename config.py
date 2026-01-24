class Config:
    # 'PLANT_TO_RUN' can be 'Bathtub', 'Cournot' or 'plant3'
    PLANT_TO_RUN = "Bathtub"
    # 'CONTROLLER_TO_USE' can be 'Classic' or 'NeuralNet'
    CONTROLLER_TO_USE = "Classic"

    CONTROL_SIGNAL = 0.5
    DISTURBANCE = 0.1

    # Classic controller configuration
    PID_KP_START = 0.1
    PID_KI_START = 0.01
    PID_KD_START = 0.01

    LEARNING_RATE = 0.01
    NUM_EPOCHS = 200
    TIMESTEPS_PER_EPOCH = 100

    # --- Bathtub ---
    BATHTUB_AREA = 10.0
    BATHTUB_DRAIN_C = 0.1
    BATHTUB_H0 = 1.0
    BATHTUB_TARGET = 5.0

    # --- Cournot ---
    COURNOT_P_MAX = 10.0
    COURNOT_CM = 0.1
    COURNOT_TARGET_PROFIT = 2.5