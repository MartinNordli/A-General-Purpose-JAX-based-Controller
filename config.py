class Config:

    # ==========================================
    #  Plant and controller to use
    # ==========================================

    PLANT_TO_RUN = "Drone" # Bathtub, Cournot or Drone
    CONTROLLER_TO_USE = "NeuralNet" # Classic or NeuralNet

    # ==========================================
    # General configurations
    # ==========================================

    CONTROL_SIGNAL = 0.5
    DISTURBANCE = 0.1 # 0.1 for bathtub

    # CLASSIC: LR: 0.01 was good for bathtub, 0.001 for cournot.
    # NeuralNet: LR: 0.0001, 400 epochs.
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 30000
    TIMESTEPS_PER_EPOCH = 1000

    # ========================================= =
    # Classic controller configurations
    # ==========================================

    PID_KP_START = 0.5 # 0.1
    PID_KI_START = 0.01 # 0.01
    PID_KD_START = 8.0 #0.01

    # ==========================================
    # NeuralNet controller configurations
    # ==========================================

    # [5, 5] means 2 hidden layers with 5 neurons each.
    NN_LAYER_SIZES = [10,10,10,10] 
    
    # Activationfunctions: "relu", "sigmoid", eller "tanh"
    NN_ACTIVATION = "relu"

    # ==========================================
    # Bathtub configurations
    # ==========================================

    BATHTUB_AREA = 10.0
    BATHTUB_DRAIN_C = 0.1
    BATHTUB_H0 = 1.0
    BATHTUB_TARGET = 5.0

    # ==========================================
    # Cournot configurations
    # ==========================================

    COURNOT_Q1_START = 0.1
    COURNOT_Q2_START = 0.6
    COURNOT_P_MAX = 10.0
    COURNOT_CM = 0.1
    COURNOT_TARGET_PROFIT = 2.5

    # ==========================================
    # Drone configurations
    # ==========================================
    DRONE_MASS = 1.0
    DRONE_INITIAL_HEIGHT = 20.0
    DRONE_TARGET = 50.0