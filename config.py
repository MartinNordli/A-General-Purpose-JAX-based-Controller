class Config:

    # ==========================================
    #  Plant and controller to use
    # ==========================================

    PLANT_TO_RUN = "Cournot" # Bathtub, Cournot or plant3
    CONTROLLER_TO_USE = "NeuralNet" # Classic or NeuralNet

    # ==========================================
    # General configurations
    # ==========================================

    CONTROL_SIGNAL = 0.5
    DISTURBANCE = 0.01 # 0.1 for bathtub

    # CLASSIC: 0.01 was good for bathtub, 0.001 for cournot.
    # NeuralNet: 
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 400
    TIMESTEPS_PER_EPOCH = 100

    # ==========================================
    # Classic controller configurations
    # ==========================================

    PID_KP_START = 0.1
    PID_KI_START = 0.01
    PID_KD_START = 0.01

    # ==========================================
    # NeuralNet controller configurations
    # ==========================================

    # [5, 5] means 2 hidden layers with 5 neurons each.
    NN_LAYER_SIZES = [5] 
    
    # Activationfunctions: "relu", "sigmoid", eller "tanh"
    NN_ACTIVATION = "tanh"

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
