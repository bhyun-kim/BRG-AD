


"""
Baseline configuration for bridge monitoring simulation 

"""

sim_config = dict(

    # 1. Beam 
    beam = dict(
        length = 3,                         # elment beam length in m
        elasticity = float(206.e+9),        # Young's modulus : Steel
        mass = 7850,                        # steel mass density of unit volumn
        damping_factor =0.01,               # damping factor
        num_elements = 16,                  # number of elements in FEM model 
        rot_spring = 0.0,  # additional rotational stiffness at both supports
    ),
    
    # 1-1. Beam section 
    section = dict( 
        type = 'box', 
        height = float(2), # height in m
        width= float(2), # width in m
        thickness = 0.1 # thickness
    ),

    # 2. Sensor 
    sensor = dict( 
        sns_loc = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,] ,
        fc = 50,
        fs = 2.56*50, 
        # 2-1. acc and strain related params
        Ra = 0.05,  # RMS noise level for acceleration
        acc_noise_ref = 6, # acc noise level is determined using this sensor
        Rs = 0.05, # RMS noise level for strain
        disp_noise_ref = 1, # strain noise level is determined using this sensor
    ),
    
    
    # 3. time 
    duration = 3600,

    # 4. Exteranl Load 
    load_sources = [
        dict(
            axial_load_range = (5000, 10000), 
            noise_ratio = 0.03, 
            velocity_range = (10, 20),
            period = 20,
            num_of_cars_per_period = 10,
            num_of_axis = 2,
            wheelbase = 1,
            ),
    ],

    # 5. Anomaly Setting 
    anomaly = [
        dict(
            type = 'over_loading',
            axial_load_range = (100000, 150000), # DB-24 axial load limit 
            noise_ratio = 0.03, 
            velocity_range = (5, 15),
            period = 500,
            num_of_cars_per_period =  1,
            num_of_axis = 3,
            wheelbase = 2,
        ),
        dict(
            type = 'section_loss',
            loss_type = 'moment_of_inertia', # support for 1. moment of inertia, 2. section 
            loss_factor = 0.9, # 
            loss_location = 7
        ),
        dict(
            type = 'sns_noise',
            noise_type = 'drift',        # factors for drift is preset in the function  
            num_of_anomalies = 6,
            anomaly_sns_loc = [True, False, False, False] # has to have same length with sns_loc
        ),
        dict(
            type = 'sns_noise',          # factors for weak noise is preset in the function 
            noise_type = 'weak_noise',
            num_of_anomalies = 6,
            anomaly_sns_loc = [True, False, False, False] # has to have same length with sns_loc
        ),
        dict(
            type = 'sns_noise',         # factors for strong noise is preset in the function 
            noise_type = 'strong_noise',
            num_of_anomalies = 6,
            anomaly_sns_loc = [True, False, False, False] # has to have same length with sns_loc
        )]





)
