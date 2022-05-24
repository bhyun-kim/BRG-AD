cfg = dict(
    num_data=500,
    beam = dict(
        length=0.2,
        elasticity=206.0E+9,
        mass=7850,
        damping_factor=0.01,
        num_elements=16,
        rot_spring=0.0
    ),
    section = dict(
        type="rectangle", 
        height=float(10e-3), 
        width=float(50e-3), 
    ),
    sensor = dict(
        sns_loc=[4, 7, 11, 15],
        freq_cutoff=50,
        freq_sample=128, 
        acc_noise_ratio=0.05, 
        acc_noise_ref=6, 
        disp_noise_ratio=0.05, 
        disp_noise_ref=6 
    ),
    duration=20,
    load_sources=[
        dict(
            axial_load_range=[10, 100], 
            noise_ratio=0.03, 
            velocity_range=[0.3, 1.0],
            num_of_cars=3,
            num_of_axis=2,
            wheelbase=0.1
        ),   
    ],
    anomaly=[
        dict(
            type="section_loss",
            subtype="moment_of_inertia",
            loss_factor=0.9, 
            loss_location=7,
            num_of_anomalies=100
        ),
        dict(
            type="sns_noise",
            subtype="drift",
            num_of_anomalies=100,
            anomaly_sns_loc=[True, True, True, True] 
        ),
        dict(
            type="sns_noise",      
            subtype="weak_noise",
            num_of_anomalies=100,
            anomaly_sns_loc=[True, True, True, True] 
        ),
        dict(
            type="sns_noise",         
            subtype="strong_noise",
            num_of_anomalies=100,
            anomaly_sns_loc=[True, True, True, True] 
        )
    ]
)
