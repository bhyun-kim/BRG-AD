cfg = dict(
    beam = dict(
        length = 3,
        elasticity =206.0E+9,
        mass=7850,
        damping_factor=0.01,
        num_elements=16,
        rot_spring=0.0
    ),
    section = dict(
        type="box", 
        height=2.0, 
        width=2.0, 
        thickness=0.1 
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
    sim_duration=3600,
    load_sources=[
        dict(
            axial_load_range=[5000, 10000], 
            noise_ratio=0.03, 
            velocity_range=[10, 20],
            period=20,
            num_of_cars_per_period=10,
            num_of_axis=2,
            wheelbase=1
        ),   
    ],
    anomalies=[
        dict(
            type="over_loading",
            axial_load_range=[100000, 150000], 
            noise_ratio=0.03, 
            velocity_range=[5, 15],
            period=500,
            num_of_cars_per_period= 1,
            num_of_axis=3,
            wheelbase=2
        ),
        dict(
            type="section_loss",
            loss_type="moment_of_inertia",
            loss_factor=0.9, 
            loss_location=7
        ),
        dict(
            type="sns_noise",
            noise_type="drift",
            num_of_anomalies=6,
            anomaly_sns_loc=[True, False, False, False] 
        ),
        dict(
            type="sns_noise",      
            noise_type="weak_noise",
            num_of_anomalies=6,
            anomaly_sns_loc=[True, False, False, False] 
        ),
        dict(
            type="sns_noise",         
            noise_type="strong_noise",
            num_of_anomalies=6,
            anomaly_sns_loc=[True, False, False, False] 
        )
    ]
)
