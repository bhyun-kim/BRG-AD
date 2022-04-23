import os
import click
import logging

from utils.config import Config
from types import SimpleNamespace

from fem import FEM 
from scipy import signal
from lib.signal_process import band_limited_noise
from lib.excitation import load_source, generate_excitation
from lib.anomaly import decide_anomal_type, apply_sns_noise, apply_section_loss, apply_over_loading

#########################
##### Configuration #####
#########################

@click.command()
@click.argument('data_path', type=click.Path(exists=True))

@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--num_data', type=int, default= 2000, help='Number of data to create')
        

# Beam model cfg
@click.option('--length', type=float, default=0.2, help='Element length in meter')
@click.option('--elastictiy', type=float, default=206.e+9, help="Young's modulus of element (default: steel)" )
@click.option('--mass', type=float, default=7850, help='density of element')
@click.option('--damping_factor', type=float, default=0.01, help='damping factor')

# Beam model size cfg
@click.option('--height', type=float, default=float(10e-3), help='Beam height in meter')
@click.option('--width', type=float, default=float(50e-3), help='Beam width in meter')

# Beam model element cfg
@click.option('--num_elements', type=int, default=16, help='Number of element in FEM model')
@click.option('--sns_loc', type=list, default=[3, 7, 11, 15], help='Location of sensors.')

# Time related cfg
@click.option('--sim_duration', type=int, default=20, help='Duration of FEM simulation in seconds')
@click.option('--freq_sample', type=float, default=128, help='Sampling frequency of FEM simulation')
@click.option('--freq_cutoff', type=float, default=50, help='Cutoff frequency for anti aliasing filter')

# Exitation cfg
@click.option('--axis_loads', type=list(float), default=[5., 10., 15.], help='Axis loads in N' )
@click.option('--velocities', type=list(float), default=[0.6, 0.8, 1.0], help='Velocities of axis in m/s') 
@click.option('--load_noise_ratio', type=float, default=0.03, help='Axis load noise ratio')

# Measurement noise cfg
@click.option('--acc_noise_ratio', type=float, default=0.05, help='RMS noise level for acceleration')
@click.option('--acc_noise_ref', type=int, default=6, help='Acc noise level is determined using this sensor')
@click.option('--disp_noise_ratio', type=float, default=0.05, help='RMS noise level for displacement')
@click.option('--disp_noise_ref', type=int, default=6, help='Displacement noise level is determined using this sensor')

# Outlier cfg
@click.option('--outlier_class_names', type=list, default=[], help='List of outlier class names')
@click.option('--num_outliers', type=list(int), default=[0], 
               help="Number of outliers in the dataset, Its lenght should be equal to outlier_class_names'.")

# Section loss cfg
@click.option('--section_loss_ratio', type=float, default=0.9, help='Section loss ratio to be remained')
@click.option('--section_loss_loc', type=int, default=7, help='Section loss location')

# Additional cfg
@click.option('--rot_spring', type=float, default=0.0, help='Additional rotational stiffness at both supports')


def main(
    data_path, load_config, num_data, 
    length, elastictiy, mass, damping_factor, height, width,
    num_elements, sns_loc, sim_duration, freq_sample, freq_cutoff,
    axis_loads, velocities, load_noise_ratio, acc_noise_ratio, 
    acc_noise_ref, disp_noise_ratio, disp_noise_ref, outlier_class_names,
    num_outliers, section_loss_ratio, section_loss_loc, rot_spring
    ):

    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    os.makedirs(data_path, exist_ok=True)
    log_file = data_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)


    data = dict()
    data['is_anomal'] = False 
    data['anomal_type'] = None
    data['sns_loc'] = sns_loc
    data['anomal_type'] = [None] * len(sns_loc) * 2 
    data['acc'] = None 
    data['disp'] = None
    data['sample_rate'] = fs

    # convert height, width, area, moment_inertia to list 
    height_list = [height]*num_elements
    width_list = [width]*num_elements
    area_list = [area]*num_elements
    moment_inertia_list = [moment_inertia]*num_elements

    data = decide_anomal_type(data, anomaly_ratio, anomal_types)

    section_properties = (height_list, width_list, area_list, moment_inertia_list)

    # apply section loss        
    section_properties = apply_section_loss(data, section_properties, section_loss_loc, section_loss_factor)
    height_list, width_list, area_list, moment_inertia_list= section_properties
    

    # create fem model
    model = FEM(length, height_list, width_list, area_list, moment_inertia_list, elasticity, 
                mass, damping_factor, num_elements)

    model.assemble_stiff_mat()
    model.assemble_mass_mat()
    # model.apply_rot_spring()
    model.do_static_condensation()
    model.create_damping_mat()
    model.create_ss()

    static_load = random.choice(static_loads)
    static_load_input = apply_over_loading(data, static_load)
    data['load_source'] = static_load

    velocity = random.choice(velocities)
    load = load_source(velocity, static_load_input, noise_ratio)

    data['velocity'] = velocity

    starting_time = 0 # random.choice([0, 1, 2])
    data['starting_time'] = starting_time
    
    time_mat, load_mat = generate_excitation(model, load, 0, duration, fs, starting_time = starting_time)

    # solve differential equation
    _, y, s = signal.lsim(model.ss, load_mat.T, time_mat)

    # TODO 
    # figure out meaning of elliptic filter 
    # create and apply elliptic filter
    b, a = signal.ellip(8, 0.1, 90, 2*np.pi*fc, analog=True,) # SISO low pass filter
    a1, b1, c1, d1 = signal.tf2ss(b, a)
    eliptic_filter = signal.StateSpace(a1, b1, c1, d1) 
    filetered_y = np.zeros_like(y)
    for i in range(0, y.shape[1]): 
        _, y_, s_ = signal.lsim(eliptic_filter, y[:, i], time_mat)
        filetered_y[:, i] = y_
    
    acc_noise_ref_data = filetered_y[:, acc_noise_ref-1+15]
    strain_noise_ref_data = filetered_y[:, strain_noise_ref-1]
    
    acc_noise = Ra*np.std(filetered_y[:, acc_noise_ref-1+15])*np.random.randn(acc_noise_ref_data.shape[0]) 
    strain_noise = Rs*np.std(filetered_y[:, strain_noise_ref-1])*np.random.randn(strain_noise_ref_data.shape[0])   
    # save the create data 
    data['acc'] = [filetered_y[:, i-1+15] + acc_noise for i in sns_loc]
    data['disp'] = [filetered_y[:, i-1] + strain_noise for i in sns_loc]

    data['acc_true'] = [y[:, i-1+15] for i in sns_loc]
    data['disp_true'] = [y[:, i-1] for i in sns_loc]
    
    # if  data['anomal_type'] in ['drift', 'weak_noise', 'strong_noise'] : 
    data = apply_sns_noise(data)

    os.makedirs(data_save_path, exist_ok=True)
    file_save = open(os.path.join(data_save_path, 'data_{}.pkl'.format(str(num+1).zfill(5))), 'wb')
    pickle.dump(data, file_save)
    file_save.close()
