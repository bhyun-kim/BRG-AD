import os
import click
import logging
import pickle
import json 

from tqdm import tqdm
from copy import deepcopy
from importlib import import_module

from lib.fem import create_fem_model, run_fem_simulation
from lib.excitation import create_load_matrix
from lib.anomaly import apply_sns_noise_v2
from lib.utils import cvt_moduleToDict, cvt_pathToModule


#########################
##### Configuration #####
#########################

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('cfg_path', type=click.Path(exists=True))


def main(data_path, cfg_path):

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

    # import configuration
    _cfg_mod = cvt_pathToModule(cfg_path)
    _mod = import_module(_cfg_mod)
    _var = cvt_moduleToDict(_mod)
    cfg_source = _var['cfg']
    cfg_pretty = json.dumps(cfg_source, indent=4, sort_keys=True)

    logger.info(f"\n{cfg_pretty}")

    num_data = cfg_source['num_data']

    logger.info(f"Number of the Total Simulation: {num_data}.")

    num_total_ano = 0
    ano_idx_list =[]

    cfg_anomalies = cfg_source.pop('anomaly')
    
    for a_idx, cfg_ano in enumerate(cfg_anomalies):
        
        ano_type = cfg_ano['type']
        ano_subtype = cfg_ano['subtype']
        num_ano = cfg_ano['num_of_anomalies']

        num_total_ano += num_ano
        logger.info(f"Number of Anomaly {ano_type}, {ano_subtype}: {num_ano}.")
        ano_idx_list += [a_idx]*num_ano

    ano_idx_list += [-1]*(num_data-num_total_ano)
    

    for s_idx, anomal_num in tqdm(enumerate(ano_idx_list), desc="Simulation Progress"):

        cfg = deepcopy(cfg_source)

        if anomal_num > -1:
            is_ano = True
            cfg['anomaly'] = [cfg_anomalies[anomal_num]]
        else : 
            is_ano = False
            cfg['anomaly'] = [dict(type=None)]

        model = create_fem_model(cfg['beam'], cfg['section'], cfg['anomaly'])
        time_mat, load_mat_sum, norm_load_mat_sum, overload_mat_sum = create_load_matrix(model, cfg,)
        response = run_fem_simulation(model, cfg, time_mat, load_mat_sum)
        
        if cfg['anomaly'][0]['type'] == "sns_noise" : 
            response = apply_sns_noise_v2(response, cfg)

        data = dict()
        data['is_anomal'] = is_ano
        data['sns_loc'] = cfg['sensor']['sns_loc']
        data['acc'] = response['acc']
        data['disp'] = response['disp']
        data['sample_rate'] = cfg['sensor']['freq_sample']

        if cfg['anomaly'][0]['type'] == "sns_noise" :
            data['anomal_type'] = cfg['anomaly'][0]['subtype']
            data['anomal_loc'] = cfg['anomaly'][0]['anomaly_sns_loc'] * 2

        elif cfg['anomaly'][0]['type'] == "section_loss" : 
            data['anomal_type'] = cfg['anomaly'][0]['subtype']

        else : 
            data['anomal_type'] = None

        os.makedirs(data_path, exist_ok=True)

        data_save_path = os.path.join(data_path, 'data_{}.pkl'.format(str(s_idx+1).zfill(4)))
        with open(data_save_path, 'wb') as f :
            pickle.dump(data, f)

if __name__ == '__main__':
    main()