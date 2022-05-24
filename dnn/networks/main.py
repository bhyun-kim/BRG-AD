from .cnn1d.base import base_CNN_1D_Autoencoder, base_CNN_1D_Encoder
from .cnn1d.residual import Residual_CNN_1D_Autoencoder, Residual_CNN_1D_Encoder
from .cnn1d.fullyConv1D import fullyConv_1D_Autoencoder, fullyConv_1D_Encoder
from .cnn2d.fullyConv2D import fullyConv_2D_Autoencoder, fullyConv_2D_Encoder
from .cnn2d.base import base_CNN_2D_Encoder, base_CNN_2D_Autoencoder

def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = (
        'base_CNN_1D_Autoencoder', 'base_CNN_2D_Autoencoder', 'Residual_CNN_1D_Autoencoder', 'fullyConv_1D_Autoencoder',
        'fullyConv_2D_Autoencoder'
    )
    assert net_name in implemented_networks

    en_net = None

    if net_name == 'base_CNN_2D_Autoencoder' : 
        en_net = base_CNN_2D_Encoder()
        
    if net_name == 'base_CNN_1D_Autoencoder' : 
        en_net = base_CNN_1D_Encoder()

    if net_name == 'Residual_CNN_1D_Autoencoder' : 
        en_net = Residual_CNN_1D_Encoder()

    if net_name == 'fullyConv_1D_Autoencoder' : 
        en_net = fullyConv_1D_Encoder()
    
    if net_name == 'fullyConv_2D_Autoencoder' : 
        en_net = fullyConv_2D_Encoder()
    
    if en_net:
        return en_net
    else :
        assert net_name, f"Network {net_name} is not in the implemented network list."



def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""
    
    ae_net = None

    if net_name == 'base_CNN_1D_Autoencoder':
        ae_net = base_CNN_1D_Autoencoder()

    if net_name == 'base_CNN_2D_Autoencoder':
        ae_net = base_CNN_2D_Autoencoder()

    if net_name == 'Residual_CNN_1D_Autoencoder':
        ae_net = Residual_CNN_1D_Autoencoder()

    if net_name == 'fullyConv_1D_Autoencoder' : 
        ae_net = fullyConv_1D_Autoencoder()
        
    if net_name == 'fullyConv_2D_Autoencoder' : 
        ae_net = fullyConv_2D_Autoencoder()

    if ae_net:
        return ae_net
    else :
        assert net_name, f"Network {net_name} is not in the implemented network list."
