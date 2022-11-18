from models.DC_CDN import DC_CDN
from models.CDCNs import CDCN, CDCNpp

def build_model(cfg):
    network = None

    if cfg['model']['base'] == 'CDCNpp':
        network = CDCNpp()
    elif cfg['model']['base'] == 'CDCN':
        network = CDCN()
    elif cfg['model']['base'] == 'DC_CDN':
        network = DC_CDN()
    else:
        raise NotImplementedError
    
    return network
