import pytorch_lightning as pt

class ARModule(pt.LightningModule):
    """ AutoRegression Module
        aiming to generate input 
    """

    def __init__(self,config):
        super(ARModule, self).__init__()
        
