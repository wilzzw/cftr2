import numpy as np
import pandas as pd

from utils.dataset import read_trajdata, write_trajdata

class nbdcom:
    def __init__(self, traj_ids):
        # TODO: not very elegant to sort here
        self.traj_ids = np.sort(traj_ids)

    def load_data(self, data_loc, df=True, **read_kwargs):
        # Currently does not support stride
        if read_kwargs.get("stride") is not None:
            raise ValueError("nbdcom does not support stride")
        
        com_data = read_trajdata(data_loc, traj_ids=self.traj_ids, **read_kwargs)
        output_dict = {"data": com_data[0], "nframes": com_data[1], "traj_ids": com_data[2]}

        # Append with extra requested attributes
        if len(read_kwargs.get('attrs', [])) > 0:
            output_dict.update({attr: com_data[3+i] for i, attr in enumerate(read_kwargs['attrs'])})

        if df:
            # Make a dataframe
            output_dict['data'] = pd.DataFrame()
            output_dict['data']['traj_id'] = np.repeat(output_dict['traj_ids'], np.array(output_dict['nframes']))
            output_dict['data']['timestep'] = np.concatenate([np.arange(nframes) for nframes in output_dict['nframes']])
            output_dict['data'][['x', 'y', 'z']] = com_data[0]

        return output_dict
    
    def load_allcom(self):
        self.nbd1com = {}
        self.nbd1com['overall'] = self.load_data("nbds/nbd1/com")['data']
        self.nbd1com['alpha'] = self.load_data("nbds/nbd1/com_alpha")['data']
        self.nbd1com['beta'] = self.load_data("nbds/nbd1/com_beta")['data']

        self.nbd2com = {}
        self.nbd2com['overall'] = self.load_data("nbds/nbd2/com")['data']
        self.nbd2com['alpha'] = self.load_data("nbds/nbd2/com_alpha")['data']
        self.nbd2com['beta'] = self.load_data("nbds/nbd2/com_beta")['data']
        
