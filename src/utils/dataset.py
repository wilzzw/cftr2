import os

import numpy as np
import h5py

def read_dataset(hdf5: str, datadir: str, *attrs):
    with h5py.File(os.path.expanduser(hdf5), "r") as f:
        dataset = f[datadir]
        data = dataset[:]
        if len(attrs) > 0:
            attrs_data = {attr: dataset.attrs[attr] for attr in attrs}
            return data, attrs_data
        return data

def write_data(hdf5: str, datadir: str, data, expand=False, **attrs):
    with h5py.File(os.path.expanduser(hdf5), "r+") as f:
        # f.create_dataset(datadir, data=data)
        # TODO: temporary fix for resizing; it is not the most secure way to do it
        if expand:
            f[datadir].resize(data.shape)
        f[datadir][:] = data
        for key, val in attrs.items():
            f[datadir].attrs[key] = val
    return

def read_trajdata(datadir: str, hdf5='~/cftr2/results/data/cftr.hdf5',
                  traj_ids=None, stride=1, subselect=None, attrs=[]):
    
    with h5py.File(os.path.expanduser(hdf5), "r") as f:

        dataset = f[datadir]
        
        try:
            nframes_avail = dataset.attrs['nframes']
        except KeyError:
            print(f"nframes not an attribute of {datadir}")
            nframes_avail = None

        # TODO: do one way or the other, not both
        traj_ids_avail = dataset.attrs['traj_ids']
        # traj_ids_avail = f['traj_ids'][:]

        # Read all traj_ids data if traj_ids is None
        if traj_ids is None:
            traj_ids = traj_ids_avail
        else:
            # Available trajectories
            # Does not mean there is data for all trajectories! (still depends on nframes_avail)
            invalid_traj_ids = np.setdiff1d(traj_ids, traj_ids_avail)

            if len(invalid_traj_ids) > 0:
                print(f"traj_ids={invalid_traj_ids} not in dataset, ending")
                return

        traj_id_where, = np.where(np.in1d(traj_ids_avail, traj_ids))
        nframes_onrecord = nframes_avail[traj_id_where]

        # Read data
        data = [dataset[traj_where, np.arange(0,traj_nframes,stride)] for traj_where, traj_nframes in zip(traj_id_where, nframes_onrecord)]
        
        if subselect is None:
            data = np.concatenate(data)
        else:
            # h5py does not support fancy indexing yet
            data = np.concatenate([dat[:, subselect] for dat in data])

        nframes_loaded = [len(np.arange(0,traj_nframes,stride)) for traj_nframes in nframes_onrecord]

        retrieved = (data, nframes_loaded, traj_ids)
        # Append with extra requested attributes
        if len(attrs) > 0:
            retrieved = retrieved + tuple([dataset.attrs[attr] for attr in attrs])
            
        return retrieved

def write_trajdata(traj_id, datadir: str, data, 
                   hdf5='~/cftr2/results/data/cftr.hdf5', overwrite=False):
    with h5py.File(os.path.expanduser(hdf5), "r+") as f:
        dat = f[datadir]
        traj_ids_avail = dat.attrs['traj_ids']

        if traj_id not in traj_ids_avail:
            print(f"traj_id={traj_id} not in dataset, ending")
            return

        nframes_avail = dat.attrs['nframes']
        traj_id_where, = np.where(traj_ids_avail == traj_id)
        nframes_onrecord = nframes_avail[traj_id_where[0]]

        if nframes_onrecord > 0:
            print(f"traj_id={traj_id} already has data of {nframes_onrecord} frames on record")

            if nframes_onrecord > len(data):
                if overwrite:
                    print(f"overwriting {nframes_onrecord} frames of data with {len(data)} frames of data")
                else:
                    print(f"traj_id={traj_id} has more data on record than the data to be written, ending")
                    return

        dat[traj_id_where[0], :len(data)] = data
        nframes_avail[traj_id_where[0]] = len(data)

        dat.attrs['nframes'] = nframes_avail
    return

def read_attr(datadir: str, *attrs, hdf5='~/cftr2/results/data/cftr.hdf5'):
    with h5py.File(os.path.expanduser(hdf5), "r") as f:
        dat = f[datadir]

        if len(attrs) > 1:
            return {attr: dat.attrs[attr] for attr in attrs}
        return dat.attrs[attrs[0]]

def write_attr(datadir: str, hdf5='~/cftr2/results/data/cftr.hdf5', **attrs):
    with h5py.File(os.path.expanduser(hdf5), "r+") as f:
        dat = f[datadir]
        for key, val in attrs.items():
            dat.attrs[key] = val
    return

def export_data(groupdir: str, hdf5_filedir: str, output_grp=None):
    if output_grp is None:
        output_grp = f"{groupdir}.h5"
    with h5py.File(hdf5_filedir, "r") as f, h5py.File(output_grp, 'w') as f_out:
        data_group = f[groupdir]
        f_out.copy(data_group, groupdir)
        return

def import_data(groupdir: str, input_filedir: str, hdf5_filedir: str, renamed_group=None):
    # TODO: renamed_group does not work as intended
    with h5py.File(hdf5_filedir, "r+") as f, h5py.File(input_filedir, 'r') as f_input:
        group = f_input[groupdir]
        if groupdir in f.keys():
            if renamed_group is None:
                print(f"{groupdir} already in {hdf5_filedir}, ending")
            else:
                f.copy(group, renamed_group)
            return
        f.copy(group, groupdir)
        return

def create_dataset(name: str, shape: tuple, dtype: str, data=None,
                   hdf5='~/cftr2/results/data/cftr.hdf5', resizable=False,
                   **attrs):
    with h5py.File(os.path.expanduser(hdf5), "a") as f:
        create_kwargs = {'shape': shape, 'dtype': dtype}
        if data is not None:
            create_kwargs['data'] = data
            create_kwargs['shape'] = data.shape
        if resizable:
            create_kwargs['maxshape'] = (None,)*len(shape)
        f.create_dataset(name, **create_kwargs)
        
        for key, val in attrs.items():
            f[name].attrs[key] = val
    return

def rename_dataset(hdf5: str, name: str, rename: str):
    with h5py.File(hdf5, "r+") as f:
        f[rename] = f[name]
        del f[name]
    return

def expand_dataset(datadir: str, traj_ids, hdf5='~/cftr2/results/data/cftr.hdf5'):
    # Not the same as resizing; sizes are by default fixed in my way of doing things
    # Only expand in traj_ids dimension
    
    traj_ids = np.array(traj_ids)
    hdf5 = os.path.expanduser(hdf5)

    # Step 0: read the target dataset
    with h5py.File(hdf5, "r") as f:
        dataset = f[datadir]
        dtype = dataset.dtype

        # # TODO: temporary solution to get traj_ids if attribute is not available
        # try:
        #     traj_ids_avail = dataset.attrs['traj_ids']
        # except KeyError:
        #     traj_ids_avail = f['helix_axis/r1131-1139'].attrs['traj_ids']
        #     assert len(traj_ids_avail) == dataset.shape[0]
        traj_ids_avail = dataset.attrs['traj_ids']
        
        nframes_avail = dataset.attrs['nframes']
        data = dataset[:]

    # Step 1 see what new traj_ids are not in the dataset
    traj_ids_toadd = np.setdiff1d(traj_ids, traj_ids_avail)
    if len(traj_ids_toadd) == 0:
        print("no new traj_ids to add, ending")
        return
    new_traj_ids = np.concatenate((traj_ids_avail, traj_ids_toadd))
    new_nframes = np.concatenate((nframes_avail, np.zeros(len(traj_ids_toadd), dtype=int)))

    # Step 2: create new dataset with new traj_ids with new size
    new_shape = (len(traj_ids_avail)+len(traj_ids_toadd),) + data.shape[1:]
    create_dataset(f"{datadir}.new_dataset", shape=new_shape, dtype=dtype, hdf5=hdf5, 
                   traj_ids=new_traj_ids, nframes=new_nframes)
    
    # Step 3: copy data from old dataset to new dataset
    for i, t in enumerate(traj_ids_avail):
        # 240201 critical bug fix
        traj_nframes = nframes_avail[i]
        write_trajdata(t, f"{datadir}.new_dataset", data[i,:traj_nframes], hdf5=hdf5, overwrite=True)

    # Step 4: delete old dataset
    delete_grp(datadir, hdf5=hdf5)

    # Step 5: rename new dataset to old dataset
    rename_dataset(hdf5, f"{datadir}.new_dataset", datadir)

    return

def create_grp(grp: str, 
               hdf5=os.path.expanduser('~/cftr2/results/data/cftr.hdf5'), **attrs):
    with h5py.File(hdf5, "a") as f:
        f.create_group(grp)
        for key, val in attrs.items():
            f[grp].attrs[key] = val
    return

def delete_grp(grp: str, 
               hdf5=os.path.expanduser('~/cftr2/results/data/cftr.hdf5')):
    with h5py.File(hdf5, "r+") as f:
        del f[grp]
    return