import numpy as np

from sequence.genseqmap import protein_residue_alignment
from database.query import project_sql, protmodel2prot

sql = project_sql()

def interval_selection(*intuple):
    resids = np.array([r for r in range(intuple[0], intuple[1]+1)])
    return resids

# Combine multiple interval selections; return an array of residues
def combine_from_intervals(*intervals):
    combined_resids = np.concatenate([interval_selection(*interval) for interval in intervals])
    return combined_resids

# Get chains of a protein model
# Returns a list of tuples of begin and end resids for each chain
def get_protein_chain_range(protein_model_id):
    resids_avail = sql.fetch_all(table='protein_chains', 
                                 items=('begin', 'end'), 
                                 protein_model_id=protein_model_id)
    return resids_avail

# Residues actually available in a protein model
# Remove any deletions
# Returns a numpy array of residues
# TODO: the next few functions were not yet tested against protein_models that are of different proteins (e.g. zebrafish CFTR)
def model_residavail(protein_model_id):
    protein_chains = get_protein_chain_range(protein_model_id)
    if len(protein_chains) > 0:
        resid_list = combine_from_intervals(*protein_chains)
        return resid_list
    # Try to find whether the model is related to another protein model but with mutations
    mutation_records = sql.fetch_all(table='mutations', 
                                     items=('parent_model_id', 'position', 'resid_to'),
                                     protein_model_id=protein_model_id)
    if len(mutation_records) > 0:
        # There should be only one parent model
        parent_model_ids = set([model_id for model_id, _, _ in mutation_records])
        assert len(parent_model_ids) == 1

        resid_list = model_residavail(list(parent_model_ids)[0])
        for _, position, resid_to in mutation_records:
            if resid_to is None:
                # This is a deletion mutation
                resid_list = np.delete(resid_list, np.where(resid_list == position))
        return resid_list
    else:
        raise ValueError(f"protein_model_id {protein_model_id} does not have any residue information on record")

# Get the residue mapping between two protein models
def protmodel_residue_alignment(protein_model_id1, protein_model_id2):
    # Get the residue mapping between the two proteins on which the protein models are based
    protein_id1 = protmodel2prot(protein_model_id1)
    protein_id2 = protmodel2prot(protein_model_id2)
    protein_residmap = protein_residue_alignment(protein_id1, protein_id2)

    # Residues that are actually available in the protein models
    residavail1 = model_residavail(protein_model_id1)
    residavail2 = model_residavail(protein_model_id2)

    # Filter out residues that are not available in both protein models
    residmap = {r1: r2 for r1, r2 in protein_residmap.items() if (r1 in residavail1) and (r2 in residavail2)}
    return residmap

def mapped_interval(model_id1, model_id2, interval_defmodel, start_res, end_res):
    residue_mapping = protmodel_residue_alignment(model_id1, model_id2)
    if interval_defmodel == model_id1:
        mapped_respairs = {r1: r2 for r1, r2 in residue_mapping.items() if r1 >= start_res and r1 <= end_res}
    elif interval_defmodel == model_id2:
        mapped_respairs = {r1: r2 for r1, r2 in residue_mapping.items() if r2 >= start_res and r2 <= end_res}
    else:
        raise ValueError("interval_defmodel should be one of the protein models")
    return mapped_respairs

def advanced_combine(model_id1, model_id2, interval_defmodel, *intervals):
    mapped_intervals = [mapped_interval(model_id1, model_id2, interval_defmodel, *interval) for interval in intervals]
    # Combine the dictionaries in mapped_intervals
    combined_residmap = {}
    for resmap in mapped_intervals:
        combined_residmap.update(resmap)
        
    resids1 = list(combined_residmap.keys())
    resids2 = list(combined_residmap.values())
    return resids1, resids2

# TODO: This is a bit of a mess; mda or mdtraj have different conventions for selecting residues; split this into two functions
def select_resids_str(resids, package="mda"):
    if package == "mda":
        return "(resid " + " ".join([str(r) for r in resids]) + ")"
    return "(resSeq " + " ".join([str(r) for r in resids]) + ")"

def select_CA_str(selstr: str):
    return f'protein and name CA and ({selstr})' 

def select_resCA(resids, package="mda"):
    return select_CA_str(select_resids_str(resids, package))

def select_domains(domain_intervals, package="mda"):
    return select_resids_str(combine_from_intervals(*domain_intervals), package)

def select_domainsCA(domain_intervals, package="mda"):
    return select_CA_str(select_domains(domain_intervals, package))