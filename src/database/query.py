import sqlite3
import json
import os

import pandas as pd
import numpy as np

from utils.core_utilities import always_array

db = sqlite3.connect(os.path.expanduser("~/cftr2/projects.sqlite3"))
db.row_factory = sqlite3.Row
db = db.cursor()

# Core query string for trajectories
JOINT_TRAJ_TABLE = """trajectories
                join mdp using(mdp_id)
                join gro using(gro_id)
                join top using(top_id)
                join compositions using(comp_id) 
                join protein_component using(comp_id)
                join conditions using(cond_id) 
                join parameters using(param_id) 
                join protein_models using(protein_model_id) 
                join proteins using(protein_id)"""

_ambiguous_columns = {"comp_id": "gro.comp_id"}

def _resolve_column_ambiguity(column_name):
    return _ambiguous_columns.get(column_name, column_name)

### Query string processing functions ###
def quote(string):
    # Modify to fit sqlite3 syntax
    return f"'{string}'"

# "SELECT * FROM $table"
def select_from(table):
    selstring = f"SELECT * FROM {table}"
    return selstring

# "JOIN $table USING($on)"
def join(table, **join_tables_on):
    table = quote(table)
    for join_table, on in join_tables_on.items():
        table = f"{table} JOIN {quote(join_table)} USING({on})"
    return table

# "WHERE $category=$value; or $category IS NULL"
def where(category, value):
    # Handle None/Null
    if value is None:
        wherestring = f"{category} IS NULL"
        return wherestring
    
    # Handle text values; could be 'NULL'
    elif isinstance(value, str):
        if value.upper() == "NULL":
            wherestring = f"{category} IS NULL"
        else:
            # Text values should be decorated with extra ''
            value = quote(value)
            wherestring = f"{category}={value}"
        return wherestring

    # Handle ranges of values
    elif isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Range should be specified as an iterable of two values.")
        if value[0] is not None:
            wherestring = f"{category}>{value[0]}"
            if value[1] is not None:
                wherestring = f"{wherestring} AND {category}<{value[1]}"
        elif value[1] is not None:
            wherestring = f"{category}<{value[1]}"
        else:
            raise ValueError("(None, None) is not a valid range.")
        return wherestring
    
    # Handle ranges of values; end inclusive
    elif isinstance(value, list):
        if len(value) != 2:
            raise ValueError("Range should be specified as an iterable of two values.")
        if value[0] is not None:
            wherestring = f"{category}>={value[0]}"
            if value[1] is not None:
                wherestring = f"{wherestring} AND {category}<={value[1]}"
        elif value[1] is not None:
            wherestring = f"{category}<={value[1]}"
        else:
            raise ValueError("[None, None] is not a valid range.")
        return wherestring
    
    # Handle single values
    else:
        wherestring = f"{category}={value}"
        return wherestring

# "SELECT * FROM $table WHERE $where_string"
def select_where(table, **specify):
    selection_string = select_from(table)
    params_where = []
    for parameter, value in specify.items():
        # already dealt with table
        if parameter == "table":
            continue
        params_where.append(where(parameter, value))
    params_where.append("true")
    # join all where clauses
    where_clause = ' AND '.join(params_where)

    selection_query = f"{selection_string} WHERE {where_clause}"
    return selection_query

# TODO: make an integrity test for the database; e.g. all grotop_id in tpr should have grotop value of 1 in gro
class project_sql:
    def __init__(self, sql_name=os.path.expanduser("~/cftr2/projects.sqlite3")):
        self.sql_name=sql_name
        self.conn = sqlite3.connect(self.sql_name)
        self.conn.row_factory = sqlite3.Row
        self.db = self.conn.cursor()

    def fetch_all(self, table, items=None, **where_filter):
        fetched_objects = self.db.execute(select_where(table=table, **where_filter)).fetchall()
        if items is None:
            fetched_items = fetched_objects
        # TODO: this is not great, any iterable of items should work
        elif isinstance(items, tuple):
            fetched_items = [tuple([row[item] for item in items]) for row in fetched_objects]
        else:
            item = items
            fetched_items = [row[item] for row in fetched_objects]
        return fetched_items
    
    def fetch_one(self, table, items=None, **where_filter):
        fetched_objects = self.fetch_all(table=table, items=items, **where_filter)
        if len(fetched_objects) == 0:
            return
        return fetched_objects[0]
    
    def insert(self, table, **insert_info):
        # # Text values should be decorated with extra ''
        for col, value in insert_info.items():
            # Handle None/Null
            if str(value).upper == 'NULL' or value is None:
                insert_info[col] = 'NULL'
                continue
            if isinstance(value, str):
                insert_info[col] = quote(value)
        insert_str = "INSERT INTO {} ({}) VALUES ({})".format(table, ",".join([str(k) for k in insert_info.keys()]), ",".join([str(v) for v in insert_info.values()]))
        self.db.execute(insert_str)

    def update(self, table, update_dict: dict, **where_filter):
        # # Text values should be decorated with extra ''
        for col, value in update_dict.items():
            if isinstance(value, str):
                update_dict[col] = quote(value)
        update_str = "UPDATE {} SET {} WHERE {}".format(table, 
                                                        ", ".join(["{}={}".format(col, value) for col, value in update_dict.items()]), 
                                                        " AND ".join([where(col, value) for col, value in where_filter.items()]))
        self.db.execute(update_str)


### Query functions ###
sql = project_sql()

# Query traj_id from the trajectories table
def get_trajid(all=False, verbose=False, **sim_params):
    if not all:
        sim_params["status"] = None

    sim_params = {_resolve_column_ambiguity(param): value for param, value in sim_params.items()}

    if verbose:
        selection_query = select_where(table=JOINT_TRAJ_TABLE, **sim_params)
        print(selection_query)

    traj_ids = sql.fetch_all(table=JOINT_TRAJ_TABLE, items="traj_id", **sim_params)
    return traj_ids

# Get attributes from trajectories
def get_trajattr(traj_ids, *attr_names, condensed=True):
    traj_ids = always_array(traj_ids)

    # Create the query
    select_traj_ids = "traj_id IN ({})".format(",".join([str(t) for t in traj_ids]))
    traj_query = f"SELECT * FROM {JOINT_TRAJ_TABLE} WHERE {select_traj_ids}"

    # Execute the query; fetchall entries
    attributes = sql.db.execute(traj_query).fetchall()
    attribute_df = pd.DataFrame([dict(row) for row in attributes])

    if len(attr_names) != 0:
        attribute_df = attribute_df[list(attr_names)]
        # Give singular values if condensed, and only one traj_id, and only one attribute queried
        if condensed and len(attr_names) == 1 and len(traj_ids) == 1:
            attribute_df = attribute_df.values[0][0]
    elif condensed:
        # Condensed means no other id columns, except traj_id
        no_other_ids = [attr for attr in attribute_df.columns if not attr.endswith("_id") or attr == "traj_id"]
        attribute_df = attribute_df[no_other_ids]
    return attribute_df

def comp2protmodel(comp_id):
    protein_model_id = sql.fetch_one("protein_component", "protein_model_id", comp_id=comp_id)
    return protein_model_id

def gro2comp(gro_id):
    comp_id = sql.fetch_one("gro", "comp_id", gro_id=gro_id)
    return comp_id

def get_comp_with_mol(*molecules):
    for mol in molecules:
        return sql.fetch_all("mol_component JOIN molecule_models USING(mol_model_id)", items="comp_id", model_name=mol)

# Recursion function used to get the ultimate origin of the gro
# TODO: re-implementation idea: grotop 1-to-1 with comp_id
def gro2grotop(gro_id):
    # Query the traj_id that the gro_id extends from
    traj_id_from = sql.fetch_one("gro", "extend_from", gro_id=gro_id)
    # If there is no traj_id, then the gro_id is the a grotop
    # Meaning it is not extended from other trajectories
    if traj_id_from is None:
        return gro_id
    
    # Query the gro_id of the source trajectory
    source_gro = traj2gro(traj_id_from)
    # 230112 added an extra block handling extended but mutated composition, or slightly changed composition (e.g. number of water)
    comp_id = gro2comp(gro_id)
    # "Extended" but not the same (identical) anymore
    if gro2comp(source_gro) != comp_id:
        # Try to find grotop with the same composition
        same_composition_gro_ids = sql.fetch_all(table="gro", items="gro_id", grotop=1, comp_id=comp_id)
        # I think this is a new composition. There should be one with grotop=1
        # If not, then we have a problem and raise an error
        if len(same_composition_gro_ids) == 0:
            raise ValueError(f"Composition for grotop gro_id={gro_id} is not found in the database.")
        # Take the first one
        grotop_id = sorted(same_composition_gro_ids)[0]
        return grotop_id
    return gro2grotop(source_gro)

def protmodel2prot(protein_model_id):
    protein_id = sql.fetch_one("protein_models", "protein_id", protein_model_id=protein_model_id)
    return protein_id

def traj2comp(traj_id):
    return gro2comp(traj2gro(traj_id))

def traj2gro(traj_id):
    gro_id = sql.fetch_one("trajectories", "gro_id", traj_id=traj_id)
    return gro_id

def traj2grotop(traj_id):
    gro_id = traj2gro(traj_id)
    grotop_id = gro2grotop(gro_id)
    return grotop_id

def traj2mdp(traj_id):
    mdp_id = sql.fetch_one("trajectories", "mdp_id", traj_id=traj_id)
    return mdp_id

def traj2prot(traj_id):
    return protmodel2prot(traj2protmodel(traj_id))

def traj2protmodel(traj_id):
    prot_model_id = comp2protmodel(gro2comp(traj2gro(traj_id)))
    return prot_model_id

def traj2top(traj_id):
    top_id = sql.fetch_one("trajectories", "top_id", traj_id=traj_id)
    return top_id

def traj2tprtop(traj_id):
    grotop_id = traj2grotop(traj_id)
    top_id = traj2top(traj_id)
    tpr_id = sql.fetch_one("tpr", "tpr_id", grotop_id=grotop_id, top_id=top_id)
    return tpr_id


def traj_group(sim_id):
    description, notes = sql.fetch_one("simulations", items=("description", "notes"), sim_id=sim_id)
    print(f"Description: {description}")
    print(f"Notes: {notes}")

    traj_ids = sql.fetch_all("trajectory_groups", items="traj_id", sim_id=sim_id)
    # New: ignore status not None for the trajectory due to issues with the trajectory
    traj_ids = np.intersect1d(get_trajid(all=False), traj_ids)
    return traj_ids

def get_mfa_id(protein_id1, protein_id2):
    protali_id = sql.fetch_one(table='protein_sequence_alignment', 
                               items='mfa_id', 
                               protein_id1=protein_id1, protein_id2=protein_id2)

    # If not found, try the other way around
    if protali_id is None:
        protali_id = sql.fetch_one(table='prot_alignments', 
                                   items='mfa_id', 
                                   protein_id1=protein_id2, protein_id2=protein_id1)

        if protali_id is None:
            raise ValueError(f"Protein alignment not found for protein_id1={protein_id1} and protein_id2={protein_id2}")
        else:
            protali_id = -1 * protali_id
            
    return protali_id

def get_translocation(exclusion=True, **translocate_info):
    qstr = "select * from translocations"
    if len(translocate_info) == 0:
    # New 230717: exclude curate code > 1
        if exclusion:
            where_str = " WHERE curated<=1"
        else:
            where_str = ""
    else:
        where_str = " WHERE curated<=1 AND "
        params_where = []
        for param, value in translocate_info.items():
            params_where.append("{}={}".format(param, value))
        where_str += " AND ".join(params_where)
    qstr = qstr + where_str

    translocate_entries = db.execute(qstr).fetchall()
    return translocate_entries

def transloc_record(traj_id):
    analysis_history = db.execute("SELECT * FROM transloc_history WHERE traj_id={}".format(traj_id)).fetchall()
    if len(analysis_history) == 0:
        return
    if len(analysis_history) > 1:
        print("Warning: More than one records on transloc_history was found for traj_id=%d; check integrity of the database." % traj_id)
    return analysis_history[0] # Regardless, get the first one

def transloc_total_time(traj_id):
    recorded = transloc_record(traj_id)
    if recorded is None:
        return
    ntsteps = recorded['nframes']
    stepsize = recorded['stepsize']
    total_time = (ntsteps - 1) * stepsize
    return total_time

def transloc_analyzed(traj_id):
    # On record and nframes is greater than one (don't divide by zero)
    return (transloc_record(traj_id) is not None) and (transloc_total_time(traj_id) is not None) and transloc_total_time(traj_id) > 0

def transloc_path(gro_id):
    transloc_path_id = db.execute("SELECT transloc_path FROM gro WHERE gro_id=?", (gro_id,)).fetchone()
    return transloc_path_id['transloc_path']

# Get the protein domain/range definitions from protein, specified by protein_id
def get_protdef(protein_id, def_id=1):
    # Extract the json file
    prot_defs = open(os.path.expanduser("~/cftr2/data/protein_defs.json"), "r")
    prot_defs_info = json.loads(prot_defs.read())
    prot_defs.close()

    # Get the one that matches 'protein_id' and 'def_id'
    for protdef in prot_defs_info:
        if protdef["protein_id"] == protein_id:
            if protdef["def_id"] == def_id:
                # Remove _ids
                protdef = {name: interval for name, interval in protdef.items() if name[-3:] != "_id"}
                return protdef

    # Definition not found
    print("Information requested not found")
    return