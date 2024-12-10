def beads(xyz, save_path):
    vmd = open(save_path, "w")
    for point in xyz:
        x, y, z = point
        vmd.write("graphics top sphere {"+"{} {} {}".format(x, y, z)
                    +"}\n")
    vmd.close()

# Used to export snapshots specified by traj_id and timestep (tf)
# Writes a gmx-based script to be run on Niagara to generate snapshots
# TODO: dt should be parsed; I should add those to the database
def export_snapshots(tf, script='extract_snapshots.sh', concat_name="state", 
                     output_grp="Protein", dt=1000):
    """
    tf: trajectory frame
    xyz: coordinates
    save_path: path to save snapshots
    nframes: number of frames to save
    """
    with open(script, "w") as f:
        i = 0
        for traj_id, timestep in tf:
            i += 1
            f.write(f"echo {output_grp} | gmx -quiet trjconv -f ~/project/research/trajectories/{traj_id}_md_fixed_dt{dt}.xtc ")
            f.write(f"-s ~/cftr2/mdsim/step6.0_minimization_11.tpr ")
            f.write(f"-o ~/cftr2/tmp/snapshot{i}.gro")
            f.write(f" -b {timestep}000 -e {timestep}000\n")
            
            # Erase timestep info
            f.write(f"sed -i -e 's/t=//g' ~/cftr2/tmp/snapshot{i}.gro\n")
            f.write(f"sed -i -e 's/step=//g' ~/cftr2/tmp/snapshot{i}.gro\n")
            f.write(f"echo {output_grp} | gmx -quiet trjconv -f ~/cftr2/tmp/snapshot{i}.gro -s ~/cftr2/mdsim/step6.0_minimization_11.tpr -o ~/cftr2/tmp/snapshot{i}.xtc\n")
            f.write("\n")
        
        f.write(f"gmx trjcat -f ~/cftr2/tmp/snapshot{{1..{i}}}.xtc -o {concat_name}.xtc -cat\n")
        