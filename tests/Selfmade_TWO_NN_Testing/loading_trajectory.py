#Strucutre of the code: 
#Read in coords and trajectory and store it as pandas frame for verification 
#make possibility to switch between representationss: cart vs. internal distances
#Compute Nearest neighbour RMSD in cartesian; Save time index of nearest neighbour; 
# Reproduce time-lag plots in simpler ways 

# Project structre: Main-file for coordination 
# file for loading in 
# File for data-computation
# file for plotting 



import MDAnalysis as mda
import pandas as pd
import os
import time 
from MDAnalysis.analysis import distances
import numpy as np
from MDAnalysis.analysis import align

def load_trajectory(gro_path, xtc_path, coord_type="internal", selection="not name H*", num_frames=None, step = 100):
    """
    Loads an MD trajectory, prints a verification DataFrame of the selected atoms 
    for the first frame, and extracts a feature matrix based on coord_type.
    """
    print(f"Loading topology: {gro_path}")
    print(f"Loading trajectory: {xtc_path}")

    u = mda.Universe(gro_path, xtc_path)
    atoms = u.select_atoms(selection) # discard H-atoms
    n_atoms = len(atoms)
    
    
    # --- Pandas Verification (Frame 0) ---
    data = {
        'AtomID': atoms.indices + 1,
        'AtomName': atoms.names,
        'ResName': atoms.resnames,
        'ResID': atoms.resids,
        'X (Å)': atoms.positions[:, 0],
        'Y (Å)': atoms.positions[:, 1],
        'Z (Å)': atoms.positions[:, 2]
    }
    df = pd.DataFrame(data)
    print(f"\n--- Verification: First 10 Selected Atoms (Out of {n_atoms}) ---")
    print(df.head(10))
    print("-------------------------------------------------------\n")
    
    # --- Feature Extraction ---
    total_frames = len(u.trajectory)
    frames_to_process = min(num_frames, total_frames) if num_frames else total_frames

    #traj_slice = u.trajectory[0:frames_to_process:step]
    #n_frames_out = len(traj_slice)
    
    print(f"Processing {frames_to_process} frames for {coord_type} coordinates...")

    
    
    if coord_type == "internal":
        n_features = n_atoms * (n_atoms - 1) // 2
        feature_matrix = np.zeros((frames_to_process, n_features), dtype=np.float32)
        
        for i, ts in enumerate(u.trajectory[:frames_to_process]):
            feature_matrix[i, :] = distances.self_distance_array(atoms.positions)


            
    elif coord_type == "cartesian":
        #create reference frame
        ref = mda.Universe(gro_path, xtc_path)
        ref_atoms = ref.select_atoms(selection)


        n_features = n_atoms * 3
        feature_matrix = np.zeros((frames_to_process, n_features), dtype=np.float32)

        print(feature_matrix)
        print(feature_matrix.shape)

        for i, ts in enumerate(u.trajectory[:frames_to_process]):
            align.alignto(atoms, ref_atoms, select=selection)
            feature_matrix[i, :] = atoms.positions.flatten()

        print(feature_matrix)
    else:
        raise ValueError("coord_type must be either 'internal' or 'cartesian'")
        

    
    return feature_matrix, atoms

# Example usage assuming the folder is 'data':
#gro_file = os.path.join('Data/input_files', 'coord.gro')
#xtc_file = os.path.join('Data/trajectories', 'replica7.xtc') # Replace 'traj.xtc' with your actual filename
#df, universe = load_and_verify_trajectory(gro_file, xtc_file)



def load_traj_step(gro_path, xtc_path, coord_type="internal", selection="not name H*", num_frames=None, step=100):
    """
    Loads an MD trajectory, prints a verification DataFrame of the selected atoms 
    for the first frame, and extracts a feature matrix based on coord_type.
    """
    print(f"Loading topology: {gro_path}")
    print(f"Loading trajectory: {xtc_path}")

    u = mda.Universe(gro_path, xtc_path)
    atoms = u.select_atoms(selection)  # discard H-atoms
    n_atoms = len(atoms)
    
    # --- Pandas Verification (Frame 0) ---
    data = {
        'AtomID': atoms.indices + 1,
        'AtomName': atoms.names,
        'ResName': atoms.resnames,
        'ResID': atoms.resids,
        'X (Å)': atoms.positions[:, 0],
        'Y (Å)': atoms.positions[:, 1],
        'Z (Å)': atoms.positions[:, 2]
    }
    df = pd.DataFrame(data)
    print(f"\n--- Verification: First 10 Selected Atoms (Out of {n_atoms}) ---")
    print(df.head(10))
    print("-------------------------------------------------------\n")
    
    # --- Feature Extraction ---
    total_frames = len(u.trajectory)
    stop_frame = min(num_frames, total_frames) if num_frames is not None else total_frames

    # MDAnalysis trajectory slice: [start:stop:step]
    traj_slice = u.trajectory[0:stop_frame:step]
    n_frames_out = len(traj_slice)
    
    print(f"Processing {n_frames_out} frames (sampled every {step} steps up to frame {stop_frame}) for {coord_type} coordinates...")

    if coord_type == "internal":
        n_features = n_atoms * (n_atoms - 1) // 2
        feature_matrix = np.zeros((n_frames_out, n_features), dtype=np.float32)
        
        for i, ts in enumerate(traj_slice):
            feature_matrix[i, :] = distances.self_distance_array(atoms.positions)

    elif coord_type == "cartesian":
        # Load only the topology file as reference to avoid re-reading the XTC trajectory into memory
        ref = mda.Universe(gro_path)
        ref_atoms = ref.select_atoms(selection)

        n_features = n_atoms * 3
        feature_matrix = np.zeros((n_frames_out, n_features), dtype=np.float32)

        for i, ts in enumerate(traj_slice):
            align.alignto(atoms, ref_atoms, select=selection)
            feature_matrix[i, :] = atoms.positions.flatten()

    else:
        raise ValueError("coord_type must be either 'internal' or 'cartesian'")
        
    return feature_matrix, atoms