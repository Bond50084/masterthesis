import loading_trajectory as lt
import os
import numpy as np
import compute_dnn as nn
import visualization as vis

def main():

    COORDINATE_TYPE = "cartesian" 
    NUM_FRAMES = 1000000
    STEP = 200
    SELECTION = "not name H*"
    gro_file = 'Data/input_files/coord.gro'
    xtc_file = 'Data/trajectories/replica7.xtc'

    print(f"--- Running Pipeline with {COORDINATE_TYPE.upper()} coordinates ---")


    #FIRST STEP: LOADING DATA
    #feature_matrix, selected_atoms = lt.load_trajectory(
    #    gro_file, 
    #    xtc_file, 
    #    coord_type=COORDINATE_TYPE, 
    #    selection=SELECTION,
    #    num_frames=NUM_FRAMES,
    #    step = STEP
    #)
    feature_matrix, selected_atoms = lt.load_traj_step(
        gro_file, 
        xtc_file, 
        coord_type=COORDINATE_TYPE, 
        selection=SELECTION,
        num_frames=NUM_FRAMES,
        step = STEP
    )

    u = selected_atoms.universe
    dt = u.trajectory.dt
    D, frames = nn.compute_aligned_rmsd_matrix(u, selection=SELECTION, stop=NUM_FRAMES, step = STEP)
    #np.save("rmsd_matrix.npy", D)

    nn_df = nn.first_second_nearest_neighbours(D, frames, dt=dt)
    print("HIEER")
    print(nn_df.head(20))


    # THIRD STEP: PLOTTING (Figure 1a + 1c)
    vis.plot_figure1_left(D, nn_df, save_path="fig1_left.png", in_nm=True)
    vis.plot_poisson_tests(D, n_balls=100, save_path="fig2_3_poisson.png", in_nm=True)  # Figure 2a, 2b, 3a
    vis.plot_twonn(nn_df["mu"].to_numpy(), save_path="fig2_3_two_nn.png")  # Figure 2c, 3b





    d_hat = nn.twonn_mle(nn_df["mu"].to_numpy())
    vis.plot_x_uniformity(nn_df, d_hat, save_path="fig6_x_uniform.png")




    

if __name__ == "__main__":
    main()