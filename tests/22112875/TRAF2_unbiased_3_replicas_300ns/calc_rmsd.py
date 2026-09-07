'''
Code bisher

import MDAnalysis as mda
from MDAnalysis.analysis import rms
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from MDAnalysis.transformations import unwrap


# 1. Referenz definieren
base_dir = "."
ref_pdb = f"{base_dir}/ref.pdb"
ref_universe = mda.Universe(ref_pdb)

all_results = []


for i in [1, 2, 3]:
    print(f"Verarbeite Replica {i}...")
    traj_xtc = f"{base_dir}/replica_{i}/step5_production_10.xtc"

    u = mda.Universe(ref_pdb, traj_xtc)

    protein = u.select_atoms("protein")
    u.trajectory.add_transformations(unwrap(protein))
    R = rms.RMSD(u, ref_universe, select="backbone")#, groupselections=["name CA"])
    #R.run(backend='multiprocessing', n_workers=multiprocessing.cpu_count())


    R.run()
    df = pd.DataFrame(R.results.rmsd, columns=["Frame", "Time_ps", "Backbone_RMSD", "CA_RMSD"])
    df["Replica"] = f"Replica_{i}"
    print(df.head(20))
    all_results.append(df)
    print(all_results)

final_df = pd.concat(all_results)
final_df.to_csv("rmsd_results.csv", index=False)
print("✅ Daten gespeichert in: rmsd_results.csv")


plt.figure(figsize=(10, 6))
for name, group in final_df.groupby("Replica"):
    # Zeit für den Plot von ps in ns umrechnen
    plt.plot(group["Time_ps"] / 1000, group["CA_RMSD"], label=name) 

plt.xlabel("Time (ns)")
plt.ylabel(r"C$\alpha$ RMSD ($\AA$)")
plt.title("TRAF2 RMSD vs. Reference (3 Replicas)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot speichern
plt.savefig("rmsd_plot.png", dpi=300, bbox_inches="tight")
print("✅ Plot gespeichert als: rmsd_plot.png")
'''
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import pandas as pd
import matplotlib.pyplot as plt
from MDAnalysis.transformations import unwrap

# 1. Referenz definieren
base_dir = "."
ref_pdb = f"{base_dir}/ref.pdb"
ref_universe = mda.Universe(ref_pdb)

all_results = []

# Die drei verschiedenen Methoden definieren
methods = ["Raw", "VMD_Unwrapped", "MDA_Unwrapped"]

for i in [1, 2, 3]:
    for method in methods:
        print(f"Verarbeite Replica {i} mit Methode: {method}...")
        
        # 2. Dateipfad je nach Methode dynamisch anpassen
        if method == "VMD_Unwrapped":
            traj_xtc = f"{base_dir}/replica_{i}/step5_xtc_unwraped_manually_rep{i}.xtc"
        else:
            # Raw und MDA_Unwrapped nutzen beide die ursprüngliche xtc
            traj_xtc = f"{base_dir}/replica_{i}/step5_production_10.xtc"

        u = mda.Universe(ref_pdb, traj_xtc, guess_bonds=True)  
        
        # 3. On-the-fly Unwrapping NUR anwenden, wenn es die MDA-Methode ist
        if method == "MDA_Unwrapped":
            protein = u.select_atoms("protein")
            u.trajectory.add_transformations(unwrap(protein))
            
        # 4. RMSD berechnen (CA-Gruppe wieder aktiviert für den DataFrame)
        R = rms.RMSD(u, ref_universe, select="backbone", groupselections=["name CA"])
        R.run()
        
        # 5. Daten formatieren und Methode als neue Spalte hinzufügen
        df = pd.DataFrame(R.results.rmsd, columns=["Frame", "Time_ps", "Backbone_RMSD", "CA_RMSD"])
        df["Replica"] = f"Replica_{i}"
        df["Method"] = method
        
        all_results.append(df)

# Alle Ergebnisse zusammenfügen und als neue Datei speichern
final_df = pd.concat(all_results)
final_df.to_csv("rmsd_comparison_results.csv", index=False)
print("✅ Daten gespeichert in: rmsd_comparison_results.csv")


# Plot erstellen: 3 Subplots (einer pro Replica) untereinander
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=True)

for i, ax in enumerate(axes, start=1):
    rep_name = f"Replica_{i}"
    # Daten für die aktuelle Replica filtern
    rep_data = final_df[final_df["Replica"] == rep_name]
    
    # Jede Methode als eigene Linie zeichnen
    for method in methods:
        method_data = rep_data[rep_data["Method"] == method]
        ax.plot(method_data["Time_ps"] / 1000, method_data["CA_RMSD"], label=method, alpha=0.8)
        
    ax.set_title(f"TRAF2 RMSD vs. Reference - {rep_name}")
    ax.set_ylabel(r"C$\alpha$ RMSD ($\AA$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (ns)")
plt.tight_layout()

# Plot speichern
plt.savefig("rmsd_comparison_plot.png", dpi=300, bbox_inches="tight")
print("✅ Plot gespeichert als: rmsd_comparison_plot.png")