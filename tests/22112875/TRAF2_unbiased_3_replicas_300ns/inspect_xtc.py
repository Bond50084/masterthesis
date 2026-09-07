import MDAnalysis as mda

# Pfade zu den Dateien im aktuellen Ordner
ref_pdb = "ref.pdb"
traj_xtc = "replica_1/step5_production_10.xtc"

# Universe laden
u = mda.Universe(ref_pdb, traj_xtc)

print("=== Allgemeine Trajektorien-Infos ===")
print(f"Anzahl der Frames:   {len(u.trajectory)}")
print(f"Anzahl der Atome:    {len(u.atoms)}")
print(f"Gesamtzeit:          {u.trajectory[-1].time} ps")
print(f"Zeit pro Frame (dt): {u.trajectory.dt} ps\n")

# Wir springen zum ersten Frame (Index 0)
u.trajectory[0]

print("=== Was steht in einem einzelnen Frame? (z.B. Frame 0) ===")
print(f"Aktueller Frame:    {u.trajectory.frame}")
print(f"Simulationszeit:    {u.trajectory.time} ps")
# Die Dimensionen sind [x, y, z, alpha, beta, gamma]
print(f"Simulationsbox:     {u.dimensions[:3]} Å\n")

print("=== Koordinaten der ersten 5 Atome in diesem Frame ===")
print(f"{'Index':<7} {'Residuum':<10} {'Atom':<7} {'X (Å)':<10} {'Y (Å)':<10} {'Z (Å)':<10}")
print("-" * 60)

for i in range(5):
    atom = u.atoms[i]
    x, y, z = u.atoms.positions[i]
    print(f"{atom.index:<7} {atom.resname:<10} {atom.name:<7} {x:<10.3f} {y:<10.3f} {z:<10.3f}")
