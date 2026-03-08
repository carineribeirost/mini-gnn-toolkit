"""
Molecular fingerprints and Coulomb matrices for classical ML baselines.
No JAX or jraph dependency — these are pure NumPy/RDKit/scipy.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def morgan_fingerprints(smiles_list: list[str], n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Return (N, n_bits) float32 array of Morgan fingerprint bit vectors."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
        else:
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            fps.append(np.array(bv, dtype=np.float32))
    return np.stack(fps)


def coulomb_matrix_eigvals(smiles_list: list[str], max_atoms: int) -> np.ndarray:
    """
    Return (N, max_atoms) float32 array of sorted Coulomb matrix eigenvalues.

    The Coulomb matrix C is defined as:
        C_ii  = 0.5 * Z_i^2.4
        C_ij  = Z_i * Z_j / |R_i - R_j|  (i != j)

    Eigenvalues are sorted by absolute value descending and padded/truncated
    to max_atoms. Molecules with more atoms than max_atoms raise an error.
    """
    from scipy.linalg import eigvalsh

    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(np.zeros(max_atoms, dtype=np.float32))
            continue

        mol = Chem.AddHs(mol)
        try:
            from rdkit.Chem import AllChem as _AC
            _AC.EmbedMolecule(mol, randomSeed=0)
            _AC.MMFFOptimizeMolecule(mol)
        except Exception:
            results.append(np.zeros(max_atoms, dtype=np.float32))
            continue

        conf   = mol.GetConformer()
        atoms  = mol.GetAtoms()
        n      = mol.GetNumAtoms()

        if n > max_atoms:
            raise ValueError(
                f"Molecule has {n} atoms but max_atoms={max_atoms}. "
                "Increase max_atoms in the baseline config."
            )

        zs  = np.array([a.GetAtomicNum() for a in atoms], dtype=np.float64)
        pos = np.array([conf.GetAtomPosition(i) for i in range(n)], dtype=np.float64)

        C = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            C[i, i] = 0.5 * zs[i] ** 2.4
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                C[i, j] = C[j, i] = zs[i] * zs[j] / (d + 1e-8)

        eigs = eigvalsh(C)
        eigs = np.sort(np.abs(eigs))[::-1]         # descending absolute value
        padded = np.zeros(max_atoms, dtype=np.float32)
        padded[:n] = eigs.astype(np.float32)
        results.append(padded)

    return np.stack(results)
