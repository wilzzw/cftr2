import numpy as np
from MDAnalysis.analysis.align import rotation_matrix

# My custom function that works on coordinate data in large arrays
# Bypassing loading trajectories of any kind
# Borrowing from MDAnalysis
def transform(X, R, t):
    """Transform coordinates X by rotation matrix R and translation vector t.

    Parameters
    ----------
    X : numpy.ndarray, shape=(n_atoms, 3)
        Coordinates to transform.
    R : numpy.ndarray, shape=(3, 3)
        Rotation matrix.
    t : numpy.ndarray, shape=(3,)
        Translation vector.

    Returns
    -------
    X : numpy.ndarray, shape=(n_atoms, 3)
        Transformed coordinates.

    """
    # Center the coordinates X first
    X -= np.mean(X, axis=0)
    return np.dot(X, R.T) + t

def align2parts(X, Xref, index, index_ref):
    """Align two parts of a structure.

    Parameters
    ----------
    X : numpy.ndarray, shape=(n_frames, n_atoms, 3)
        Coordinates of the structures to align.
    Xref : numpy.ndarray, shape=(n_atoms, 3)
        Coordinates of the reference structure.
    index : numpy.ndarray, shape=(n_index,)
        Array indices of the atom dimension/axis in the first part.
    index_ref : numpy.ndarray, shape=(n_index,)
        Array indices of the atoms in the second part.

    Returns
    -------
    Xtr : numpy.ndarray, shape=(n_atoms, 3)
        Transformed coordinates.

    """

    # index and index_ref should be the same length
    assert len(index) == len(index_ref)

    # Xcom = np.mean(X[index], axis=0)
    Xcom = np.mean(X[:,index], axis=-2)
    # Add an axis to Xcom to allow broadcasting
    Xcom = Xcom[:, np.newaxis, :]
    Xrefcom = np.mean(Xref[index_ref], axis=-2)
    # Move X to the center of mass of selected parts
    Xtr = X - Xcom

    # Compute the rotation matrix to align the two parts
    # My attempt to vectorize it
    # Unfortunately np.apply_along_axis does not work with function that takes multidimensional arrays
    # np.apply_over_axes is not what I am looking for either
    # The trick is to linearize the target array and then reshape it back inside the function
    def v_rotmat(x):
        x = x.reshape(len(index), 3)
        R, _ = rotation_matrix(x, (Xref-Xrefcom)[index_ref])
        return R
    R = np.apply_along_axis(v_rotmat, 1, Xtr[:,index].reshape(len(Xtr), len(index)*3))

    # R, _ = rotation_matrix(Xtr[index], (Xref-Xrefcom)[index_ref])
    # Apply the rotation matrix to the whole structure
    # Fancy einsum to do the matrix multiplication for each pair of matrices along axis 0
    Xtr = np.einsum('ijk,ikl->ijl', Xtr, np.transpose(R, axes=[0,2,1]))

    # Move the center of mass of X to the center of mass of Xref
    Xtr = Xtr + Xrefcom
    return Xtr
