# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np


def LDA(data: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Fisher's Linear Discriminant Analysis. See https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher's_linear_discriminant and http://www.facweb.iitkgp.ac.in/~sudeshna/courses/ml08/lda.pdf for descriptions of the method.

    >>> data = np.array([  # First class
    ...     [1, 2],
    ...     [1.5, 2.7],
    ...     [2.2, 2],
    ...     [2.5, 3],
    ...     [3.5, 3.1],
    ...     [4, 3.5],
    ...     # Second class
    ...     [2.5, 4.5],
    ...     [3.2, 4.8],
    ...     [5, 5.2],
    ...     [5.5, 5],
    ...     [5.9, 5.2],
    ...     [6.6, 6],
    ... ])
    >>> labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> coeff, proj, latent = LDA(data, labels)
    >>> np.round(
    ...     np.abs(latent / np.sum(latent)), 2
    ... )  # Explained discrimination (only the first axis is relevant when using two classes)
    array([1., 0.])
    >>> coeff[:, 0]  # First eigenvector
    array([ 0.24558339, -0.96937547])
    >>> proj[0, 0].item()  # Projection of the first data point
    -1.6931675538135342
    >>> np.matmul(
    ...     [[1, 2], [1.5, 1.5]], coeff[:, 0]
    ... )  # The eigenvectors are stored in the columns of coeff (here only the first column is meaningful)
    array([-1.69316755, -1.08568813])

    Args:
        data: matrix containing all data points (observations in rows, variables in columns) [nObservations x nVariables].
        labels: vector with the class labels [nObservations].

    Returns:
        coeff: matrix with the eigenvectors as columns [nVariables x nVariables].
        proj: projection of the data points onto the new axes [nObservations x nVariables].
        latent: vector with the eigenvalues [nVariables].
    """
    assert data.shape[0] == len(labels), "Each data point (= row of the data matrix) must have an associated label"
    assert len(np.unique(labels)) >= 2, "At least two classes are required"

    # Gather data
    nFeatures = data.shape[1]
    withinScatter = np.zeros((nFeatures, nFeatures))
    betweenScatter = np.zeros((nFeatures, nFeatures))

    overallMean = np.mean(data, axis=0)

    # Iterate over all classes to calculate the combined scatter matrices
    for c in np.unique(labels):
        dataClass = data[labels == c, :]

        # Variance to the common mean value
        meanClass = np.mean(dataClass, axis=0)
        meanDiff = meanClass - overallMean
        betweenScatter = betweenScatter + dataClass.shape[0] * np.outer(meanDiff, meanDiff)

        # Variance in each class
        covClass = np.cov(dataClass, rowvar=False)
        withinScatter = withinScatter + covClass

    # Perform LDA
    scatter = np.matmul(
        np.linalg.pinv(withinScatter), betweenScatter
    )  # Using the pseudo-inverse matrix gives stabler results

    # Sort the eigenvalues descendingly (https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt)
    eigenvalues, eigenvectors = np.linalg.eig(scatter)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvectors, np.matmul(data, eigenvectors), eigenvalues
