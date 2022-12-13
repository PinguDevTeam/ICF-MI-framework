"""
This is a python class for tensors in TT-format.
Through this object you can compute TT-decomposition of multidimensional arrays in
one shot using [TTSVD algorithm](https://epubs.siam.org/doi/epdf/10.1137/090752286)
or incrementally using [TT-ICE algorithm](https://arxiv.org/abs/2211.12487).

Furthermore, this object allows exporting and importing TT-cores using native
format (`.ttc`) and intermediary formats (`.txt`).
"""
import warnings
import time

import numpy as np

from .utils import ttsvd
from pickle import dump, load


class ttObject:
    """
    Python object for tensors in Tensor-Train format.

    This object computes the TT-decomposition of multidimensional arrays using
    `TTSVD`_, `TT-ICE`_, and `TT-ICE*`_.
    It furthermore contains an inferior method, ITTD, for benchmarking purposes.

    This object handles various operations in TT-format such as dot product and norm.
    Also provides supporting operations for uncompressed multidimensional arrays such
    as reshaping and transpose.
    Once the tensor train approximation for a multidimensional array is computed, you
    can compute projection of appropriate arrays onto the TT-cores, reconstruct
    projected or compressed tensors and compute projection/compression accuracy.

    You can also save/load tensors as `.ttc` or `.txt` files.
    Attributes
    ----------
    inputType: type
        Type of the input data. This determines how the object is initialized.
    originalData:
        Original multidimensional input array. Generally will not be stored
        after computing
        an initial set of TT-cores.
    method: :obj:`str`
        Method of computing the initial set of TT-cores. Currently only accepts
        `'ttsvd'` as input
    ttEpsilon: :obj:`float`
        Desired relative error upper bound.
    ttCores: :obj:`list` of :obj:`numpy.array`
        Cores of the TT-decomposition. Stored as a list of numpy arrays.
    ttRanks: :obj:`list` of :obj:`int`
        TT-ranks of the decomposition. Stored as a list of integers.
    nCores: :obj:`int`
        Number of TT-cores in the decomposition.
    nElements: :obj:`int`
        Number of entries present in the current `ttObject`.
    originalShape: :obj:`tuple` or :obj:`list`
        Original shape of the multidimensional array.
    reshapedShape: :obj:`tuple` or :obj:`list`
        Shape of the multidimensional array after reshaping. Note that
        this attribute is only meaningful before computing a TT-decomposition
    indexOrder: :obj:`list` of :obj:`int`
        Keeps track of original indices in case of transposing the input array.
    .. _TTSVD:
        https://epubs.siam.org/doi/epdf/10.1137/090752286
        _TT-ICE:
        https://arxiv.org/abs/2211.12487
        _TT-ICE*:
        https://arxiv.org/abs/2211.12487

    """

    def __init__(
        self,
        data,
        epsilon: float = None,
        keepData: bool = False,
        samplesAlongLastDimension: bool = True,
        method: str = "ttsvd",
    ) -> None:
        """
        Initializes the ttObject.

        Parameters
        ----------
        data: :obj:`numpy.array` or :obj:`list`
            Main input to the ttObject. It can either be a multidimensional
            `numpy array` or `list of numpy arrays`.
            If list of numpy arrays are presented as input, the object will interpret it
            as the TT-cores of an existing decomposition.
        epsilon: :obj:`float`, optional
            The relative error upper bound desired for approximation. Optional for cases
            when `data` has type `list`.
        keepData: :obj:`bool`, optional
            Optional boolean variable to determine if the original array will be kept
            after compression. Set to `False` by default.
        samplesAlongLastDimension: :obj:`bool`, optional
            Boolean variable to ensure if the samples are stacked along the last
            dimension. Assumed to be `True` since it is one of the assumptions in
            TT-ICE.
        method: :obj:`str`, optional
            Determines the computation method for tensor train decomposition of
            the multidimensional array presented as `data`. Set to `'ttsvd'` by default.

            Currently the package only has support for ttsvd, additional support such as
            `ttcross` might be included in the future.

        """
        # self.ttRanks=ranks
        self.ttCores = None
        self.nCores = None
        self.nElements = None
        self.inputType = type(data)
        self.method = method
        self.keepOriginal = keepData
        self.originalData = data
        self.samplesAlongLastDimension = samplesAlongLastDimension
        if self.inputType is np.memmap:
            self.inputType = np.ndarray

        if self.inputType == np.ndarray:
            self.ttEpsilon = epsilon
            self.originalShape = list(data.shape)
            self.reshapedShape = self.originalShape
            self.indexOrder = [idx for idx in range(len(self.originalShape))]
        elif self.inputType == list:
            self.nCores = len(data)
            self.ttCores = data
            self.reshapedShape = [core.shape[1] for core in self.ttCores]
            self.updateRanks()
        else:
            raise TypeError("Unknown input type!")

    @property
    def coreOccupancy(self) -> None:
        """
        :obj:`list` of :obj:`float`: A metric showing the *relative rank* of each
        TT-core. This metric is used for a heuristic enhancement tool in `TT-ICE*`
        algorithm
        """
        try:
            return [
                core.shape[-1] / np.prod(core.shape[:-1]) for core in self.ttCores[:-1]
            ]
        except ValueError:
            warnings.warn(
                "No TT cores exist, maybe forgot calling object.ttDecomp?", Warning
            )
            return None

    @property
    def compressionRatio(self) -> float:
        """
        :obj:`float`: A metric showing how much compression with respect to the
        original multidimensional array is achieved.
        """
        originalNumEl = 1
        compressedNumEl = 0
        for core in self.ttCores:
            originalNumEl *= core.shape[1]
            compressedNumEl += np.prod(core.shape)
        return originalNumEl / compressedNumEl

    def changeShape(self, newShape: tuple or list) -> None:
        """
        Function to change shape of input tensors and keeping track of the reshaping.
        Reshapes `originalData` and saves the final shape in `reshapedShape`

        Note
        ----
        A simple `numpy.reshape` would be sufficient for this purpose but in order to
        keep track of the shape changes the `reshapedShape` attribute also needs to be
        updated accordingly.

        Parameters
        ----------
        newShape:obj:`tuple` or `list`
            New shape of the tensor

        Raises
        ------
        warning
            If an attempt is done to modify the shape after computing a
            TT-decomposition.
            This is important since it will cause incompatibilities in other functions
            regarding the shape of the uncompressed tensor.

        """
        if self.ttCores is not None:
            warnings.warning(
                "Warning! You are reshaping the original data after computing a\
                TT-decomposition! We will proceed without reshaping self.originalData!!"
            )
            return None
        self.reshapedShape = newShape
        self.originalData = np.reshape(self.originalData, self.reshapedShape)
        self.reshapedShape = list(self.originalData.shape)
        # Changed reshapedShape to a list for the trick in ttICEstar
        if self.samplesAlongLastDimension:
            self.singleDataShape = self.reshapedShape[:-1]
            # This line assumes we keep the last index as the samples index and don't
            # interfere with its shape

    def computeTranspose(self, newOrder: list) -> None:
        """
        Transposes the axes of `originalData`.

        Similar to `changeShape`, a simple
        `numpy.transpose` would be sufficient for this purpose but in order to
        keep track of the transposition order `indexOrder` attribute also needs
        to be updated accordingly.

        Parameters
        ----------
        newOrder:obj:`list`
            New order of the axes.

        Raises
        ------
        ValueError
            When the number of transposition axes are not equal to the number of
            dimensions of `originalData`.
        """
        assert self.inputType == np.ndarray and self.ttCores is None
        if len(newOrder) != len(self.indexOrder):
            raise ValueError(
                "size of newOrder and self.indexOrder does not match. \
                    Maybe you forgot reshaping?"
            )
        self.indexOrder = [self.indexOrder[idx] for idx in newOrder]
        self.originalData = self.originalData.transpose(newOrder)
        self.reshapedShape = list(self.originalData.shape)

    def saveData(
        self, fileName: str, directory="./", justCores=True, outputType="ttc"
    ) -> None:
        """
        Writes the computed TT-cores to an external file.

        Parameters
        ----------
        fileName:obj:`str`

        directory:obj:`str`
            Location to save files with respect to the present working directory.
        justCores:obj:`bool`
            Boolean variable to determine if `originalData` will be discarded
            or not while saving.
        outputType:obj:`str`
            Type of the output file. `ttc` for pickled `ttObject`, `txt` for
            individual text files for each TT-core.
        """
        if justCores:
            if outputType == "ttc":
                with open(directory + fileName + ".ttc", "wb") as saveFile:
                    temp = ttObject(self.ttCores)
                    for attribute in vars(self):
                        if attribute != "originalData":
                            setattr(temp, attribute, eval(f"self.{attribute}"))
                    dump(temp, saveFile)
            elif outputType == "txt":
                for coreIdx, core in enumerate(self.ttCores):
                    np.savetxt(
                        directory + f"{fileName}_{coreIdx}.txt",
                        core.reshape(-1, core.shape[-1]),
                        header=f"{core.shape[0]} {core.shape[1]} {core.shape[2]}",
                        delimiter=" ",
                    )
            elif outputType == "npy":
                for coreIdx, core in enumerate(self.ttCores):
                    np.save(
                        directory + f"{fileName}_{coreIdx}.npy",
                        core,
                    )
            else:
                raise ValueError(f"Output type {outputType} is not supported!")
        else:
            if outputType == "txt":
                raise ValueError(
                    ".txt type outputs are only supported for justCores=True!!"
                )
            if self.method == "ttsvd":
                with open(directory + fileName + ".ttc", "wb") as saveFile:
                    dump(self, saveFile)
            else:
                raise ValueError("Unknown Method!")

    @staticmethod
    def loadData(fileName: str, numCores=None) -> "ttObject":
        """
        Loads data from a `.ttc` or `.txt` file


        Static method to load TT-cores into a ttObject object.
        Note
        ----
        If data is stored in {coreFile}_{coreIdx}.txt format,
        the input fileName should just be coreFile.txt

        Parameters
        ----------
        fileName:obj:`str`
            Name of the file that will be loaded.
        numCores:obj:`int`
            Number of cores that the resulting `ttObject` will have
            (Only required when input data format is `.txt`)
        """
        fileExt = fileName.split(".")[-1]
        if fileExt == "ttc":
            with open(fileName, "rb") as f:
                dataSetObject = load(f)
            return dataSetObject
        elif fileExt == "txt":
            if numCores is None:
                raise ValueError("Number of cores are not defined!!")
            fileBody = fileName.split(".")[0]
            coreList = []
            for coreIdx in range(numCores):
                with open(f"{fileBody}_{coreIdx}.{fileExt}"):
                    coreShape = f.readline()[2:-1]
                    coreShape = [int(item) for item in coreShape.split(" ")]
                coreList.append(
                    np.loadtxt(f"{fileBody}_{coreIdx}.{fileExt}").reshape[coreShape]
                )
            return ttObject(coreList)
        elif fileExt == "npy":
            if numCores is None:
                raise ValueError("Number of cores are not defined!!")
            fileBody = fileName.split(".")[0]
            coreList = []
            for coreIdx in range(numCores):
                coreList.append(np.load(f"{fileBody}_{coreIdx}.{fileExt}"))
            return ttObject(coreList)
        else:
            raise ValueError(f"{fileExt} files are not supported!")

    def projectTensor(self, newData: np.array, upTo=None) -> np.array:
        """
        Projects tensors onto basis spanned by TT-cores.

        Given a tensor with appropriate dimensions, this function leverages
        the fact that TT-cores obtained through `TTSVD`_ and `TT-ICE`_ are
        column-orthonormal in the mode-2 unfolding.

        Note
        ----
        This function will not yield correct results if the TT-cores are
        not comprised of orthogonal vectors.

        Parameters
        ----------
        newData:obj:`np.aray`
            Tensor to be projected
        upTo:obj:`int`, optional
            Index that the projection will be terminated. If an integer is
            passed as this parameter, `newTensor` will be projected up to
            (not including) the core that has index `upTo`. Assumes 1-based
            indexing.

         .. _TTSVD:
            https://epubs.siam.org/doi/epdf/10.1137/090752286
            _TT-ICE:
            https://arxiv.org/abs/2211.12487
            _TT-ICE*:
            https://arxiv.org/abs/2211.12487
        """
        for coreIdx, core in enumerate(self.ttCores):
            if (coreIdx == len(self.ttCores) - 1) or coreIdx == upTo:
                break
            newData = (
                core.reshape(np.prod(core.shape[:2]), -1).transpose()
            ) @ newData.reshape(
                self.ttRanks[coreIdx] * self.ttCores[coreIdx].shape[1], -1
            )
        return newData

    def reconstruct(self, projectedData, upTo=None):
        """
        Reconstructs tensors using TT-cores.

        Assumes that `projectedData` is a slice from the last
        TT-core.
        While reconstructing any projected tensor from  `projectTensor`,
        this function leverages the fact that TT-cores obtained through
        `TTSVD`_ and `TT-ICE`_ are column-orthonormal in the mode-2
        unfolding.

        Note
        ----
        This function might not yield correct results for projected tensors
        if the TT-cores are not comprised of orthogonal vectors.

        Parameters
        ----------
        projectedData:obj:`np.aray`
            A tensor slice (or alternatively an array)
        upTo:obj:`int`, optional
            Index that the reconstruction will be terminated. If an integer is
            passed as this parameter, `projectedData` will be projected up to
            (not including) the core that has index `upTo`. Assumes 1-based
            indexing.
        .. _TTSVD:
            https://epubs.siam.org/doi/epdf/10.1137/090752286
            _TT-ICE:
            https://arxiv.org/abs/2211.12487

        """
        if upTo is None:
            upTo = len(self.ttCores) - 1  # Write the core index in 1-indexed form!!
        for core in self.ttCores[:upTo][::-1]:
            projectedData = np.tensordot(core, projectedData, axes=(-1, 0))
        return projectedData

    def updateRanks(self) -> None:
        """
        Updates the ranks of the `ttObject` after incremental updates.
        """
        self.ttRanks = [1]
        for core in self.ttCores:
            self.ttRanks.append(core.shape[-1])
        return None

    def computeRelError(self, newTensor: np.array) -> np.array:
        """
        Computes relative error by projecting data onto TT-cores.

        This function computes the error induced by projecting `data` onto the TT-cores
        of `ttObject`. The last index of `newTensor` is assumed to be the number of
        individual observations.

        Note
        ----
        - In order to compute the projection onto the TT-cores, the dimensions of `data`
        should match that of `ttObject`.
        - If a single observation will be passed as `newTensor`, an additional
        index/dimension should be introduced either through reshaping or [:,None]

        Parameters
        ----------
        newTensor:obj:`np.array`
            Tensor for which the projection error is computed

        Returns
        -------
        relError:obj:`np.array`
            Array of relative errors
        """
        elementwiseNorm = np.linalg.norm(newTensor, axis=0)
        for _ in range(len(newTensor.shape) - 2):
            elementwiseNorm = np.linalg.norm(elementwiseNorm, axis=0)
        projectedData = self.projectTensor(newTensor)
        reconstructedData = self.reconstruct(projectedData).reshape(newTensor.shape)
        difference = newTensor - reconstructedData
        differenceNorm = np.linalg.norm(difference, axis=0)
        for _ in range(len(difference.shape) - 2):
            differenceNorm = np.linalg.norm(differenceNorm, axis=0)
        relError = differenceNorm / elementwiseNorm
        return relError

    def computeRecError(self, data: np.array, start=None, finish=None) -> None:
        """
        Function to compute relative error by reconstructing data from slices
        of TT-cores.
        Currently not implemented.
        """
        if finish is None:
            finish = start + 1
        rec = self.reconstruct(self.ttCores[-1][:, start:finish, :]).reshape(
            self.reshapedShape[:-1] + [-1]
        )
        data = data.reshape(self.reshapedShape[:-1] + [-1])
        diff = data - rec
        elementwiseNorm = np.linalg.norm(data, axis=0)
        for _ in range(len(data.shape) - 2):
            elementwiseNorm = np.linalg.norm(elementwiseNorm, axis=0)
        diffNorm = np.linalg.norm(diff, axis=0)
        for _ in range(len(diff.shape) - 2):
            diffNorm = np.linalg.norm(diffNorm, axis=0)
        recError = diffNorm / elementwiseNorm
        return recError

    def ttDecomp(self, norm=None, dtype=np.float32) -> "ttObject.ttCores":
        """
        Computes TT-decomposition of a multidimensional array using `TTSVD`_ algorithm.

        Currently only supports `ttsvd` as method. In the future additional formats may
        be covered.

        Parameters
        ----------
        norm:obj:`float`, optional
            Norm of the tensor to be compressed
        dtype:obj:`type`, optional
            Desired data type for the compression. Intended to allow lower precision
            if needed.

        Raises
        ------
        ValueError
            When `method` is not one of the admissible methods.


        The following attributes are modified as a result of this function:
        -------
        - `ttObject.ttCores`
        - `ttObject.ttRanks`
        - `ttObject.compressionRatio`

        .. _TTSVD:
            https://epubs.siam.org/doi/epdf/10.1137/090752286
        """
        if norm is None:
            norm = np.linalg.norm(self.originalData)
        if self.method == "ttsvd":
            startTime = time.time()
            self.ttRanks, self.ttCores = ttsvd(
                self.originalData, norm, self.ttEpsilon, dtype=dtype
            )
            self.compressionTime = time.time() - startTime
            self.nCores = len(self.ttCores)
            self.nElements = 0
            for cores in self.ttCores:
                self.nElements += np.prod(cores.shape)
            if not self.keepOriginal:
                self.originalData = None
            return None
        else:
            raise ValueError("Method unknown. Please select a valid method!")
