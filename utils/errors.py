class InvalidReconstructionException(Exception):
    pass


class InvalidComponentNumberException(InvalidReconstructionException):
    pass


class NonInvertibleEigenvectorException(InvalidReconstructionException):
    pass


class InvalidRunningOptionError(Exception):
    pass


class InvalidSubsetTrajectory(Exception):
    pass


class InvalidProteinTrajectory(Exception):
    pass


class InvalidKernelName(Exception):
    pass


class ModelNotFittedError(Exception):
    pass