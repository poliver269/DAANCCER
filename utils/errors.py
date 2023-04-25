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
