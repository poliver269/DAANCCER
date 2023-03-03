class InvalidReconstructionException(Exception):
    pass


class InvalidComponentNumberException(InvalidReconstructionException):
    pass


class NonInvertibleEigenvectorException(InvalidReconstructionException):
    pass
