class DataBallPyWarning(Warning):
    "Warning class specific for databallpy"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
