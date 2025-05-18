from packages.math import math_utils

class QueternionConfig:
    def GetInputSize(self):
        return 5
    def GetEulerAngles(self, data):
        return math_utils.euler_from_quat(data)

class RotationMatrixConfig:
    def GetInputSize(self):
        return 6
    def GetEulerAngles(self, data):
        return math_utils.get_euler_from_matrix(data)
    
def get_config(arguments) -> QueternionConfig | RotationMatrixConfig:
    if arguments.data_representation == "quaternion":
        return QueternionConfig()
    elif arguments.data_representation == "rotation_matrix":
        return RotationMatrixConfig()
    else:
        raise ValueError(f"Unknown input type: {arguments.data_representation}")