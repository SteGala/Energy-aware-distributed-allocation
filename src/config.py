from enum import Enum
import logging

class Utility(Enum):
    LGF = 1
    STEFANO = 2
    ALPHA_BW_GPU = 3
    ALPHA_GPU_CPU = 4
    ALPHA_GPU_BW = 5
    POWER = 6
    SGF = 7
    UTIL = 8
    
class DebugLevel(Enum):
    TRACE = 5
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    
class SchedulingAlgorithm(Enum):
    FIFO = 1
    SDF = 2 # shortest duration first

# create an enum to represent the possible types of GPUS
# the idea is to represent the types of GPU in ascending order of performance
# i.e., NVIDIA > AMD > INTEL so when we receive the request for an AMD GPU
# it can be executed on an NVIDIA and AMD GPU but not on an INTEL GPU
class GPUType(Enum):
    T4 = 1
    P100 = 2
    V100 = 3
    MISC = 4
    #V100M32 = 4 sarebbe 4 e misc 5
    
class ApplicationGraphType(Enum):
    LINEAR = 1
    GRAPH20 = 2
    GRAPH40 = 3
    GRAPH60 = 4
    
class GPUSupport:
    
    @staticmethod
    def get_gpu_type(gpu_type):
        """
        Returns the GPUType enum corresponding to the string `gpu_type`.

        Args:
            gpu_type (str): The GPU type.

        Returns:
            GPUType: The GPUType enum corresponding to `gpu_type`.
        """
        if gpu_type == "T4":
            return GPUType.T4
        elif gpu_type == "P100":
            return GPUType.P100
        elif gpu_type == "V100":
            return GPUType.V100
        # elif gpu_type == "V100M32":
        #     return GPUType.V100M32
        else:
            return GPUType.MISC
    
    
    @staticmethod
    def can_host(gpu_type1, gpu_type2):
        """
        Determines whether a GPU of type `gpu_type1` can host a GPU of type `gpu_type2`.

        Args:
            gpu_type1 (GPUType): The type of the host GPU.
            gpu_type2 (GPUType): The type of the job GPU.

        Returns:
            bool: True if `gpu_type1` can host `gpu_type2`, False otherwise.
        """
        return True
        # if gpu_type2.value == GPUType.V100.value:
        #     if gpu_type1.value == GPUType.V100.value:
        #         return True
        #     else:
        #         return False
        # return True
    
        
        

    
    @staticmethod
    def get_compute_resources(gpu_type):
        """
        Returns the number of CPUs and GPUs available for a given GPU type.

        Args:
            gpu_type (GPUType): The type of GPU to get compute resources for.

        Returns:
            Tuple[int, int]: A tuple containing the number of CPUs and GPUs available.
        """
        cpu = [96, 96, 64, 96]
        gpu = [2, 8, 2, 8]

        if gpu_type == GPUType.T4:
            return cpu[0], gpu[0]
        elif gpu_type == GPUType.P100:
            return cpu[2], gpu[2]
        elif gpu_type == GPUType.V100:
            return cpu[3], gpu[3]
        # elif gpu_type == GPUType.V100M32:
        #     return cpu[4], gpu[4]
        else: #MISC
            return cpu[1], gpu[1]
        
    @staticmethod
    def get_GPU_corrective_factor(gpu_type1, gpu_type2, decrement=0.15):
        """
        Returns the corrective factor for the GPU of type `gpu_type1` to host a GPU of type `gpu_type2`.

        Args:
            gpu_type1 (GPUType): The type of the node GPU.
            gpu_type2 (GPUType): The type of the job GPU.
            decrement (float, optional): The decrement factor if the GPUs don't match. Defaults to 0.15.

        Returns:
            float: The corrective factor for the GPU of type `gpu_type1` to host a GPU of type `gpu_type2`.
        """
        difference = gpu_type2.value - gpu_type1.value
        return 1 - (difference * decrement)