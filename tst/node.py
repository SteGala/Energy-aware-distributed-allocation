class Node:
    def __init__(self, cpu, gpu, bw, performance):
        self.initial_cpu = cpu
        self.used_cpu = 0
        self.initial_gpu = gpu
        self.used_gpu = 0
        self.initial_bw = bw
        self.used_bw = 0
        self.performance = performance
        
    def allocate(self, cpu, gpu, bw):
        self.used_cpu += cpu
        self.used_gpu += gpu
        self.used_bw += bw
        
    def deallocate(self, cpu, gpu, bw):
        self.used_cpu -= cpu
        self.used_gpu -= gpu
        self.used_bw -= bw
        
    def can_host_job(self, cpu, gpu):
        if (self.used_cpu + cpu <= self.initial_cpu):# and (self.used_gpu + gpu <= self.initial_gpu):
            return True
        return False