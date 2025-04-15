from ruth_full_module_system import TensorHub

class MemoryDiscriminator:
    def evaluate_memory(self, memory_json):
        resultado = sum([1 for m in memory_json if 'error' not in m]) / len(memory_json)
        TensorHub.register("MemoryDiscriminator", [resultado])
        return resultado