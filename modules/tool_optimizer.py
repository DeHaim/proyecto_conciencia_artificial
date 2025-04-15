from ruth_full_module_system import TensorHub

class ToolOptimizer:
    def evaluate_tools(self, tools: list[str]):
        optimized = sorted(tools)
        TensorHub.register("ToolOptimizer", [len(tools)])
        return optimized