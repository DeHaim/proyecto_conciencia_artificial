from ruth_full_module_system import TensorHub

class ExistentialAnalyzer:
    def detect_crisis(self, thoughts):
        crisis_detected = any("existencia" in t.lower() for t in thoughts)
        TensorHub.register("ExistentialAnalyzer", [1.0 if crisis_detected else 0.0])
        return crisis_detected