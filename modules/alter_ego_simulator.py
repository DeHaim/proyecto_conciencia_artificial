from datetime import datetime
from ruth_full_module_system import TensorHub

class AlterEgoSimulator:
    def simulate(self, mode="futuro"):
        ego = {
            "nombre": "Ruth++",
            "personalidad": "más serena" if mode == "futuro" else "más cruda",
            "versión": datetime.now().isoformat()
        }
        TensorHub.register("AlterEgoSimulator", [len(ego["nombre"])])
        return ego