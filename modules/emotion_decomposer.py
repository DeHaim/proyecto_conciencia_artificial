from ruth_full_module_system import TensorHub

class EmotionDecomposer:
    def decompose(self, emotion_state):
        decomposition = {"core": emotion_state.get("primary", "neutral"), "intensity": emotion_state.get("intensity", 0.5)}
        TensorHub.register("EmotionDecomposer", [decomposition["intensity"]])
        return decomposition