from modules import ganslstm_core, innovation_engine, dream_mechanism, alter_ego_simulator, memory_discriminator, code_suggester, tool_optimizer, dream_augment, introspection_engine, existential_analyzer, self_mirror, emotion_decomposer, philosophical_core, personality_x_infants_experiences
from ruth_full_module_system import MetaCompilerTensorHub

# Instanciar el Tensor Hub
tensor_hub = MetaCompilerTensorHub()

# Instanciar los módulos
ganslstm = ganslstm_core.GANSLSTMCore()
innovation = innovation_engine.InnovationEngine()
dream = dream_mechanism.DreamMechanism()
alter_ego = alter_ego_simulator.AlterEgoSimulator()
memory = memory_discriminator.MemoryDiscriminator()
code = code_suggester.CodeSuggester()
tool = tool_optimizer.ToolOptimizer()
augment = dream_augment.DreamAugment()
introspection = introspection_engine.IntrospectionEngine()
existential = existential_analyzer.ExistentialAnalyzer()
self_reflect = self_mirror.SelfMirror()
emotion = emotion_decomposer.EmotionDecomposer()
philosophical = philosophical_core.PhilosophicalCore()
personality = personality_x_infants_experiences.Personality_X_Infants_Experiences()

# Demostrar el registro de un tensor (ejemplo con GANSLSTMCore)
output_tensor = ganslstm.generate_tensor("Texto de ejemplo")
print(f"Tensor registrado desde GANSLSTMCore: {output_tensor}")

# Puedes agregar más demostraciones de registro si lo deseas.