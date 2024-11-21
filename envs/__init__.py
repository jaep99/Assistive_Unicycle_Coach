from gymnasium.envs.registration import register

register(
    id='UnicyclePendulumStudent-v0',
    entry_point='envs.unicycle_pendulum_student_3d_v0:UnicyclePendulumStudent',
)