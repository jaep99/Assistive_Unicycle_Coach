from gymnasium.envs.registration import register

register(
    id='UnicyclePendulumTrajectory-v0',
    entry_point='envs.unicycle_pendulum_trajectory_3d_v0:UnicyclePendulumTrajectory',
)