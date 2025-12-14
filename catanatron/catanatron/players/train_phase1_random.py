import gymnasium as gym
import numpy as np
import torch as th
import os
import shutil
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from weighted_random import WeightedRandomPlayer

# 🛑 SAFETY
th.distributions.Distribution.set_default_validate_args(False)

# ==========================================
# 💎 REWARD SHAPING WRAPPER
# ==========================================
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, vp_reward=0.1):
        super().__init__(env)
        self.vp_reward = vp_reward
        self.last_vp = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_vp = self._get_vp()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_vp = self._get_vp()
        vp_diff = current_vp - self.last_vp
        
        # Intermediate Reward
        if vp_diff > 0:
            reward += vp_diff * self.vp_reward
            self.last_vp = current_vp
            
        # 🟢 WIN BONUS: Massive spike for reaching 10 VP
        if (terminated or truncated) and current_vp >= 10:
            reward += 10.0  # <--- Make winning worth 100x more than a settlement
            
        return obs, reward, terminated, truncated, info

    def _get_vp(self):
        try:
            return self.env.unwrapped.game.state.player_state.get('P0_ACTUAL_VICTORY_POINTS', 0)
        except:
            return 0

# ==========================================
# 2. SETUP
# ==========================================
def mask_fn(env: gym.Env) -> np.ndarray:
    env = env.unwrapped
    if hasattr(env, "get_valid_actions"): valid_ids = env.get_valid_actions()
    else: valid_ids = getattr(env, "valid_actions", [])
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid_ids] = True
    return np.array(mask, dtype=bool)

def make_env():
    import gymnasium as gym
    from gymnasium.envs.registration import register
    try: from catanatron_gym.envs import CatanatronEnv
    except ImportError:
        try: from catanatron.gym.envs import CatanatronEnv
        except ImportError: pass

    if "catanatron/Catanatron-v0" not in gym.envs.registry:
        register(id="catanatron/Catanatron-v0", entry_point=CatanatronEnv, max_episode_steps=500)

    from catanatron import RandomPlayer, Color
    env_config = {
        "victory_points": 10,
        "map_type": "random",
        "enemies": [WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.ORANGE), WeightedRandomPlayer(Color.WHITE)]
    }
    
    env = gym.make("catanatron/Catanatron-v0", config=env_config)
    env = RewardShapingWrapper(env, vp_reward=0.1) 
    env = Monitor(env) 
    env = ActionMasker(env, mask_fn)
    return env

# ==========================================
# 3. TRAINING LOOP (RESUMABLE)
# ==========================================
def train_phase_1_shaped():
    n_envs = 8 
    print(f"🚀 Starting Phase 1: Shaped Reward Training vs Random Bots")

    # Files to look for
    model_path = "ppo_catanatron_4p_10vp.zip"
    stats_path = "vec_normalize_4p.pkl"

    # Create Base Envs
    venv = SubprocVecEnv([make_env for _ in range(n_envs)]) 
    
    # 🟢 1. SETUP / RESUME NORMALIZATION
    if os.path.exists(stats_path):
        print(f"📈 Found existing stats: {stats_path}. Loading...")
        env = VecNormalize.load(stats_path, venv)
        # Important: Reset training mode and clips after loading
        env.training = True
        env.norm_reward = True
        env.clip_obs = 5.0
        env.clip_reward = 10.0
        env.epsilon = 1e-8
    else:
        print("⚠️ No stats found. Creating fresh Normalization.")
        env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=10.0, epsilon=1e-8)

    # 🟢 2. CONFIG
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ortho_init=True,
    )

    # 🟢 3. SETUP / RESUME MODEL
    if os.path.exists(model_path):
        print(f"🧠 Found existing model: {model_path}. Resuming...")
        
        # We pass env to ensure it attaches correctly
        # We pass custom_objects if we want to enforce specific LR/Clips on the loaded model
        custom_objects = {
            "learning_rate": 3e-4, 
            "clip_range": 0.2,
            "max_grad_norm": 0.5
        }
        model = MaskablePPO.load(model_path, env=env, custom_objects=custom_objects)
        reset_timesteps = False # Continue log count
    else:
        print("⚠️ No model found. Starting FRESH training.")
        model = MaskablePPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=3e-4,    
            max_grad_norm=0.5,     
            gae_lambda=0.95,
            clip_range=0.2,
            batch_size=512,
            n_epochs=10,
            policy_kwargs=policy_kwargs
        )
        reset_timesteps = True # Start log count from 0

    # 4. START TRAINING
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path='./checkpoints/', name_prefix='ppo_phase1')

    try:
        # Add 3 Million steps to whatever we already have
        model.learn(total_timesteps=3_000_000, callback=checkpoint_callback, reset_num_timesteps=reset_timesteps)
        
        print("✅ Phase 1 Complete!")
        model.save("ppo_catanatron_4p_10vp")
        env.save("vec_normalize_4p.pkl")

    except KeyboardInterrupt:
        print("🛑 Interrupted! Saving progress...")
        model.save("ppo_catanatron_4p_10vp")
        env.save("vec_normalize_4p.pkl")

if __name__ == "__main__":
    train_phase_1_shaped()