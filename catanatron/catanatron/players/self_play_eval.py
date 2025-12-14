import gymnasium as gym
import numpy as np
import os
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.envs.registration import register

# 🛑 SAFETY
th.distributions.Distribution.set_default_validate_args(False)

# ==========================================
# 1. CONFIG
# ==========================================
MODEL_FILE = "ppo_catanatron_4p_10vp.zip" # The model you want to test
STATS_FILE = "vec_normalize_4p.pkl"       # The matching stats file
N_GAMES = 100                             # Number of games to play

# ==========================================
# 2. SETUP HELPER FUNCTIONS
# ==========================================
def register_env():
    CatanatronEnv = None
    try:
        from catanatron_gym.envs import CatanatronEnv
    except ImportError:
        try:
            from catanatron.gym.envs import CatanatronEnv
        except ImportError:
            pass

    if CatanatronEnv is not None and "catanatron/Catanatron-v0" not in gym.envs.registry:
        register(id="catanatron/Catanatron-v0", entry_point=CatanatronEnv, max_episode_steps=1000)

def mask_fn(env: gym.Env) -> np.ndarray:
    env = env.unwrapped
    if hasattr(env, "get_valid_actions"): valid_ids = env.get_valid_actions()
    else: valid_ids = getattr(env, "valid_actions", [])
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid_ids] = True
    return np.array(mask, dtype=bool)

def make_eval_env():
    from catanatron import RandomPlayer, Color
    # 🟢 OPPONENTS: Random Bots (Baseline)
    env_config = {
        "enemies": [RandomPlayer(Color.RED), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)],
        "map_type": "random", 
        "victory_points": 10
    }
    env = gym.make("catanatron/Catanatron-v0", config=env_config)
    env = ActionMasker(env, mask_fn)
    return env

def get_base_env(vec_env):
    env = vec_env
    while hasattr(env, "venv"): env = env.venv
    return env.envs[0]

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================
def main():
    print(f"🔍 Looking for: {MODEL_FILE}")
    if not os.path.exists(MODEL_FILE):
        print("❌ ERROR: Model file not found.")
        return
    if not os.path.exists(STATS_FILE):
        print("❌ ERROR: Stats file (pkl) not found. Evaluation will fail/crash without it.")
        return

    register_env()

    # 1. Load Normalization Stats
    print("📈 Loading Normalization Stats...")
    venv = DummyVecEnv([lambda: make_eval_env()])
    env = VecNormalize.load(STATS_FILE, venv)
    env.training = False     # Do not update stats during eval
    env.norm_reward = False  # We want to see raw game results

    # 2. Load Model
    print("🧠 Loading Neural Network...")
    model = MaskablePPO.load(MODEL_FILE, env=env)
    
    # Access base env to get action masks
    base_env = get_base_env(env)

    print(f"\n🚀 Playing {N_GAMES} Games vs Random Bots...")
    print("-" * 65)
    print(f"{'GAME':<6} | {'STATUS':<10} | {'MOVES':<6} | {'RWD':<6} | {'RESULT'}")
    print("-" * 65)

    wins = 0
    
    for i in range(N_GAMES):
        obs = env.reset()
        dones = np.array([False])
        moves = 0
        final_reward = 0

        while not dones[0]:
            # Get Mask
            single_mask = mask_fn(base_env)
            mask_batch = single_mask[None, :]

            # Predict (Deterministic = Best Move)
            action, _ = model.predict(obs, deterministic=True, action_masks=mask_batch)
            obs, rewards, dones, infos = env.step(action)
            moves += 1

            # Capture reward when game ends (before auto-reset in DummyVecEnv)
            if dones[0]:
                final_reward = rewards[0]

        # Determine Result from reward signal
        # Catanatron returns +1.0 for WIN, -1.0 for LOSS
        if final_reward > 0:
            result = "🏆 WIN"
            wins += 1
        else:
            result = "💀 LOSS"

        status = "🟢 ALIVE" if moves > 12 else "🔴 DEAD"

        # Print row
        print(f"#{i+1:<5} | {status:<10} | {moves:<6} | {final_reward:+.1f} | {result}")

    print("-" * 65)
    print(f"📊 FINAL WIN RATE: {wins}/{N_GAMES} ({wins/N_GAMES*100:.1f}%)")

if __name__ == "__main__":
    main()