import os
import shutil
import gymnasium as gym
import numpy as np
import torch as th
import time
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from catanatron.models.player import Player
from catanatron.models.enums import ActionType

try: from catanatron.game import Action
except ImportError: from catanatron.models import Action

# 🛑 SAFETY
th.distributions.Distribution.set_default_validate_args(False)

# ==========================================
# 📝 LOGGING WRAPPER (Now tracks Opponent Version)
# ==========================================
class GameLoggerWrapper(gym.Wrapper):
    def __init__(self, env, log_file="catan_scores.txt"):
        super().__init__(env)
        self.log_file = log_file
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("Timestamp,P0_VP,P1_VP,P2_VP,P3_VP,Winner,Opp_Gen\n")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            try:
                state = self.env.unwrapped.game.state.player_state
                p0 = state.get('P0_ACTUAL_VICTORY_POINTS', 0)
                p1 = state.get('P1_ACTUAL_VICTORY_POINTS', 0)
                p2 = state.get('P2_ACTUAL_VICTORY_POINTS', 0)
                p3 = state.get('P3_ACTUAL_VICTORY_POINTS', 0)
                scores = [p0, p1, p2, p3]
                winner_idx = scores.index(max(scores))
                winner_name = "Agent" if winner_idx == 0 else f"Opponent_{winner_idx}"
                
                # 🟢 DEBUG: Get Opponent Generation
                opp_gen = "Unknown"
                try:
                    # Iterate to find a PPOOpponent that isn't us
                    players = []
                    if hasattr(self.env.unwrapped, 'players'): players = self.env.unwrapped.players
                    elif hasattr(self.env.unwrapped, 'game'): players = self.env.unwrapped.game.players
                    
                    for p in players:
                        if isinstance(p, PPOOpponent) and p.color != self.env.unwrapped.p0.color:
                            opp_gen = str(p.generation)
                            break
                except: pass

                # Print to terminal
                icon = "🏆" if winner_idx == 0 else "💀"
                # print(f"{icon} Game Over [OppGen: v{opp_gen}]: Agent={p0} | Opps=[{p1}, {p2}, {p3}]")
                
                with open(self.log_file, "a") as f:
                    timestamp = int(time.time())
                    f.write(f"{timestamp},{p0},{p1},{p2},{p3},{winner_name},{opp_gen}\n")
            except Exception as e:
                print(f"Logging error: {e}")
        return obs, reward, terminated, truncated, info

# ==========================================
# 💎 REWARD SHAPING
# ==========================================
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_vp = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_vp = self._get_vp()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_vp = self._get_vp()
        
        vp_diff = current_vp - self.last_vp
        if vp_diff > 0:
            reward += vp_diff * 0.1
            self.last_vp = current_vp
            
        if (terminated or truncated) and current_vp >= 10:
            reward += 20.0 
            
        reward -= 0.001
        return obs, reward, terminated, truncated, info

    def _get_vp(self):
        try: return self.env.unwrapped.game.state.player_state.get('P0_ACTUAL_VICTORY_POINTS', 0)
        except: return 0

# ==========================================
# ⚡ MASK & HELPER
# ==========================================
def mask_fn(env: gym.Env) -> np.ndarray:
    env = env.unwrapped
    if hasattr(env, "get_valid_actions"): valid_ids = env.get_valid_actions()
    else: valid_ids = getattr(env, "valid_actions", [])
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid_ids] = True
    return np.array(mask, dtype=bool)

def normalize_action(action):
    normalized = action
    if normalized.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.MOVE_ROBBER:
        val = action.value
        if isinstance(val, tuple) and len(val) == 3: 
            return Action(action.color, action.action_type, val[0])
        return Action(action.color, action.action_type, val)
    elif normalized.action_type == ActionType.BUILD_ROAD:
        return Action(action.color, action.action_type, tuple(sorted(action.value)))
    elif normalized.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.DISCARD:
        return Action(action.color, action.action_type, None)
    return normalized

# ==========================================
# 🤖 PPO OPPONENT (Debuggable)
# ==========================================
class PPOOpponent(Player):
    def __init__(self, color, model_path, env_config=None):
        super().__init__(color)
        self.model_path = model_path
        self.stats_path = "vec_normalize_selfplay.pkl"
        self.generation = 0 
        
        dummy_config = env_config.copy() if env_config else {}
        dummy_config["map_type"] = "random"
        from catanatron import RandomPlayer, Color
        dummy_config["enemies"] = [RandomPlayer(Color.RED), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)]
        
        self.env = gym.make("catanatron/Catanatron-v0", config=dummy_config)
        self._load_actions_array()
        self.load_model_and_stats()

    def _load_actions_array(self):
        self.action_list = None
        try:
            func_get_valid = self.env.unwrapped.get_valid_actions
            to_action_space_func = func_get_valid.__globals__.get('to_action_space')
            if to_action_space_func:
                vals = to_action_space_func.__globals__
                if 'ACTIONS_ARRAY' in vals: self.action_list = vals['ACTIONS_ARRAY']
        except: pass
        if self.action_list is None: raise ValueError("CRITICAL: Could not locate ACTIONS_ARRAY.")

    def load_model_and_stats(self):
        # 🟢 REMOVED SILENT TRY/EXCEPT.
        try:
            self.model = MaskablePPO.load(self.model_path)
            if os.path.exists(self.stats_path):
                self.venv = DummyVecEnv([lambda: self.env])
                self.venv = VecNormalize.load(self.stats_path, self.venv)
                self.venv.training = False
                self.venv.norm_reward = False
            else: self.venv = None
        except Exception as e:
            print(f"⚠️ OPPONENT LOAD FAILED: {e}")
            pass

    def reload_model(self, new_path):
        self.model_path = new_path
        self.load_model_and_stats()
        self.generation += 1 
        print(f"   ✅ [PPOOpponent {self.color}] UPDATED to v{self.generation}")

    def decide(self, game, playable_actions):
        try:
            self.env.unwrapped.game = game.copy()
            self.env.unwrapped.p0.color = self.color
            
            raw_obs = self.env.unwrapped._get_observation()
            obs = np.array([raw_obs])
            if self.venv: obs = self.venv.normalize_obs(obs)

            valid_ids = self.env.unwrapped.get_valid_actions()
            mask = np.zeros(self.env.action_space.n, dtype=bool)
            mask[valid_ids] = True
            
            action_idx, _ = self.model.predict(obs, action_masks=[mask], deterministic=True)
            action_idx = action_idx[0] if self.venv else action_idx
            
            (target_type, target_value) = self.action_list[action_idx]
            for real_action in playable_actions:
                normalized = normalize_action(real_action)
                if normalized.action_type == target_type and normalized.value == target_value:
                    return real_action
        except: pass
        return playable_actions[0]

# ==========================================
# ⚔️ GATED SELF-PLAY CALLBACK
# ==========================================
class GatedSelfPlayCallback(BaseCallback):
    def __init__(self, check_freq=50_000, win_threshold=0.28, 
                 model_path="self_play_model.zip", 
                 latest_save_path="ppo_selfplay_latest.zip",
                 stats_path="vec_normalize_selfplay.pkl", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.win_threshold = win_threshold
        self.model_path = model_path
        self.latest_save_path = latest_save_path
        self.stats_path = stats_path
        self.last_check_step = 0
        self.base_eval_env = None 
        self.eval_env = None

    def _init_callback(self) -> None:
        self.base_eval_env = DummyVecEnv([make_env])

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_check_step >= self.check_freq:
            self.last_check_step = self.num_timesteps
            
            self.model.save(self.latest_save_path)
            if self.training_env is not None:
                self.training_env.save(self.stats_path)

            print(f"\n⚔️  EVALUATION (Step {self.num_timesteps}): Testing Agent vs Current Opponents...")
            
            if self.training_env is not None:
                self.eval_env = VecNormalize.load(self.stats_path, self.base_eval_env)
                self.eval_env.training = False 
                self.eval_env.norm_reward = False 
            else:
                self.eval_env = self.base_eval_env
            
            n_eval_episodes = 100
            wins = 0
            
            for _ in range(n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                while not done:
                    base_gym_env = self.base_eval_env.envs[0]
                    valid_ids = base_gym_env.unwrapped.get_valid_actions()
                    mask = np.zeros(base_gym_env.action_space.n, dtype=bool)
                    mask[valid_ids] = True
                    mask_batch = mask[None, :] 

                    action, _ = self.model.predict(obs, action_masks=mask_batch, deterministic=True)
                    obs, rewards, dones, infos = self.eval_env.step(action)
                    done = dones[0]
                    
                    if float(rewards[0]) > 10.0: 
                        wins += 1

            win_rate = wins / n_eval_episodes
            print(f"   📊 Result: {wins}/{n_eval_episodes} Wins (Win Rate: {win_rate:.2%})")

            if win_rate >= self.win_threshold:
                print(f"   ✅ SUCCESS! Agent is stronger. UPDATING OPPONENTS.")
                
                # 1. Save new model
                self.model.save(self.model_path) 
                
                # 🟢 WAIT FOR DISK I/O (Crucial fix for multiprocess race condition)
                time.sleep(2.0)
                
                # 2. Reload Opponents
                self.training_env.env_method("update_opponents", self.model_path)
                self.base_eval_env.env_method("update_opponents", self.model_path)
            else:
                print(f"   ❌ FAILED. Keeping current opponents.")

        return True

class SelfPlayEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def update_opponents(self, model_path):
        # 🟢 UPDATED LOGIC: Update ANY player found in the structure
        players_to_update = []
        base = self.env.unwrapped
        
        # Collect from all possible locations
        if hasattr(base, 'players'): players_to_update.extend(base.players)
        if hasattr(base, 'game') and hasattr(base.game, 'players'): players_to_update.extend(base.game.players)
        if hasattr(base, 'enemies'): players_to_update.extend(base.enemies)
        if hasattr(base, 'config') and 'enemies' in base.config: players_to_update.extend(base.config['enemies'])
        
        # Deduplicate by ID
        seen_ids = set()
        updated_count = 0
        
        for p in players_to_update:
            if id(p) in seen_ids: continue
            seen_ids.add(id(p))
            
            if isinstance(p, PPOOpponent):
                p.reload_model(model_path)
                updated_count += 1
        
        print(f"[SelfPlayWrapper] Updated {updated_count} unique opponents in this environment.")

# ==========================================
# ⚙️ CONFIG & MAIN
# ==========================================
ENV_CONFIG = {"victory_points": 10}

def make_env():
    import gymnasium as gym
    from gymnasium.envs.registration import register
    try: from catanatron_gym.envs import CatanatronEnv
    except ImportError:
        try: from catanatron.gym.envs import CatanatronEnv
        except ImportError: pass

    if "catanatron/Catanatron-v0" not in gym.envs.registry:
        register(id="catanatron/Catanatron-v0", entry_point=CatanatronEnv, max_episode_steps=1000)

    opponent_path = "self_play_model.zip"
    if not os.path.exists(opponent_path):
        if os.path.exists("ppo_catanatron_4p_10vp.zip"):
            shutil.copy("ppo_catanatron_4p_10vp.zip", opponent_path)

    from catanatron import Color
    enemies = [
        PPOOpponent(Color.RED, opponent_path, env_config=ENV_CONFIG),
        PPOOpponent(Color.ORANGE, opponent_path, env_config=ENV_CONFIG),
        PPOOpponent(Color.WHITE, opponent_path, env_config=ENV_CONFIG)
    ]

    full_config = ENV_CONFIG.copy()
    full_config["enemies"] = enemies
    
    env = gym.make("catanatron/Catanatron-v0", config=full_config)
    env = RewardShapingWrapper(env)
    env = GameLoggerWrapper(env, log_file="catan_scores.txt")
    env = Monitor(env)
    env = SelfPlayEnvWrapper(env)
    env = ActionMasker(env, mask_fn)
    return env

def train_self_play():
    n_envs = 6 
    print(f"🚀 Starting Gated Self-Play Training (Threshold: 28% Wins)")
    
    if not os.path.exists("ppo_catanatron_4p_10vp.zip"):
        print("❌ CRITICAL: Need 'ppo_catanatron_4p_10vp.zip' to start!")
        return
    
    shutil.copy("ppo_catanatron_4p_10vp.zip", "self_play_model.zip")
    if os.path.exists("vec_normalize_4p.pkl") and not os.path.exists("vec_normalize_selfplay.pkl"):
        shutil.copy("vec_normalize_4p.pkl", "vec_normalize_selfplay.pkl")

    venv = SubprocVecEnv([make_env for _ in range(n_envs)]) 
    
    if os.path.exists("vec_normalize_selfplay.pkl"):
        print("📈 Loading self-play stats...")
        env = VecNormalize.load("vec_normalize_selfplay.pkl", venv)
        env.training = True
        env.norm_reward = True
        env.clip_obs = 5.0
        env.clip_reward = 20.0
    else:
        env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=20.0)

    print("🧠 Loading Agent...")
    custom_objects = {
        "learning_rate": 1e-4, 
        "clip_range": 0.2,
        "ent_coef": 0.01
    }
    model = MaskablePPO.load("ppo_catanatron_4p_10vp.zip", env=env, custom_objects=custom_objects)

    self_play_cb = GatedSelfPlayCallback(
        check_freq=50_000,       
        win_threshold=0.28,      
        model_path="self_play_model.zip",
        latest_save_path="ppo_selfplay_latest.zip"
    )
    checkpoint_cb = CheckpointCallback(save_freq=100_000, save_path='./checkpoints/', name_prefix='ppo_selfplay')

    try:
        model.learn(total_timesteps=5_000_000, callback=[self_play_cb, checkpoint_cb], reset_num_timesteps=False)
        model.save("ppo_catanatron_selfplay_final")
        env.save("vec_normalize_selfplay.pkl")
    except KeyboardInterrupt:
        print("🛑 Interrupted.")
        model.save("ppo_catanatron_selfplay_final")
        env.save("vec_normalize_selfplay.pkl")

if __name__ == "__main__":
    train_self_play()