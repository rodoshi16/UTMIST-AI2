import os
import optuna
import pandas as pd
from functools import partial

from train_agent import (
    RecurrentPPOAgent, train, TrainLogging,
    danger_zone_reward, damage_interaction_reward, in_state_reward, 
    holding_more_than_3_kets, on_win_reward, AttackState,
    RewardManager, RewTerm, SaveHandler, SaveHandlerMode, OpponentsCfg
)
from environment.agent import BasedAgent
from environment.environment import CameraResolution


def objective(trial):
    """
    Defines one optimization trial.
    Each trial uses a unique combination of reward weights.
    """
    # --- Suggest weights for each reward term ---
    dmg = trial.suggest_float("damage_interaction_reward", 0.5, 2.0)
    dz = trial.suggest_float("danger_zone_reward", 0.05, 0.5)
    atk = trial.suggest_float("penalize_attack_reward", -0.15, -0.02)
    hold = trial.suggest_float("holding_more_than_3_kets", -0.5, 0.5)

    # --- Build Reward Manager ---
    reward_manager = RewardManager(
        {
            'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=dz),
            'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=dmg),
            'penalize_attack_reward': RewTerm(
                func=in_state_reward, 
                weight=atk, 
                params={'desired_state': AttackState}
            ),
            'holding_more_than_3_kets': RewTerm(
                func=holding_more_than_3_kets, 
                weight=hold
            ),
        },
        {
            'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
        }
    )

    # --- Initialize agent, save handler, and opponent ---
    my_agent = RecurrentPPOAgent()
    save_path = f"checkpoints/optuna_trial_{trial.number}"
    os.makedirs(save_path, exist_ok=True)

    save_handler = SaveHandler(
        agent=my_agent,
        save_freq=10_000,
        save_path=save_path,
        run_name=f"optuna_trial_{trial.number}",
        mode=SaveHandlerMode.FORCE,
    )

    opponent_cfg = OpponentsCfg(opponents={'based_agent': (1.0, partial(BasedAgent))})

    # --- Train briefly (for evaluation) ---
    try:
        train(
            my_agent,
            reward_manager,
            save_handler,
            opponent_cfg,
            CameraResolution.LOW,
            train_timesteps=100_000,
            train_logging=TrainLogging.NONE
        )
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        return -999  # Penalize crashed runs

    # --- Load training rewards ---
    log_path = os.path.join(save_path, "monitor.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path, skiprows=1)
        mean_reward = df['r'].mean()
        print(f"Trial {trial.number}: mean_reward={mean_reward:.2f}")
        return mean_reward
    else:
        print(f"❌ Trial {trial.number} failed: log file not found.")
        return -999  # Penalize missing logs