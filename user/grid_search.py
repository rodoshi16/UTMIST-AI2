import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from train_agent import RecurrentPPOAgent, train, TrainLogging, danger_zone_reward, damage_interaction_reward, in_state_reward, on_win_reward, AttackState, RewardManager, RewTerm, SaveHandler, SaveHandlerMode, OpponentsCfg
from functools import partial
from environment.agent import BasedAgent
from environment.environment import CameraResolution
from itertools import product

"""
Grid Search over reward function weights for training an RL agent.
"""

# Create an array to store results of Grid Search training
results = []

# Define search space
search_space = {
    'damage_interaction_reward': [0.5, 1.0],
    'danger_zone_reward': [0.1, 0.5],
    'penalize_attack_reward': [-0.1, -0.04],
}

for dmg, dz, atk in product(
    search_space['damage_interaction_reward'],
    search_space['danger_zone_reward'],
    search_space['penalize_attack_reward']
):
    print(f"\n🚀 Training with dmg={dmg}, dz={dz}, atk={atk}")

    reward_manager = RewardManager(
        {
            'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=dz),
            'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=dmg),
            'penalize_attack_reward': RewTerm(func=in_state_reward, weight=atk, params={'desired_state': AttackState}),
        },
        {
            'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=50)),
        }
    )

    my_agent = RecurrentPPOAgent()
    save_path = f'checkpoints/gridsearch_d{dmg}_z{dz}_a{atk}'
    save_handler = SaveHandler(
        agent=my_agent,
        save_freq=1000,
        save_path=save_path,
        run_name=f'exp_d{dmg}_z{dz}_a{atk}',
        mode=SaveHandlerMode.FORCE
    )

    opponent_cfg = OpponentsCfg(opponents={'based_agent': (1.0, partial(BasedAgent))})

    # Run training
    try:
        train(
            my_agent,
            reward_manager,
            save_handler,
            opponent_cfg,
            CameraResolution.LOW,
            train_timesteps=2000,
            train_logging=TrainLogging.PLOT
        )

        # Load training log (usually saved as monitor.csv)
        import os
        log_path = os.path.join(save_path, "monitor.csv")
        if os.path.exists(log_path):
            df = pd.read_csv(log_path, skiprows=1)
            mean_reward = df['r'].mean()  # average episode reward
        else:
            mean_reward = None

        results.append({
            "damage_interaction_reward": dmg,
            "danger_zone_reward": dz,
            "penalize_attack_reward": atk,
            "mean_reward": mean_reward,
            "checkpoint": save_path
        })
    except Exception as e:
        print(f"❌ Failed for dmg={dmg}, dz={dz}, atk={atk}: {e}")

# Save results to CSV for inspection
pd.DataFrame(results).to_csv("gridsearch_results.csv", index=False)

# Print the best one
best = max(results, key=lambda x: x["mean_reward"] or float('-inf'))
print("\n🏆 Best configuration:")
print(best)