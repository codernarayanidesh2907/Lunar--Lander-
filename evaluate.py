


import gymnasium as gym
from stable_baselines3 import PPO
import csv

# Load trained model
model = PPO.load("models/lunar_lander_ppo500K")

# Create environment
env = gym.make("LunarLander-v3", render_mode="human")

# Evaluation settings
total_episodes = 30

safe_landings = 0
crashes = 0
total_rewards = 0

# Create CSV file
with open("evaluation_results.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Episode",
        "Reward",
        "Status",
        "X_Position",
        "Y_Position"
    ])

    print("\nStarting AI Evaluation...\n")

    for episode in range(total_episodes):

        observation, info = env.reset()
        episode_reward = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        # Check final landing state
        base_env = env.unwrapped

        safe = not base_env.game_over and not base_env.lander.awake

        if safe:
            safe_landings += 1
            status = "SAFE LANDING"
        else:
            crashes += 1
            status = "UNSUCCESSFUL"

        # Record final position
        x_position = float(base_env.lander.position.x)
        y_position = float(base_env.lander.position.y)

        total_rewards += episode_reward

        # Write episode data into CSV
        writer.writerow([
            episode + 1,
            round(episode_reward, 2),
            status,
            round(x_position, 2),
            round(y_position, 2)
        ])

        print(
            f"Episode {episode + 1}: "
            f"{status} | Reward: {episode_reward:.2f}"
        )

env.close()

# Final report
print("\n========== EVALUATION REPORT ==========")

print(f"Total Episodes: {total_episodes}")
print(f"Safe Landings: {safe_landings}")
print(f"Unsuccessful: {crashes}")

print(
    f"Safe Landing Rate: "
    f"{safe_landings / total_episodes * 100:.1f}%"
)

print(f"Average Reward: {total_rewards / total_episodes:.2f}")

print("\nResults saved to evaluation_results.csv")
print("=======================================")
