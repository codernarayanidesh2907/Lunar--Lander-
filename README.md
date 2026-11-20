# 🌕 Lunar Lander — Reinforcement Learning Dashboard

An interactive Lunar Lander simulation powered by **Reinforcement Learning (PPO)**, featuring a real-time desktop dashboard built with Python and Tkinter.

The project uses a trained PPO agent to control a lunar lander in the Gymnasium environment while visualizing simulation activity and telemetry data.

## 🚀 Features

- **PPO Agent:** Uses Proximal Policy Optimization from Stable-Baselines3.
- **Model Evaluation:** Evaluates the performance of the trained PPO agent.
- **Live Simulation:** Displays the actual Lunar Lander environment in the dashboard.
- **Real-Time Telemetry:** Tracks altitude, vertical speed, and horizontal speed.
- **Live Graph:** Visualizes telemetry changes throughout the simulation.
- **Desktop GUI:** Built using Tkinter with a space-tech interface.
- **Pre-trained Model:** Includes a trained PPO model for evaluation.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Gymnasium | Lunar Lander environment |
| Stable-Baselines3 | PPO reinforcement learning |
| NumPy | Numerical operations |
| Matplotlib | Telemetry visualization |
| Tkinter | Desktop GUI |
| Pillow | Simulation image rendering |
| Box2D | Physics simulation |

## 📁 Project Structure

```text
Lunar-Lander-RL/
│
├── dashboard.py
├── evaluate.py
├── requirements.txt
├── .gitignore
│
└── models/
    └── lunar_lander_ppo500K.zip
```

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/Lunar-Lander-RL.git
cd Lunar-Lander-RL
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment (Windows)**

```bash
.venv\Scripts\activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

## 🎮 Run the Dashboard

```bash
python dashboard.py
```

Click **START SIMULATION** to launch the PPO-controlled Lunar Lander.

## 📊 Dashboard

The dashboard provides:
- Live Lunar Lander visualization
- Altitude telemetry
- Vertical and horizontal speed readings
- Real-time telemetry graph

<img width="1903" height="977" alt="image" src="https://github.com/user-attachments/assets/64cc5291-5312-47ab-b132-9305923237bc" />

<!-- Save your screenshot as screenshots/dashboard.png -->

## 🧠 Reinforcement Learning

The project uses **Proximal Policy Optimization (PPO)**, a policy-gradient reinforcement learning algorithm.

The trained agent interacts with the Lunar Lander environment, selects actions, and attempts to achieve a successful landing.



## 👩‍💻 Author : Narayani Deshpande 

