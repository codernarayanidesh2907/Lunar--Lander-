
import tkinter as tk
import gymnasium as gym
from stable_baselines3 import PPO
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# Load trained PPO model
MODEL_PATH = "models/lunar_lander_ppo500K.zip"

model = PPO.load(MODEL_PATH)

# Create Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

print("PPO Model Loaded Successfully!")

# Main window
root = tk.Tk()
root.title("Lunar Lander | RL Dashboard")
root.geometry("1200x850")
root.configure(bg="#0B1220")

# Header
header = tk.Frame(root, bg="#0B1220")
header.pack(fill="x", padx=25, pady=20)

title = tk.Label(
    header,
    text="LUNAR LANDER",
    font=("Orbitron", 24, "bold"),
    fg="#22D3EE",
    bg="#0B1220",
    justify="center"
)
title.pack(fill="x", pady=5)




# Main content area
content = tk.Frame(root, bg="#111C2E")
content.pack(fill="both", expand=True, padx=25, pady=10)

welcome = tk.Label(
    content,
    text=" TELEMETRY DATA",
    font=("Orbitron", 16, "bold"),
    fg="#E2E8F0",
    bg="#111C2E"
)
welcome.pack(pady=30)

status = tk.Label(
    content,
    text="SYSTEM READY  •  AWAITING START",
    font=("Orbitron", 11),
    fg="#22D3EE",
    bg="#111C2E"
)
status.pack(pady=10)


# Telemetry Cards
telemetry_frame = tk.Frame(content, bg="#111C2E")
telemetry_frame.pack(fill="x", padx=35, pady=35)

def create_card(parent, title, value, unit, column):
    card = tk.Frame(
        parent,
        bg="#17263A",
        padx=20,
        pady=20
    )
    card.grid(row=0, column=column, padx=12, sticky="nsew")

    heading = tk.Label(
        card,
        text=title,
        font=("Segoe UI", 11, "bold"),
        fg="#94A3B8",
        bg="#17263A"
    )
    heading.pack(pady=5)

    reading = tk.Label(
        card,
        text=value,
        font=("Segoe UI", 25, "bold"),
        fg="#22D3EE",
        bg="#17263A"
    )
    reading.pack(pady=5)

    unit_label = tk.Label(
        card,
        text=unit,
        font=("Segoe UI", 10),
        fg="#94A3B8",
        bg="#17263A"
    )
    unit_label.pack()

    return reading

# Make all three cards equal in width
for i in range(3):
    telemetry_frame.grid_columnconfigure(i, weight=1)

altitude_value = create_card(
    telemetry_frame, "ALTITUDE", "0.00", "meters", 0
)

vertical_speed_value = create_card(
    telemetry_frame, "VERTICAL SPEED", "0.00", "m/s", 1
)

horizontal_speed_value = create_card(
    telemetry_frame, "HORIZONTAL SPEED", "0.00", "m/s", 2
)
# Lower Dashboard Area
lower_frame = tk.Frame(content, bg="#111C2E")
lower_frame.pack(fill="both", expand=True, padx=15, pady=5)

lower_frame.grid_columnconfigure(0, weight=1)
lower_frame.grid_columnconfigure(1, weight=1)
lower_frame.grid_rowconfigure(0, weight=1)


# Actual Lunar Lander Display
simulation_frame = tk.Frame(
    lower_frame,
    bg="#17263A",
    padx=10,
    pady=10
)

simulation_title = tk.Label(
    simulation_frame,
    text="LIVE LANDER SIMULATION",
    font=("Segoe UI", 12, "bold"),
    fg="#22D3EE",
    bg="#17263A"
)
simulation_frame.grid(
    row=0, column=0, sticky="nsew", padx=8, pady=5
)
simulation_title.pack(pady=5)

# Image display area
lander_display = tk.Label(
    simulation_frame,
    bg="#0B1220"
)
lander_display.pack(pady=5)

# Live Telemetry Graph
graph_frame = tk.Frame(lower_frame, bg="#111C2E")

graph_frame.grid(
    row=0, column=1, sticky="nsew", padx=8, pady=5
)

# Create Matplotlib figure
fig = Figure(figsize=(5, 3.2), dpi=100)
fig.subplots_adjust(bottom=0.22)
fig.patch.set_facecolor("#111C2E")

ax = fig.add_subplot(111)
ax.set_facecolor("#17263A")

# Graph styling
ax.set_title(
    "LIVE TELEMETRY",
    color="#E2E8F0",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel("Simulation Steps", color="#94A3B8")
ax.set_ylabel("Normalized Value", color="#94A3B8")

ax.tick_params(colors="#94A3B8")
ax.grid(color="#334155", linestyle="--", alpha=0.5)

for spine in ax.spines.values():
    spine.set_color("#334155")

# Prepare three graph lines
altitude_line, = ax.plot([], [], color="#22D3EE", label="Altitude")
vertical_line, = ax.plot([], [], color="#3B82F6", label="Vertical Speed")
horizontal_line, = ax.plot([], [], color="#FBBF24", label="Horizontal Speed")

ax.legend(
    facecolor="#17263A",
    labelcolor="#E2E8F0",
    loc="upper right"
)

# Embed graph in Tkinter
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

# Store telemetry history
steps = []
altitude_history = []
vertical_history = []
horizontal_history = []



# Simulation Control
simulation_running = False
observation = None

def start_simulation():
    global simulation_running, observation

    if simulation_running:
        return

    observation, info = env.reset()
    simulation_running = True

    status.config(
        text="SIMULATION RUNNING  •  PPO AGENT ACTIVE"
    )

    update_simulation()


def update_simulation():
    global observation, simulation_running

    if not simulation_running:
        return

    # PPO agent predicts the next action
    action, _ = model.predict(
        observation,
        deterministic=True
    )

    # Take one simulation step
    observation, reward, terminated, truncated, info = env.step(action)


    # Get actual Lunar Lander game frame
    frame = env.render()
    print(frame.shape)

    # Convert frame into a Tkinter-compatible image
    game_image = Image.fromarray(frame)
    game_image = game_image.resize(
    (500, 333),
    Image.Resampling.LANCZOS
)
    

    photo = ImageTk.PhotoImage(game_image)
    print("Image created successfully!")

    # Display the live game frame
    lander_display.config(image=photo)
    print("Display size:", lander_display.winfo_width(), lander_display.winfo_height())
    print("Image sent to Tkinter")

    lander_display.image = photo
   
    # Update telemetry cards
    altitude_value.config(
        text=f"{observation[1]:.3f}"
    )

    horizontal_speed_value.config(
        text=f"{observation[2]:.3f}"
    )

    vertical_speed_value.config(
        text=f"{observation[3]:.3f}"
    )
    
    # Store live telemetry data
    steps.append(len(steps))
    altitude_history.append(float(observation[1]))
    vertical_history.append(float(observation[3]))
    horizontal_history.append(float(observation[2]))

    # Update graph lines
    altitude_line.set_data(steps, altitude_history)
    vertical_line.set_data(steps, vertical_history)
    horizontal_line.set_data(steps, horizontal_history)

    # Automatically adjust graph axes
    ax.relim()
    ax.autoscale_view()

    # Refresh graph
    canvas.draw_idle()


    # Check whether the episode has ended
    if terminated or truncated:
        simulation_running = False
        status.config(
            text="EPISODE COMPLETED  •  PRESS START TO RESTART"
        )
        return

    # Schedule the next simulation step
    root.after(50, update_simulation)


# Start Button
start_button = tk.Button(
    content,
    text="START SIMULATION",
    font=("Segoe UI", 12, "bold"),
    bg="#22D3EE",
    fg="#0B1220",
    activebackground="#67E8F9",
    relief="flat",
    padx=25,
    pady=12,
    cursor="hand2",
    command=start_simulation
)

start_button.pack(before=lower_frame, pady=10)


root.update_idletasks()
# Start the application
root.mainloop()
