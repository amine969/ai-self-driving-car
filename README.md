# Self-Driving Car Simulation using Genetic Algorithm & Neural Networks 🏎️🧠

This project is a **2D Self-Driving Car Simulation** built from scratch in Python using **Pygame** and **NumPy**. It demonstrates how an Artificial Intelligence (AI) can learn to navigate a race track autonomously without any human intervention or pre-programmed pathing.

The cars are driven by a custom **Neural Network** acting as the "brain," and the learning process is optimized over multiple generations using a **Genetic Algorithm (Evolutionary AI)**.

---

## 🚀 Key Features

* **Custom Neural Network Architecture:** Built purely with NumPy (no heavy frameworks like TensorFlow/PyTorch), featuring dense layers and ReLU activation functions.
* **Ray-Casting Sensor System:** Each car simulates 5 proximity sensors (angles: -90°, -45°, 0°, 45°, 90°) to perceive its distance from track boundaries in real-time.
* **Genetic Algorithm Optimization:** Cars that travel furthest and fastest gain higher *fitness*. The best-performing network is selected to breed the next generation with slight random mutations.
* **Persistent Memory:** Includes automatic checkpoint saving (`brain.pkl`). The simulation saves the best-performing brain so it can resume learning or showcase its peak performance instantly upon restart.
* **Precise Mask Collisions:** Uses Pygame's pixel-perfect mask system to handle precise track boundary collisions.

---

## 🛠️ How It Works (The AI Logic)

1. **Perception (Inputs):** The car casts 5 rays to measure distances to walls. These distances, combined with the car's current velocity, form a **6-dimensional input vector**.
2. **Decision Making (Brain):** The input is fed into a 3-layer neural network:
   * `Input Layer` (6) ➔ `Hidden Layer 1` (5) with ReLU
   * `Hidden Layer 1` (5) ➔ `Hidden Layer 2` (4) with ReLU
   * `Hidden Layer 2` (4) ➔ `Output Layer` (3)
3. **Action (Outputs):** The 3 output nodes determine whether the car should **Turn Left**, **Turn Right**, or **Accelerate**.
4. **Evolution:** If a car crashes or the generation timer expires, the generation ends. The top car’s weights and biases are copied, mutated slightly by adding Gaussian noise, and used to repopulate the next generation of 100 cars.

---

## 📁 Project Structure

* `car.py` - Contains the Pygame environment, game loop, ray-casting logic, and the evolutionary algorithm logic.
* `network.py` - Houses the custom Neural Network layers (`layer_Dense`), activation functions (`Activation_ReLU`), and forward propagation logic.
* `utils.py` - (Dependency) Helper functions for image scaling and center-based rotation adjustments.
* `imgs/` - Directory holding the visual assets (car sprites and track layouts).

---

## 💻 Installation & Setup

### Prerequisites
Make sure you have Python 3.x installed along with the required libraries:
```bash
pip install pygame numpy
```

### Running the Simulation
1. Clone this repository:
   ```bash
   git clone https://github.com
   cd YOUR_REPOSITORY_NAME
   ```
2. Ensure your track image is named `track9.png` inside an `imgs` folder.
3. Run the main script:
   ```bash
   python car.py
   ```

---

## 📈 Future Improvements / Roadmap
* [ ] Add a dynamic UI overlay to display current top fitness, live neural network weights visualizer, and mutation rate.
* [ ] Implement more complex track layouts to test the AI's adaptability.
* [ ] Optimize the ray-casting algorithm to support larger populations smoothly.
