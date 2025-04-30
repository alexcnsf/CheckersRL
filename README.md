# CheckersRL

A reinforcement learning implementation for the game of Checkers.

## Setup Instructions

### 1. Environment Setup

First ensure you have Python 3.x installed on your system. Then you can run the commands in your terminal:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Requirements

The project requires several dependencies for deep learning, scientific computing, and data processing. Install them using:

```bash
pip install -r requirements.txt
```

### 3. Running the Training Process

The training process involves two main components:

#### 3.1 Training the Model (`training.py`)

To start the training process:
```bash
python training.py
```

The script supports four training modes, controlled by modifying the `Oppenent` variable in `training.py`:
- `"itself"` - Self-play training
- `"curriculum"` - Adaptive difficulty opponent
- `"minmax"` - Minimax algorithm opponent
- `"random"` - Random move opponent

To switch between modes:
1. Open `training.py`
2. Find this section at the bottom of the file:
```python
if "__main__" == __name__ :
    Oppenent = "curriculum"  # Change this value to your desired mode
    print(Oppenent)
    GetModel(Oppenent=Oppenent)
```
3. Change `"curriculum"` to any of: `"itself"`, `"minmax"`, or `"random"`
4. Save and run `python training.py`

Each mode will:
- Save the model to `models/{mode}aa.keras`
- Generate statistics in `winrates_{mode}.csv`
- Create visualization in `final_results/training/`

#### 3.2 Model Evaluation (`round_robin.ipynb`)

To evaluate trained models:

1. Navigate to and open `round_robin.ipynb`
2. Run all cells in the notebook

The notebook will:
- Load trained models from different checkpoints
- Conduct round-robin tournaments with 100 games between each agent
- Generate performance comparisons
- Visualize results with matplotlib

### Project Structure

- `checkers.py`: Core game implementation with game logic and state management
- `training.py`: Main reinforcement learning training script
- `round_robin.ipynb`: Jupyter notebook for model evaluation
- `requirements.txt`: Complete list of project dependencies