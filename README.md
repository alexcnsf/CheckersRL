# CheckersRL

A reinforcement learning implementation for the game of Checkers.

## Setup Instructions

### 1. Environment Setup

First, ensure you have Python 3.x installed on your system. Then:

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

If you encounter any issues with specific packages:
- For TensorFlow: Make sure you have compatible CUDA drivers if using GPU
- For PyTorch: The installation might need to be adjusted based on your CUDA version
- For Jupyter: Additional kernel setup might be needed

### 3. Running the Training Process

The training process involves two main components:

#### 3.1 Training the Model (`training.py`)

To start the training process:
```bash
python training.py
```

This will:
- Initialize the checkers environment
- Create a neural network model for Q-learning
- Train the model through self-play
- Save training progress and model weights
- Generate performance plots

The training script will automatically:
- Save model weights periodically
- Generate training statistics in CSV format
- Create performance plots showing win rates and rewards

#### 3.2 Model Evaluation (`round_robin.ipynb`)

To evaluate trained models:

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to and open `round_robin.ipynb`
3. Run all cells in the notebook

The notebook will:
- Load trained models from different checkpoints
- Conduct round-robin tournaments
- Generate performance comparisons
- Visualize results with matplotlib

### Project Structure

- `checkers.py`: Core game implementation with game logic and state management
- `training.py`: Main reinforcement learning training script
- `round_robin.ipynb`: Jupyter notebook for model evaluation
- `requirements.txt`: Complete list of project dependencies

### Dependencies Overview

The project uses several key packages:

#### Core Scientific Computing
- numpy, scipy, pandas: For numerical computations and data handling
- matplotlib: For visualization
- scikit-learn: For machine learning utilities

#### Deep Learning
- tensorflow & keras: Primary deep learning framework
- torch, torchvision, torchaudio: PyTorch support
- tensorboard: For training visualization

#### Development Tools
- Jupyter ecosystem: For interactive development and visualization
- tqdm: For progress tracking
- rich: For enhanced terminal output

### Notes

- Training duration depends on your hardware capabilities
- GPU acceleration is recommended for faster training
- Models are saved periodically during training
- Hyperparameters can be adjusted in `training.py`
- The round-robin evaluation helps track improvement across training iterations

### Troubleshooting

If you encounter any issues:
1. Ensure your virtual environment is activated
2. Verify all dependencies are correctly installed
3. Check Python and CUDA versions if using GPU
4. Make sure you have sufficient disk space for model checkpoints

## Contributing

Feel free to submit issues and enhancement requests!

## Features

- 10x10 Checkers board implementation
- Multiple AI agents:
  - Random agent
  - MinMax agent
  - Curriculum learning agent
- Game state features extraction for machine learning
- Board compression for efficient state representation
- Support for both regular pieces and queens
- Score tracking and game state management

## Prerequisites

- Python 3.x
- pip (Python package installer)
- virtualenv (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd checkers_rl2
```

2. Create and activate a virtual environment:
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

- `checkers.py`: Main game implementation
  - Checkers game logic
  - AI agents
  - State management
  - Feature extraction

## Usage

### Basic Game Play

```python
from checkers import Checkers

# Initialize a new game
game = Checkers()

# Play a random game
result = game.playrandom(verbose=True)

# Play with specific agents
result = game.playRandomMinMax(Bot1="random", Bot2="minmax", verbose=True)
```

### Available Agents

1. Random Agent (`"random"`)
   - Makes random valid moves
   - Good for baseline testing

2. MinMax Agent (`"minmax"`)
   - Uses minmax algorithm for decision making
   - Can be configured with different depths

3. Curriculum Agent (`"curriculum"`)
   - Adapts its strategy based on performance
   - Gradually increases difficulty

### Feature Extraction

The game provides several features for machine learning:
- Board state
- Piece counts
- Position-based features
- Game metrics

```python
# Get features for a specific player
features = game.GetFeatures(player=1, verbose=True)

# Compress board state
compressed_board = game.CompressBoard(player=1, board=game.board)
```

## Development

### Running Tests

```python
# Run the main game
python checkers.py
```


## Dependencies

The project uses several key packages:
- numpy: Numerical computations
- tensorflow/keras: Deep learning
- scikit-learn: Machine learning utilities
- torch: PyTorch for additional ML capabilities
- matplotlib: Visualization

All dependencies are listed in `requirements.txt`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request




