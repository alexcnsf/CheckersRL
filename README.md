# Checkers Reinforcement Learning

This project implements a Checkers game environment with reinforcement learning capabilities. It includes various AI agents (random, minmax) and provides features for training and evaluating AI models.

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

### Adding New Agents

1. Implement your agent class
2. Add it to the available agents in `playRandomMinMax`
3. Test against existing agents

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

## License

[Add your license information here]

## Acknowledgments

[Add any acknowledgments or references here]




