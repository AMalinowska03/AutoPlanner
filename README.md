# _Automation of personalized task planning taking into account fatigue and circadian rhythm_ | Master's Thesis Project

This project compares **Reinforcement Learning** vs. **NSGA** for zero-configuration, personalized task planning.

Instead of asking users to define their productivity patterns, the system automatically infers working habits, fatigue, and circadian rhythms from observed behavior. Simulated users with varying behavioral profiles are used to evaluate:

1. **Personalization Accuracy:** Adaptation to implicit user habits.
2. **Efficiency & Utility:** Reduction in total execution time and energy drain.
3. **Dynamic Adaptability:** Real-time plan adjustment in response to unexpected schedule disruptions.

[Jump to Installation](#installation)

## **Structure**

- [**Models**](./models) - implementation of planner models
- [**Simulation**](./simulation) - variables for user profiles and simulating their task execution
- [**Experiments**](./experiments) - core of the project, running experiment logic
- [**Data**](./data) - base data management and generating augmented data for bigger sample size
- [**Evaluation**](./evaluation) - calculating metrics, plotting and result aggregation

#### Project elements

- users
- tasks
- disruptors

## **Models**

<!-- comparison to base of sorted tasks -->

### `RLPlanner` PPO algorithm

<!-- assumptions -->
<!-- elements -->
<!-- flow -->

### `GAPlanner` - NSGA-III algorithm

<!-- assumptions -->
<!-- elements -->
<!-- flow -->

## **Simulation**

#### User parameters

#### Daily flow

#### Weekly flow

## **Experimets**

### Faze I - plan efficiency and personalization effectiveness

### Faze II - disruptor responsiveness

## **Evaluation**

<!-- measures used -->
<!-- present main results -->
<!-- link the thesis -->

---

<a id="installation"></a>

## Installation

Requires **Python 3.12+** and an NVIDIA GPU with appropriate CUDA drivers (optional, but recommended).

### 1. Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 2. PyTorch (with CUDA support)

Install the CUDA 12.4-enabled build of PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. Project Dependencies

Install all core project dependencies:

```bash
pip install -r requirements.txt
```

### 4. Spinning Up Setup

Zarejestruj lokalny pakiet `spinningup` w środowisku wirtualnym w trybie edytowalnym (flaga `--no-deps` zapobiega próbie instalacji przestarzałych pakietów zdefiniowanych w oryginalnym repozytorium):

```bash
# Windows (PowerShell)
pip install --no-deps -e .\spinningup

# Linux / macOS
pip install --no-deps -e ./spinningup
```
