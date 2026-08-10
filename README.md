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

## Instalation

Required python 3.12+

```bash
python -m venv .venv
# Linux/MacOS
source .venv/bin/activate
# Windows Powershell
.venv\Scripts\Activate.ps1
```

Run the following commands to set up the environment:

```bash
pip install -r requirements.txt
```
