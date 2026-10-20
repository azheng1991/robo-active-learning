# Active Learning for Robotics: Sim-to-Real Hard Negative Mining

This repository demonstrates an end-to-end active learning pipeline for training reliable robotic policies using the Robosuite simulation environment and the Panda robot arm.

## The Problem
Standard Behavioral Cloning models perform well on average but fail catastrophically in edge-case physical states. In this project, we observed that our Baseline model succeeded in most standard pick-and-place trajectories, but completely failed if the cube started on the far left edge of the bin.

## The Solution
By mining hard-negative failure states in simulation and retraining the model on expert demonstrations initialized from those exact states, we achieved perfect reliability on a fraction of the compute.

### Performance Metrics

| Metric | Baseline Model | Improved Model (Active Learning) |
| :--- | :--- | :--- |
| **Initial Expert Demonstrations** | 100 | 100 |
| **Targeted Demonstrations Added** | 0 | 50 (mined from failures) |
| **Total Dataset Size** | 100 | 150 |
| **Overall Success Rate** | 68% | **100%** |
| **Success in Edge-Case States** | 0% | **100%** |

## Repository Structure
- \data_generation/\: Scripts for generating the baseline expert datasets and the targeted datasets based on edge cases.
- \evaluation/\: Contains the hard negative states mined from the baseline model's failures.
- \isualization/\: Contains the scripts to render side-by-side video comparisons of the Baseline vs. Improved model.

## Why This Matters (The Sim-to-Real Gap)
This highlights the core superpower of simulation. In the real world, physically resetting a robot arm 50 times to collect edge-case data would take an engineer days. In simulation, we evaluate the model, find its weaknesses, generate targeted data, and retrain it—all entirely autonomously.

Bridging the " Sim-to-Real gap\
