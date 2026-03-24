# Hacker News — Show HN

## Title
Show HN: deltatau-audit – Find timing bugs in RL agents before deployment

## URL
https://github.com/maruyamakoju/deltatau-audit

## Text (for self-post, optional)
RL agents trained in simulation often fail when deployed to hardware with different timing — different frame rates, observation delays, jitter. This is one of the most common failure modes in robotics RL, but there's no standard tool to test for it.

deltatau-audit is a CLI tool that audits your trained model against 6 timing perturbation scenarios in one command. It also auto-fixes fragile models by retraining with timing augmentation, and includes formal verification (Lipschitz bounds, interval bound propagation) for mathematical safety guarantees.

Works with Stable-Baselines3, CleanRL, HuggingFace Hub, and any PyTorch model. Integrates with GitHub Actions CI.

`pip install deltatau-audit && deltatau-audit demo`
