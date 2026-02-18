# Twitter/X Thread

[1/6] Your PPO agent scores 990 on HalfCheetah. Add a 1-step observation delay — a timing mismatch any real deployment has — and it scores 38. That's -96%. The model didn't change. The world did.

[2/6] Why? Standard RNNs treat all timesteps as equal. They have no concept of how much time passed between steps. Sensor lag, variable control rates, jitter — the hidden state can't adapt. It assumes a clock that doesn't exist in deployment.

[3/6] deltatau-audit finds and fixes this in one command:

deltatau-audit fix-sb3 --algo ppo \
  --model my_model.zip \
  --env HalfCheetah-v5

Rewraps the env with speed randomization. Retrains. Audits. No architecture changes required.

[4/6] Before vs after fix-sb3 on HalfCheetah:

Delay:   990 → 38  (−96%) ... fix → PASS (+146pp)
Jitter:  −75% drop ... fix → +93pp recovery
5x speed: negative return ... fix → +50pp

Statistically significant (95% bootstrap CI, all 4 scenarios).

[5/6] Try it now:

pip install "deltatau-audit[demo]"
python -m deltatau_audit demo cartpole

Colab: https://colab.research.google.com/github/maruyamakoju/deltatau-audit/blob/main/notebooks/quickstart.ipynb
GitHub: https://github.com/maruyamakoju/deltatau-audit

[6/6] CI integration: add --ci to any audit command.

Returns exit code 0 (pass) / 1 (warn) / 2 (fail) + ci_summary.json.

Catches timing regressions in your training pipeline before they reach deployment. Works as a GitHub Action step.
