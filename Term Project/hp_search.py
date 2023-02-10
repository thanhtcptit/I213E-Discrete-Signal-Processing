import os
import subprocess

import optuna

from train import train


hp_configs = {
    "batch_size": 32,
    "optim": ["adam", "sgd"],
    "lr": [1e-4, 1e-2],
    "n_epochs": [30, 80],

    "kernel_size": 3,
    "pool_size": [(4, 2), (5, 2)],
    "dropout": [0.2, 0.8]
}
n_trial = 200

past_trials = []

def run_trial(trial):
    trial_configs = {}
    for k, v in hp_configs.items():
        if isinstance(v, list):
            if len(v) == 2:
                if isinstance(v[0], int):
                    trial_configs[k] = trial.suggest_int(k, v[0], v[1])
                elif isinstance(v[0], float):
                    trial_configs[k] = trial.suggest_float(k, v[0], v[1])
                else:
                    trial_configs[k] = trial.suggest_categorical(k, v)
            else:
                trial_configs[k] = trial.suggest_categorical(k, v)
        else:
            trial_configs[k] = v
    print(trial_configs)
    if trial_configs in past_trials:
        raise optuna.TrialPruned()
    past_trials.append(trial_configs)

    return train(trial_configs["n_epochs"], trial_configs["batch_size"], trial_configs["lr"],
                 trial_configs["optim"], trial_configs["kernel_size"], trial_configs["pool_size"],
                 trial_configs["dropout"], log=False)


def save_best_model(study, trial):
    if study.best_trial.number == trial.number:
        os.replace("cnn.pt", "best_cnn.pt")


study = optuna.create_study(study_name="gtzan_melspec", direction="maximize")
study.optimize(run_trial, n_trials=n_trial, gc_after_trial=True,
               callbacks=[save_best_model])
print("Number of finished trials: ", len(study.trials))

df = study.trials_dataframe()
print(df)

print("Best trial:")
trial = study.best_trial

print(" - Value: ", trial.value)
print(" - Params: ")
for key, value in trial.params.items():
    print("  - {}: {}".format(key, value))
