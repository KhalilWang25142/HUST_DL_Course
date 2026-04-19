# W&B Sweep Usage

Initialize the sweep:

```bash
wandb sweep sweeps/mnist_accuracy_bayes.yaml
```

Start one agent with a 10-run budget:

```bash
wandb agent <entity/project/sweep_id> --count 10
```

Run multiple agents in separate terminals to parallelize the sweep. Split the budget manually to avoid exceeding 10 total runs:

```bash
wandb agent <entity/project/sweep_id> --count 5
```
