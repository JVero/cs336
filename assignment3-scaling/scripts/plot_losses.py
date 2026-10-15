import json

from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Row:
    parameters: int
    compute_budget: float
    final_loss: float

if __name__ == "__main__":
    fig = plt.figure()
    if_path = Path("./data/isoflops_curves.json")
    with open(if_path, "r") as f:
        data = json.load(f)
        rows = [Row(**row) for row in data]
    budgets = np.array([row.compute_budget for row in rows])
    
    # group data by compute budget
    profiles = {}
    unique_budgets = set(budgets)
    for budget in unique_budgets:
        profiles[budget] = [[],[]]
    for row in rows:
        profiles[row.compute_budget][0].append(row.parameters)
        profiles[row.compute_budget][1].append(row.final_loss)
        
    # Find optimum per compute budget
    n_opt_by_budget = {}
    min_loss_by_budget = {}
    for budget, profile in profiles.items():
        profile_params: list[int] = profile[0]
        profile_losses: list[float] = profile[1]
        parabola_coeffs = np.polyfit(np.log10(profile_params), profile_losses, deg=2)
        parabola = np.poly1d(parabola_coeffs)
        n_opt = int(10**(-parabola_coeffs[1]/(2*parabola_coeffs[0])))
        n_opt_by_budget[budget] = n_opt
        min_loss_by_budget[budget] = parabola(np.log10(n_opt))
    
    budgets = np.array(list(n_opt_by_budget.keys()))
    log_budgets = np.log10(budgets)
    optimum_model_size = np.array([n_opt_by_budget[budget] for budget in budgets])
    log_optimum_model_size = np.log10(optimum_model_size)

    size_from_budget = np.poly1d(np.polyfit(log_budgets, log_optimum_model_size, deg=1))
    budget = np.linspace(18, 25)

    size23 = size_from_budget(23)
    size24 = size_from_budget(24)
    
    print(f"For 10^23 total flops, you'd want a model with 10^{size23} = {int(10**size23)} params")
    print(f"For 10^24 total flops, you'd want a model with 10^{size24} = {int(10**size24)} params")
    
    plt.plot(log_budgets, log_optimum_model_size, 'o')
    plt.plot(budget, size_from_budget(budget))
    plt.xlabel("log 10 compute budget")
    plt.ylabel("log 10 Optimal model size")
    plt.savefig("./scripts/model_size.png")
    
    plt.close()
    ### Calculate D from C and N
    ds = []
    print(log_budgets)
    for log_budget in log_budgets:
        ds.append(10**log_budget / (6 * 10**size_from_budget(log_budget)))
    log_d = np.log10(ds)
    plt.plot(log_budgets, log_d, 'o')
    tokens_from_budget = np.poly1d(np.polyfit(log_budgets, log_d, deg=1))
    
    d23 = tokens_from_budget(23)
    d24 = tokens_from_budget(24)
    
    print(f"For 10^23 total flops, you'd want 10^{d23} = {int(10**d23)} tokens")
    print(f"For 10^24 total flops, you'd want 10^{d24} = {int(10**d24)} tokens")
    plt.xlabel("log 10 compute budget")
    plt.ylabel("log 10 optimal tokens for fixed compute budget")
    plt.plot(budget, tokens_from_budget(budget), 'r-')
    plt.savefig("./scripts/optimal_tokens.png")
    
    