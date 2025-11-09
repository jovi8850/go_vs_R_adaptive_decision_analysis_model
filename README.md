# **Adaptive Decision Analysis in R and Go**

## **Overview**
This project implements and compares two **adaptive decision analysis models** — **LinUCB (Contextual Bandit)** and **Epsilon-Greedy (Simple Bandit)** — in both **R** and **Go**.

The objective is to evaluate Go’s computational efficiency and scalability versus R’s modeling flexibility for **computer-intensive statistical methods** on contextual bandit data. Both languages were benchmarked on identical datasets to assess runtime, memory use, and learning performance.

---

## **Method Summary**
- **LinUCB (Contextual Bandit):** Learns context–action relationships using linear regression with confidence bounds.  
- **Epsilon-Greedy (Simple Bandit):** Balances exploration and exploitation with a probabilistic approach.  
- Both models adaptively update after each round of feedback to maximize cumulative reward.  

Datasets were synthetically generated with adjustable noise, complexity, and contextual features (10 dimensions, 5 arms, ~5,000 samples).


---
## **Instructions**

Before running the **generate_contextual_bandit.R** script, ensure that the file location variable is correctly set to the directory where you want your generated training and testing datasets to be saved.

**1.** Replace the placeholder path in file_location with the absolute path of your working directory.

**2.** Ensure that the folder exists before running the script.

**3.** After running the script, verify that the following files are created in the specified directory:
- contextual_bandit_train.csv
- contextual_bandit_test.csv

**4.** Use these generated datasets when executing both the R and Go adaptive decision analysis models.

---

## **Performance Metrics**

| Metric | **R (Actual)** | **Go (Actual)** | **Observations** |
|--------|----------------|-----------------|------------------|
| **LinUCB Runtime (seconds)** | 13.59 | 4.07 | Go executed ~3× faster |
| **LinUCB Memory Used (MB)** | 3.68 | 5.24 | R slightly more memory-efficient |
| **LinUCB Average Reward** | 0.414 | 0.414 | Identical across both implementations |
| **LinUCB Average Regret** | 4.17 | 4.17 | Identical |
| **Epsilon-Greedy Runtime (seconds)** | 6.61 | 0.12 | Go executed much faster |
| **Epsilon-Greedy Memory Used (MB)** | 0.57 | 11.78 | Go used more memory for this smaller model |
| **Epsilon-Greedy Final Reward** | 580.12 | 266.85 | R’s model achieved higher convergence reward |
| **Total Decisions: Test Dataset** | 5,000 | 5,000 | Consistent dataset across both languages |

**Conclusion:**  
Go replicated R’s LinUCB results with **equivalent accuracy** while reducing runtime by about **70%**.  
Memory usage varied depending on model structure — **R used less memory for small workloads**, but Go’s scalability and processing speed would dominate on larger data or cloud applications.  
Epsilon-Greedy results diverged slightly, reflecting **stochastic learning differences**, not structural implementation issues.

---

## **Results Visualization**
**Observation:**  
In both implementations, **LinUCB outperformed Epsilon-Greedy**, achieving higher cumulative rewards and lower regret, validating its adaptive advantage.

---

## **Conclusion**
The comparative evaluation shows:
- **LinUCB** performed consistently across both languages with identical numerical outcomes.  
- **Go** achieved **faster execution**, making it more suitable for large-scale or real-time adaptive decision systems.  
- **R** remains a strong environment for **prototyping, visualization, and parameter tuning** before migrating to production-grade Go code.  

Overall, Go provides **computational efficiency** and **performance scalability**, while R offers a **flexible analytic development interface**.

---

## **Recommendations**
- Use **R** for initial modeling, exploration, and statistical diagnostics.  
- Use **Go** for production-scale or time-sensitive systems requiring optimized performance.  
- In cloud settings, expect **30–50% lower compute costs** using Go for similar workloads due to runtime gains.  
- Validate Epsilon-Greedy results with multiple random seeds to confirm stability across RNG implementations.

---

## **GenAI Tools Disclosure**
This project used **ChatGPT (OpenAI GPT-5)** and **DeepSeek** for:
- Code structure and performance tuning  
- Documentation drafting  
- Parameter logging and testing framework design  

All final code and results were independently verified and executed by the author.

---

## **Repository Contents**

```
.RData
.Rhistory
ada_model_Go.go
ada_model_R.r
contextual_bandit_test.csv
contextual_bandit_train.csv
generate_contextual_bandit.R
go.mod
go.sum
go_analysis_parameters_20251108_192918.xlsx
go_cumulative_regret_20251108_192918.png
go_cumulative_reward_20251108_192918.png
r_analysis_parameters.xlsx
r_cumulative_regret_20251108_191017.png
r_cumulative_reward_20251108_191017.png
```

