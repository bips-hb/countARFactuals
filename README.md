# countARFactuals
This repository contains the code for the countARFactuals paper.

```bash
    .
    ├── R_experiments                   # simulations study
    │   ├── run_experiment.R            # run study/generate counterfactuals
    │   ├── utils_experiment.R          # helper function to run study
    │   ├── create_cfe_dir.R            # extract results 
    │   ├── aggregate_results.R         # aggregate results
    │   ├── res.RDS                     # contains counterfactuals
    │   ├── res_agg.RDS                 # contains aggregated results
    │   ├── evaluate_results.R          # reproduce plots of paper
    ├── counterfactuals_R_package       # adapted counterfactuals package
    │   ├── R/MOCClassif.R              # Algo 1 - ARF integrated in MOC
    │   ├── R/CountARFactualClassif.R   # Algo 2 - stand alone ARF method
    ├── python
    ├── real_world_example     
    └── ...
``` 
