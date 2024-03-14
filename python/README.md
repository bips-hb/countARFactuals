# Synthetic Data Generation and Likelihood Evaluation

## Data Generation

There are two scripts for data generation. `illustrative.py` for simple illustrative 2d datasets (two sines, cassini, pawelczyk), as well as `dgp.py` for (semi-)automatically generated high-dimensional mixed data sampled from a randomly generated Bayesian Network.

Both scripts can be called from the command line to generate data.

To generate the illustrative datasets, call

```bash
python python/illustrative.py savepath --batch_size batch_size
```
where we used `python/synthetic` as a `savepath` and `100000` as `batch_size`.

To generate the BN based data, call

```bash
python python/dgp_v2.py --batch_size batch_size
```
Again we used `100000` as batch size.

## Evaluate Likelihood

To evaluate the likelihood of generated counterfactuals, the `get_likelihood.py` script can be used. From the root directory in the repo, call

```bash
python python/get_likelihood dgpname cfpath
```
where `dgpname` is the name of the DGP, i.e. in `['twosines', 'cassini', 'pawelczyk', 'bn_5_v2', 'bn_10_v2', 'bn_20']`.
The `cfpath` is the path to the `*_cfs.csv` file containing the counterfactuals for which the likelihood should be evaluated.
A new file is generated ending in `_cfs_with_log_probs.csv` containing a column with the likelihoods.

## Visualizations

Visualizations of the dataset can be found in the folder 'visualizations/'.