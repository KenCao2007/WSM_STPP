Code for **"Score Matching for Estimating Finite Point Processes"**

---

### Dataset

Download the dataset from:  
https://drive.google.com/drive/folders/1J6jLIxNn5LRQCghv24wzOGHWZ6V0_0Jr?usp=sharing

---

### Reproducing Results

**Table 3**

To reproduce Table 3 in our paper, go to `scripts/table` and run, for example:

```bash
bash run.sh --seeds 1,2,3 --dataset "Earthquake" --estimator "wsm"
```

**Figure 2**
To reproduce Figure 2, go to scripts/figure and run:

```bash
bash Hawkes2_figure.sh
```

**Figure 3**
To reproduce Figure 3, go to scripts/figure and run:
```bash 
bash Earthquake_figure.sh
```

### Citation

If you find this code useful in your research, please consider citing our paper.

### Acknowledgements
This repository is built on top of the **SMASH** codebase.
We also adapt plotting and synthetic data generation code from **AutoInt** and **NSTPP**.