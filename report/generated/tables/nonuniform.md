| Model      | Criterion   | Policy          | Balanced Acc   | Mel Sens      | Mel AUROC     | DCR         |   n |
|:-----------|:------------|:----------------|:---------------|:--------------|:--------------|:------------|----:|
| deit_small | Magnitude   | binned_default  | 0.758 ± 0.047  | 0.452 ± 0.064 | 0.858 ± 0.034 | 2.50 ± 0.87 |  19 |
| deit_small | Magnitude   | binned_k2       | 0.747 ± 0.045  | 0.408 ± 0.053 | 0.858 ± 0.008 | 2.42 ± 0.37 |  16 |
| deit_small | Magnitude   | binned_k3       | 0.758 ± 0.047  | 0.452 ± 0.064 | 0.858 ± 0.034 | 2.50 ± 0.87 |  19 |
| deit_small | Magnitude   | binned_k5       | 0.772 ± 0.006  | 0.468 ± 0.048 | 0.857 ± 0.026 | 2.18 ± 0.15 |  16 |
| deit_small | Magnitude   | continuous_t0.5 | 0.733 ± 0.031  | 0.397 ± 0.014 | 0.851 ± 0.021 | 1.82 ± 0.16 |  16 |
| deit_small | Magnitude   | continuous_t1   | 0.652 ± 0.120  | 0.240 ± 0.119 | 0.793 ± 0.047 | 2.14 ± 1.35 |  19 |
| deit_small | Magnitude   | continuous_t2   | 0.623 ± 0.010  | 0.197 ± 0.128 | 0.791 ± 0.027 | 1.70 ± 0.54 |  16 |
| deit_small | Magnitude   | dense           | 0.914 ± 0.032  | 0.894 ± 0.030 | 0.964 ± 0.011 | 0.00 ± 0.00 |  19 |
| deit_small | Magnitude   | learnable       | 0.813 ± 0.001  | 0.645 ± 0.051 | 0.892 ± 0.010 | 1.77 ± 0.48 |  16 |
| deit_small | Magnitude   | obs_like        | 0.569 ± 0.011  | 0.051 ± 0.007 | 0.826 ± 0.019 | 1.79 ± 0.21 |  16 |
| deit_small | Magnitude   | uniform         | 0.787 ± 0.025  | 0.569 ± 0.053 | 0.881 ± 0.019 | 1.87 ± 0.42 |  19 |
| deit_small | Wanda       | binned_default  | 0.841 ± 0.036  | 0.727 ± 0.068 | 0.918 ± 0.012 | 2.62 ± 0.61 |  19 |
| deit_small | Wanda       | binned_k2       | 0.817 ± 0.029  | 0.739 ± 0.002 | 0.913 ± 0.006 | 1.72 ± 0.23 |  16 |
| deit_small | Wanda       | binned_k3       | 0.841 ± 0.036  | 0.727 ± 0.068 | 0.918 ± 0.012 | 2.62 ± 0.61 |  19 |
| deit_small | Wanda       | binned_k5       | 0.857 ± 0.003  | 0.741 ± 0.034 | 0.925 ± 0.006 | 2.53 ± 0.55 |  16 |
| deit_small | Wanda       | continuous_t0.5 | 0.830 ± 0.007  | 0.706 ± 0.065 | 0.919 ± 0.005 | 2.22 ± 1.27 |  16 |
| deit_small | Wanda       | continuous_t1   | 0.755 ± 0.075  | 0.579 ± 0.110 | 0.891 ± 0.045 | 2.58 ± 3.26 |  19 |
| deit_small | Wanda       | continuous_t2   | 0.762 ± 0.019  | 0.540 ± 0.055 | 0.894 ± 0.002 | 1.57 ± 0.53 |  16 |
| deit_small | Wanda       | dense           | 0.914 ± 0.032  | 0.894 ± 0.030 | 0.964 ± 0.011 | 0.00 ± 0.00 |  19 |
| deit_small | Wanda       | learnable       | 0.871 ± 0.004  | 0.825 ± 0.048 | 0.946 ± 0.005 | 2.35 ± 0.62 |  16 |
| deit_small | Wanda       | obs_like        | 0.605 ± 0.017  | 0.200 ± 0.020 | 0.839 ± 0.016 | 1.50 ± 0.06 |  16 |
| deit_small | Wanda       | uniform         | 0.612 ± 0.018  | 0.539 ± 0.041 | 0.837 ± 0.019 | 0.77 ± 0.09 |  19 |