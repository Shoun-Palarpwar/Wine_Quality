# Held-out test results

```text
     Model  Accuracy  Weighted F1  Macro F1  Balanced accuracy    MAE  Within one point  Quadratic weighted kappa
     dummy    0.4265       0.2550    0.0997             0.1667 0.7243            0.8603                    0.0000
  baseline    0.6066       0.5879    0.2887             0.2889 0.4265            0.9669                    0.5563
  balanced    0.5846       0.5797    0.3241             0.3346 0.4669            0.9522                    0.5800
regression    0.6066       0.5894    0.3126             0.3026 0.4154            0.9779                    0.5796
     tuned    0.6029       0.5985    0.3332             0.3456 0.4522            0.9485                    0.5892
```

Model selection uses training cross-validation only. These test results describe one fixed split; they do not establish a performance ceiling.
