| Model      |   Sparsity | vs (criterion)   | Metric            |   n |   Mean diff (mag − x) |           p |   95% CI low |   95% CI high | p<0.05   |
|:-----------|-----------:|:-----------------|:------------------|----:|----------------------:|------------:|-------------:|--------------:|:---------|
| deit_tiny  |        0.3 | wanda            | balanced_acc      |   3 |            0.0297766  | 0.0450215   |  0.00124201  |   0.0583111   | ✓        |
| deit_tiny  |        0.3 | wanda            | mel_sensitivity   |   3 |           -0.023657   | 0.0493412   | -0.0471861   |  -0.000127953 | ✓        |
| deit_tiny  |        0.3 | wanda            | bcc_sensitivity   |   3 |           -0.0154971  | 0.236371    | -0.0489059   |   0.0179118   |          |
| deit_tiny  |        0.3 | wanda            | akiec_sensitivity |   3 |            0.0186343  | 0.181702    | -0.0156054   |   0.0528739   |          |
| deit_tiny  |        0.3 | taylor           | balanced_acc      |   3 |            0.0451516  | 0.00108282  |  0.03373     |   0.0565731   | ✓        |
| deit_tiny  |        0.3 | taylor           | mel_sensitivity   |   3 |           -0.0421958  | 0.147034    | -0.111248    |   0.0268564   |          |
| deit_tiny  |        0.3 | taylor           | bcc_sensitivity   |   3 |            0.0459649  | 0.0687969   | -0.00659025  |   0.0985201   |          |
| deit_tiny  |        0.3 | taylor           | akiec_sensitivity |   3 |            0.0782407  | 0.0293669   |  0.0148522   |   0.141629    | ✓        |
| deit_tiny  |        0.3 | random           | balanced_acc      |   3 |            0.681566   | 0.000295936 |  0.570167    |   0.792965    | ✓        |
| deit_tiny  |        0.3 | random           | mel_sensitivity   |   3 |            0.53822    | 0.0327696   |  0.0833216   |   0.993119    | ✓        |
| deit_tiny  |        0.3 | random           | bcc_sensitivity   |   3 |            0.828684   | 0.00307205  |  0.529652    |   1.12772     | ✓        |
| deit_tiny  |        0.3 | random           | akiec_sensitivity |   3 |            0.539699   | 0.192188    | -0.484651    |   1.56405     |          |
| deit_tiny  |        0.5 | wanda            | balanced_acc      |   3 |            0.384295   | 0.0020758   |  0.263035    |   0.505555    | ✓        |
| deit_tiny  |        0.5 | wanda            | mel_sensitivity   |   3 |           -0.37804    | 0.000493616 | -0.451413    |  -0.304668    | ✓        |
| deit_tiny  |        0.5 | wanda            | bcc_sensitivity   |   3 |            0.285058   | 0.000308145 |  0.237831    |   0.332286    | ✓        |
| deit_tiny  |        0.5 | wanda            | akiec_sensitivity |   3 |            0.55625    | 0.00485824  |  0.321102    |   0.791398    | ✓        |
| deit_tiny  |        0.5 | taylor           | balanced_acc      |   3 |            0.152973   | 0.00741759  |  0.0779939   |   0.227951    | ✓        |
| deit_tiny  |        0.5 | taylor           | mel_sensitivity   |   3 |           -0.334625   | 0.000106449 | -0.373461    |  -0.295789    | ✓        |
| deit_tiny  |        0.5 | taylor           | bcc_sensitivity   |   3 |            0.152953   | 0.00151794  |  0.109568    |   0.196338    | ✓        |
| deit_tiny  |        0.5 | taylor           | akiec_sensitivity |   3 |            0.266204   | 0.00168964  |  0.187894    |   0.344514    | ✓        |
| deit_tiny  |        0.5 | random           | balanced_acc      |   3 |            0.590049   | 0.000759094 |  0.457667    |   0.722431    | ✓        |
| deit_tiny  |        0.5 | random           | mel_sensitivity   |   3 |           -0.288586   | 0.046923    | -0.5698      |  -0.00737187  | ✓        |
| deit_tiny  |        0.5 | random           | bcc_sensitivity   |   3 |            0.859942   | 6.33248e-05 |  0.776043    |   0.94384     | ✓        |
| deit_tiny  |        0.5 | random           | akiec_sensitivity |   3 |            0.568634   | 0.00326327  |  0.359138    |   0.778131    | ✓        |
| deit_tiny  |        0.7 | wanda            | balanced_acc      |   3 |            0.119549   | 0.00145964  |  0.0860866   |   0.153011    | ✓        |
| deit_tiny  |        0.7 | wanda            | mel_sensitivity   |   3 |           -0.987494   | 9.57492e-08 | -0.99854     |  -0.976449    | ✓        |
| deit_tiny  |        0.7 | wanda            | bcc_sensitivity   |   3 |            0.919532   | 3.71538e-05 |  0.844454    |   0.994611    | ✓        |
| deit_tiny  |        0.7 | wanda            | akiec_sensitivity |   3 |            0.0703704  | 0.00396272  |  0.0426494   |   0.0980914   | ✓        |
| deit_tiny  |        0.7 | taylor           | balanced_acc      |   3 |           -0.0105579  | 0.458358    | -0.0501438   |   0.0290279   |          |
| deit_tiny  |        0.7 | taylor           | mel_sensitivity   |   3 |           -0.48333    | 0.0003419   | -0.566249    |  -0.40041     | ✓        |
| deit_tiny  |        0.7 | taylor           | bcc_sensitivity   |   3 |            0.370789   | 0.00181269  |  0.259072    |   0.482506    | ✓        |
| deit_tiny  |        0.7 | taylor           | akiec_sensitivity |   3 |            0.0634259  | 0.00750734  |  0.0322061   |   0.0946458   | ✓        |
| deit_tiny  |        0.7 | random           | balanced_acc      |   3 |            0.114689   | 0.00055991  |  0.0914658   |   0.137913    | ✓        |
| deit_tiny  |        0.7 | random           | mel_sensitivity   |   3 |           -0.966661   | 1.46344e-05 | -1.0245      |  -0.908827    | ✓        |
| deit_tiny  |        0.7 | random           | bcc_sensitivity   |   3 |            0.919532   | 3.71538e-05 |  0.844454    |   0.994611    | ✓        |
| deit_tiny  |        0.7 | random           | akiec_sensitivity |   3 |            0.0703704  | 0.00396272  |  0.0426494   |   0.0980914   | ✓        |
| deit_small |        0.3 | wanda            | balanced_acc      |   3 |            0.0169712  | 0.00721648  |  0.00873296  |   0.0252095   | ✓        |
| deit_small |        0.3 | wanda            | mel_sensitivity   |   3 |            0.0436067  | 0.00142793  |  0.0314917   |   0.0557217   | ✓        |
| deit_small |        0.3 | wanda            | bcc_sensitivity   |   3 |            0.012193   | 0.0866315   | -0.00324192  |   0.0276279   |          |
| deit_small |        0.3 | wanda            | akiec_sensitivity |   3 |            0.0293981  | 0.271549    | -0.040213    |   0.0990093   |          |
| deit_small |        0.3 | taylor           | balanced_acc      |   3 |            0.00129546 | 0.40869     | -0.00300753  |   0.00559844  |          |
| deit_small |        0.3 | taylor           | mel_sensitivity   |   3 |            0.0163654  | 2.5208e-06  |  0.0158207   |   0.01691     | ✓        |
| deit_small |        0.3 | taylor           | bcc_sensitivity   |   3 |           -0.00555556 | 0.391002    | -0.0232358   |   0.0121247   |          |
| deit_small |        0.3 | taylor           | akiec_sensitivity |   3 |            0.0479167  | 0.107565    | -0.0191533   |   0.114987    |          |
| deit_small |        0.3 | random           | balanced_acc      |   3 |            0.670341   | 4.26449e-05 |  0.613031    |   0.727651    | ✓        |
| deit_small |        0.3 | random           | mel_sensitivity   |   3 |            0.909142   | 0.000150871 |  0.790571    |   1.02771     | ✓        |
| deit_small |        0.3 | random           | bcc_sensitivity   |   3 |            0.587398   | 0.0416521   |  0.0417976   |   1.133       | ✓        |
| deit_small |        0.3 | random           | akiec_sensitivity |   3 |            0.23669    | 0.306138    | -0.375376    |   0.848756    |          |
| deit_small |        0.5 | wanda            | balanced_acc      |   3 |            0.181573   | 0.00695711  |  0.0945598   |   0.268587    | ✓        |
| deit_small |        0.5 | wanda            | mel_sensitivity   |   3 |            0.0184156  | 0.463132    | -0.0514501   |   0.0882814   |          |
| deit_small |        0.5 | wanda            | bcc_sensitivity   |   3 |            0.380643   | 0.00152473  |  0.27251     |   0.488777    | ✓        |
| deit_small |        0.5 | wanda            | akiec_sensitivity |   3 |           -0.133333   | 0.124514    | -0.333753    |   0.0670862   |          |
| deit_small |        0.5 | taylor           | balanced_acc      |   3 |            0.0146368  | 0.157232    | -0.010186    |   0.0394596   |          |
| deit_small |        0.5 | taylor           | mel_sensitivity   |   3 |           -0.281112   | 4.07933e-06 | -0.292096    |  -0.270128    | ✓        |
| deit_small |        0.5 | taylor           | bcc_sensitivity   |   3 |            0.0696784  | 0.0234646   |  0.0178386   |   0.121518    | ✓        |
| deit_small |        0.5 | taylor           | akiec_sensitivity |   3 |            0.0868056  | 0.00900228  |  0.0412446   |   0.132367    | ✓        |
| deit_small |        0.5 | random           | balanced_acc      |   3 |            0.649968   | 0.000302598 |  0.542938    |   0.756999    | ✓        |
| deit_small |        0.5 | random           | mel_sensitivity   |   3 |            0.588871   | 0.00025677  |  0.497097    |   0.680644    | ✓        |
| deit_small |        0.5 | random           | bcc_sensitivity   |   3 |            0.463363   | 0.211738    | -0.468605    |   1.39533     |          |
| deit_small |        0.5 | random           | akiec_sensitivity |   3 |            0.716667   | 0.00280496  |  0.466006    |   0.967328    | ✓        |
| deit_small |        0.7 | wanda            | balanced_acc      |   3 |            0.00937354 | 0.17514     | -0.0074948   |   0.0262419   |          |
| deit_small |        0.7 | wanda            | mel_sensitivity   |   3 |            0          | 1           |  0           |   0           |          |
| deit_small |        0.7 | wanda            | bcc_sensitivity   |   3 |            0.966257   | 1.07221e-06 |  0.942075    |   0.990439    | ✓        |
| deit_small |        0.7 | wanda            | akiec_sensitivity |   3 |           -0.928125   | 9.47128e-06 | -0.976152    |  -0.880098    | ✓        |
| deit_small |        0.7 | taylor           | balanced_acc      |   3 |           -0.270292   | 2.79691e-05 | -0.290365    |  -0.250219    | ✓        |
| deit_small |        0.7 | taylor           | mel_sensitivity   |   3 |           -0.341301   | 0.00034505  | -0.400035    |  -0.282567    | ✓        |
| deit_small |        0.7 | taylor           | bcc_sensitivity   |   3 |            0.208918   | 0.0147612   |  0.0779572   |   0.339879    | ✓        |
| deit_small |        0.7 | taylor           | akiec_sensitivity |   3 |           -0.174884   | 0.010855    | -0.273004    |  -0.0767643   | ✓        |
| deit_small |        0.7 | random           | balanced_acc      |   3 |            0.00287232 | 0.0587368   | -0.000198199 |   0.00594285  |          |
| deit_small |        0.7 | random           | mel_sensitivity   |   3 |            0          | 1           |  0           |   0           |          |
| deit_small |        0.7 | random           | bcc_sensitivity   |   3 |            0.788889   | 0.0334171   |  0.117039    |   1.46074     | ✓        |
| deit_small |        0.7 | random           | akiec_sensitivity |   3 |           -0.300926   | 0.295257    | -1.05809     |   0.456241    |          |