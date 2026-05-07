| Model      |   Sparsity | vs (criterion)   | Metric            |   n |   Mean diff (mag − x) |           p |   95% CI low |   95% CI high | p<0.05   |
|:-----------|-----------:|:-----------------|:------------------|----:|----------------------:|------------:|-------------:|--------------:|:---------|
| deit_tiny  |        0.3 | wanda            | balanced_acc      |   3 |           0.0297766   | 0.0450215   |  0.00124201  |   0.0583111   | ✓        |
| deit_tiny  |        0.3 | wanda            | mel_sensitivity   |   3 |          -0.023657    | 0.0493412   | -0.0471861   |  -0.000127953 | ✓        |
| deit_tiny  |        0.3 | wanda            | bcc_sensitivity   |   3 |          -0.0154971   | 0.236371    | -0.0489059   |   0.0179118   |          |
| deit_tiny  |        0.3 | wanda            | akiec_sensitivity |   3 |           0.0186343   | 0.181702    | -0.0156054   |   0.0528739   |          |
| deit_tiny  |        0.3 | taylor           | balanced_acc      |   3 |           0.0451516   | 0.00108282  |  0.03373     |   0.0565731   | ✓        |
| deit_tiny  |        0.3 | taylor           | mel_sensitivity   |   3 |          -0.0421958   | 0.147034    | -0.111248    |   0.0268564   |          |
| deit_tiny  |        0.3 | taylor           | bcc_sensitivity   |   3 |           0.0459649   | 0.0687969   | -0.00659025  |   0.0985201   |          |
| deit_tiny  |        0.3 | taylor           | akiec_sensitivity |   3 |           0.0782407   | 0.0293669   |  0.0148522   |   0.141629    | ✓        |
| deit_tiny  |        0.3 | random           | balanced_acc      |   3 |           0.681566    | 0.000295936 |  0.570167    |   0.792965    | ✓        |
| deit_tiny  |        0.3 | random           | mel_sensitivity   |   3 |           0.53822     | 0.0327696   |  0.0833216   |   0.993119    | ✓        |
| deit_tiny  |        0.3 | random           | bcc_sensitivity   |   3 |           0.828684    | 0.00307205  |  0.529652    |   1.12772     | ✓        |
| deit_tiny  |        0.3 | random           | akiec_sensitivity |   3 |           0.539699    | 0.192188    | -0.484651    |   1.56405     |          |
| deit_tiny  |        0.5 | wanda            | balanced_acc      |   5 |           0.420459    | 4.26653e-22 |  0.403829    |   0.437089    | ✓        |
| deit_tiny  |        0.5 | wanda            | mel_sensitivity   |   5 |          -0.415859    | 8.90556e-22 | -0.432961    |  -0.398757    | ✓        |
| deit_tiny  |        0.5 | wanda            | bcc_sensitivity   |   5 |           0.355696    | 8.10726e-18 |  0.331926    |   0.379465    | ✓        |
| deit_tiny  |        0.5 | wanda            | akiec_sensitivity |   5 |           0.601252    | 2.32885e-19 |  0.568016    |   0.634489    | ✓        |
| deit_tiny  |        0.5 | taylor           | balanced_acc      |   5 |           0.173404    | 1.27213e-18 |  0.162909    |   0.183898    | ✓        |
| deit_tiny  |        0.5 | taylor           | mel_sensitivity   |   5 |          -0.296126    | 1.59442e-11 | -0.340034    |  -0.252218    | ✓        |
| deit_tiny  |        0.5 | taylor           | bcc_sensitivity   |   5 |           0.224068    | 7.42602e-10 |  0.182453    |   0.265682    | ✓        |
| deit_tiny  |        0.5 | taylor           | akiec_sensitivity |   5 |           0.287228    | 1.37133e-20 |  0.273572    |   0.300885    | ✓        |
| deit_tiny  |        0.5 | random           | balanced_acc      |   5 |           0.618208    | 8.74493e-25 |  0.600583    |   0.635834    | ✓        |
| deit_tiny  |        0.5 | random           | mel_sensitivity   |   5 |          -0.445412    | 3.1056e-13  | -0.498285    |  -0.392539    | ✓        |
| deit_tiny  |        0.5 | random           | bcc_sensitivity   |   5 |           0.894475    | 1.75534e-29 |  0.880059    |   0.908892    | ✓        |
| deit_tiny  |        0.5 | random           | akiec_sensitivity |   5 |           0.612642    | 2.41578e-20 |  0.582623    |   0.64266     | ✓        |
| deit_tiny  |        0.7 | wanda            | balanced_acc      |   5 |           0.120116    | 6.51258e-17 |  0.111138    |   0.129093    | ✓        |
| deit_tiny  |        0.7 | wanda            | mel_sensitivity   |   5 |          -0.981827    | 7.80575e-47 | -0.983758    |  -0.979896    | ✓        |
| deit_tiny  |        0.7 | wanda            | bcc_sensitivity   |   5 |           0.902312    | 4.61873e-32 |  0.891676    |   0.912947    | ✓        |
| deit_tiny  |        0.7 | wanda            | akiec_sensitivity |   5 |           0.0450138   | 5.84699e-10 |  0.0367727   |   0.0532549   | ✓        |
| deit_tiny  |        0.7 | taylor           | balanced_acc      |   5 |           0.000460182 | 0.940645    | -0.0123054   |   0.0132257   |          |
| deit_tiny  |        0.7 | taylor           | mel_sensitivity   |   5 |          -0.398946    | 8.39611e-10 | -0.473589    |  -0.324304    | ✓        |
| deit_tiny  |        0.7 | taylor           | bcc_sensitivity   |   5 |           0.454436    | 1.29209e-14 |  0.409177    |   0.499696    | ✓        |
| deit_tiny  |        0.7 | taylor           | akiec_sensitivity |   5 |           0.0403462   | 9.4296e-10  |  0.0327445   |   0.0479479   | ✓        |
| deit_tiny  |        0.7 | random           | balanced_acc      |   5 |           0.119144    | 3.33769e-17 |  0.110553    |   0.127734    | ✓        |
| deit_tiny  |        0.7 | random           | mel_sensitivity   |   5 |          -0.97766     | 7.35357e-36 | -0.984932    |  -0.970389    | ✓        |
| deit_tiny  |        0.7 | random           | bcc_sensitivity   |   5 |           0.902312    | 4.61873e-32 |  0.891676    |   0.912947    | ✓        |
| deit_tiny  |        0.7 | random           | akiec_sensitivity |   5 |           0.0450138   | 5.84699e-10 |  0.0367727   |   0.0532549   | ✓        |
| deit_small |        0.3 | wanda            | balanced_acc      |   3 |           0.0169712   | 0.00721648  |  0.00873296  |   0.0252095   | ✓        |
| deit_small |        0.3 | wanda            | mel_sensitivity   |   3 |           0.0436067   | 0.00142793  |  0.0314917   |   0.0557217   | ✓        |
| deit_small |        0.3 | wanda            | bcc_sensitivity   |   3 |           0.012193    | 0.0866315   | -0.00324192  |   0.0276279   |          |
| deit_small |        0.3 | wanda            | akiec_sensitivity |   3 |           0.0293981   | 0.271549    | -0.040213    |   0.0990093   |          |
| deit_small |        0.3 | taylor           | balanced_acc      |   3 |           0.00129546  | 0.40869     | -0.00300753  |   0.00559844  |          |
| deit_small |        0.3 | taylor           | mel_sensitivity   |   3 |           0.0163654   | 2.5208e-06  |  0.0158207   |   0.01691     | ✓        |
| deit_small |        0.3 | taylor           | bcc_sensitivity   |   3 |          -0.00555556  | 0.391002    | -0.0232358   |   0.0121247   |          |
| deit_small |        0.3 | taylor           | akiec_sensitivity |   3 |           0.0479167   | 0.107565    | -0.0191533   |   0.114987    |          |
| deit_small |        0.3 | random           | balanced_acc      |   3 |           0.670341    | 4.26449e-05 |  0.613031    |   0.727651    | ✓        |
| deit_small |        0.3 | random           | mel_sensitivity   |   3 |           0.909142    | 0.000150871 |  0.790571    |   1.02771     | ✓        |
| deit_small |        0.3 | random           | bcc_sensitivity   |   3 |           0.587398    | 0.0416521   |  0.0417976   |   1.133       | ✓        |
| deit_small |        0.3 | random           | akiec_sensitivity |   3 |           0.23669     | 0.306138    | -0.375376    |   0.848756    |          |
| deit_small |        0.5 | wanda            | balanced_acc      |   5 |           0.176495    | 1.70032e-16 |  0.162603    |   0.190387    | ✓        |
| deit_small |        0.5 | wanda            | mel_sensitivity   |   5 |           0.0294407   | 1.13902e-05 |  0.0189759   |   0.0399055   | ✓        |
| deit_small |        0.5 | wanda            | bcc_sensitivity   |   5 |           0.397586    | 7.65662e-22 |  0.381365    |   0.413806    | ✓        |
| deit_small |        0.5 | wanda            | akiec_sensitivity |   5 |          -0.16354     | 7.80822e-09 | -0.198622    |  -0.128459    | ✓        |
| deit_small |        0.5 | taylor           | balanced_acc      |   5 |           0.0186871   | 0.000110047 |  0.0106306   |   0.0267436   | ✓        |
| deit_small |        0.5 | taylor           | mel_sensitivity   |   5 |          -0.271956    | 1.5058e-18  | -0.288563    |  -0.255348    | ✓        |
| deit_small |        0.5 | taylor           | bcc_sensitivity   |   5 |           0.0385781   | 3.0667e-05  |  0.0237096   |   0.0534465   | ✓        |
| deit_small |        0.5 | taylor           | akiec_sensitivity |   5 |           0.14583     | 4.28312e-06 |  0.0977817   |   0.193879    | ✓        |
| deit_small |        0.5 | random           | balanced_acc      |   5 |           0.655976    | 8.60909e-28 |  0.642996    |   0.668956    | ✓        |
| deit_small |        0.5 | random           | mel_sensitivity   |   5 |           0.570682    | 1.64917e-21 |  0.546433    |   0.594931    | ✓        |
| deit_small |        0.5 | random           | bcc_sensitivity   |   5 |           0.445066    | 0.000751499 |  0.212675    |   0.677457    | ✓        |
| deit_small |        0.5 | random           | akiec_sensitivity |   5 |           0.748274    | 1.21562e-21 |  0.71699     |   0.779559    | ✓        |
| deit_small |        0.7 | wanda            | balanced_acc      |   5 |          -0.0036976   | 0.0647346   | -0.00764462  |   0.000249431 |          |
| deit_small |        0.7 | wanda            | mel_sensitivity   |   5 |           0           | 1           |  0           |   0           |          |
| deit_small |        0.7 | wanda            | bcc_sensitivity   |   5 |           0.948201    | 6.83659e-36 |  0.941176    |   0.955226    | ✓        |
| deit_small |        0.7 | wanda            | akiec_sensitivity |   5 |          -0.974357    | 1.57594e-30 | -0.988189    |  -0.960526    | ✓        |
| deit_small |        0.7 | taylor           | balanced_acc      |   5 |          -0.283409    | 1.83311e-23 | -0.292899    |  -0.27392     | ✓        |
| deit_small |        0.7 | taylor           | mel_sensitivity   |   5 |          -0.361802    | 2.26435e-19 | -0.381772    |  -0.341832    | ✓        |
| deit_small |        0.7 | taylor           | bcc_sensitivity   |   5 |           0.203103    | 1.58759e-13 |  0.179876    |   0.226331    | ✓        |
| deit_small |        0.7 | taylor           | akiec_sensitivity |   5 |          -0.15832     | 2.37566e-13 | -0.176836    |  -0.139805    | ✓        |
| deit_small |        0.7 | random           | balanced_acc      |   5 |           0.000574465 | 0.0834608   | -8.36659e-05 |   0.0012326   |          |
| deit_small |        0.7 | random           | mel_sensitivity   |   5 |           0           | 1           |  0           |   0           |          |
| deit_small |        0.7 | random           | bcc_sensitivity   |   5 |           0.557778    | 8.65658e-05 |  0.322496    |   0.793059    | ✓        |
| deit_small |        0.7 | random           | akiec_sensitivity |   5 |          -0.0601852   | 0.248009    | -0.16588     |   0.0455101   |          |
| deit_tiny  |        0.2 | wanda            | balanced_acc      |   2 |           0.0188452   | 1.14361e-13 |  0.0172454   |   0.020445    | ✓        |
| deit_tiny  |        0.2 | wanda            | mel_sensitivity   |   2 |           0.0203742   | 4.49293e-08 |  0.0160665   |   0.0246819   | ✓        |
| deit_tiny  |        0.2 | wanda            | bcc_sensitivity   |   2 |           0.0250811   | 1.78249e-05 |  0.0164213   |   0.0337408   | ✓        |
| deit_tiny  |        0.2 | wanda            | akiec_sensitivity |   2 |          -0.00819672  | 0.00150177  | -0.0127077   |  -0.00368575  | ✓        |
| deit_tiny  |        0.2 | taylor           | balanced_acc      |   2 |           0.0134441   | 5.70335e-12 |  0.0119515   |   0.0149367   | ✓        |
| deit_tiny  |        0.2 | taylor           | mel_sensitivity   |   2 |          -0.00719314  | 0.0308995   | -0.01363     |  -0.000756265 | ✓        |
| deit_tiny  |        0.2 | taylor           | bcc_sensitivity   |   2 |           0.029754    | 2.12798e-07 |  0.0226623   |   0.0368456   | ✓        |
| deit_tiny  |        0.2 | taylor           | akiec_sensitivity |   2 |           0.0263796   | 3.74189e-11 |  0.0230413   |   0.0297179   | ✓        |
| deit_tiny  |        0.2 | random           | balanced_acc      |   2 |           0.600465    | 6.07521e-22 |  0.586298    |   0.614633    | ✓        |
| deit_tiny  |        0.2 | random           | mel_sensitivity   |   2 |           0.662094    | 2.82104e-29 |  0.65703     |   0.667159    | ✓        |
| deit_tiny  |        0.2 | random           | bcc_sensitivity   |   2 |           0.622163    | 9.06377e-18 |  0.594242    |   0.650084    | ✓        |
| deit_tiny  |        0.2 | random           | akiec_sensitivity |   2 |           0.782152    | 2.6558e-11  |  0.685516    |   0.878788    | ✓        |
| deit_tiny  |        0.4 | wanda            | balanced_acc      |   2 |           0.10978     | 6.29525e-20 |  0.106248    |   0.113312    | ✓        |
| deit_tiny  |        0.4 | wanda            | mel_sensitivity   |   2 |          -0.0276056   | 3.22405e-06 | -0.035811    |  -0.0194002   | ✓        |
| deit_tiny  |        0.4 | wanda            | bcc_sensitivity   |   2 |           0.0750048   | 2.98125e-11 |  0.0656627   |   0.0843469   | ✓        |
| deit_tiny  |        0.4 | wanda            | akiec_sensitivity |   2 |           0.221023    | 7.11802e-14 |  0.202856    |   0.23919     | ✓        |
| deit_tiny  |        0.4 | taylor           | balanced_acc      |   2 |           0.0648534   | 2.42975e-09 |  0.0538079   |   0.075899    | ✓        |
| deit_tiny  |        0.4 | taylor           | mel_sensitivity   |   2 |          -0.0162037   | 0.00390558  | -0.0263425   |  -0.00606495  | ✓        |
| deit_tiny  |        0.4 | taylor           | bcc_sensitivity   |   2 |           0.0544059   | 7.80765e-07 |  0.0400251   |   0.0687867   | ✓        |
| deit_tiny  |        0.4 | taylor           | akiec_sensitivity |   2 |           0.0809282   | 5.91366e-08 |  0.0634561   |   0.0984002   | ✓        |
| deit_tiny  |        0.4 | random           | balanced_acc      |   2 |           0.692235    | 4.1655e-23  |  0.678578    |   0.705892    | ✓        |
| deit_tiny  |        0.4 | random           | mel_sensitivity   |   2 |           0.0574878   | 0.374869    | -0.0764855   |   0.191461    |          |
| deit_tiny  |        0.4 | random           | bcc_sensitivity   |   2 |           0.940921    | 1.19601e-29 |  0.934125    |   0.947718    | ✓        |
| deit_tiny  |        0.4 | random           | akiec_sensitivity |   2 |           0.793697    | 1.96287e-24 |  0.780926    |   0.806467    | ✓        |
| deit_tiny  |        0.6 | wanda            | balanced_acc      |   2 |           0.328479    | 3.75452e-21 |  0.319727    |   0.337232    | ✓        |
| deit_tiny  |        0.6 | wanda            | mel_sensitivity   |   2 |          -0.827116    | 1.13734e-23 | -0.842079    |  -0.812152    | ✓        |
| deit_tiny  |        0.6 | wanda            | bcc_sensitivity   |   2 |           0.880793    | 2.72752e-32 |  0.876553    |   0.885034    | ✓        |
| deit_tiny  |        0.6 | wanda            | akiec_sensitivity |   2 |           0.213346    | 3.02349e-18 |  0.204451    |   0.222241    | ✓        |
| deit_tiny  |        0.6 | taylor           | balanced_acc      |   2 |           0.0458166   | 1.17557e-16 |  0.0433743   |   0.0482589   | ✓        |
| deit_tiny  |        0.6 | taylor           | mel_sensitivity   |   2 |          -0.628214    | 7.6057e-12  | -0.699365    |  -0.557063    | ✓        |
| deit_tiny  |        0.6 | taylor           | bcc_sensitivity   |   2 |           0.206132    | 3.46542e-14 |  0.189998    |   0.222266    | ✓        |
| deit_tiny  |        0.6 | taylor           | akiec_sensitivity |   2 |          -0.0533941   | 0.03265     | -0.101749    |  -0.00503937  | ✓        |
| deit_tiny  |        0.6 | random           | balanced_acc      |   2 |           0.344019    | 1.29455e-48 |  0.343884    |   0.344154    | ✓        |
| deit_tiny  |        0.6 | random           | mel_sensitivity   |   2 |          -0.834347    | 2.19736e-22 | -0.85274     |  -0.815954    | ✓        |
| deit_tiny  |        0.6 | random           | bcc_sensitivity   |   2 |           0.88313     | 1.23933e-35 |  0.880584    |   0.885675    | ✓        |
| deit_tiny  |        0.6 | random           | akiec_sensitivity |   2 |           0.213346    | 3.02349e-18 |  0.204451    |   0.222241    | ✓        |
| deit_small |        0.2 | wanda            | balanced_acc      |   2 |           0.00579461  | 1.12185e-06 |  0.0042171   |   0.00737213  | ✓        |
| deit_small |        0.2 | wanda            | mel_sensitivity   |   2 |           0.0378979   | 7.52148e-14 |  0.0347711   |   0.0410247   | ✓        |
| deit_small |        0.2 | wanda            | bcc_sensitivity   |   2 |          -0.0046729   | 0.00150177  | -0.00724457  |  -0.00210122  | ✓        |
| deit_small |        0.2 | wanda            | akiec_sensitivity |   2 |           0.00294389  | 0.378281    | -0.00396732  |   0.0098551   |          |
| deit_small |        0.2 | taylor           | balanced_acc      |   2 |           0.00516648  | 1.24322e-05 |  0.00343965  |   0.00689331  | ✓        |
| deit_small |        0.2 | taylor           | mel_sensitivity   |   2 |           0.0237221   | 5.81774e-09 |  0.0194158   |   0.0280284   | ✓        |
| deit_small |        0.2 | taylor           | bcc_sensitivity   |   2 |          -8.32667e-17 | 1           | -0.00363689  |   0.00363689  |          |
| deit_small |        0.2 | taylor           | akiec_sensitivity |   2 |           0.0122951   | 0.040969    |  0.000575247 |   0.0240149   | ✓        |
| deit_small |        0.2 | random           | balanced_acc      |   2 |           0.472254    | 5.00723e-18 |  0.451888    |   0.492621    | ✓        |
| deit_small |        0.2 | random           | mel_sensitivity   |   2 |           0.852043    | 5.15263e-31 |  0.847053    |   0.857034    | ✓        |
| deit_small |        0.2 | random           | bcc_sensitivity   |   2 |           0.0191207   | 5.41391e-07 |  0.0142136   |   0.0240279   | ✓        |
| deit_small |        0.2 | random           | akiec_sensitivity |   2 |           0.59559     | 2.52933e-19 |  0.57456     |   0.61662     | ✓        |
| deit_small |        0.4 | wanda            | balanced_acc      |   2 |           0.0551241   | 1.10672e-13 |  0.0504548   |   0.0597934   | ✓        |
| deit_small |        0.4 | wanda            | mel_sensitivity   |   2 |           0.107036    | 1.35957e-08 |  0.0863481   |   0.127724    | ✓        |
| deit_small |        0.4 | wanda            | bcc_sensitivity   |   2 |           0.0833015   | 2.71048e-16 |  0.0786043   |   0.0879988   | ✓        |
| deit_small |        0.4 | wanda            | akiec_sensitivity |   2 |          -0.00294389  | 0.378281    | -0.0098551   |   0.00396732  |          |
| deit_small |        0.4 | taylor           | balanced_acc      |   2 |           0.0313145   | 3.4311e-23  |  0.0307046   |   0.0319243   | ✓        |
| deit_small |        0.4 | taylor           | mel_sensitivity   |   2 |          -0.0304178   | 6.24251e-08 | -0.0370121   |  -0.0238236   | ✓        |
| deit_small |        0.4 | taylor           | bcc_sensitivity   |   2 |          -0.0106332   | 0.0570527   | -0.0216284   |   0.000361987 |          |
| deit_small |        0.4 | taylor           | akiec_sensitivity |   2 |           0.0978411   | 1.93179e-06 |  0.0699833   |   0.125699    | ✓        |
| deit_small |        0.4 | random           | balanced_acc      |   2 |           0.704246    | 1.29155e-19 |  0.680473    |   0.72802     | ✓        |
| deit_small |        0.4 | random           | mel_sensitivity   |   2 |           0.860805    | 9.8452e-31  |  0.855541    |   0.866069    | ✓        |
| deit_small |        0.4 | random           | bcc_sensitivity   |   2 |           0.460042    | 0.00204583  |  0.196577    |   0.723507    | ✓        |
| deit_small |        0.4 | random           | akiec_sensitivity |   2 |           0.368737    | 0.00800642  |  0.111548    |   0.625926    | ✓        |
| deit_small |        0.6 | wanda            | balanced_acc      |   2 |           0.15103     | 2.67443e-21 |  0.147096    |   0.154964    | ✓        |
| deit_small |        0.6 | wanda            | mel_sensitivity   |   2 |           0.14172     | 9.09552e-16 |  0.13305     |   0.150391    | ✓        |
| deit_small |        0.6 | wanda            | bcc_sensitivity   |   2 |           0.777942    | 2.27638e-21 |  0.757894    |   0.79799     | ✓        |
| deit_small |        0.6 | wanda            | akiec_sensitivity |   2 |          -0.451397    | 4.14104e-16 | -0.477587    |  -0.425206    | ✓        |
| deit_small |        0.6 | taylor           | balanced_acc      |   2 |          -0.105203    | 2.31127e-10 | -0.120338    |  -0.090068    | ✓        |
| deit_small |        0.6 | taylor           | mel_sensitivity   |   2 |          -0.525214    | 4.58134e-18 | -0.54773     |  -0.502698    | ✓        |
| deit_small |        0.6 | taylor           | bcc_sensitivity   |   2 |           0.139615    | 2.71028e-12 |  0.124891    |   0.154339    | ✓        |
| deit_small |        0.6 | taylor           | akiec_sensitivity |   2 |          -0.114465    | 0.0432741   | -0.224989    |  -0.00394183  | ✓        |
| deit_small |        0.6 | random           | balanced_acc      |   2 |           0.383788    | 1.90205e-27 |  0.379901    |   0.387675    | ✓        |
| deit_small |        0.6 | random           | mel_sensitivity   |   2 |           0.204335    | 7.98227e-16 |  0.191943    |   0.216727    | ✓        |
| deit_small |        0.6 | random           | bcc_sensitivity   |   2 |           0.485123    | 0.00174401  |  0.212998    |   0.757248    | ✓        |
| deit_small |        0.6 | random           | akiec_sensitivity |   2 |           0.412607    | 1.08191e-15 |  0.387066    |   0.438148    | ✓        |