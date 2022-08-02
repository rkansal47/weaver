# Weaver

Clone of original weaver repo for HWW trainings.

## For new plots

e.g.

```bash 
python new_plots/make_plots.py --inferences-dir /hwwtaggervol/inferences/rk/04_18_ak8_qcd_oneweight/ --plot-dir /hwwtaggervol/plots/rk/04_18_ak8_qcd_oneweight/
```

## For testing new models

Change the `--data-train` input to wherever your small sample files are located.

```bash
python train.py --data-train '../sample_data/*/*.root' --gpus "" --num-workers 0 --data-config data/cm/ak15_4q_flat_eta_genHm_pt300.yaml --network-config networks/particle_net_pf_sv_4_layers_pyg_ef.py --model-prefix models/ef_test/ef_test --batch-size 8 --log logs/ef_test.log
```
