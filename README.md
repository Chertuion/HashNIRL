# HashNIRL



# Reproduce results
In the below we give a brief introduction of the commands and their usage in our code. 
# To obtain results of ERM
simply run
'''
python main.py --model 1dcnn
'''
with corresponding datasets and model specifications.

# A a simplistic example to run with HashNIRL for Chunjian_area dataset
```
python main.py --dataset Chunjian_area --batch_size 64 --epochs 10 --lr 0.0005 --threshold 0.05 --window_size 2 --rate 0.75 --batch_rate 0.1 --drop_rate 0.4 --invCont_coe 1.6
```

# Running with the baselines:
- To test with invarTSmodel, simply run `python main.py --model invartsmodel`
- To test with DivIL, simply run `baseline_models/DivIL/NIROOD/NIR_DivIL.py`
- To test with MSGNet, simply run `baseline_models/MSGNet/run_longExp.py`
- To test with OOD-TV-IRM, simply run `python baseline_models/OOD-TV-IRM/OOD-TV/OOD-TV_NIR/main.py`
- To test with invariant learning baselines, specify `--num_envs=2` and use `--irm_opt` to be `irm`, `vrex`, `eiil` or `ib-irm` to specify the methods.

