# HashNIRL

## 🔧 Environment Setup

To run the code successfully, please set up the environment by installing all required dependencies:

'''
conda env create -f environment.yaml
'''

This will create a new Conda environment with the necessary packages.

---

## 📦 Reproducing Results

Below is a brief introduction to the commands used to run different models in our codebase.

### To obtain results of ERM
simply run

'''
python main.py --model 1dcnn
'''

with corresponding datasets and model specifications.

### A simplistic example
To run with HashNIRL for Chunjian_area dataset

```
python main.py --dataset Chunjian_area --batch_size 64 --epochs 10 --lr 0.0005 --threshold 0.05 --window_size 2 --rate 0.75 --batch_rate 0.1 --drop_rate 0.4 --invCont_coe 1.6
```

### Running with the baselines:
- To test with invarTSmodel, simply run `python main.py --model invartsmodel`
- To test with DivIL, simply run `python baseline_models/DivIL/NIROOD/NIR_DivIL.py`
- To test with MSGNet, simply run `python baseline_models/MSGNet/run_longExp.py`
- To test with OOD-TV-IRM, simply run `python baseline_models/OOD-TV-IRM/OOD-TV/OOD-TV_NIR/main.py`
- To test with invariant learning baselines, specify `--num_envs=2` and use `--irm_opt` to be `irm`, `vrex`, `eiil` or `ib-irm` to specify the methods.

