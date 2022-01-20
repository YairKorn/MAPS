# Pseudo Sequential - MARL Architecture
  
! Under work !
In case of import errors, add the next command to ~/.bashsrc:
```
export PYTHONPATH="${PYTHONPATH}:./src"
```

After finished, delete all occurences of #$


## Installation instructions
Installation instruction from PSeq repository

Follow the first 2 steps from [PyMARL](https://github.com/oxwhirl/pymarl) :  
1. Build the Dockerfile using     
```shell 
cd docker 
bash build.sh 
``` 
2. Set up StarCraft II and SMAC:    
```shell 
bash install_sc2.sh
 ``` 
This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.  

Finally, copy the StarCraft II maps used in the [DCG paper](https://arxiv.org/abs/1910.00091):
```shell 
cp -f maps/* ./3rdparty/StarCraftII/Maps/Melee 
```
### Using an existing PyMARL copy
Both algorithms and environments can be used with existing versions of  [PyMARL](https://github.com/oxwhirl/pymarl)  as well. 

The __environments__ and the __maps__ can be imported by copying: 
```
maps/coverage_maps/*
src/config/envs/adv_coverage.yaml
src/envs/adv_coverage.py
```
and by appending `src/envs/__init__.py` with: 
```
from .adv_coverage import AdversarialCoverage
REGISTRY["mrac"] = partial(env_fn, env=AdversarialCoverage)
```
The __algorithms__ can be imported by copying:
```
src/action_model/*
src/components/mcts_buffer.py
src/config/algs/pseq.yaml
src/controllers/pseq_controller.py
src/learners/pseq_learner.py
src/learners/TDn_learner.py
```
by appending `src/controllers/__init__.py` with:
```
from .pseq_controller import PSeqMAC
REGISTRY["pseq"] = PSeqMAC
```
by appending `src/learners/__init__.py` with:
```
from .pseq_learner import PSeqLearner
REGISTRY["pseq_learner"] = PSeqLearner
```

# To Be Continued...

## Replicate the experiments  
As in the [PyMARL](https://github.com/oxwhirl/pymarl) framework, all experiments are run like this:  
```shell  
python3 src/main.py --config=$ALG --env-config=$ENV with $PARAMS  
```  
The experiments should be run in a Docker container, to avoid installing complicated dependencies:  
```shell  
bash run.sh $GPU python3 src/main.py --config=$ALG --env-config=$ENV with $PARAMS  
```  
The `sacred` logs containing the results will be stored as `json` files in the `results` folder.  

### Parameters for different algorithms

| Algorithms | `$ALG`        | `$PARAMS`                   | Comment |
| ---------- | ------------- | --------------------------- |  ------ |
| DCG 	     | `dcg`         |                             | DCG w/o low-rank approximation |
| DCG-S      | `dcg`         | `duelling=True`             | DCG with privileged bias function |
| DCG (rank `$K`) | `dcg`    | `cg_payoff_rank=$K`         | DCG with low-rank approximation |
| DCG (nps)  | `dcg_noshare` |                             | DCG w/o parameter sharing |
| CG         | `cg`          |                             | DCG (nps) with central observations |
| QTRAN      | `qtran`       |                             | from PyMARL |
| QMIX       | `qmix`        |                             | from PyMARL |
| IQL        | `iql`         |                             | from PyMARL |
| VDN (pymarl) | `vdn`       |                             | from PyMARL |
| VDN (dcg)  | `dcg`         | `cg_edges=vdn`              | VDN using DCG classes |
| VDN-S      | `dcg`         | `cg_edges=vdn duelling=True` | VDN with privileged bias function |
| LRQ (rank `$K`) | `lrq`    | `low_rank=$K`               | Low-rank joint Q-value |

All DCG variants and CG can have the following topologies:

| Topologies | `$ALG`                   | `$PARAMS`        |
| ---------- | ------------------------ | ---------------- | 
| DCG 	     | `dcg`/`dcg_noshare`/`cg` |                  |
| CYCLE      | `dcg`/`dcg_noshare`/`cg` | `cg_edges=cycle` |
| LINE       | `dcg`/`dcg_noshare`/`cg` | `cg_edges=line`  |
| STAR       | `dcg`/`dcg_noshare`/`cg` | `cg_edges=star`  |
| VDN        | `dcg`/`dcg_noshare`/`cg` | `cg_edges=vdn`   |
| `$N` rand. edges | `dcg`/`dcg_noshare`/`cg` | `cg_edges=$N`  |
| given topology | `dcg`/`dcg_noshare`/`cg` | `cg_edges=$LIST` |

`$LIST` must be a list of tuples of node indices (starting with 0), for example, `$LIST="[(0,1),(1,2),(2,3),(3,0)]"` for a cycle of 4 agents.


### Parameters for individual plots

| Experiment | `$ENV`        | `$PARAMS`                     | Task                        |
| ---------- | ------------- | ----------------------------- | --------------------------- |
| Fig. 2a    | `rel_overgen` | `env_args.miscapture_punishment=0`     | Relative overgeneralization |
| Fig. 2b    | `rel_overgen` | `env_args.miscapture_punishment=-1`    | Relative overgeneralization |
| Fig. 2c    | `rel_overgen` | `env_args.miscapture_punishment=-1.25` | Relative overgeneralization |
| Fig. 2d    | `rel_overgen` | `env_args.miscapture_punishment=-1.5`  | Relative overgeneralization |
| Fig. 3a-c  | `rel_overgen` | `env_args.miscapture_punishment=-2`    | Relative overgeneralization |
| Fig. 4a-c  | `ghost_hunt`  |                               | Artificial decentralization |
| Fig. 5, 8a | `sc2`         | `env_args.map_name=MMM2`      | StarCraft II                |
| Fig. 8b    | `sc2`         | `env_args.map_name=so_many_baneling` | StarCraft II         |
| Fig. 8c    | `sc2`         | `env_args.map_name=8m_vs_9m`  | StarCraft II                |
| Fig. 8d    | `sc2`         | `env_args.map_name=3s_vs_5z`  | StarCraft II                |
| Fig. 8e    | `sc2`         | `env_args.map_name=3s5z`      | StarCraft II                |
| Fig. 8f    | `sc2`         | `env_args.map_name=micro_focus` | StarCraft II              |

  
## Citing DCG   
If you use DCG in your research, or any other implementation provided here that is not included in [PyMARL](https://github.com/oxwhirl/pymarl), please cite the [DCG paper](https://arxiv.org/abs/1910.00091): 
  
*W. B&ouml;hmer, V. Kurin and S. Whiteson. Deep Coordination Graphs, to appear at the International Conference on Machine Learning (ICML), 2020. URL: https://arxiv.org/abs/1910.00091*  
  
In BibTeX format:  
```tex  
@InProceedings{boehmer2020dcg,  
    title = {Deep Coordination Graphs}, 
    author = {Wendelin B\"ohmer and Vitaly Kurin and Shimon Whiteson}, 
    booktitle = {International Conference on Machine Learning}, 
    url = {https://arxiv.org/abs/1910.00091}, 
    year = {2020},
}  
```  
As DCG uses [PyMARL](https://github.com/oxwhirl/pymarl), you should also cite the [SMAC paper](https://arxiv.org/abs/1902.04043):

_M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019._

In BibTeX format:
```
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```
  
## License  
  
Code licensed under the Apache License v2.0
