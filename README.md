# DynAmo

The official PyTorch implementation of the arXiv Paper <b>Unsupervised Detection of Gradual Behavioural Drifts with Dynamic Clustering and Hyperbox Trajectory Evolution Monitoring
</b>
Visit my webpage for more details

![DynAmo](dynamo.png?raw=true "DynAmo workflow")

# Content
```
.
|   dynamo.png
|   LICENSE
|   README.md
+---data
|   +---E-Linus
|   |       P1GD.csv
|   |       P1GI.csv
|   |       P2GD.csv
|   |       P2GI.csv     
|   \---SmallerDatasets
|           ARAS.csv
|           PolimiHouse.csv
|           VanKastareen.csv
+---Docker
|       Dockerfile
|       os_requirements.txt
|       requirements.txt 
+---lib
|   \---pyclee
|           clusters.py
|           dyclee.py
|           forgetting.py
|           plotting.py
|           types.py       
+---res
|   +---E-Linus
|   |       config.json
|   |       
|   \---SmallerDatasets
|           config.json      
\---src
    |   main.py
    |   utils.py
    |   __init__.py
    +---eval
    |       eval_strategy.py
    |       metrics.py
    |       __init__.py  
    +---experiments
    |       optimize.py    
    \---prediction_strategy
        |   dynamo.py
        |   __init__.py
        +---divergency
        |       tests.py
        |       __init__.py 
        +---ensemble
        |       trackers.py
        |       __init__.py    
        \---voting
                consensus_functions.py
                __init__.py
```

# Citation
Please use the following citation:
```
@article{prenkaj2023unsupervised,
  title={Unsupervised Detection of Behavioural Drifts with Dynamic Clustering and Trajectory Analysis},
  author={Prenkaj, Bardh and Velardi, Paola},
  journal={arXiv preprint arXiv:2302.06228},
  year={2023}
}
```
