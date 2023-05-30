# DynAmo

The official implementation of the arXiv Paper <b>Unsupervised Detection of Gradual Behavioural Drifts with Dynamic Clustering and Hyperbox Trajectory Evolution Monitoring
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

# Setup
## Environment

Build the docker image:
```
cd Docker
docker build -f ./Dockerfile  -t <your_docker_repo>/dynamo:v0 .
```

Run the docker image:
```
docker run -it --rm --user $(id -u):$(id -g) --shm-size=1024M -v $PWD:/home/workspace <your_docker_repo>/dynamo:vo /bin/bash -c "cd /home/workspace/src && python3 main.py <configuration_file_path> <dataset_file_path> ./<output_file_name>.csv"
```
Example:
```
docker run -it --rm --user $(id -u):$(id -g) --shm-size=1024M -v $PWD:/home/workspace <your_docker_repo>/dynamo:vo /bin/bash -c "cd /home/workspace/src && python3 main.py ../res/E-Linus/config.json ../data/E-Linus/P1GD.csv ./P1GD_evaluation.csv"
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
