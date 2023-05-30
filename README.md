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
docker run -it --rm --user $(id -u):$(id -g) --shm-size=1024M -v $PWD:/home/workspace <your_docker_repo>/dynamo:v0 /bin/bash -c "cd /home/workspace/src && python3 main.py <configuration_file_path> <dataset_file_path> ./<output_file_name>.csv"
```
Example:
```
docker run -it --rm --user $(id -u):$(id -g) --shm-size=1024M -v $PWD:/home/workspace <your_docker_repo>/dynamo:v0 /bin/bash -c "cd /home/workspace/src && python3 main.py ../res/E-Linus/config.json ../data/E-Linus/P1GD.csv ./P1GD_evaluation.csv"
```

## Configuration Files
⚠️ UPDATE 30/05/2023 ⚠️

Each dataset should have their own configuration files. In this repository, you'll find the E-Linus and SmallerDatasets JSON configuration files that setup DynAmo's execution according to the best hyperparameters reported in the paper.

To write your configuration file, follow these steps:

1. Create a JSON file under the ```res``` directory. E.g. ```cd res && mkdir <your_dataset_name> && cd <your_dataset_name> && touch config.json```
2. The ```config.json``` file should must contain these 3 keys: ```data```, ```dynamo```, and ```eval_strategy```
3. ```data``` contains the necessary information on how to process the dataset (e.g., where the label is situated) and the generation of the trajectory via the DyClee [1] method
4. ```dynamo``` contains the necessary information to run DynAmo (e.g., ```lookup_size```, ```drift_detection_threshold```)
5. ```eval_strategy``` contains a list of evaluation metrics that are used to measure the performance

We invite you to see ```../res/E-Linus/config.json``` for an example of a configuration file. Since DynAmo is a flexible framework, you can modify this configuration file according to your needs and domain adaptations.

[1] Roa, N.B., Travé-Massuyès, L. and Grisales-Palacio, V.H., 2019. DyClee: Dynamic clustering for tracking evolving environments. Pattern Recognition, 94, pp.162-186.

# Bayesian Optimisation
If you wish to perform your own Bayesian optimisation - i.e., use custom hyperparamter search spaces - follow these steps:

1. ```docker run -it --rm --user $(id -u):$(id -g) --shm-size=1024M -v $PWD:/home/workspace <your_docker_repo>/dynamo:v0 /bin/bash```
2. ```cd /home/workspace/src/experiments```
3. ```python3 optimize.py <path_to_config_file> <path_to_dataset_folder> --trial_num <trial_num> --timeout <timeout> --seed <seed> --drift_detection_threshold <two_comma_sep_values> --lookup_size <two_comma_sep_values> --limit_per_window <two_comma_sep_values> --window_moving_step <two_comma_sep_values>```
4. In addition to the previous arguments, the ```optimize.py``` script can take the optional arguments ```--plot``` if you want to see plots of the hyperparamter optimisation history, and ```--store``` if you want to store the best average performances after the optimisation finishes
5. Not specifying the optional arguments (i.e., those with preceding ```--```) will execute the Bayesian optimisation as done in the paper.

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

# Contacts
Ask <a href="mailto:bardhprenkaj95@gmail.com">bardhprenkaj95@gmail.com</a> for any doubts.

# Acknowledgement
We build upon <a href="https://github.com/harenbrs/pyclee">PyClee</a>.
