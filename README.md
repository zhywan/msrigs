# Multi-Stage Re-Identification Game Solver (MSRIGS)
[![DOI](https://zenodo.org/badge/257110497.svg)](https://zenodo.org/badge/latestdoi/257110497)

## Description:

This program finds the best solution for sharing genomic attributes, Y-chromosome short tandem repeats (Y-STRs), under an economically motivated adversary's surname inference and re-identification two-stage attack based on a Stackelberg game model named Multi-Stage Re-Identification Game (MSRIG). The attack is introduced by Gymrek et al. in 2013 [1]. In the first stage of the attack, the adversary infers the surname of a target from the target's Y-STR data with the help of a published genetic genealogy dataset (e.g., Ysearch). In the second stage of the attack, the adversary re-identifies the target with the help of a public demographic dataset (e.g., a voter registration list) by linking upon the inferred surname and other demographic attributes (e.g., year of birth and state of residence).

The game theoretic protection model is based upon Wan et al.'s Re-identification Game introduced in 2015 [2] and Wan et al.'s Genomic Privacy Game introduced in 2017 [3]. Two players in the game are a data subject and an adversary. The adversary's strategy is either to attack or not to attack, for each target. The data subject's strategy space is dependent upon specific scenarios in consideration. Two game scenarios are mainly considered in this model: the opt-in game scenario and the masking game scenario. In the opt-in game scenario, we assume that the data subject's strategy is either to share the entire data record or not. In the masking game scenario, we assume that the data subject can decide whether to share or not for each attribute in the data record.

## References:

This code is partially based on our journal paper: 

[0] Z. Wan, Y. Vorobeychik, W. Xia, Y. Liu, M. Wooders, J. Guo, Z. Yin, E. W. Clayton, M. Kantarcioglu, and B. Malin. "Using game theory to thwart multi-stage privacy intrusions when sharing data" (under review).

Other published articles essential for understanding the software are as follows:

[1] M. Gymrek, A. L. McGuire, D. Golan, E. Halperin, and Y. Erlich. Identifying personal genomes by surname inference. Science, 339(6117): 321–324, 2013

[2] Z. Wan, Y. Vorobeychik, W. Xia, E. W. Clayton, M. Kantarcioglu, R. Ganta, R. Heatherly, and B. A. Malin. A game theoretic framework for analyzing re-identification risk. PloS one, 10(3): e0120592, 2015.

[3] Z. Wan, Y. Vorobeychik, W. Xia, E. W. Clayton, M. Kantarcioglu, and B. Malin. Expanding Access to Large-Scale Genomic Data While Promoting Privacy: A Game Theoretic Approach. The American Journal of Human Genetics, 100(2): 316–322, 2017.

## Software Structure:

### Preprocessing -- Large-scale population simulation

Source code file:

            /Population_Simulation/Population_Simulation_Final.ipynb

Data files:

            /Population_Simulation/Birth_Age_statistics.txt
            
            /Population_Simulation/census_lastnames_freq.txt
            
            /Population_Simulation/Income_statistics.txt
            
            /Population_Simulation/Pop_Increase_stats.txt
            
            /Population_Simulation/State_statistics.txt

Supplementary files:

            /Population_Simulation/census_lastnames_info_top1000.csv
            
            /Population_Simulation/state_stats_info.csv
            
            /Population_Simulation/state_stats_info.xlsx

### Preprocessing -- *k*-Anonymization baseline

Source code file:

            /DemoKAnonymization/src/edu/vanderbilt/mc/hiplab/demokanonymization/DemoKAnonymization.java

Data files:
            /DemoKAnonymization/exp/2058/data/hierarchy/hierarchy_State.csv
            
            /DemoKAnonymization/exp/2058/data/hierarchy/hierarchy_YOB.csv
            
            /DemoKAnonymization/exp/2058/data/hierarchy/hierarchy_STR1.csv
            
            ...
            
            /DemoKAnonymization/exp/2058/data/hierarchy/hierarchy_STR12.csv
            
            /DemoKAnonymization/exp/2058/data/weighted_entropy/i0.csv
            
            ...
            
            /DemoKAnonymization/exp/2058/data/weighted_entropy/i99.csv
            
            /DemoKAnonymization/exp/2058/data/target_data/i0.csv
            
            ...
            
            /DemoKAnonymization/exp/2058/data/target_data/i99.csv

Output files:

            /DemoKAnonymization/exp/2058/output_strategy/i0.csv
            
            ...
            
            /DemoKAnonymization/exp/2058/output_strategy/i99.csv

Library package file (not included, accessible from https://arx.deidentifier.org/downloads/):

            /DemoKAnonymization/lib/libarx-3.9.0.jar

### Experiments based on a large-scale simulated population

Source code files:

            /msrigs_simulation_mainexperiment.py
            
            /msrigs_functions.py

Data files:

            /data/simu/birth_year1.txt
            
            /data/simu/birth_year2.txt
            
            /data/simu/birth_year3.txt
            
            /data/simu/ped1.txt
            
            /data/simu/ped2.txt
            
            /data/simu/ped3.txt
            
            /data/simu/state1.txt
            
            /data/simu/state2.txt
            
            /data/simu/state3.txt
            
            /data/simu/surname1.txt
            
            /data/simu/surname2.txt
            
            /data/simu/surname3.txt

Visualization code files:

            /msrigs_drawfig2_mainresult.py
            
            /msrigs_drawfig3_optimalstrategies.py

### Sensitivity analysis on eight parameters based on simulated datasets

Source code files:

            /msrigs_simulation_sensitivityanalysis_mchanging.py
            
            /msrigs_simulation_sensitivityanalysis_missinglevelchanging.py
            
            /msrigs_simulation_sensitivityanalysis_thetachanging.py
            
            /msrigs_simulation_sensitivityanalysis_ngchanging.py
            
            /msrigs_simulation_sensitivityanalysis_nichanging.py
            
            /msrigs_simulation_sensitivityanalysis_losschanging.py
            
            /msrigs_simulation_sensitivityanalysis_benefitchanging.py
            
            /msrigs_simulation_sensitivityanalysis_costchanging.py
            
            /msrigs_functions.py

Data files:

            /data/simu/birth_year1.txt
            
            /data/simu/birth_year2.txt
            
            /data/simu/birth_year3.txt
            
            /data/simu/ped1.txt
            
            /data/simu/ped2.txt
            
            /data/simu/ped3.txt
            
            /data/simu/state1.txt
            
            /data/simu/state2.txt
            
            /data/simu/state3.txt
            
            /data/simu/surname1.txt
            
            /data/simu/surname2.txt
            
            /data/simu/surname3.txt

Visualization code files:

            /msrigs_drawfig4_sensitivityanalysis.py

### Sensitivity analysis on three settings based on simulated datasets

Source code files:

            /msrigs_simulation_homogeneityconstraint.py
            
            /msrigs_simulation_mainexperiment.py (by changing parameters)
            
            /msrigs_functions.py

Data files:

            /data/simu/birth_year1.txt
            
            /data/simu/birth_year2.txt
            
            /data/simu/birth_year3.txt
            
            /data/simu/ped1.txt
            
            /data/simu/ped2.txt
            
            /data/simu/ped3.txt
            
            /data/simu/state1.txt
            
            /data/simu/state2.txt
            
            /data/simu/state3.txt
            
            /data/simu/surname1.txt
            
            /data/simu/surname2.txt
            
            /data/simu/surname3.txt

Visualization code file:

            /msrigs_drawfig4_sensitivityanalysis.py

### Robustness analysis on three parameters based on simulated datasets

Source code files:

            /msrigs_simulation_robustanalysis_cost.py
            
            /msrigs_simulation_robustanalysis_ng.py
            
            /msrigs_simulation_robustanalysis_ni.py
            
            /msrigs_functions.py

Data files:

            /data/simu/birth_year1.txt
            
            /data/simu/birth_year2.txt
            
            /data/simu/birth_year3.txt
            
            /data/simu/ped1.txt
            
            /data/simu/ped2.txt            
            
            /data/simu/ped3.txt
            
            /data/simu/state1.txt
            
            /data/simu/state2.txt
            
            /data/simu/state3.txt
            
            /data/simu/surname1.txt
            
            /data/simu/surname2.txt
            
            /data/simu/surname3.txt

Visualization code file:

            /msrigs_drawfigs5_robustanalysis.py

### Experiments based on Craig Venter’s data and the Ysearch dataset

Source code files:

            /msrigs_realdata.py
            
            /msrigs_functions.py

Data files:

            /data/Venter.txt
            
            /data/Ysearch.txt
            
            /data/Ysearch_ID.txt
            
            /data/MU.txt

Visualization code file:

            /msrigs_drawfigs7_venterstrategies.py

Preprocessing code files (supplementary):

            /msrigs_realdata_sanitization_substitution.py
            
            /msrigs_realdata_sanitization_shuffle.py

### Usefulness and fairness analyses

Source code files:

            /msrigs_simulation_analysis_usefulness_and_fairness.py
            
            /msrigs_functions.py

Visualization code files:

            /msrigs_drawfigs1_usefulness.py
            
            /msrigs_drawfigs2_fairness_wrt_usefulness.py

### Sensitivity analysis on the minority-support factor based on simulated datasets

Source code files:

            /msrigs_simulation_mainexperiment.py (by changing parameters)
            
            /msrigs_functions.py

Data files:

            /data/simu/birth_year1.txt
            
            /data/simu/birth_year2.txt
            
            /data/simu/birth_year3.txt
            
            /data/simu/ped1.txt
            
            /data/simu/ped2.txt
            
            /data/simu/ped3.txt
            
            /data/simu/state1.txt
            
            /data/simu/state2.txt
            
            /data/simu/state3.txt
            
            /data/simu/surname1.txt
            
            /data/simu/surname2.txt
            
            /data/simu/surname3.txt

Visualization code file:

            /msrigs_drawfigs6_sensitivityminoritysupport.py

## Software Disclaimer:

MSRIGS is a free software; you can redistribute it or modify it under the terms of the GNU General Public License. 

MSRIGS is distributed in the hope that it will be useful, but without any warranty. To use it in your research, please cite our journal paper under review (mentioned above).

## Authors:

Zhiyu Wan

## Copyright:

Copyright 2020-2021 Zhiyu Wan

## Questions:

For any questions, please contact me via zhiyu dot wan AT vanderbilt dot edu.
