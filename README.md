# MSRIGS
Multi-Stage Re-Identification Game Solver (MSRIGS)

## Description:

This program finds the best solution for sharing Y-chromesome genomic attribures (Y-STRs) under an economically motivated adversary's surname inference and re-identification two-stage attack based on a Stackelberg game model named Surname Inference and Re-Identification Game. The attack is introduced by Gymrek et al. in 2013. In the first stage of the attack, the adversary infers the surname of a target from the target's Y-STR DNA sequence with the help of a published genetic geneaolgy dataset (e.g. Ysearch). In the second stage of the attack, the adversary re-identifies the target with the help of a public demogrphic dataset (e.g. voter registration list) by linking upon the inferred surname and other demographic attributes (e.g., birth year and State).

The game theoretic protection model is based upon Wan et al.'s Re-identification Game introduced in 2015 and Wan et al.'s Genomic Privacy Game introduced in 2017. Two players in the game are a data provider and an adversary. The adversary's strategy is either to attack or not, for each target. The data provider's strategy space is dependent upon specific scenarios in consideration. Two game scenarios are considered in this model: the "opt-in" game scenario, and the "masking" game scenario. In the "opt-in" game scenario, we assume the data provider's strategy is either to share the entire data reocrd or not. In the "masking" game scenario, we assume the data provider can decide either to share or not for each attribute in the data record.

## References:

This code is partially based on our journal paper: 

[0] Zhiyu Wan, Yevgeniy Vorobeychik, Weiyi Xia, Ellen Wright Clayton, Murat Kantarcioglu, and Bradley Malin. "Preventing Biomedical Data Re-identification in the Face of Multi-stage Privacy Attacks Through Game Theory" that is uncer review.

Other published articles essential for understanding the software are as follows:

[1] M. Gymrek, A. L. McGuire, D. Golan, E. Halperin, and Y. Erlich. Identifying personal genomes by surname inference. Science, 339(6117):321–324, 2013

[2] Z. Wan, Y. Vorobeychik, W. Xia, E. W. Clayton, M. Kantarcioglu, R. Ganta, R. Heatherly, and B. A. Malin. A game theoretic framework for analyzing re-identification risk. PloS one, 10(3): e0120592, 2015.

[3] Z. Wan, Y. Vorobeychik, W. Xia, E. W. Clayton, M. Kantarcioglu, and B. Malin. Expanding Access to Large-Scale Genomic Data While Promoting Privacy: A Game Theoretic Approach. The American Journal of Human Genetics, 100(2):316–322, 2017.

## Software Disclaimer:

MSRIGS is a free software; you can redistribute it or modify it under the terms of the GNU General Public License. 

MSRIGS is distributed in the hope that is will be useful, but without any warranty. To use it in your research, please cite our journal paper uner review mentioned above.

## Authors:

Zhiyu Wan

## Copyright:

Copyright 2020 Zhiyu Wan

## Questions:

For any questions, please contact me via zhiyu dot wan AT vanderbilt dot edu
