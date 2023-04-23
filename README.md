# KAEL_WSDM23
Active Ensemble Learning for Knowledge Graph Error Detection, WSDM2023, Singapore

## Framework
We propose a new framwork to effectively combine a set of off-the-shelf KG error detection algorithms with minimum human annotations.<br>
It adaptively updates the ensemble learning policy in each iteration based on active queries as:<br>
<img src="/images/KAEL_running.jpg" alt="KAEL_running" width="300"/><br>
<img src="/images/KAEL.jpg" alt="KAEL" width="600"/>

## Usage
- Ranking Files
Run your chosen base error detectors on the raw dataset that contain errors, i.e., triples labeled as 1.<br>
Generate ranking files according to different scoring functions in different base detectors.
- Apply TransE to raw dataset and maintain a unified embedding table;
- Prepare an embedding file for each ranking file, make sure same entity/triple shares the same embedding vector;
- Check directories;
- run main.py.


## Reference in BibTex:
@inproceedings{dong2023active,
Title={Active ensemble learning for knowledge graph error detection},
Author={Dong, Junnan and Zhang, Qinggang and Huang, Xiao and Tan, Qiaoyu and Zha, Daochen and Zihao, Zhao},
Booktitle={Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
Year={2023}}
