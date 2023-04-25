# KAEL_WSDM23
Active Ensemble Learning for Knowledge Graph Error Detection, WSDM2023, Singapore

## Framework
We propose a new framwork to effectively combine a set of off-the-shelf KG error detection algorithms with minimum human annotations.<br>
It adaptively updates the ensemble learning policy in each iteration based on active queries as:<br>
<img src="/images/KAEL_running.jpg" alt="KAEL_running" width="300"/><br>
<img src="/images/KAEL.jpg" alt="KAEL" width="600"/>

## Model
- A Three-Stage Scheme based on the tailored MAB<br>
1. `initialize`: Initializes the parameters by prioritizing the overlaps from all base detectors.
2. `ranking`: Ranks the triples for each arm and temporarily store the best triples and their expectations for current arm.
3. `train`: Trains the tailored MAB within the remaining opportunities.
4. `application`: Applies the trained model and parameters for errors within the remaining iterations.

## Usage
- Ranking Files<br>
Run your chosen base error detectors on raw datasets that contain errors, i.e., triples labeled as 1.<br>
Generate ranking files according to different scoring functions in different base detectors.
- Apply TransE to raw dataset and maintain a unified embedding table;
- Prepare an embedding file for each ranking file, make sure same entity/triple shares the same embedding vector;
- Check directories;
- Instantialize the model and define parameters, e.g., limited opportunities in `param.py`;
- run main.py.

## Reference in BibTex:
@inproceedings{dong2023active,<br>
Title={Active ensemble learning for knowledge graph error detection},<br>
Author={Dong, Junnan and Zhang, Qinggang and Huang, Xiao and Tan, Qiaoyu and Zha, Daochen and Zihao, Zhao},<br>
Booktitle={Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},<br>
Year={2023}}
