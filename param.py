import torch

log_folder = "./log"
path = './UMLSdata'  # './WN18data'  './FB15K237data'   './UMLSdata'
k = 4  # number of arms
topk = 326 if path == './UMLSdata' else 4650 if path == './WN18data' else 16281  # 326 4650
ratio = 0.2  # ration in [0.2,0.6,1], refers to 1%,3%,5%
oppratio = 0.5  # opportunity/overlaps to control limited opportunities
alpha = 100
eye = 300 if topk == 326 else 150  # 150 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
