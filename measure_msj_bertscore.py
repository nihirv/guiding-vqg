from TextGenerationEvaluationMetrics.multiset_distances import MultisetDistances
from TextGenerationEvaluationMetrics.bert_distances import FBD

import pandas as pd
import csv
import json


gts = []
preds = []

# data = pd.read_csv("wbs/bert_vqg_step3.tsv")
# print("What is the man wearing on his face ?"[:-1].lower().strip())

tsv_file = open("wbs/bert_vqg_step3.tsv")
read_tsv = csv.reader(tsv_file, delimiter="\t")
for row in read_tsv:
    gts.append(row[-2].lower())
    preds.append(row[-1].lower())

#with open("/data/nv419/machine_drive/guided-vqg/data/processed/explicit_model_outputs.json", "r") as f:
#    data = json.load(f)


#for k,v in data.items():
#    for item in v:
#        gg = item["real_q"][:-1]
#        pp = item["generated_q"][:-1]
#        gts.append(gg)
#        preds.append(pp)
print(len(gts))


with open("gts_wbs.txt", "w") as f:
    for item in gts:
        f.write(item+'\n')

with open("preds_wbs.txt", "w") as f:
    for item in preds:
        f.write(item+'\n')


# msd = MultisetDistances(references=gts)
# msj_distance = msd.get_jaccard_score(sentences=preds)
# print(msj_distance)

# fbd = FBD(references=gts, model_name="bert-base-uncased", bert_model_dir="/homes/zw5018/.cache/huggingface/transformers/")
# fbd_distance_sentences = fbd.get_score(sentences=preds)
# print(fbd_distance_sentences)



