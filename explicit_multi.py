import json

gts_json = "/data2/zwang/data/vqg/v2_OpenEnded_mscoco_val2014_questions.json"


with open(gts_json, 'r') as f:
    gts_json = json.load(f)


gts_dict = {}
for item in gts_json["questions"]:
    img_id = item["image_id"]
    question = item["question"]

    if img_id not in gts_dict:
        gts_dict[img_id] = [question.lower()]
    else:
        gts_dict[img_id].append(question.lower())


pred_json = "/data/zwang/projects/controllable-vqg/data/processed/explicit_model_outputs.json"
with open(pred_json, 'r') as f:
    pred_json = json.load(f)        
    
pred_list = []
gts_list = []

pred_dict = {}
for k,v in pred_json.items():
    img_id = int(k)
    questions = [item["generated_q"] for item in v]
    if img_id not in pred_dict:
        pred_dict[img_id] = questions

    for item in questions:
        pred_list.append(item)
        gts = gts_dict[img_id]
        gts_list.append(gts)

print(len(gts_list))
print(len(pred_list))

from nlg_eval.nlgeval import NLGEval
nlge = NLGEval(no_glove=True, no_skipthoughts=True)
scores = nlge.compute_metrics(ref_list=gts_list, hyp_list=pred_list)

print(scores)
