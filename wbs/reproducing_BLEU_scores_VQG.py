from nltk.translate.bleu_score import corpus_bleu
from nlg_eval.nlgeval import NLGEval

path = '' # put your path here

file_names = [
              'bert_vqg_caption_unfreeze.tsv',
              'bert_vqg_image_caption_from_image_freeze.tsv',
              'bert_vqg_step1txt.tsv',
              'bert_vqg_step2bis.tsv',
              'bert_vqg_step3.tsv'
]


preproc = lambda x: x.lower().split()

nlge = NLGEval(no_glove=True, no_skipthoughts=True)
for file_name in file_names:
    d_srcs = dict()
    with open(path + file_name, 'r') as f:
        lines = f.readlines()

    for line in lines:
        src, _, ref, hyp = line.split('\t')

        ref = preproc(ref)
        hyp = preproc(hyp)
        ref = str(' '.join(ref))
        hyp = str(' '.join(hyp))
        d_srcs[src] = {'refs': ref, 'hyp': hyp}
        
    refss = [d_srcs[src]['refs'] for src in d_srcs]
    hyps = [d_srcs[src]['hyp'] for src in d_srcs]

    assert len(refss) == len(hyps)


    scores=nlge.compute_metrics(ref_list=[refss], hyp_list=hyps)
    print(scores)
