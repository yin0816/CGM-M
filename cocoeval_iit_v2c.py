import os
import glob
import sys
import pickle

import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import python3 coco-caption
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# All used metrics
# METRICS = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
METRICS = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L", "CIDEr"]

# From COCOEval code
class COCOScorer(object):
    def __init__(self):
        print('init COCO-EVAL scorer')
            
    def score(self, GT, RES, IDs, result_file):
        self.eval = {}
        self.imgToEval = {}
        gts = {}
        res = {}
        for ID in IDs:
            gts[ID] = GT[ID]
            res[ID] = RES[ID]
        #print('Tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # Set up scorers
        #print('Setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # Compute scores
        eval = {}
        self.final_results = []
        for scorer, method in scorers:
            print('Computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, IDs, m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, IDs, method)
                print("%s: %0.3f"%(method, score))
        
        print()
        # Collect scores by metrics
        for metric in METRICS:
            self.final_results.append(self.eval[metric])
        self.final_results = np.array(self.final_results)

        return self.eval
    
    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

def read_prediction_file(prediction_file):
    """Helper function to read generated prediction files.
    """
    # Create dicts for ground truths and predictions
    gts_dict, pds_dict = {}, {}
    f = open(prediction_file, 'r')
    lines = f.read().split('\n')
    f.close()

    for i in range(0, len(lines) - 4, 4):
        id_line = lines[i+1]
        gt_line = lines[i+2]
        pd_line = lines[i+3]
            
        # Build individual ground truth dict
        curr_gt_dict = {}
        curr_gt_dict['image_id'] = id_line
        curr_gt_dict['cap_id'] = 0 # only 1 ground truth caption
        curr_gt_dict['caption'] = gt_line
        gts_dict[id_line] = [curr_gt_dict]
            
        # Build current individual prediction dict
        curr_pd_dict = {}
        curr_pd_dict['image_id'] = id_line
        curr_pd_dict['caption'] = pd_line
        pds_dict[id_line] = [curr_pd_dict]
    
    return gts_dict, pds_dict

def test_iit_v2c():
    """Helper function to test on IIT-V2C dataset.
    """
    # Get all generated predicted files
    prediction_files = sorted(glob.glob(os.path.join(ROOT_DIR, 'checkpoints', 'prediction', '*.txt')))
    
    scorer = COCOScorer()
    max_scores = np.zeros((len(METRICS), ), dtype=np.float32)
    max_file = None
    for prediction_file in prediction_files:

        file_name = prediction_file.split('\\')[-1]
        print('file_name =', file_name)

        gts_dict, pds_dict = read_prediction_file(prediction_file)
        ids = list(gts_dict.keys())
        scorer.score(gts_dict, pds_dict, ids, prediction_file)
        if np.sum(scorer.final_results) > np.sum(max_scores):
            max_scores = scorer.final_results
            max_file = prediction_file

    print('Maximum Score with file', max_file)
    for i in range(len(max_scores)):
        print('%s: %0.3f' % (METRICS[i], max_scores[i]))


def val_iit_v2c(file_path):
    """Helper function to test on IIT-V2C dataset.
    """
    if not hasattr(val_iit_v2c, 'max_scores'):  # hasattr函数的第一个变量为当前函数名，第二个为变量名，加单引号
        val_iit_v2c.max_scores = 0  # 注意之后使用这个变量时一定要在变量名前加  函数名.

    scorer = COCOScorer()
    val_iit_v2c.max_scores = np.zeros((len(METRICS),), dtype=np.float32)

    gts_dict, pds_dict = read_prediction_file(file_path)
    ids = list(gts_dict.keys())
    scorer.score(gts_dict, pds_dict, ids, None)

    if np.sum(scorer.final_results) > np.sum(val_iit_v2c.max_scores):
        val_iit_v2c.max_scores = scorer.final_results

    # scorer_sum = np.sum(scorer.final_results)

    # print('Maximum Score with file', max_file)
    # for i in range(len(max_scores)):
    #     print('%s: %0.3f' % (METRICS[i], max_scores[i]))

    return scorer.final_results


def cocoevalcap_test():
    scorer = COCOScorer()
    # curr_gt_dict = {}
    # curr_gt_dict['image_id'] = id_line
    # curr_gt_dict['cap_id'] = 0  # only 1 ground truth caption
    # curr_gt_dict['caption'] = gt_line
    # gts_dict[id_line] = [curr_gt_dict]
    #
    # # Build current individual prediction dict
    # curr_pd_dict = {}
    # curr_pd_dict['image_id'] = id_line
    # curr_pd_dict['caption'] = pd_line
    # pds_dict[id_line] = [curr_pd_dict]

    # [{'image_id': '0', 'cap_id': 0, 'caption': 'bothhand open egg carton'}]

    # ref = {'1': [{'image_id': '0', 'caption': 'go down the stairs and stop at the bottom'}]}
    # gt = {'1': [{'image_id': '0', 'cap_id': 0, 'caption': 'Walk down the steps and stop at the bottom'}]}
    gt = {
            # '1': [{'image_id': '0', 'cap_id': 0, 'caption': 'bothhand open egg carton'}],
            '2': [{'image_id': '1', 'cap_id': 0, 'caption': 'lefthand take fruit'}],
          }
    ref = {
            # '1': [{'image_id': '0', 'caption': 'bothhand open egg carton'}],
            '2': [{'image_id': '1', 'caption': 'lefthand take'}],  # bothhand open egg carton  lefthand take fruit'
           }


    # from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score

    # def meteor_score(
    #         references,
    #         hypothesis,
    #         preprocess=str.lower,
    #         stemmer=PorterStemmer(),
    #         wordnet=wordnet,
    #         alpha=0.9,
    #         beta=3,
    #         gamma=0.5,
    # ):
    # import nltk
    # nltk.download('wordnet')

    gt_meteor = ['bothhand open egg']
    ref_meteor = 'bothhand open egg carton'

    meteor_score_nltk = round(meteor_score(gt_meteor, ref_meteor), 4)
    print('meteor_score_nltk =', meteor_score_nltk)

    gts_dict, pds_dict = gt, ref
    ids = list(gts_dict.keys())
    scorer.score(gts_dict, pds_dict, ids, None)
    final_results = scorer.final_results

    for i in range(len(METRICS)):
        print('%s: %0.3f' % (METRICS[i], final_results[i]))

def nltk_bleu():
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    chencherry = SmoothingFunction()
    reference = [['bothhand', 'open', 'egg']]
    candidate = ['bothhand', 'open', 'egg', 'carton']
    score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method7)
    print(score)

    print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
    print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
    print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
    print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))



if __name__ == '__main__':
    # test_iit_v2c()
    # print('evaluate_COCO Done!')
    cocoevalcap_test()
    # nltk_bleu()

    # import nltk
    # nltk.download('wordnet')
    pass