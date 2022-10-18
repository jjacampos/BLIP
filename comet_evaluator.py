import re
import random
import json
import copy
import pdb
import pdb
import bert_score
import nltk
import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
import pdb

MEMORY_BREAK = "<MM_BREAK>"
START_OF_MULTIMODAL_CONTEXTS = '<SOM>'
END_OF_MULTIMODAL_CONTEXTS = '<EOM>'
START_OF_API_CALL = '=> <SOAC>:'
END_OF_API_CALL = '<EOAC>'
START_OF_API_RESULT = '<SOAR>'
END_OF_API_RESULT = '<EOAR>'
START_OF_RESPONSE = "<SOR>"
END_OF_SENTENCE = '<EOS>'
SYSTEM = '<SYSTEM>'
USER = '<USER>'


class COMETEvaluator:
    
    def __init__(self):
        self.bert_scorer = bert_score.BERTScorer(lang="en")

    def _normalize_sentence(self, sentence):
        """Normalize the sentences and tokenize."""
        return nltk.tokenize.word_tokenize(sentence.lower())

    def _rec_prec_f1(self, n_correct, n_true, n_pred):
        rec = n_correct / n_true if n_true != 0 else 0
        prec = n_correct / n_pred if n_pred != 0 else 0
        f1 = 2 * prec * rec / (prec + rec) \
             if (prec + rec) != 0 else 0
        return rec, prec, f1

    def _evaluate_frame(self, true_frame, pred_frame, strict=True, lowercase=False):
        """
        If strict=True,
            For each dialog_act (frame), set(slot values) must match.
            If dialog_act is incorrect, its set(slot values) is considered wrong.
        """
        count_dict = self._initialize_count_dict()
        count_dict['n_frames'] += 1

        # Compare Dialog Actss
        true_act = true_frame['act'] if 'act' in true_frame else None
        pred_act = pred_frame['act'] if 'act' in pred_frame else None
        if not lowercase:
            b_correct_act = true_act == pred_act
        else:
            # Lowercase evaluation.
            b_correct_act = true_act.lower() == pred_act.lower()
        count_dict['n_correct_acts'] += b_correct_act
        count_dict['n_true_acts'] += 'act' in true_frame
        count_dict['n_pred_acts'] += 'act' in pred_frame

        # Compare Slots
        if not lowercase:
            true_frame_slot_values = {f'{k}={v}' for k, v in true_frame.get('slots', [])}
            pred_frame_slot_values = {f'{k}={v}' for k, v in pred_frame.get('slots', [])}
        else:
            true_frame_slot_values = {
                f'{k}={v}'.lower() for k, v in true_frame.get('slots', [])
            }
            pred_frame_slot_values = {
                f'{k}={v}'.lower() for k, v in pred_frame.get('slots', [])
            }

        count_dict['n_true_slots'] += len(true_frame_slot_values)
        count_dict['n_pred_slots'] += len(pred_frame_slot_values)
        
        if strict and not b_correct_act:
            pass
        else:
            count_dict['n_correct_slots'] += \
                                             len(true_frame_slot_values.intersection(pred_frame_slot_values))

        # Compare Request slots
        true_frame_request_slot_values = {rs for rs in true_frame.get('request_slots', [])}
        pred_frame_request_slot_values = {rs for rs in pred_frame.get('request_slots', [])}

        if not lowercase:
            true_frame_request_slot_values = {rs for rs in true_frame.get('request_slots', [])}
            pred_frame_request_slot_values = {rs for rs in pred_frame.get('request_slots', [])}
        else:
            true_frame_request_slot_values = {rs.lower() for rs in true_frame.get('request_slots', [])}
            pred_frame_request_slot_values = {rs.lower() for rs in pred_frame.get('request_slots', [])}

        count_dict['n_true_request_slots'] += len(true_frame_request_slot_values)
        count_dict['n_pred_request_slots'] += len(pred_frame_request_slot_values)

        if strict and not b_correct_act:
            pass
        else:
            count_dict['n_correct_request_slots'] += len(true_frame_request_slot_values.intersection(pred_frame_request_slot_values))

        # Compare Objects
        true_frame_object_values = {object_id for object_id in true_frame.get('memories', [])}
        pred_frame_object_values = {object_id for object_id in pred_frame.get('memories', [])}

        count_dict['n_true_objects'] += len(true_frame_object_values)
        count_dict['n_pred_objects'] += len(pred_frame_object_values)

        if strict and not b_correct_act:
            pass
        else:
            count_dict['n_correct_objects'] += len(true_frame_object_values.intersection(pred_frame_object_values))

        # Joint
        count_dict['n_correct_beliefs'] += (b_correct_act and true_frame_slot_values == pred_frame_slot_values and true_frame_request_slot_values == pred_frame_request_slot_values and true_frame_object_values == pred_frame_object_values)

        return count_dict

    
    def _evaluate_turn(self, true_turn, pred_turn, lowercase=False):
        count_dict = self._initialize_count_dict()

        # Must preserve order in which frames appear.
        for frame_idx in range(len(true_turn)):
            # For each frame
            true_frame = true_turn[frame_idx]
            if frame_idx >= len(pred_turn):
                pred_frame = {}
            else:
                pred_frame = pred_turn[frame_idx]

                count_dict = self._add_dicts(
                    count_dict,
                    self._evaluate_frame(
                        true_frame, pred_frame, strict=False, lowercase=lowercase
                    )
                )

        return count_dict
    

    def _add_dicts(self, d1, d2):
        return {k: d1[k] + d2[k] for k in d1}


    def _d_f1(self, n_true, n_pred, n_correct):
        # 1/r + 1/p = 2/F1
        # dr / r^2 + dp / p^2 = 2dF1 /F1^2
        # dF1 = 1/2 F1^2 (dr/r^2 + dp/p^2) 
        dr = self._b_stderr(n_true, n_correct)
        dp = self._b_stderr(n_pred, n_correct)
        
        r = n_correct / n_true if n_true else 0
        p = n_correct / n_pred if n_pred else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        
        d_f1 = 0.5 * f1**2 * (dr / r**2 + dp / p**2) if p * r != 0 else 0
        return d_f1
    
    def _evaluate_from_flat_list(self, d_true, d_pred, lowercase=False):
        
        c = self._initialize_count_dict()

        # Count # corrects & # wrongs
        for i in range(len(d_true)):
            true_turn = d_true[i]
            pred_turn = d_pred[i]
            turn_evaluation = self._evaluate_turn(true_turn, pred_turn, lowercase=lowercase)

            c = self._add_dicts(c, turn_evaluation)

        try:
            # Calculate metrics
            joint_accuracy = c['n_correct_beliefs'] / c['n_frames']
        
            act_rec, act_prec, act_f1 = self._rec_prec_f1(
                n_correct=c['n_correct_acts'],
                n_true=c['n_true_acts'],
                n_pred=c['n_pred_acts'])
        
            slot_rec, slot_prec, slot_f1 = self._rec_prec_f1(
                n_correct=c['n_correct_slots'],
                n_true=c['n_true_slots'],
                n_pred=c['n_pred_slots'])
        
            request_slot_rec, request_slot_prec, request_slot_f1 = self._rec_prec_f1(
                n_correct=c['n_correct_request_slots'],
                n_true=c['n_true_request_slots'],
                n_pred=c['n_pred_request_slots'])        
        
            object_rec, object_prec, object_f1 = self._rec_prec_f1(
                n_correct=c['n_correct_objects'],
                n_true=c['n_true_objects'],
                n_pred=c['n_pred_objects'])
        
            # Calculate std err
            act_f1_stderr = self._d_f1(c['n_true_acts'], c['n_pred_acts'], c['n_correct_acts'])
            slot_f1_stderr = self._d_f1(c['n_true_slots'], c['n_pred_slots'], c['n_correct_slots'])
            request_slot_f1_stderr = self._d_f1(c['n_true_request_slots'], c['n_pred_request_slots'], c['n_correct_request_slots'])        
            object_f1_stderr = self._d_f1(c['n_true_objects'], c['n_pred_objects'], c['n_correct_objects'])
        
            return {
                'joint_accuracy': joint_accuracy,
                'act_rec': act_rec,
                'act_prec': act_prec,
                'act_f1': act_f1,
                'act_f1_stderr': act_f1_stderr,
                'slot_rec': slot_rec,
                'slot_prec': slot_prec,
                'slot_f1': slot_f1,
                'slot_f1_stderr': slot_f1_stderr,
                'request_slot_rec': request_slot_rec,
                'request_slot_prec': request_slot_prec,
                'request_slot_f1': request_slot_f1,
                'request_slot_f1_stderr': request_slot_f1_stderr,        
                'object_rec': object_rec,
                'object_prec': object_prec,
                'object_f1': object_f1,
                'object_f1_stderr': object_f1_stderr,        
            }
        except:
            return c

    def _parse_flattened_results(self, lines):
        results = []
        for line in lines:
            parsed = self._parse_flattened_result(line)
            results.append(parsed)
        return results

    def _b_arr(self, n_total, n_pos):
        out = np.zeros(int(n_total))
        out[:int(n_pos)] = 1.0
        return out
    
    def _b_stderr(self, n_total, n_pos):
        return np.std(self._b_arr(n_total, n_pos)) / np.sqrt(n_total)
    
    def _initialize_count_dict(self):
        c = {
            'n_frames': 0.0,
            'n_true_acts': 0.0,
            'n_pred_acts': 0.0,
            'n_correct_acts': 0.0,
            'n_true_slots': 0.0,
            'n_pred_slots': 0.0,
            'n_correct_slots': 0.0,
            'n_true_request_slots': 0.0,
            'n_pred_request_slots': 0.0,
            'n_correct_request_slots': 0.0,
            'n_true_objects': 0.0,
            'n_pred_objects': 0.0,
            'n_correct_objects': 0.0,
            'n_correct_beliefs': 0.0,
        }
        return copy.deepcopy(c)
    
    def _parse_flattened_result(self, to_parse):
        dialog_act_regex = re.compile(r'([\w:?.?]*)  *\[(.*)\] *\(([^\]]*)\) *\<([^\]]*)\>')
        slot_regex = re.compile(r'([A-Za-z0-9_.-:]*)  *= ([^,]*)')
        request_regex = re.compile(r'([A-Za-z0-9_.-:]+)')
        object_regex = re.compile(r'([A-Za-z0-9]+)')
        
        belief = []

        splits = to_parse.strip().split(END_OF_API_CALL)
        # to_parse: 'DIALOG_ACT_1 : [ SLOT_NAME = SLOT_VALUE, ... ] ...'
        if len(splits) == 2:
            for dialog_act in dialog_act_regex.finditer(to_parse):
                d = {
                    'act': dialog_act.group(1),
                    'slots': [],
                    'request_slots': [],
                    'memories': []
                }

                for slot in slot_regex.finditer(dialog_act.group(2)):
                    d['slots'].append(
                        [
                            slot.group(1).strip(),
                            slot.group(2).strip()
                        ]
                    )

                for request_slot in request_regex.finditer(dialog_act.group(3)):
                    d['request_slots'].append(request_slot.group(1).strip())

                for object_id in object_regex.finditer(dialog_act.group(4)):
                    d['memories'].append(object_id.group(1).strip())

                if d != {}:
                    belief.append(d)
        return belief

    
    def evaluate(self, predicts, answers):

        # Get the response generation and API prediction examples
        response_predicts = []
        response_answers = []
        dst_predicts = []
        dst_answers = []

        for predict, answer in zip(predicts, answers):
            if END_OF_API_CALL in answer:
                dst_predicts.append(predict)
                dst_answers.append(answer)
            else:
                response_predicts.append(predict)
                response_answers.append(answer)

        # EVALUATE DST
        dst_predicts_parsed = self._parse_flattened_results(dst_predicts)
        dst_answers_parsed = self._parse_flattened_results(dst_answers)
        
        dst_results = self._evaluate_from_flat_list(dst_predicts_parsed, dst_answers_parsed)

        # EVALUATE RESPONSE GENERATION
        # Compute BLEU scores.
        bleu_scores = []
        # Smoothing function.
        chencherry = nltk.translate.bleu_score.SmoothingFunction()

        bert_scores = []
        iterator =  tqdm.tqdm(zip(response_predicts, response_answers), desc="Evaluating response generation")
        for response, gt_response in iterator:
            # Response generation evaluation.
            try:
                gt_response_clean = self._normalize_sentence(gt_response)
                response_clean = self._normalize_sentence(response)
                bleu_score = nltk.translate.bleu_score.sentence_bleu(
                    [gt_response_clean],
                    response_clean,
                    smoothing_function=chencherry.method7,
                )
                bleu_scores.append(bleu_score)
                '''
                _, _, bert_f1 = self.bert_scorer.score(
                    [" ".join(response_clean)], [" ".join(gt_response_clean)]
                )
                bert_scores.append(bert_f1.item())
                '''
            except:
                print(gt_response)
                print("-->", response)


        bleu_score_mean = np.mean(bleu_scores)
        bleu_score_err = np.std(bleu_scores) / np.sqrt(len(bleu_scores))
        #bert_score_mean = np.mean(bert_scores)
        #bert_score_err = np.std(bert_scores) / np.sqrt(len(bert_scores))
        print(f'DST results: {dst_results}')
        rand_index = random.randrange(len(response_predicts))
        print(f'Random response generation prediction/gt: {response_predicts[rand_index]}/{response_answers[rand_index]}')
        rand_index = random.randrange(len(dst_predicts))
        print(f'Random response generation prediction/gt: {dst_predicts[rand_index]}/{dst_answers[rand_index]}')
        
        print(f"# BLEU evaluations: {len(bleu_scores)}")
        print(f"BLEU Score: {bleu_score_mean:.4f} +- {bleu_score_err}")
        #print(f"BERT Score: {bert_score_mean:.4f} +- {bert_score_err}")

        return {'bleu_score': bleu_score_mean,
                'dst_results': dst_results,}
                #'bert_score': bert_score_mean


        
        


        
            
        
        
        
        
