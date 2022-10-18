from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint

import torch
import pdb
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import json

class BLIP_COMET(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 special_tokens_file = '/fsx/jacampos/data/comet/split_v2/processed/mem_dials_gpt2_special_tokens.json'
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()
        self.tokenizer.add_special_tokens(json.load(open(special_tokens_file, 'r')))
        
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        
        decoder_config = BertConfig.from_json_file(med_config)        
        self.text_decoder = BertLMHeadModel(config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

    def forward(self, images, predict, images_mask, target=None, n=None, evaluate=True, use_images=True):
        images_mask = images_mask.to(images.device).unsqueeze(-1)
        #Get visual features
        B, N, C, W, H = images.size()
        images = images.view(-1, C, W, H)
        image_embeds = self.visual_encoder(images)
        features_shape = image_embeds.shape[-1]
        #Mean features over the context, normalization added similar to video
        image_embeds = image_embeds.view(B, N, -1) 
        #Apply the mask
        image_embeds = image_embeds * images_mask 
        image_embeds = F.normalize(image_embeds.mean(dim=1))
        image_embeds = image_embeds.view(B, -1, features_shape)
        
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(images.device)
            
        predict = self.tokenizer(predict, padding='longest',return_tensors="pt").to(images.device)
        predict.input_ids[:,0] = self.tokenizer.enc_token_id
        target = self.tokenizer(target, padding='longest', return_tensors="pt").to(images.device) 
        target.input_ids[:,0] = self.tokenizer.bos_token_id
        targets = target.input_ids.masked_fill(target.input_ids == self.tokenizer.pad_token_id, -100)      

        if use_images:
            predict_output = self.text_encoder(predict.input_ids, 
                                           attention_mask = predict.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,
                                           return_dict = True)

        else:
            image_atts = torch.zeros(image_embeds.size()[:-1],dtype=torch.long).to(images.device)            
            predict_output = self.text_encoder(predict.input_ids,
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,
                                           attention_mask = predict.attention_mask, 
                                           return_dict = True)


        predict_states = predict_output.last_hidden_state
        predict_atts = predict.attention_mask

        target_output = self.text_decoder(target.input_ids, 
                                          attention_mask = target.attention_mask, 
                                          encoder_hidden_states = predict_states,
                                          encoder_attention_mask = predict_atts,
                                          labels = targets,
                                          return_dict = True,   
                                          reduction = 'none')
            
        loss = target_output.loss
        loss = loss.mean()

        output = {'loss': loss}
        
        if evaluate:
            predict_output = self.text_encoder(predict.input_ids, 
                                                attention_mask = predict.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True) 
            
            num_beams = 3
            predict_states = predict_output.last_hidden_state.repeat_interleave(num_beams,dim=0)
            predict_atts = torch.ones(predict_states.size()[:-1],dtype=torch.long).to(predict_states.device)
            model_kwargs = {"encoder_hidden_states": predict_states, "encoder_attention_mask":predict_atts}
                
            bos_ids = torch.full((image_embeds.size(0),1),fill_value=self.tokenizer.bos_token_id,device=images.device)
                
            generations = self.text_decoder.generate(input_ids=bos_ids,
                                                     max_length=10,
                                                     min_length=1,
                                                     num_beams=num_beams,
                                                     eos_token_id=self.tokenizer.sep_token_id,
                                                     pad_token_id=self.tokenizer.pad_token_id, 
                                                     **model_kwargs)
                
            answers = []
            for generation in generations:
                answer = self.tokenizer.decode(generation)    
                answers.append(answer)
            output['answers'] = answers

        return output
 
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')   
        
        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques,k)

        max_topk_ids = log_probs_sum.argmax(dim=1) 
        max_ids = topk_ids[max_topk_ids>=0,max_topk_ids]

        return max_ids
    
    
def blip_comet(pretrained='',**kwargs):
    model = BLIP_COMET(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
#         assert(len(msg.missing_keys)==0)
    return model  


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
        
        
