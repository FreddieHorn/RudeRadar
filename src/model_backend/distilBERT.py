import torch.nn as nn
import torch
import numpy as np
import lightning as L

from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torchmetrics import R2Score

class TextAugmenterForBert:
    """Stores tokenizer denoted by tokenizer_name and possesses augment data method, which tokenizes 
    comments which the dataset consists of.
    Args:
        tokenizer_name (str): name of the tokenizer to retrieve from AutoTokenizer
        df (DataFrame): dataset ['comment_body', 'offensiveness score']
    """
    def __init__(self, tokenizer_name: str, df) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.df = df
        #adding tokens from our dataset that are not already in the tokenizer - RUINED RESULTS!!
        # token_list = tokenize_words(self.df, "comment_body")
        # num_added_toks = self.tokenizer.add_tokens(token_list)
        # print(f"Added {num_added_toks} to the tokenizer")
    
    def encode_data(self, posts_included = True):
        """Tokenizes input data using tokenizer declared in __init__. 

        Args:
            posts_included (bool, optional): Whether we want to include posts or not. Defaults to True.

        Returns:
            input_ids, labels, attention_mask: Tokenized input sentences, rudeness level and attention mask. This format
            is required for transformers BERT-like models.
        """
        if posts_included:
            encoded_corpus = self.tokenizer(self.df.post_title.to_list(), self.df.comment_body.to_list(),
                                add_special_tokens=True,
                                padding="max_length",
                                truncation=True,
                                max_length=200,
                                return_attention_mask=True)
        else:
            encoded_corpus = self.tokenizer(self.df.comment_body.to_list(),
                                add_special_tokens=True,
                                padding="max_length",
                                truncation=True,
                                max_length=200,
                                return_attention_mask=True)

        input_ids = encoded_corpus['input_ids']
        attention_mask = encoded_corpus['attention_mask']
        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)
        labels = self.df.offensiveness_score.to_numpy()
        return input_ids, labels, attention_mask
    
class Regressor(L.LightningModule):
    """This module consists of distilBERT with regression layer on top of it so it suits the given task of determining the offensiveness score. 

    Args:
        bertlike_model: model from the big family of BERTs. distilBERT is used, but it is capable of handling other models (like RoBERTA) 
        with a little to no tweaks in the code.
        total_training_steps (int): number of training steps. This number is used in a scheduler which modifies the lr based on the current
        training step
        dropout (float, optional): Dropout in a regression layer placed on top of distilBERT. Defaults to 0.2.
        lr (float, optional): learning rate parameter. Defaults to 1e-3.
    """
    def __init__(self, bertlike_model, total_training_steps = 0, dropout=0.2, lr=1e-3,
                 accuracy_threshold: float = 0.05) -> None:
        super().__init__()
        D_in, D_out = 768, 1 #bert (or its derivatives) has 768 outputs 
        self.model = bertlike_model
        #self.model.enable_input_require_grads()
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(D_in, D_out)) #can experiment with bigger regression part, buuut freezing bert would be needed 
        self.loss = nn.MSELoss()
        self.R2 = R2Score()
        self.MAE = nn.L1Loss()
        self.lr = lr
        self.drop = nn.Dropout(dropout)
        self.total_steps = total_training_steps #param for get_linear_schedule_with_warmup
        self.threshold = accuracy_threshold
        
        # # save hyper-parameters to self.hparams (auto-logged by W&B)
        # self.save_hyperparameters()

    def forward(self, input_ids, attention_masks):
        outputs = self.model(input_ids, attention_masks)
        pooled_output = outputs[0].mean(dim=1) #Last layer hidden-state of the first token of the sequence (classification token) (but also can be used for regression)
        output = self.regressor(pooled_output)
        return torch.tanh(output)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        #scheduler is crucial. it deals with fine-tuning instability described in https://arxiv.org/pdf/2006.04884.pdf
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*self.total_steps), num_training_steps=self.total_steps)
        return [optimizer], [scheduler] 
    
    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        batch_inputs, batch_masks, batch_labels = \
                        tuple(b for b in train_batch)
        outputs = self(batch_inputs, batch_masks)
        loss = self.loss(outputs.squeeze(1).float(), 
                        batch_labels.float())
        std = torch.std(outputs.squeeze(1).float())
        self.log("train_loss", loss)
        self.log("train_std", std) #for experiments. dont include in official version
        return loss
    
    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT:
        batch_inputs, batch_masks, batch_labels = \
                        tuple(b for b in val_batch)
        outputs = self(batch_inputs, batch_masks)
        val_loss = self.loss(outputs.squeeze(1).float(), 
                        batch_labels.float())
        std = torch.std(outputs.squeeze(1).float())
        self.log("val_loss", val_loss)
        self.log("val_std", std) #for experiments. dont include in official version
        return val_loss
    
    def test_step(self, test_batch, batch_idx):
        batch_inputs, batch_masks, batch_labels = \
                        tuple(b for b in test_batch)
        outputs = self(batch_inputs, batch_masks)
        mae_loss = self.MAE(outputs.squeeze(1).float(), 
                        batch_labels.float())
        r2_score = self.R2(outputs.squeeze(1).float(), #warning! using batch_size = 8 made the last test step have only one sample in 
                            batch_labels.float())
        std = torch.std(outputs.squeeze(1).float())   #preds and batch labels. r2 needs > 1 samples so I increased batch size to 16
        accuracy = torch.sum(torch.abs(batch_labels - outputs.squeeze(1)) < self.threshold).item() / len(
            batch_labels
        )
        self.log("test accuracy", accuracy)
        self.log("mae_loss", mae_loss)
        self.log("r2_score", r2_score)
        self.log("test_std", std) #for experiments. dont include in official version
        return mae_loss, r2_score
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch_inputs, batch_masks = \
                        tuple(b for b in batch)
        return torch.tanh(self(batch_inputs, batch_masks))