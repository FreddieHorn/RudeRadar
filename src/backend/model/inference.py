from .distilBERT import Regressor
from .data_processing import DataPreprocessor

from transformers import AutoTokenizer, DistilBertModel
import torch

class RudenessDeterminator:
    def __init__(self) -> None:
        model_name = 'distilbert-base-uncased'
        checkpoint_path = "../../checkpoints/DistilBERT-NAT-NP-WITH_SCHEDULER_RUN2_NO_FREEZE-epoch=19-val_loss=0.03.ckpt"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        distilbert = DistilBertModel.from_pretrained(model_name, num_labels = 1)

        self.processor = DataPreprocessor()
        self.model = Regressor.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=torch.device(self.device), bertlike_model = distilbert)
        self.model.to(self.device)
        self.model.eval()

    def measure_rudeness(self, text):
        cleaned_text = self.processor.processBERT(text)
        encoded_text = self.tokenizer(cleaned_text,
                        add_special_tokens=True,
                        padding="max_length",
                        truncation=True,
                        max_length=200,
                        return_attention_mask=True,
                        return_tensors="pt")
        
        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            prediction = self.model(input_ids, attention_mask)

        

        return prediction.item()