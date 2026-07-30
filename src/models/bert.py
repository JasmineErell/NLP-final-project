import torch
from typing import List, Dict, Any
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

from src.models.common import BaseMaskedModel 

class BertPretrainedFineTuner(BaseMaskedModel):
    """
    Cross-domain transfer learning: Fine-tunes pre-trained English BERT on symbolic music.
    """

    def __init__(
        self, 
        name: str = "MusicBERT_Transfer", 
        is_factorized: bool = False, 
        output_dir: str = "./music_bert_transfer_results" 
    ):
        super().__init__(name) 
        self.is_factorized = is_factorized 
        self.output_dir = output_dir 
        
        # Route hardware automatically for PyTorch acceleration
        self.device = torch.device( 
            "mps" if torch.backends.mps.is_available()  
            else "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        
        self.tokenizer = None 
        self.model = None 

    def fit(self, train_sequences: List[List[str]], **kwargs) -> Dict[str, Any]:
        # 1. Load Pre-trained English Tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        # 2. Extract and Inject Custom Musical Vocabulary
        # Flatten the training sequences to find all unique musical tokens
        unique_music_tokens = list(set([token for seq in train_sequences for token in seq]))
        self.tokenizer.add_tokens(unique_music_tokens)

        # 3. Format Custom Data for Hugging Face
        max_len = 512 if self.is_factorized else 256 

        def tokenize_function(examples): 
            return self.tokenizer( 
                examples["tokens"],   
                is_split_into_words=True,   
                padding="max_length",   
                truncation=True,   
                max_length=max_len  
            )

        raw_dataset = Dataset.from_dict({"tokens": train_sequences})  
        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["tokens"])  

        # 4. Load Pre-trained Weights and Resize Embeddings
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)  

        # 5. Setup Masking Collator
        data_collator = DataCollatorForLanguageModeling(  
            tokenizer=self.tokenizer,  
            mlm=True,  
            mlm_probability=0.15  
        )

        # 6. Define Training Arguments
        epochs = kwargs.get("epochs", 5)  
        batch_size = kwargs.get("batch_size", 16) # Lowered default batch size due to larger model size

        training_args = TrainingArguments(  
            output_dir=self.output_dir,  
            overwrite_output_dir=True,  
            num_train_epochs=epochs,  
            per_device_train_batch_size=batch_size,  
            save_steps=10_000,  
            save_total_limit=2,  
            prediction_loss_only=True,  
            logging_steps=500  
        )

        # 7. Execute Training
        trainer = Trainer(  
            model=self.model,  
            args=training_args,  
            data_collator=data_collator,  
            train_dataset=tokenized_dataset,  
        )

        trainer.train()  
        trainer.save_model(f"{self.output_dir}/final_model")  
        
        return {"status": "success", "loss": trainer.state.best_metric}  

    def predict_token_probabilities(self, sequence: List[str], target_index: int) -> Dict[str, float]:  
        if self.model is None or self.tokenizer is None:  
            raise RuntimeError("Model has not been trained or loaded yet. Call fit() first.")  

        self.model.eval()  

        masked_sequence = sequence.copy()  
        masked_sequence[target_index] = self.tokenizer.mask_token  
        
        inputs = self.tokenizer(  
            masked_sequence,   
            is_split_into_words=True,   
            return_tensors="pt"  
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  

        input_ids = inputs["input_ids"][0]  
        mask_token_id = self.tokenizer.mask_token_id  
        
        mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]  
        if len(mask_indices) == 0:  
            raise ValueError("Mask token not found in the tokenized sequence.")  
        
        actual_target_index = mask_indices[0].item()  

        with torch.no_grad(): 
            outputs = self.model(**inputs) 
            logits = outputs.logits 

        mask_logits = logits[0, actual_target_index, :] 
        probabilities = torch.nn.functional.softmax(mask_logits, dim=0) 

        vocab = self.tokenizer.get_vocab() 
        id_to_token = {v: k for k, v in vocab.items()} 
        
        prob_dict = {} 
        for token_id, prob in enumerate(probabilities.tolist()):
            token_str = id_to_token.get(token_id, "[UNK]") 
            if token_str not in ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]: 
                prob_dict[token_str] = prob 

        return prob_dict