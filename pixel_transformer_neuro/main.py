from pixel_transformer_neuro.training.training import run_training
from pixel_transformer_neuro.models.attention_null_model import AttentionNullModel
from pixel_transformer_neuro.models.config_attention_null import model_config
from dotenv import load_dotenv
import os
# Load .env file
load_dotenv()

# Set WANDB API key
wandb_api_key = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = wandb_api_key 

if __name__ == "__main__":
    dataset_path_dict = {
        "embeddings": "/home/maria/LuckyMouse/pixel_transformer_neuro/data/processed/google_vit-base-patch16-224_embeddings_softmax.pkl",
        "neural": "/home/maria/LuckyMouse/pixel_transformer_neuro/data/processed/hybrid_neural_responses.npy"
    }

    wandb_config = {
        "project": "pixel-transformer-neuro",
        "name": "attention-null",
        "model": "AttentionNullModel",
        "description": "Null model with self-attention and ViT softmax embeddings"
    }

    run_training(
        model_class=AttentionNullModel,
        model_config=model_config,
        dataset_path_dict=dataset_path_dict,
        wandb_config=wandb_config,
        num_epochs=20,
        batch_size=128,
        learning_rate=1e-3
    )
