import torch
import torch.nn as nn

class FeatureTokenizer(nn.Module):
    def __init__(self, cat_cardinalities, num_continuous, dim):
        super().__init__()
        # List of embeddings for each categorical feature
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cardinality, embedding_dim=dim)
            for cardinality in cat_cardinalities
        ])
        
        # Linear projection layers to map continuous features into the same dimension vector space
        self.cont_projections = nn.ModuleList([
            nn.Linear(1, dim) for _ in range(num_continuous)
        ])
        
        # Classification [CLS] token initialized randomly
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x_cat, x_cont):
        batch_size = x_cat.size(0)
        tokens = []
        
        # Tokenize Categorical Data
        for i, embed_layer in enumerate(self.cat_embeddings):
            tokens.append(embed_layer(x_cat[:, i]).unsqueeze(1))
            
        # Tokenize Continuous Data
        for i, proj_layer in enumerate(self.cont_projections):
            cont_feat = x_cont[:, i].unsqueeze(1) # shape: (batch, 1)
            tokens.append(proj_layer(cont_feat).unsqueeze(1))
            
        # Concatenate all feature tokens: shape (batch, num_features, dim)
        x = torch.cat(tokens, dim=1)
        
        # Expand and prepend the [CLS] token to the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # shape: (batch, num_features + 1, dim)
        return x

class TabularTransformer(nn.Module):
    def __init__(self, cat_cardinalities, num_continuous, dim=32, depth=4, heads=4, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(cat_cardinalities, num_continuous, dim)
        
        # Encoder Stack built from native PyTorch Transformer Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # MLP Prediction Head attached specifically to the [CLS] token output
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1) # Outputting raw logits for binary classification (Fraud vs. Legitimate)
        )

    def forward(self, x_cat, x_cont):
        # 1. Tokenize inputs into a unified representation
        x = self.tokenizer(x_cat, x_cont)
        
        # 2. Pass through multi-head attention stack
        x = self.transformer_encoder(x)
        
        # 3. Pull out the [CLS] token (index 0) to feed into the classifier
        cls_output = x[:, 0, :]
        logits = self.mlp_head(cls_output)
        return logits
