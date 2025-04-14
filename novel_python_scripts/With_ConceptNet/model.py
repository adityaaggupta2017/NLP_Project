import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

class SimpleAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(SimpleAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value)
        output = self.out_proj(context)
        return output


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, features, adj_matrix):
        h = torch.mm(features, self.W)  # (N, out_features)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        a_input = a_input.view(N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return h_prime


class ContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True, dropout=0.1):
        super(ContextEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, 
                            bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, token_embeddings, attention_mask):
        # Compute lengths from attention mask, clamp to minimum of 1 to avoid zero-length sequences
        lengths = torch.clamp(attention_mask.sum(dim=1), min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(token_embeddings, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, (h_n, _) = self.lstm(packed)
        num_directions = 2 if self.lstm.bidirectional else 1
        h_n = h_n.view(self.lstm.num_layers, num_directions, token_embeddings.size(0), self.lstm.hidden_size)
        h_final = torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=1)
        return h_final


class ImprovedDeceptionModel(nn.Module):
    def __init__(self, model_name, use_game_scores=True):
        super(ImprovedDeceptionModel, self).__init__()
        # Pretrained transformers for current message and context
        self.transformer = RobertaModel.from_pretrained(model_name)
        self.context_transformer = RobertaModel.from_pretrained(model_name)
        
        # Freeze first 2 encoder layers for both transformers
        for layer in self.transformer.encoder.layer[:2]:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in self.context_transformer.encoder.layer[:2]:
            for param in layer.parameters():
                param.requires_grad = False
        
        self.hidden_size = self.transformer.config.hidden_size
        self.use_game_scores = use_game_scores
        self.conceptnet_dim = 300  # Size of Numberbatch embeddings
        
        self.dropout = nn.Dropout(0.3)
        self.context_encoder = ContextEncoder(input_dim=self.hidden_size, hidden_dim=self.hidden_size//2, 
                                              num_layers=1, bidirectional=True, dropout=0.1)
        
        # Adjust input dimension for graph attention to include ConceptNet features
        self.combined_dim = self.hidden_size + self.context_encoder.output_dim + self.conceptnet_dim
        self.gat1 = GraphAttentionLayer(self.combined_dim, 256)
        self.gat2 = GraphAttentionLayer(256, 256)
        
        # Game score processing branch
        if use_game_scores:
            self.score_proj = nn.Sequential(
                nn.Linear(1, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            fusion_dim = self.combined_dim + 256 + 32
        else:
            fusion_dim = self.combined_dim + 256
        
        self.feature_norm = nn.LayerNorm(fusion_dim)
        
        # Ensemble of classifier heads
        self.classifier1 = nn.Linear(fusion_dim, 2)  # from fused features
        self.classifier2 = nn.Linear(self.combined_dim, 2)  # current+context+conceptnet
        self.classifier3 = nn.Linear(256, 2)  # from graph features
        self.ensemble_weights = nn.Parameter(torch.ones(3))
    
    def forward(self, input_ids, attention_mask, context_input_ids=None, 
                context_attention_mask=None, game_scores=None, batch_adj_matrix=None, conceptnet_features=None):
        # Current message representation
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_features = self.dropout(pooled_output)
        
        # Context processing: encode context tokens and summarize via LSTM
        if context_input_ids is not None and torch.sum(context_input_ids) > 0:
            ctx_outputs = self.context_transformer(input_ids=context_input_ids, attention_mask=context_attention_mask)
            ctx_token_embeddings = ctx_outputs.last_hidden_state  
            context_summary = self.context_encoder(ctx_token_embeddings, context_attention_mask)
        else:
            batch_size = text_features.size(0)
            context_summary = torch.zeros(batch_size, self.context_encoder.output_dim, device=text_features.device)
        
        # Combine current message, context summary, and ConceptNet features
        combined_features = torch.cat([text_features, context_summary, conceptnet_features], dim=1)
        
        # Graph-based relational features
        if batch_adj_matrix is not None:
            graph_features = self.gat1(combined_features, batch_adj_matrix)
            graph_features = F.elu(graph_features)
            graph_features = self.gat2(graph_features, batch_adj_matrix)
        else:
            graph_features = torch.zeros(combined_features.size(0), 256, device=combined_features.device)
        
        # Fuse features: combined (text+context+conceptnet), graph, and optionally game scores
        if self.use_game_scores and game_scores is not None:
            score_features = self.score_proj(game_scores.unsqueeze(1))
            all_features = torch.cat([combined_features, graph_features, score_features], dim=1)
        else:
            all_features = torch.cat([combined_features, graph_features], dim=1)
        
        all_features = self.feature_norm(all_features)
        logits1 = self.classifier1(all_features)
        logits2 = self.classifier2(combined_features)
        logits3 = self.classifier3(graph_features)
        ensemble_weights = F.softmax(self.ensemble_weights, dim=0)
        final_logits = (ensemble_weights[0] * logits1 +
                        ensemble_weights[1] * logits2 +
                        ensemble_weights[2] * logits3)
        return final_logits