import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import os

def parse_proof_to_graph(proof_text: str, problem_text: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(0, type="problem", content=problem_text)
    steps = [s.strip() for s in proof_text.split('\n') if s.strip()]
    if not steps:
        steps = [s.strip() for s in proof_text.split('.') if s.strip()]
    for i, step in enumerate(steps, start=1):
        if len(step) > 10:
            graph.add_node(i, type="step", content=step)
            if i > 1:
                graph.add_edge(i-1, i, type="sequential")
            else:
                graph.add_edge(0, i, type="derives")
    goal_id = len(graph.nodes)
    graph.add_node(goal_id, type="goal", content="conclusion")
    if len(graph.nodes) > 2:
        graph.add_edge(len(graph.nodes) - 2, goal_id, type="concludes")
    return graph

class GraphEncoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512, output_tokens=8, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_tokens = output_tokens
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(input_dim, hidden_dim, heads=4))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim * 4, hidden_dim, heads=4))
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_tokens * input_dim)
        )
    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        tokens = self.readout(x)
        tokens = tokens.view(-1, self.output_tokens, self.input_dim)
        return tokens

class TextEmbedder:
    def __init__(self, device='cuda'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.device = device
    def __call__(self, texts: list) -> torch.Tensor:
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.to(self.device)

class GraphProofDataset(Dataset):
    def __init__(self, parquet_path: str, text_embedder, max_samples=None):
        self.df = pd.read_parquet(parquet_path)
        if max_samples:
            self.df = self.df.iloc[:max_samples]
        self.text_embedder = text_embedder
        print(f"Loaded {len(self.df)} examples from {parquet_path}")
    def __len__(self):
        return len(self.df) * 2
    def __getitem__(self, idx):
        row_idx = idx // 2
        is_pos = (idx % 2 == 0)
        row = self.df.iloc[row_idx]
        if is_pos:
            question = row['pos_question']
            response = row['pos_response']
            truth_value = row['pos_truth_value']
        else:
            question = row['neg_question']
            response = row['neg_response']
            truth_value = row['neg_truth_value']
        graph = parse_proof_to_graph(response, question)
        node_texts = [graph.nodes[n]['content'] for n in graph.nodes]
        node_embeddings = self.text_embedder(node_texts)
        edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t()
        if edge_index.numel() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = Data(
            x=node_embeddings,
            edge_index=edge_index,
            truth_value=torch.tensor([1.0 if truth_value else 0.0], dtype=torch.float32)
        )
        return data

def train_graph_encoder(train_dataset, val_dataset, graph_encoder, num_epochs=5, batch_size=16, lr=1e-4, device='cuda'):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: Batch.from_data_list(batch)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: Batch.from_data_list(batch)
    )
    optimizer = torch.optim.AdamW(graph_encoder.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    classifier = nn.Linear(graph_encoder.output_tokens * graph_encoder.input_dim, 1).to(device)
    for epoch in range(num_epochs):
        graph_encoder.train()
        classifier.train()
        train_loss = 0.0
        train_acc = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            tokens = graph_encoder(batch)
            tokens_flat = tokens.view(tokens.size(0), -1)
            logits = classifier(tokens_flat).squeeze(-1)
            loss = criterion(logits, batch.truth_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_acc += (preds == batch.truth_value).float().mean().item()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        graph_encoder.eval()
        classifier.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                tokens = graph_encoder(batch)
                tokens_flat = tokens.view(tokens.size(0), -1)
                logits = classifier(tokens_flat).squeeze(-1)
                loss = criterion(logits, batch.truth_value)
                val_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_acc += (preds == batch.truth_value).float().mean().item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="deeptheorem/train.parquet")
    parser.add_argument("--val_data", type=str, default="deeptheorem/test.parquet")
    parser.add_argument("--output", type=str, default="graph_encoder_pretrained")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_tokens", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    text_embedder = TextEmbedder(device=device)
    train_dataset = GraphProofDataset(args.train_data, text_embedder, args.max_samples)
    val_dataset = GraphProofDataset(args.val_data, text_embedder, max_samples=min(1000, len(train_dataset)//10))
    graph_encoder = GraphEncoder(input_dim=384, hidden_dim=512, output_tokens=args.output_tokens, num_layers=3).to(device)
    print(f"Graph encoder: {sum(p.numel() for p in graph_encoder.parameters())/1e6:.2f}M parameters")
    print(f"Output: {args.output_tokens} tokens of dimension 384")
    train_graph_encoder(train_dataset, val_dataset, graph_encoder, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
    torch.save({
        'graph_encoder': graph_encoder.state_dict(),
        'config': {
            'input_dim': 384,
            'hidden_dim': 512,
            'output_tokens': args.output_tokens,
            'num_layers': 3
        }
    }, f"{args.output}/graph_encoder_pretrained.pt")
    print(f"Saved to {args.output}/graph_encoder_pretrained.pt")
