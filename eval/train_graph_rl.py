import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
import argparse
from tqdm import tqdm
import os
import json

class ProofGraph:
    def __init__(self, problem: str):
        self.graph = nx.DiGraph()
        self.problem = problem
        self.step_count = 0
        self.graph.add_node(0, type="problem", content=problem, emb=None)
        self.graph.add_node(1, type="goal", content="prove_or_disprove", emb=None)
        
    def add_step(self, content: str, parents: list = None) -> int:
        node_id = len(self.graph.nodes)
        self.graph.add_node(node_id, type="step", content=content, emb=None)
        if parents:
            for p in parents:
                self.graph.add_edge(p, node_id, type="derives")
        else:
            self.graph.add_edge(0, node_id, type="derives")
        self.step_count += 1
        return node_id
    
    def to_pyg(self, text_embedder) -> Data:
        node_texts = [self.graph.nodes[n]['content'] for n in self.graph.nodes]
        embeddings = text_embedder(node_texts)
        edge_index = torch.tensor(list(self.graph.edges), dtype=torch.long).t()
        if edge_index.numel() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=embeddings, edge_index=edge_index)

class GraphEncoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, output_tokens=4):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=4)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=1)
        self.output_tokens = output_tokens
        self.readout = nn.Linear(hidden_dim, output_tokens * input_dim)
        
    def forward(self, data: Data) -> torch.Tensor:
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)
        x = global_mean_pool(x, batch)
        x = self.readout(x)
        return x.view(-1, self.output_tokens, 384)

class TextEmbedder:
    def __init__(self, device='cuda'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.device = device
    
    def __call__(self, texts: list) -> torch.Tensor:
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.to(self.device)

class Reasoner:
    def __init__(self, model_path: str):
        self.model = LLM(model=model_path, dtype=torch.float16, tensor_parallel_size=1, gpu_memory_utilization=0.5)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def generate(self, problem: str, current_steps: list, n_samples: int = 4, temperature: float = 1.0) -> list:
        context = f"Problem: {problem}\n\nCurrent proof:\n"
        context += "\n".join([f"{i+1}. {s}" for i, s in enumerate(current_steps)])
        context += "\n\nNext step:"
        messages = [{"role": "user", "content": context}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=256, n=n_samples)
        outputs = self.model.generate([prompt], sampling_params)
        return [output.text for output in outputs[0].outputs]

class Verifier:
    def score_step(self, step: str, previous_steps: list) -> float:
        reward = 0.5
        if 20 < len(step) < 300:
            reward += 0.1
        math_keywords = ['therefore', 'thus', 'hence', 'implies', 'follows', 'because']
        if any(kw in step.lower() for kw in math_keywords):
            reward += 0.1
        if '\\boxed{' in step:
            reward += 0.2
        if not any(step.lower() in prev.lower() for prev in previous_steps[-3:]):
            reward += 0.1
        return min(reward, 1.0)
    
    def score_final(self, steps: list, ground_truth: str) -> float:
        full_proof = " ".join(steps).lower()
        if any(x in full_proof for x in ["\\boxed{proved}", "\\boxed{true}"]):
            pred = "proved"
        elif any(x in full_proof for x in ["\\boxed{disproved}", "\\boxed{false}"]):
            pred = "disproved"
        else:
            return 0.0
        return 1.0 if pred == ground_truth else 0.0

class GRPOTrainer:
    def __init__(self, reasoner, verifier, graph_encoder, text_embedder, lr=1e-5):
        self.reasoner = reasoner
        self.verifier = verifier
        self.graph_encoder = graph_encoder
        self.text_embedder = text_embedder
        self.optimizer = torch.optim.AdamW(graph_encoder.parameters(), lr=lr)
    
    def train_epoch(self, problems: list, n_samples: int = 8, max_steps: int = 5, batch_size: int = 4):
        total_reward = 0.0
        for batch_start in range(0, len(problems), batch_size):
            batch_end = min(batch_start + batch_size, len(problems))
            batch_problems = problems[batch_start:batch_end]
            print(f"\nTraining batch [{batch_start+1}-{batch_end}] / {len(problems)}")
            for prob_data in tqdm(batch_problems, desc="Batch"):
                problem = prob_data['problem']
                ground_truth = prob_data['ground_truth']
                trajectories = []
                for _ in range(n_samples):
                    traj = self.generate_trajectory(problem, ground_truth, max_steps)
                    trajectories.append(traj)
                rewards = [t['total_reward'] for t in trajectories]
                baseline = np.mean(rewards)
                advantages = [r - baseline for r in rewards]
                for traj, adv in zip(trajectories, advantages):
                    if adv > 0:
                        for graph_state in traj['graphs']:
                            graph_data = graph_state.to_pyg(self.text_embedder)
                            graph_tokens = self.graph_encoder(graph_data)
                            loss = graph_tokens.pow(2).mean() * (1.0 - adv)
                            loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_reward += sum(rewards) / len(rewards)
        return total_reward / len(problems)
    
    def generate_trajectory(self, problem: str, ground_truth: str, max_steps: int):
        graph = ProofGraph(problem)
        steps = []
        step_rewards = []
        graphs = []
        for _ in range(max_steps):
            candidates = self.reasoner.generate(problem, steps, n_samples=1, temperature=0.8)
            chosen_step = candidates[0]
            graph.add_step(chosen_step)
            steps.append(chosen_step)
            graphs.append(graph)
            step_reward = self.verifier.score_step(chosen_step, steps[:-1])
            step_rewards.append(step_reward)
            if '\\boxed{' in chosen_step:
                break
        final_reward = self.verifier.score_final(steps, ground_truth)
        total_reward = np.mean(step_rewards) * 0.7 + final_reward * 0.3
        return {'steps': steps, 'graphs': graphs, 'step_rewards': step_rewards, 'final_reward': final_reward, 'total_reward': total_reward}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="eval/data/fimo.jsonl")
    parser.add_argument("--reasoner", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", type=str, default="graph_rl_checkpoints")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    df = pd.read_json(args.data, lines=True)
    if args.limit:
        df = df.iloc[:args.limit]
    
    problems = []
    for _, row in df.iterrows():
        problems.append({'problem': row['prompt'], 'ground_truth': 'proved' if row['answer'] == 1 else 'disproved'})
    
    print(f"Loaded {len(problems)} training examples")
    print("\nInitializing models...")
    text_embedder = TextEmbedder()
    graph_encoder = GraphEncoder().cuda()
    reasoner = Reasoner(args.reasoner)
    verifier = Verifier()
    
    trainer = GRPOTrainer(reasoner, verifier, graph_encoder, text_embedder, lr=args.lr)
    
    print("\n" + "="*80)
    print("Starting Graph2Proof RL Training")
    print("="*80)
    
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{args.epochs}")
        print(f"{'='*80}")
        avg_reward = trainer.train_epoch(problems, n_samples=args.n_samples, max_steps=args.max_steps, batch_size=args.batch_size)
        print(f"\nEpoch {epoch+1} - Avg Reward: {avg_reward:.4f}")
        checkpoint = {'epoch': epoch + 1, 'graph_encoder': graph_encoder.state_dict(), 'optimizer': trainer.optimizer.state_dict(), 'avg_reward': avg_reward}
        torch.save(checkpoint, f"{args.output}/checkpoint_epoch_{epoch+1}.pt")
        print(f"Saved checkpoint to {args.output}/checkpoint_epoch_{epoch+1}.pt")
    
    torch.save(graph_encoder.state_dict(), f"{args.output}/graph_encoder_final.pt")
    
    config = {'reasoner_model': args.reasoner, 'n_samples': args.n_samples, 'max_steps': args.max_steps, 'epochs': args.epochs, 'final_reward': avg_reward}
    with open(f"{args.output}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Final model saved to {args.output}/graph_encoder_final.pt")
    print("="*80)
