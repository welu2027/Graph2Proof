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
import os
import json

def chunk_indices(indices, chunk_size=2):
    for i in range(0, len(indices), chunk_size):
        yield indices[i:i + chunk_size]

def extract_prediction(text):
    t = text.lower()
    if any(x in t for x in ["\\boxed{proved}", "\\boxed{\\text{proved}}", "\\boxed{true}", "\\boxed{\\text{true}}"]):
        return 1
    if any(x in t for x in ["\\boxed{disproved}", "\\boxed{\\text{disproved}}", "\\boxed{false}", "\\boxed{\\text{false}}"]):
        return 0
    return -1

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
        
    def generate_proof(self, problem: str, max_steps: int = 5, temperature: float = 0.0):
        steps = []
        for _ in range(max_steps):
            context = f"Problem: {problem}\n\nCurrent proof:\n"
            context += "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
            context += "\n\nNext step:"
            messages = [{"role": "user", "content": context}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            sampling_params = SamplingParams(temperature=temperature, max_tokens=256, n=1)
            outputs = self.model.generate([prompt], sampling_params)
            step = outputs[0].outputs[0].text
            steps.append(step)
            if '\\boxed{' in step:
                break
        return " ".join(steps)

class GraphEvaluator:
    def __init__(self, reasoner, graph_encoder, text_embedder):
        self.reasoner = reasoner
        self.graph_encoder = graph_encoder
        self.text_embedder = text_embedder
        self.results = []
    
    def evaluate(self, problems: list, batch_size: int = 10, print_every: int = 10):
        for batch_start in range(0, len(problems), batch_size):
            batch_end = min(batch_start + batch_size, len(problems))
            batch_problems = problems[batch_start:batch_end]
            print(f"\nEvaluating [{batch_start+1}-{batch_end}] / {len(problems)}")
            for prob_data in batch_problems:
                problem = prob_data['problem']
                problem_name = prob_data['problem_name']
                answer = prob_data['answer']
                proof = self.reasoner.generate_proof(problem, max_steps=5, temperature=0.0)
                prediction = extract_prediction(proof)
                self.results.append({'problem_name': problem_name, 'answer': answer, 'prediction': prediction, 'generation': proof})
            if batch_end % print_every == 0 or batch_end == len(problems):
                self.print_accuracy(batch_end, len(problems))
    
    def print_accuracy(self, processed: int, total: int):
        df = pd.DataFrame(self.results)
        accs = []
        for problem_name in df.problem_name.unique():
            df_prob = df[df.problem_name == problem_name]
            indices = df_prob.index.tolist()
            for chunk in chunk_indices(indices, 2):
                df_chunk = df.loc[chunk]
                all_correct = (df_chunk.answer == df_chunk.prediction).all()
                accs.append(all_correct)
        accuracy = np.mean(accs) * 100 if accs else 0.0
        print(f"[{processed}/{total}] Interim Accuracy: {accuracy:.2f}%")
    
    def final_report(self):
        df = pd.DataFrame(self.results)
        accs = []
        for problem_name in df.problem_name.unique():
            df_prob = df[df.problem_name == problem_name]
            indices = df_prob.index.tolist()
            for chunk in chunk_indices(indices, 2):
                df_chunk = df.loc[chunk]
                all_correct = (df_chunk.answer == df_chunk.prediction).all()
                accs.append(all_correct)
        accuracy = np.mean(accs) * 100 if accs else 0.0
        print("\n" + "="*80)
        print("FINAL EVALUATION RESULTS")
        print("="*80)
        print(f"Outcome Score: {accuracy:.2f}%")
        print(f"Variant-groups solved: {sum(accs)}/{len(accs)}")
        print(f"Total evaluated: {len(df)} proofs")
        print("="*80)
        return df, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="eval/data/fimo.jsonl")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--reasoner", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", type=str, default="eval_output")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit_problems", type=int, default=None)
    parser.add_argument("--problem_name", type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    df = pd.read_json(args.data, lines=True)
    if args.problem_name:
        df = df[df.problem_name == args.problem_name].reset_index(drop=True)
        print(f"Evaluating problem '{args.problem_name}' with {len(df)} variants")
    elif args.limit_problems:
        problems_list = df.problem_name.unique()[:args.limit_problems]
        df = df[df.problem_name.isin(problems_list)].reset_index(drop=True)
        print(f"Evaluating first {args.limit_problems} problems ({len(df)} variants)")
    elif args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)
    
    print(f"Total rows: {len(df)}")
    print(f"Unique problems: {df.problem_name.nunique()}")
    
    problems = []
    for _, row in df.iterrows():
        problems.append({'problem': row['prompt'], 'problem_name': row['problem_name'], 'answer': row['answer']})
    
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    text_embedder = TextEmbedder()
    graph_encoder = GraphEncoder().cuda()
    checkpoint = torch.load(args.checkpoint)
    if 'graph_encoder' in checkpoint:
        graph_encoder.load_state_dict(checkpoint['graph_encoder'])
    else:
        graph_encoder.load_state_dict(checkpoint)
    graph_encoder.eval()
    print("Model loaded successfully!")
    
    print("Initializing reasoner...")
    reasoner = Reasoner(args.reasoner)
    evaluator = GraphEvaluator(reasoner, graph_encoder, text_embedder)
    
    print("\n" + "="*80)
    print("Starting Evaluation")
    print("="*80)
    
    evaluator.evaluate(problems, batch_size=args.batch_size, print_every=args.print_every)
    
    df_results, accuracy = evaluator.final_report()
    df_results.to_json(f"{args.output}/results.jsonl", orient='records', lines=True)
    
    summary = {'accuracy': accuracy, 'total_problems': df.problem_name.nunique(), 'total_variants': len(df_results), 'checkpoint': args.checkpoint}
    with open(f"{args.output}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {args.output}/")
    
    if len(df_results) > 0:
        print("\n" + "="*80)
        print("EXAMPLE PROOFS")
        print("="*80)
        for i in range(min(2, len(df_results))):
            example = df_results.iloc[i]
            print(f"\n--- Example {i+1} ---")
            print(f"Problem: {example['problem_name']}")
            print(f"Ground Truth: {'proved' if example['answer'] == 1 else 'disproved'}")
            print(f"Prediction: {'proved' if example['prediction'] == 1 else ('disproved' if example['prediction'] == 0 else 'NONE')}")
            print(f"Correct: {'✓' if example['answer'] == example['prediction'] else '✗'}")
            print(f"\nProof:\n{example['generation'][:300]}...")
        print("="*80)
