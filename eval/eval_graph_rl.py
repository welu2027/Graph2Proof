import os
import json
import torch
from torch import nn
from torch_geometric.nn import TransformerConv, global_mean_pool
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

class GraphTransformer(nn.Module):
    def __init__(self, node_dim, hidden_dim, n_heads, n_layers):
        super().__init__()
        self.convs = nn.ModuleList([TransformerConv(node_dim if i==0 else hidden_dim, hidden_dim, heads=n_heads) for i in range(n_layers)])
        self.pool = global_mean_pool

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.pool(x, batch)

class Graph2ProofRL:
    def __init__(self, model_name="Qwen-7B", node_dim=128, hidden_dim=256, n_heads=4, n_layers=3, max_steps=20):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.graph_model = GraphTransformer(node_dim, hidden_dim, n_heads, n_layers).to(device)
        self.max_steps = max_steps
        self.node_dim = node_dim

    def build_graph(self, proof_steps):
        x = torch.randn((len(proof_steps), self.node_dim), device=device)
        edge_index = torch.tensor([[i, i+1] for i in range(len(proof_steps)-1)] + [[i+1, i] for i in range(len(proof_steps)-1)], dtype=torch.long).t().contiguous().to(device)
        batch = torch.zeros(len(proof_steps), dtype=torch.long, device=device)
        return x, edge_index, batch

    def graph_embedding(self, proof_steps):
        x, edge_index, batch = self.build_graph(proof_steps)
        return self.graph_model(x, edge_index, batch)

    def llm_step(self, context_embedding, proof_text):
        prompt = f"<think>Proof so far:\n{proof_text}\nNext step:\n<answer>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.lm.generate(**inputs, max_new_tokens=64)
        step = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:])
        return step

    def compute_reward(self, proof_steps):
        if len(proof_steps) == 0:
            return 0.0
        last_step = proof_steps[-1]
        if "contradiction" in last_step.lower():
            return -1.0
        if "boxed" in last_step.lower():
            return 1.0
        return 0.1

    def run_episode(self, problem_text):
        proof_steps = []
        rewards = []
        for _ in range(self.max_steps):
            embedding = self.graph_embedding(proof_steps)
            next_step = self.llm_step(embedding, "\n".join(proof_steps) + "\n" + problem_text)
            proof_steps.append(next_step)
            reward = self.compute_reward(proof_steps)
            rewards.append(reward)
            if reward >= 1.0:
                break
        return proof_steps, rewards

if __name__ == "__main__":
    problems = [{"id": 1, "statement": "Prove that 1+1=2"}, {"id": 2, "statement": "Prove Fermat's little theorem"}]
    agent = Graph2ProofRL()
    results = []
    for p in problems:
        proof_steps, rewards = agent.run_episode(p["statement"])
        results.append({"id": p["id"], "proof": proof_steps, "rewards": rewards})
    with open("graph2proof_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
