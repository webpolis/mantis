"""
HMST Demo Script

Demonstrates the complete HMST system with a simple example.
"""

import torch
from hmst.models import BaseMoEModel, MetaController, CriticModel, EpisodicMemorySSM
from hmst.models.meta_controller import StateSummaryEncoder
from hmst.models.critic import CriticValueNetwork
from hmst.memory import EpisodicMemory, SemanticMemory, MemoryConsolidator
from hmst.inference import HMSTInferenceEngine
from hmst.configs.model_config import get_small_config


def main():
    print("=" * 80)
    print("HMST: Hierarchical Memory-State Transformer Demo")
    print("=" * 80)

    # Use small config for demo
    config = get_small_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nDevice: {device}")
    print(f"Configuration: Small (1B parameters)")

    # Initialize models
    print("\n1. Initializing models...")

    base_model = BaseMoEModel(
        vocab_size=config.base_moe.vocab_size,
        d_model=config.base_moe.d_model,
        n_layers=config.base_moe.n_layers,
        n_heads=config.base_moe.n_heads,
        d_ff=config.base_moe.d_ff,
        n_experts=config.base_moe.n_experts,
        top_k=config.base_moe.top_k,
        max_seq_len=config.base_moe.max_seq_len,
        dropout=config.base_moe.dropout
    )

    meta_controller = MetaController(
        d_model=config.meta_controller.d_model,
        n_layers=config.meta_controller.n_layers,
        n_heads=config.meta_controller.n_heads,
        d_ff=config.meta_controller.d_ff,
        dropout=config.meta_controller.dropout,
        n_experts=config.meta_controller.n_experts,
        state_dim=config.meta_controller.state_dim
    )

    critic = CriticModel(
        vocab_size=config.critic.vocab_size,
        d_model=config.critic.d_model,
        n_layers=config.critic.n_layers,
        n_heads=config.critic.n_heads,
        d_ff=config.critic.d_ff,
        max_seq_len=config.critic.max_seq_len,
        dropout=config.critic.dropout
    )

    ssm = EpisodicMemorySSM(
        d_model=config.episodic_memory.d_model,
        d_state=config.episodic_memory.d_state,
        n_blocks=config.episodic_memory.n_blocks,
        d_conv=config.episodic_memory.d_conv,
        expand=config.episodic_memory.expand,
        max_seq_len=config.episodic_memory.max_seq_len,
        dropout=config.episodic_memory.dropout
    )

    state_encoder = StateSummaryEncoder(state_dim=config.meta_controller.state_dim)

    # Count parameters
    param_counts = base_model.count_parameters()
    print(f"  Base MoE: {param_counts['total'] / 1e9:.2f}B total, {param_counts['active'] / 1e9:.2f}B active")
    print(f"  Meta-Controller: {sum(p.numel() for p in meta_controller.parameters()) / 1e6:.1f}M")
    print(f"  Critic: {sum(p.numel() for p in critic.parameters()) / 1e9:.2f}B")

    # Initialize memory systems
    print("\n2. Initializing memory systems...")

    episodic_memory = EpisodicMemory(
        ssm_model=ssm,
        max_entries=config.episodic_memory.max_entries,
        context_window=config.episodic_memory.max_seq_len,
        device=device
    )

    semantic_memory = SemanticMemory(
        dimension=config.semantic_memory.dimension,
        max_entries=config.semantic_memory.max_entries,
        index_type=config.semantic_memory.index_type,
        use_gpu=config.semantic_memory.use_gpu and device == 'cuda'
    )

    print(f"  Episodic Memory: {episodic_memory.size()} entries")
    print(f"  Semantic Memory: {semantic_memory.size()} entries")

    # Pre-populate semantic memory with some facts
    print("\n3. Populating semantic memory with sample facts...")

    sample_facts = [
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
        "Python is a programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "The Earth orbits around the Sun.",
    ]

    for fact in sample_facts:
        # Generate random embedding (in production, use model encoding)
        embedding = torch.randn(config.semantic_memory.dimension)
        semantic_memory.add(embedding, fact)

    print(f"  Added {len(sample_facts)} facts to semantic memory")

    # Create inference engine
    print("\n4. Creating inference engine...")

    engine = HMSTInferenceEngine(
        base_model=base_model,
        meta_controller=meta_controller,
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        critic_model=critic,
        state_encoder=state_encoder,
        tokenizer=None,  # Using placeholder tokenization
        device=device
    )

    print("  Inference engine ready")

    # Run sample queries
    print("\n5. Running sample queries...")
    print("=" * 80)

    queries = [
        "What is the capital of France?",
        "Tell me about Python programming.",
        "What is 2 + 2?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)

        result = engine.generate(query, max_length=20, return_details=True)

        print(f"Response: {result['response']}")
        print(f"Path: {result['path']}")
        print(f"Uncertainty: {result['uncertainty']:.4f}")
        print(f"Verified: {result['verified']}")
        print(f"Latency: {result['latency']:.3f}s")
        print(f"Memory Used: Episodic={result['memory_used']['episodic']}, "
              f"Semantic={result['memory_used']['semantic']}")

    # Show statistics
    print("\n" + "=" * 80)
    print("Inference Statistics")
    print("=" * 80)

    stats = engine.get_stats()
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Early Exit Rate: {stats.get('early_exit_rate', 0):.2%}")
    print(f"Episodic Access Rate: {stats.get('episodic_access_rate', 0):.2%}")
    print(f"Semantic Access Rate: {stats.get('semantic_access_rate', 0):.2%}")
    print(f"Verification Rate: {stats.get('verification_rate', 0):.2%}")
    print(f"Average Latency: {stats['avg_latency']:.3f}s")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
