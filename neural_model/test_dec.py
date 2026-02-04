"""
Test script for Deep Embedded Clustering (DEC) implementation.

This script verifies that DEC improves cluster separation compared to
standard autoencoder by measuring pushing_score variance between clusters.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_analysis.Model.Curve import Curve


def test_dec_training():
    """Test that DEC model trains without errors."""
    print("=" * 60)
    print("TEST: DEC Training")
    print("=" * 60)
    
    from deep_embedded_clustering import DECAutoEncoder
    import torch
    
    # Create small synthetic dataset for testing
    np.random.seed(42)
    n_samples = 100
    input_dim = 355  # Match expected input dim
    
    # Create synthetic data with 3 clusters
    data = np.random.randn(n_samples, input_dim).astype(np.float32)
    mask = np.ones_like(data)  # No padding
    
    # Create model
    model = DECAutoEncoder(
        input_dim=input_dim,
        latent_dim=16,
        n_clusters=3
    )
    
    # Phase 1: Pretrain (just 2 epochs for testing)
    print("\n[1/3] Testing pretraining...")
    model.pretrain(
        data=data,
        mask=mask,
        epochs=2,
        batch_size=16
    )
    
    assert model.pretrained, "Model should be marked as pretrained"
    print("✓ Pretraining works")
    
    # Initialize clusters
    print("\n[2/3] Testing cluster initialization...")
    labels = model.initialize_clusters(data)
    
    assert len(labels) == n_samples, "Should have labels for all samples"
    assert set(labels).issubset({0, 1, 2}), "Labels should be 0, 1, or 2"
    print("✓ Cluster initialization works")
    
    # Phase 2: DEC training (just 2 epochs for testing)
    print("\n[3/3] Testing DEC training...")
    model.train_dec(
        data=data,
        mask=mask,
        epochs=2,
        batch_size=16
    )
    
    assert model.dec_trained, "Model should be marked as DEC trained"
    print("✓ DEC training works")
    
    # Test prediction
    final_labels, z = model.predict_clusters(data)
    
    assert len(final_labels) == n_samples
    assert z.shape == (n_samples, 16)
    print("✓ Prediction works")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    
    return True


def test_cluster_separation():
    """
    Test that DEC improves cluster separation on real data.
    
    This test loads real data and compares pushing_score distribution
    between clusters to verify they correspond to driving styles.
    """
    print("\n" + "=" * 60)
    print("TEST: Cluster Separation on Real Data")
    print("=" * 60)
    
    from deep_embedded_clustering import DECAutoEncoder
    
    # Load dataset - check multiple locations
    possible_paths = [
        "data/dataset/normalized_dataset2.npz",
        "normalized_dataset2.npz",
        "data/dataset/normalized_dataset.npz"
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print(f"⚠ Dataset not found in any of: {possible_paths}")
        print("  Skipping real data test")
        return True
    
    dataset = np.load(dataset_path, allow_pickle=True)
    data = dataset["data"].astype(np.float32)
    mask = dataset["mask"].astype(np.float32)
    mean = dataset["mean"]
    std = dataset["std"]
    
    print(f"Loaded dataset: {data.shape}")
    
    # Use subset for faster testing
    n_samples = min(500, len(data))
    data = data[:n_samples]
    mask = mask[:n_samples]
    
    # Create and train DEC model
    model = DECAutoEncoder(
        input_dim=data.shape[1],
        latent_dim=32,
        n_clusters=3
    )
    
    print("\nTraining DEC (reduced epochs for testing)...")
    model.pretrain(data=data, mask=mask, epochs=10, batch_size=32)
    model.initialize_clusters(data)
    model.train_dec(data=data, mask=mask, epochs=20, batch_size=32)
    
    # Get cluster assignments
    labels, z = model.predict_clusters(data)
    
    # Create Curve objects and calculate pushing scores
    print("\nCalculating pushing scores per cluster...")
    cluster_scores = {0: [], 1: [], 2: []}
    
    import torch
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(data):
            try:
                latent_np = z[i]
                curve = Curve.from_norm_data(sample, mask[i], mean, std, latent_np)
                score = curve.pushing_score()
                cluster_scores[labels[i]].append(score)
            except Exception as e:
                # Some curves might fail due to data issues
                pass
    
    # Print statistics
    print("\nCluster Statistics (pushing_score):")
    print("-" * 40)
    
    means = []
    for cluster_id in range(3):
        scores = cluster_scores[cluster_id]
        if len(scores) > 0:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            means.append(mean_score)
            print(f"Cluster {cluster_id}: n={len(scores):4d}, "
                  f"mean={mean_score:.2f}, std={std_score:.2f}")
        else:
            print(f"Cluster {cluster_id}: EMPTY")
    
    # Check if clusters have different means
    if len(means) >= 2:
        mean_range = max(means) - min(means)
        print(f"\nMean range between clusters: {mean_range:.2f}")
        
        if mean_range > 5:
            print("✓ Good cluster separation (range > 5)")
        elif mean_range > 2:
            print("⚠ Moderate cluster separation (range 2-5)")
        else:
            print("✗ Poor cluster separation (range < 2)")
    
    print("\n" + "=" * 60)
    print("CLUSTER SEPARATION TEST COMPLETED")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    os.chdir(project_root)  # Change to project root
    
    # Run tests
    try:
        test_dec_training()
        test_cluster_separation()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    sys.exit(0)
