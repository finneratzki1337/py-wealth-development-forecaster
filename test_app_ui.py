"""Quick test to verify the app loads and simulations work."""
from wealth_forecaster.config import default_config, canonicalize
from wealth_forecaster.simulate import simulate_paths
from wealth_forecaster.metrics import aggregate

def test_basic_simulation():
    """Test that a basic simulation runs without errors."""
    print("Testing basic simulation...")
    
    # Get default config
    cfg = canonicalize(default_config())
    
    # Run a small simulation
    cfg['runs_per_scenario'] = 10  # Small number for quick test
    
    print(f"Running simulation with {cfg['runs_per_scenario']} runs per scenario...")
    print(f"Start date: {cfg['start_date']}")
    print(f"Horizon: {cfg['horizon_years']} years")
    
    # Run simulation
    results = simulate_paths(cfg)
    
    print(f"\nSimulation completed!")
    print(f"Scenarios generated: {list(results.keys())}")
    
    # Check results structure
    for scenario, data in results.items():
        print(f"\nScenario: {scenario}")
        print(f"  - Nominal wealth shape: {data['nominal_wealth'].shape}")
        print(f"  - Real wealth shape: {data['real_wealth'].shape}")
        print(f"  - Time points shape: {data['time'].shape}")
    
    # Test aggregation
    print("\nTesting metrics aggregation...")
    metrics = aggregate(results)
    
    print("Aggregation successful!")
    print(f"Metrics keys: {list(metrics.keys())}")
    
    for scenario in metrics:
        if scenario in results:
            final_nominal = metrics[scenario]['nominal']['final']
            print(f"\n{scenario.title()} scenario final nominal wealth:")
            print(f"  - p10: ${final_nominal.get('p10', 0):,.0f}")
            print(f"  - p50: ${final_nominal.get('p50', 0):,.0f}")
            print(f"  - p90: ${final_nominal.get('p90', 0):,.0f}")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_basic_simulation()
