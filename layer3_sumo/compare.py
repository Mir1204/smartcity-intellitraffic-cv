import json
import os
from layer3_sumo.run_fixed import run_fixed_simulation
from layer3_sumo.run_adaptive import run_adaptive_simulation


def compare(rain=0, gui=False):
    """Run both simulations and produce comparison metrics."""
    print("=" * 50)
    print("Running FIXED-TIME simulation...")
    fixed = run_fixed_simulation(gui=gui)

    print("=" * 50)
    print("Running ADAPTIVE (ML) simulation...")
    adaptive = run_adaptive_simulation(rain=rain, gui=gui)

    # Compute improvements
    wait_improvement = (
        (fixed["avg_waiting_time"] - adaptive["avg_waiting_time"])
        / max(fixed["avg_waiting_time"], 0.01)
    ) * 100

    queue_improvement = (
        (fixed["avg_queue_length"] - adaptive["avg_queue_length"])
        / max(fixed["avg_queue_length"], 0.01)
    ) * 100

    print("=" * 50)
    print("COMPARISON:")
    print(f"  Avg Waiting Time:  Fixed={fixed['avg_waiting_time']}s  Adaptive={adaptive['avg_waiting_time']}s")
    print(f"  Avg Queue Length:  Fixed={fixed['avg_queue_length']}   Adaptive={adaptive['avg_queue_length']}")
    print(f"  Throughput:        Fixed={fixed['total_arrived']}      Adaptive={adaptive['total_arrived']}")
    print(f"  Waiting Time Improvement: {wait_improvement:.1f}%")
    print(f"  Queue Length Improvement:  {queue_improvement:.1f}%")

    comparison = {
        "fixed": fixed,
        "adaptive": adaptive,
        "improvement_wait_pct": round(wait_improvement, 1),
        "improvement_queue_pct": round(queue_improvement, 1),
    }

    out_path = os.path.join(os.path.dirname(__file__), "results", "comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved -> {out_path}")
    return comparison


if __name__ == "__main__":
    import sys
    use_gui = "--gui" in sys.argv
    compare(gui=use_gui)
