import os
import json
import traci

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "simulation.sumocfg")


def run_fixed_simulation(green_time=42, yellow_time=3, sim_duration=600, gui=False):
    """Run simulation with fixed-time signal control."""
    binary = "sumo-gui" if gui else "sumo"
    traci.start([binary, "-c", CONFIG_PATH, "--no-warnings"])

    total_waiting_time = 0.0
    step_count = 0
    queue_lengths = []
    cumulative_arrived = 0

    while traci.simulation.getTime() < sim_duration:
        traci.simulationStep()
        step_count += 1

        vehicles = traci.vehicle.getIDList()
        step_waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
        total_waiting_time += step_waiting

        queue = sum(1 for v in vehicles if traci.vehicle.getSpeed(v) < 0.1)
        queue_lengths.append(queue)

        cumulative_arrived += traci.simulation.getArrivedNumber()

    traci.close()

    avg_waiting = total_waiting_time / max(step_count, 1)
    avg_queue = sum(queue_lengths) / max(len(queue_lengths), 1)

    results = {
        "mode": "fixed",
        "green_time": green_time,
        "avg_waiting_time": round(avg_waiting, 2),
        "avg_queue_length": round(avg_queue, 2),
        "total_arrived": cumulative_arrived,
    }

    out_path = os.path.join(os.path.dirname(__file__), "results", "fixed_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Fixed-time results: {results}")
    return results


if __name__ == "__main__":
    import sys
    use_gui = "--gui" in sys.argv
    run_fixed_simulation(gui=use_gui)
