import os
import json
import traci
from layer2_ml.predict import predict_green_time

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "simulation.sumocfg")


def count_vehicles_on_edges(edge_ids):
    """Count vehicles by type on given edges (simulating what YOLO would see)."""
    counts = {"car_count": 0, "bus_truck_count": 0, "bike_count": 0}
    for edge_id in edge_ids:
        vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
        for vid in vehicles:
            vtype = traci.vehicle.getTypeID(vid)
            if vtype == "car":
                counts["car_count"] += 1
            elif vtype == "bus_truck":
                counts["bus_truck_count"] += 1
            elif vtype == "bike":
                counts["bike_count"] += 1
    return counts


def run_adaptive_simulation(rain=0, sim_duration=600, gui=False):
    """Run simulation with ML-predicted signal timing."""
    binary = "sumo-gui" if gui else "sumo"
    traci.start([binary, "-c", CONFIG_PATH, "--no-warnings"])

    tls_id = traci.trafficlight.getIDList()[0]

    # Inbound edges for each phase
    ns_edges = ["north_in", "south_in"]
    ew_edges = ["east_in", "west_in"]

    total_waiting_time = 0.0
    step_count = 0
    queue_lengths = []
    cumulative_arrived = 0

    current_phase = 0  # 0=NS green, 1=NS yellow, 2=EW green, 3=EW yellow
    phase_timer = 0
    yellow_time = 3

    # Initial prediction for NS phase
    counts = count_vehicles_on_edges(ns_edges)
    current_green_time = predict_green_time(
        counts["car_count"], counts["bus_truck_count"],
        counts["bike_count"], rain
    )

    while traci.simulation.getTime() < sim_duration:
        traci.simulationStep()
        step_count += 1
        phase_timer += 1

        # Collect metrics
        vehicles = traci.vehicle.getIDList()
        step_waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
        total_waiting_time += step_waiting
        queue = sum(1 for v in vehicles if traci.vehicle.getSpeed(v) < 0.1)
        queue_lengths.append(queue)
        cumulative_arrived += traci.simulation.getArrivedNumber()

        # Phase transitions
        if current_phase == 0 and phase_timer >= current_green_time:
            # NS green -> NS yellow
            traci.trafficlight.setPhase(tls_id, 1)
            current_phase = 1
            phase_timer = 0
        elif current_phase == 1 and phase_timer >= yellow_time:
            # NS yellow -> EW green; predict EW green time
            counts = count_vehicles_on_edges(ew_edges)
            current_green_time = predict_green_time(
                counts["car_count"], counts["bus_truck_count"],
                counts["bike_count"], rain
            )
            traci.trafficlight.setPhase(tls_id, 2)
            current_phase = 2
            phase_timer = 0
        elif current_phase == 2 and phase_timer >= current_green_time:
            # EW green -> EW yellow
            traci.trafficlight.setPhase(tls_id, 3)
            current_phase = 3
            phase_timer = 0
        elif current_phase == 3 and phase_timer >= yellow_time:
            # EW yellow -> NS green; predict NS green time
            counts = count_vehicles_on_edges(ns_edges)
            current_green_time = predict_green_time(
                counts["car_count"], counts["bus_truck_count"],
                counts["bike_count"], rain
            )
            traci.trafficlight.setPhase(tls_id, 0)
            current_phase = 0
            phase_timer = 0

    traci.close()

    avg_waiting = total_waiting_time / max(step_count, 1)
    avg_queue = sum(queue_lengths) / max(len(queue_lengths), 1)

    results = {
        "mode": "adaptive",
        "rain": rain,
        "avg_waiting_time": round(avg_waiting, 2),
        "avg_queue_length": round(avg_queue, 2),
        "total_arrived": cumulative_arrived,
    }

    out_path = os.path.join(os.path.dirname(__file__), "results", "adaptive_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Adaptive results: {results}")
    return results


if __name__ == "__main__":
    import sys
    use_gui = "--gui" in sys.argv
    run_adaptive_simulation(gui=use_gui)
