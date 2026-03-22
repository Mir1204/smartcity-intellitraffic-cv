import numpy as np
import pandas as pd
import os


def generate_dataset(n_samples=2000, seed=42):
    """Generate synthetic traffic dataset for green time prediction."""
    rng = np.random.RandomState(seed)

    car_count = rng.randint(0, 21, size=n_samples)
    bus_truck_count = rng.randint(0, 11, size=n_samples)
    bike_count = rng.randint(0, 26, size=n_samples)
    rain = rng.randint(0, 2, size=n_samples)

    # Gaussian noise for realistic variation
    noise = rng.normal(0, 2, size=n_samples)

    # Target: green signal time (seconds)
    green_time = (
        car_count * 2
        + bus_truck_count * 3
        + bike_count * 1
        + rain * 5
        + noise
    )

    # Clamp to realistic bounds
    green_time = np.clip(green_time, 10, 120).round(1)

    df = pd.DataFrame({
        "car_count": car_count,
        "bus_truck_count": bus_truck_count,
        "bike_count": bike_count,
        "rain": rain,
        "green_time": green_time,
    })

    out_path = os.path.join(os.path.dirname(__file__), "data", "synthetic_traffic.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    return df


if __name__ == "__main__":
    generate_dataset()
