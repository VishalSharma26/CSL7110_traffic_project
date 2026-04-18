"""Generate EDA plots for the traffic prediction report."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = Path("data/processed/sample_200k.parquet")
OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_FILE)
print(f"Loaded {len(df):,} rows")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# 1. Trip duration distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["trip_duration_minutes"].clip(0, 60), bins=50, color="#2196F3", edgecolor="white")
ax.set_xlabel("Trip Duration (minutes)")
ax.set_ylabel("Number of Trips")
ax.set_title("Distribution of Trip Duration (Jan 2024)")
plt.tight_layout()
plt.savefig(OUT_DIR / "trip_duration_dist.png")
plt.close()
print("Saved: trip_duration_dist.png")

# 2. Trips by hour of day
hourly = df.groupby("pickup_hour").size().reset_index(name="trips")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(hourly["pickup_hour"], hourly["trips"], color="#4CAF50", edgecolor="white")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Number of Trips")
ax.set_title("Trip Volume by Hour of Day")
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig(OUT_DIR / "trips_by_hour.png")
plt.close()
print("Saved: trips_by_hour.png")

# 3. Congestion level distribution
cong_counts = df["congestion_level"].value_counts()
colors = {"low": "#4CAF50", "medium": "#FF9800", "high": "#F44336"}
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(cong_counts.index, cong_counts.values,
              color=[colors.get(l, "#999") for l in cong_counts.index], edgecolor="white")
ax.set_xlabel("Congestion Level")
ax.set_ylabel("Number of Trips")
ax.set_title("Congestion Level Distribution")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{bar.get_height():,}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "congestion_distribution.png")
plt.close()
print("Saved: congestion_distribution.png")

# 4. Average speed by hour
speed_hour = df.groupby("pickup_hour")["trip_speed_mph"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(speed_hour["pickup_hour"], speed_hour["trip_speed_mph"],
        marker="o", color="#9C27B0", linewidth=2)
ax.fill_between(speed_hour["pickup_hour"], speed_hour["trip_speed_mph"],
                alpha=0.15, color="#9C27B0")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average Speed (mph)")
ax.set_title("Average Trip Speed by Hour of Day")
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig(OUT_DIR / "avg_speed_by_hour.png")
plt.close()
print("Saved: avg_speed_by_hour.png")

# 5. Weekday vs Weekend trips
day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
weekday_counts = df.groupby("pickup_weekday").size()
fig, ax = plt.subplots(figsize=(8, 4))
bar_colors = ["#2196F3"] * 5 + ["#FF5722"] * 2
ax.bar(range(7), [weekday_counts.get(i, 0) for i in range(7)],
       color=bar_colors, edgecolor="white")
ax.set_xticks(range(7))
ax.set_xticklabels(day_labels)
ax.set_xlabel("Day of Week")
ax.set_ylabel("Number of Trips")
ax.set_title("Trip Volume by Day of Week")
plt.tight_layout()
plt.savefig(OUT_DIR / "trips_by_weekday.png")
plt.close()
print("Saved: trips_by_weekday.png")

print("\nAll EDA plots saved to", OUT_DIR)
