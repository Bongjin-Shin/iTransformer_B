"""
001_Genesis Sensor Data Visualization
=====================================
Generic TSB-AD-M dataset visualizer.
Columns are auto-detected; only 'Label' is treated specially.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math
import os

# ── 데이터 로드 ──────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "dataset", "TSB-AD-M",
    "137_CreditCard_id_1_Finance_tr_500_1st_541.csv",
)
SAVE_DIR = os.path.join(os.path.dirname(__file__), "figures_CreditCard")
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nLabel distribution:\n{df['Label'].value_counts()}")
print(f"\nBasic statistics:\n{df.describe()}")

sensor_cols = [c for c in df.columns if c != "Label"]
n_sensors = len(sensor_cols)

# ── 1. 전체 시계열 + 이상 구간 하이라이트 ────────────────────
fig, axes = plt.subplots(n_sensors, 1, figsize=(20, 2.0 * n_sensors + 1), sharex=True)
fig.suptitle("CreditCard - Full Time Series with Anomaly Regions",
             fontsize=14, fontweight="bold", y=0.99)

anomaly_mask = df["Label"].values == 1
for i, col in enumerate(sensor_cols):
    ax = axes[i]
    ax.plot(df.index, df[col], linewidth=0.4, color="steelblue", label=col)
    # 이상 구간을 빨간 배경으로 표시
    ax.fill_between(df.index, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=anomaly_mask, color="red", alpha=0.15, label="Anomaly")
    # y-lim 재설정 (fill_between 이 ylim 바꿀 수 있으니)
    ymin, ymax = df[col].min(), df[col].max()
    margin = (ymax - ymin) * 0.05 if ymax != ymin else 0.5
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.fill_between(df.index, ymin - margin, ymax + margin,
                    where=anomaly_mask, color="red", alpha=0.15)
    ax.set_ylabel(col, fontsize=6, fontweight="bold")
    ax.tick_params(axis='both', labelsize=6)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time Step", fontsize=10)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
fig.savefig(os.path.join(SAVE_DIR, "01_full_timeseries.png"), dpi=150, bbox_inches="tight")
print(f"\n[Saved] 01_full_timeseries.png")

# # ── 2. 센서별 분포 (Normal vs Anomaly) ───────────────────────
# n_cols_grid = min(4, n_sensors)
# n_rows_grid = math.ceil(n_sensors / n_cols_grid)
# fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(5 * n_cols_grid, 4 * n_rows_grid))
# fig.suptitle("Sensor Value Distributions: Normal vs Anomaly",
#              fontsize=16, fontweight="bold")
# axes = axes.flatten()
#
# for i, col in enumerate(sensor_cols):
#     ax = axes[i]
#     normal_vals = df.loc[df["Label"] == 0, col].dropna()
#     anomaly_vals = df.loc[df["Label"] == 1, col].dropna()
#     ax.hist(normal_vals, bins=80, alpha=0.6, color="steelblue", label="Normal", density=True)
#     if len(anomaly_vals) > 0:
#         ax.hist(anomaly_vals, bins=80, alpha=0.6, color="crimson", label="Anomaly", density=True)
#     ax.set_title(col, fontsize=12, fontweight="bold")
#     ax.legend(fontsize=9)
#     ax.grid(True, alpha=0.3)
#
# for j in range(n_sensors, len(axes)):
#     axes[j].set_visible(False)
#
# fig.tight_layout(rect=[0, 0.02, 1, 0.96])
# fig.savefig(os.path.join(SAVE_DIR, "02_distributions.png"), dpi=150, bbox_inches="tight")
# print("[Saved] 02_distributions.png")

# # ── 3. 상관관계 히트맵 ──────────────────────────────────────
# hm_size = max(8, n_sensors * 0.7)
# fig, axes = plt.subplots(1, 2, figsize=(hm_size * 2 + 4, hm_size))
# for ax, (label_val, title) in zip(axes, [(0, "Normal"), (1, "Anomaly")]):
#     subset = df[df["Label"] == label_val][sensor_cols]
#     if len(subset) > 0:
#         corr = subset.corr()
#         im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
#         ax.set_xticks(range(n_sensors))
#         ax.set_yticks(range(n_sensors))
#         tick_fs = max(6, 10 - n_sensors // 5)
#         ax.set_xticklabels(sensor_cols, rotation=45, ha="right", fontsize=tick_fs)
#         ax.set_yticklabels(sensor_cols, fontsize=tick_fs)
#         annot_fs = max(5, 8 - n_sensors // 6)
#         for r in range(n_sensors):
#             for c in range(n_sensors):
#                 ax.text(c, r, f"{corr.iloc[r, c]:.2f}", ha="center", va="center", fontsize=annot_fs)
#         ax.set_title(f"Correlation ({title})", fontsize=13, fontweight="bold")
#     else:
#         ax.set_title(f"Correlation ({title}) – no data", fontsize=13)
#
# fig.colorbar(im, ax=axes, shrink=0.8, label="Pearson r")
# fig.tight_layout(rect=[0, 0.02, 1, 0.96])
# fig.savefig(os.path.join(SAVE_DIR, "03_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
# print("[Saved] 03_correlation_heatmap.png")

# # ── 4. Box-plot: Normal vs Anomaly ──────────────────────────
# fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(5 * n_cols_grid, 4 * n_rows_grid))
# fig.suptitle("Box Plots: Normal vs Anomaly per Sensor",
#              fontsize=16, fontweight="bold")
# axes = axes.flatten()
#
# for i, col in enumerate(sensor_cols):
#     ax = axes[i]
#     data_to_plot = [
#         df.loc[df["Label"] == 0, col].dropna().values,
#         df.loc[df["Label"] == 1, col].dropna().values,
#     ]
#     bp = ax.boxplot(data_to_plot, labels=["Normal", "Anomaly"], patch_artist=True,
#                     widths=0.5)
#     bp["boxes"][0].set_facecolor("steelblue")
#     bp["boxes"][0].set_alpha(0.6)
#     if len(data_to_plot[1]) > 0:
#         bp["boxes"][1].set_facecolor("crimson")
#         bp["boxes"][1].set_alpha(0.6)
#     ax.set_title(col, fontsize=12, fontweight="bold")
#     ax.grid(True, alpha=0.3)
#
# for j in range(n_sensors, len(axes)):
#     axes[j].set_visible(False)
#
# fig.tight_layout(rect=[0, 0.02, 1, 0.96])
# fig.savefig(os.path.join(SAVE_DIR, "04_boxplots.png"), dpi=150, bbox_inches="tight")
# print("[Saved] 04_boxplots.png")

# ── 5. 이상 구간 확대 시각화 (첫 번째 이상이 시작되는 부근) ──
anomaly_indices = df.index[df["Label"] == 1]
if len(anomaly_indices) > 0:
    first_anomaly = anomaly_indices[0]
    window = 500  # 좌우 500 스텝
    start = max(0, first_anomaly - window)
    end = min(len(df), first_anomaly + window)
    df_zoom = df.iloc[start:end]

    fig, axes = plt.subplots(n_sensors, 1, figsize=(18, 1.8 * n_sensors + 1), sharex=True)
    fig.suptitle(f"Zoomed View around First Anomaly (index {first_anomaly})",
                 fontsize=14, fontweight="bold", y=0.99)

    for i, col in enumerate(sensor_cols):
        ax = axes[i]
        ax.plot(df_zoom.index, df_zoom[col], linewidth=0.8, color="steelblue")
        ymin, ymax = df_zoom[col].min(), df_zoom[col].max()
        margin = (ymax - ymin) * 0.05 if ymax != ymin else 0.5
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.fill_between(df_zoom.index, ymin - margin, ymax + margin,
                        where=df_zoom["Label"].values == 1,
                        color="red", alpha=0.2)
        ax.set_ylabel(col, fontsize=6, fontweight="bold")
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step", fontsize=10)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(os.path.join(SAVE_DIR, "05_zoom_first_anomaly.png"), dpi=150, bbox_inches="tight")
    print("[Saved] 05_zoom_first_anomaly.png")

# # ── 6. 이동 평균 & 이동 표준편차 (Rolling Statistics) ────────
# window_size = 200
# fig, axes = plt.subplots(n_sensors, 1, figsize=(20, 2.5 * n_sensors + 1), sharex=True)
# fig.suptitle(f"Rolling Mean & Std (window={window_size}) with Anomaly Regions",
#              fontsize=16, fontweight="bold", y=0.98)
#
# for i, col in enumerate(sensor_cols):
#     ax = axes[i]
#     rolling_mean = df[col].rolling(window=window_size, center=True).mean()
#     rolling_std = df[col].rolling(window=window_size, center=True).std()
#     ax.plot(df.index, rolling_mean, linewidth=0.8, color="steelblue", label="Rolling Mean")
#     ax.fill_between(df.index,
#                     rolling_mean - 2 * rolling_std,
#                     rolling_mean + 2 * rolling_std,
#                     color="steelblue", alpha=0.15, label="±2σ")
#     ymin = (rolling_mean - 2 * rolling_std).min()
#     ymax = (rolling_mean + 2 * rolling_std).max()
#     if np.isfinite(ymin) and np.isfinite(ymax):
#         margin = (ymax - ymin) * 0.05 if ymax != ymin else 0.5
#         ax.fill_between(df.index, ymin - margin, ymax + margin,
#                         where=anomaly_mask, color="red", alpha=0.12)
#     ax.set_ylabel(col, fontsize=10, fontweight="bold")
#     ax.grid(True, alpha=0.3)
#     if i == 0:
#         ax.legend(fontsize=9, loc="upper right")
#
# axes[-1].set_xlabel("Time Step", fontsize=12)
# fig.tight_layout(rect=[0, 0.02, 1, 0.96])
# fig.savefig(os.path.join(SAVE_DIR, "06_rolling_stats.png"), dpi=150, bbox_inches="tight")
# print("[Saved] 06_rolling_stats.png")

# # ── 7. Label 비율 파이 차트 & 이상 구간 길이 히스토그램 ──────
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# # 파이 차트
# label_counts = df["Label"].value_counts().sort_index()
# colors_pie = ["steelblue", "crimson"]
# axes[0].pie(label_counts, labels=["Normal (0)", "Anomaly (1)"],
#             autopct="%1.2f%%", colors=colors_pie, startangle=140,
#             textprops={"fontsize": 12})
# axes[0].set_title("Label Distribution", fontsize=14, fontweight="bold")
#
# # 이상 구간 길이 분포
# label_arr = df["Label"].values
# diffs = np.diff(label_arr)
# starts = np.where(diffs == 1)[0] + 1   # 0→1 전환
# ends = np.where(diffs == -1)[0] + 1     # 1→0 전환
# # 첫 포인트가 anomaly 일 경우
# if label_arr[0] == 1:
#     starts = np.insert(starts, 0, 0)
# # 마지막이 anomaly 일 경우
# if label_arr[-1] == 1:
#     ends = np.append(ends, len(label_arr))
# lengths = ends[:len(starts)] - starts[:len(ends)]
#
# if len(lengths) > 0:
#     axes[1].hist(lengths, bins=max(1, min(50, len(lengths))),
#                  color="crimson", alpha=0.7, edgecolor="black")
#     axes[1].set_xlabel("Anomaly Segment Length", fontsize=12)
#     axes[1].set_ylabel("Count", fontsize=12)
#     axes[1].set_title(f"Anomaly Segment Length Distribution (n={len(lengths)})",
#                       fontsize=14, fontweight="bold")
#     axes[1].grid(True, alpha=0.3)
# else:
#     axes[1].text(0.5, 0.5, "No anomaly segments", ha="center", va="center",
#                  fontsize=14, transform=axes[1].transAxes)
#
# fig.tight_layout(rect=[0, 0.02, 1, 0.96])
# fig.savefig(os.path.join(SAVE_DIR, "07_label_stats.png"), dpi=150, bbox_inches="tight")
# print("[Saved] 07_label_stats.png")

plt.show()
print(f"\n✅ All figures saved to {os.path.abspath(SAVE_DIR)}")
