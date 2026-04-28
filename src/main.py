import sys
import os
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root or from src/
sys.path.insert(0, os.path.dirname(__file__))

from generators import generate_polygon, visualize_polygon
from algorithms import gauss_area, monte_carlo_area

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)

# ─────────────────────────────────────────────
# 1. Generate & visualise polygons N=10, 50, 100
# ─────────────────────────────────────────────
print("=" * 55)
print("Завдання 1: Генерація та візуалізація полігонів")
print("=" * 55)

polygon_sizes = [10, 50, 100]
polygons = {}
for n in polygon_sizes:
    random.seed(SEED + n)
    poly = generate_polygon(num_points=n, radius=10.0)
    polygons[n] = poly
    img_path = os.path.join(IMAGES_DIR, f'polygon_{n}.png')
    visualize_polygon(poly, filename=img_path)
    print(f"  N={n:>4}  Shapely area = {poly.area:.6f}")

# ─────────────────────────────────────────────
# 2. Verify algorithm implementations (N=50)
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("Завдання 2: Перевірка алгоритмів (N=50)")
print("=" * 55)

poly50 = polygons[50]
shapely_area = poly50.area
gauss  = gauss_area(poly50)
mc     = monte_carlo_area(poly50, num_points=100_000, seed=SEED)

print(f"  Shapely    : {shapely_area:.6f}")
print(f"  Гаус       : {gauss:.6f}  (похибка {abs(gauss - shapely_area)/shapely_area*100:.4f}%)")
print(f"  Монте-Карло: {mc:.6f}  (похибка {abs(mc - shapely_area)/shapely_area*100:.4f}%)")

# ─────────────────────────────────────────────
# 3. Monte Carlo accuracy vs number of points
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("Завдання 3: Точність Монте-Карло (N=50)")
print("=" * 55)

m_values = [100, 1_000, 10_000, 100_000]
errors   = []

for m in m_values:
    mc_area = monte_carlo_area(poly50, num_points=m, seed=SEED)
    err_pct = abs(mc_area - shapely_area) / shapely_area * 100
    errors.append(err_pct)
    print(f"  M={m:>7}  Area={mc_area:.4f}  Похибка={err_pct:.4f}%")

# Plot error curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(m_values, errors, 'o-', color='steelblue', linewidth=2, markersize=8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Кількість точок (M)', fontsize=12)
ax.set_ylabel('Відносна похибка (%)', fontsize=12)
ax.set_title('Збіжність методу Монте-Карло (N=50 вершин)', fontsize=13)
ax.grid(True, which='both', linestyle='--', alpha=0.6)
ax.set_xticks(m_values)
ax.set_xticklabels([str(m) for m in m_values])
for m, e in zip(m_values, errors):
    ax.annotate(f'{e:.2f}%', xy=(m, e), xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=10, color='darkblue')
plt.tight_layout()
err_plot_path = os.path.join(IMAGES_DIR, 'error_plot.png')
plt.savefig(err_plot_path, dpi=120)
plt.close()
print(f"\n  Графік збіжності збережено: {err_plot_path}")

# ─────────────────────────────────────────────
# 4. Benchmark
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("Завдання 4: Benchmark (Shapely / Гаус / Монте-Карло)")
print("=" * 55)

bench_sizes = [10, 50, 100, 1000]
REPS        = 5
MC_POINTS   = 100_000

bench_results = {}

for n in bench_sizes:
    random.seed(SEED + n)
    poly = generate_polygon(num_points=n, radius=10.0)

    # Shapely
    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = poly.area
    t_shapely = (time.perf_counter() - t0) / REPS

    # Gauss
    t0 = time.perf_counter()
    for _ in range(REPS):
        gauss_area(poly)
    t_gauss = (time.perf_counter() - t0) / REPS

    # Monte Carlo
    t0 = time.perf_counter()
    for _ in range(REPS):
        monte_carlo_area(poly, num_points=MC_POINTS, seed=SEED)
    t_mc = (time.perf_counter() - t0) / REPS

    bench_results[n] = (t_shapely, t_gauss, t_mc)
    print(f"  N={n:>4}  Shapely={t_shapely:.6f}s  Гаус={t_gauss:.6f}s  МК={t_mc:.6f}s")

# Plot benchmark bar chart
ns    = list(bench_results.keys())
t_sh  = [bench_results[n][0] for n in ns]
t_g   = [bench_results[n][1] for n in ns]
t_mc  = [bench_results[n][2] for n in ns]

x     = np.arange(len(ns))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width, t_sh, width, label='Shapely', color='seagreen')
b2 = ax.bar(x,         t_g,  width, label='Гаус',    color='steelblue')
b3 = ax.bar(x + width, t_mc, width, label='Монте-Карло (M=100 000)', color='tomato')

ax.set_yscale('log')
ax.set_xlabel('Кількість вершин (N)', fontsize=12)
ax.set_ylabel('Час виконання (с, лог. шкала)', fontsize=12)
ax.set_title('Порівняння швидкодії алгоритмів', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in ns])
ax.legend(fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
bench_plot_path = os.path.join(IMAGES_DIR, 'time_benchmark.png')
plt.savefig(bench_plot_path, dpi=120)
plt.close()
print(f"\n  Графік benchmark збережено: {bench_plot_path}")

# ─────────────────────────────────────────────
# 5. Export timing table for README
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("Таблиця для README:")
print("=" * 55)
print(f"| {'N':>6} | {'Shapely (с)':>12} | {'Гаус (с)':>12} | {'Монте-Карло (с)':>16} |")
print("|" + "-"*8 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*18 + "|")
for n in ns:
    ts, tg, tm = bench_results[n]
    print(f"| {n:>6} | {ts:>12.6f} | {tg:>12.6f} | {tm:>16.6f} |")

print("\n  Похибки Монте-Карло:")
print(f"| {'M':>8} | {'Похибка (%)':>12} |")
print("|" + "-"*10 + "|" + "-"*14 + "|")
for m, e in zip(m_values, errors):
    print(f"| {m:>8} | {e:>12.4f} |")

print("\nВсі завдання виконано успішно.")
