import matplotlib.pyplot as plt
import numpy as np

# Images
images = ['Img1', 'Img2', 'Img3', 'Img4', 'Img5', 'Img6', 'Img7']

# Mean Iterations ± Std
# Format: [Naive_RGB, K++_RGB, Naive_LAB, K++_LAB]
iterations_mean = np.array([
    [64.95, 70.80, 41.55, 41.70],
    [36.60, 34.50, 29.55, 17.30],
    [25.80, 28.80, 18.75, 12.00],
    [47.45, 47.35, 33.75, 29.00],
    [62.65, 64.95, 33.00, 36.70],
    [39.60, 42.55, 31.70, 26.45],
    [39.95, 28.05, 32.40, 14.40]
])
iterations_std = np.array([
    [13.50, 15.58, 10.02, 8.22],
    [11.62, 9.35, 14.43, 12.20],
    [6.92, 6.42, 5.56, 6.71],
    [13.23, 9.45, 12.16, 8.49],
    [18.60, 15.08, 6.53, 21.21],
    [8.49, 6.43, 6.91, 6.72],
    [6.53, 10.98, 5.37, 6.65]
])

# Mean Inertia ± Std
inertia_mean = np.array([
    [40205174.58, 40205174.74, 8049843.38, 8049861.56],
    [57002067.17, 59772394.24, 7781146.09, 7770052.53],
    [13137618.41, 7222228.97, 637826.30, 561909.66],
    [53009945.05, 53009706.63, 5757888.83, 5924876.06],
    [100430966.92, 100430999.88, 13093757.17, 12605148.79],
    [79427820.76, 79427820.60, 9977676.68, 10023562.60],
    [10011426.16, 10011542.27, 522729.46, 552548.03]
])
inertia_std = np.array([
    [23.85, 22.76, 33.54, 44.73],
    [4711586.02, 2464086.01, 445177.10, 604652.85],
    [5421543.98, 4732311.55, 138902.20, 57277.00],
    [323.23, 405.53, 1.65, 500958.66],
    [118.11, 104.78, 808171.92, 576999.19],
    [1189531.76, 1189532.07, 99332.54, 114135.32],
    [0.00, 348.32, 1.20, 52288.30]
])

# Plot settings
methods = ['Naive', 'K-Means++']
colors = ['skyblue', 'orange', 'lightgreen', 'red']

# --- Plot Iterations ---
plt.figure(figsize=(12,5))
for i in range(4):
    plt.errorbar(images, iterations_mean[:, i], yerr=iterations_std[:, i], 
                 fmt='-o', color=colors[i], label=f'{methods[i%2]} - {"RGB" if i<2 else "LAB"}')

plt.title('K-Means Iterations per Image')
plt.ylabel('Mean Iterations ± Std')
plt.xlabel('Image')
plt.legend()
plt.grid(True)
plt.show()

# --- Plot Inertia ---
plt.figure(figsize=(12,5))
for i in range(4):
    plt.errorbar(images, inertia_mean[:, i], yerr=inertia_std[:, i], 
                 fmt='-o', color=colors[i], label=f'{methods[i%2]} - {"RGB" if i<2 else "LAB"}')

plt.title('K-Means Inertia per Image')
plt.ylabel('Mean Inertia ± Std')
plt.xlabel('Image')
plt.yscale('log')  # Log scale for wide range of inertia
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
