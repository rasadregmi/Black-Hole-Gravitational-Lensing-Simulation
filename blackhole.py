import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation

G = 6.67430e-11
c = 3.0e8
M = 5e30
Rs = 2 * G * M / c**2

def geodesic_eq(y, phi, Rs):
    u, du_dphi = y
    if np.abs(u) < 1e-6:
        return [du_dphi, 0]
    d2u_dphi2 = 1.5 * Rs * u**2 - u
    return [du_dphi, d2u_dphi2]

phis = np.linspace(0, 2 * np.pi, 2000)
impact_params = np.linspace(3.2 * Rs, 6 * Rs, 8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.set_xlim([-8 * Rs, 8 * Rs])
ax1.set_ylim([-8 * Rs, 8 * Rs])
ax1.set_aspect('equal')
ax1.set_facecolor('black')
ax1.set_title("Black Hole Gravitational Lensing", color='white')
ax1.tick_params(colors='white')
ax1.set_xlabel("X (m)", color='white')
ax1.set_ylabel("Y (m)", color='white')

black_hole = plt.Circle((0, 0), Rs, color='black', ec='white', lw=1)
ax1.add_patch(black_hole)

light_paths = []
all_radii = []
lines = []

for i, b in enumerate(impact_params):
    u0 = 1 / b
    du0 = 0.0
    sol = odeint(geodesic_eq, [u0, du0], phis, args=(Rs,))
    r = 1 / sol[:, 0]
    r[r > 10 * Rs] = np.nan
    x = r * np.cos(phis)
    y = r * np.sin(phis)
    light_paths.append((x, y))
    all_radii.append(r)
    line, = ax1.plot([], [], color=plt.cm.plasma(i / len(impact_params)))
    lines.append(line)

ax2.set_xlim(0, len(phis))
ax2.set_ylim(0, 10 * Rs)
ax2.set_title("Radius of Light Paths Over Time", color='black')
ax2.set_xlabel("Frame")
ax2.set_ylabel("Radius (m)")

radius_lines = []
for i in range(len(impact_params)):
    radius_line, = ax2.plot([], [], color=plt.cm.plasma(i / len(impact_params)))
    radius_lines.append(radius_line)

def update(frame):
    for i, ((x, y), r, line, radius_line) in enumerate(zip(light_paths, all_radii, lines, radius_lines)):
        line.set_xdata(x[:frame])
        line.set_ydata(y[:frame])
        radius_line.set_xdata(range(frame))
        radius_line.set_ydata(r[:frame])
    return lines + radius_lines

ani = animation.FuncAnimation(fig, update, frames=len(phis), interval=10, blit=True)
plt.show()
