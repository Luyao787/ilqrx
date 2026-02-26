import numpy as np
import matplotlib.pyplot as plt


def render_quad(ax, x, y, theta, phi, 
                quad, L, l, col=None, show_ell=0.05):

  pos = np.array([x, y])
  R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])

  # Update quad endpoints
  quad_comps = tuple(v @ R.T + pos for v in quad)

  for comp in quad_comps:
    ax.plot(comp[:,0], comp[:,1],
            color=col if col is not None else 'k', linewidth=2)

  # Circumscribing sphere for quad
  pos_c = pos + R @ np.array([0., 0.15*l])
  ell = plt.Circle(pos_c, l, alpha=show_ell, color='k')
  ax.add_patch(ell)

  # Pole
  pole_new = np.array([[x, y],
                       [x + L*np.sin(phi), y - L*np.cos(phi)]])
  ax.plot(pole_new[:,0], pole_new[:,1], 'o-',
          color=col if col is not None else 'b')


def render_scene(ax, obs):
  for ob in obs:
    ax.add_patch(plt.Circle(ob[0], ob[1], color='k', alpha=0.3))
    
  ax.set_aspect('equal', adjustable='box')
  

def plot_snapshots(X_traj, dt, x_target, obstacles, quad_geo, quad_params, num_snapshots=6):
    """
    Plot snapshots of the quadrotor-pendulum system at different timestamps.
    """
    N = X_traj.shape[0]
    time_indices = np.linspace(0, N-1, num_snapshots, dtype=int)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot the full trajectory path
    ax.plot(X_traj[:, 0], X_traj[:, 1], 'b-', alpha=0.3, linewidth=1)

    
    # Color map for snapshots
    colors = plt.cm.viridis(np.linspace(0, 1, num_snapshots))
    
    render_scene(ax, obstacles)
    
    for i, time_idx in enumerate(time_indices):
        # Extract state at this time
        x_pos = X_traj[time_idx, 0]
        y_pos = X_traj[time_idx, 1]
        theta = X_traj[time_idx, 2]  # quad orientation
        phi = X_traj[time_idx, 3]    # pendulum angle
        
        # Current time
        current_time = time_idx * dt
        
        # Render quadrotor and pendulum at this snapshot
        render_quad(ax, x_pos, y_pos, theta, phi, quad_geo, quad_params.L, quad_params.l, col=colors[i], show_ell=0.02)
        
    # Mark start and end points
    ax.plot(X_traj[0, 0], X_traj[0, 1], 'go', markersize=12, label='Start', zorder=10)
    ax.plot(X_traj[-1, 0], X_traj[-1, 1], 'ro', markersize=12, label='End', zorder=10)
    ax.plot(x_target[0], x_target[1], 'r*', markersize=15, label='Target', zorder=10)
    
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Set reasonable axis limits
    x_range = [X_traj[:, 0].min() - 0.5, X_traj[:, 0].max() + 0.5]
    y_range = [X_traj[:, 1].min() - 0.5, X_traj[:, 1].max() + 0.5]
    
    # Extend limits to accommodate pendulum swing
    y_range[0] -= quad_params.L  # pendulum can swing down
    y_range[1] += quad_params.L  # pendulum can swing up
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    plt.tight_layout()
    plt.savefig("quadpend_snapshots.png", dpi=600)
    plt.show()
    
    # Print snapshot information
    print("\n=== Snapshot Information ===")
    for i, time_idx in enumerate(time_indices):
        current_time = time_idx * dt
        x_pos = X_traj[time_idx, 0]
        y_pos = X_traj[time_idx, 1]
        theta = X_traj[time_idx, 2]
        phi = X_traj[time_idx, 3]
        print(f"Snapshot {i+1} (t={current_time:.1f}s): pos=({x_pos:.2f}, {y_pos:.2f}), theta={theta:.3f}, phi={phi:.3f}")