import matplotlib.pyplot as plt

def plot_parameter_trajectory(reduced_trajectory, dimensions):
    fig = plt.figure()
    if dimensions == 2:
        plt.plot(reduced_trajectory[:, 0], reduced_trajectory[:, 1], marker='o')
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(f"Parameter Space Trajectory 2D)")
    elif dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(reduced_trajectory[:, 0], reduced_trajectory[:, 1], reduced_trajectory[:, 2], marker='o')
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.set_title(f"Parameter Space Trajectory 3D)")

    return fig

def plot_gradient_norms(gradient_norms):
    """
    Plot the tracked gradient norms over meta-iterations.
    """
    fig = plt.figure()
    plt.plot(range(len(gradient_norms)), gradient_norms)
    plt.xlabel("Meta-Iteration")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Across Meta-Training")
    return fig