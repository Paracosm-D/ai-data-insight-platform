import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. K-Means Arm
class KMeansArm:
    def __init__(self, arm_id, data, seed_indices):
        self.arm_id = arm_id
        self.data = data
        self.n_samples, self.n_features = data.shape
        self.k = len(seed_indices)

        self.centroids = data[np.array(seed_indices) - 1].copy()
        self.assignments = np.zeros(self.n_samples, dtype=int)
        self.iterations_used = 0
        self.wcss_history = []

        self._update_assignments()
        self.wcss_history.append(self.calculate_wcss())

    def _update_assignments(self):
        distances = np.linalg.norm(self.data[:, np.newaxis] - self.centroids, axis=2)
        self.assignments = np.argmin(distances, axis=1)

    def _update_centroids(self):
        for j in range(self.k):
            points_in_cluster = self.data[self.assignments == j]
            if len(points_in_cluster) > 0:
                self.centroids[j] = np.mean(points_in_cluster, axis=0)

    def calculate_wcss(self):
        wcss = 0
        for j in range(self.k):
            points_in_cluster = self.data[self.assignments == j]
            if len(points_in_cluster) > 0:
                wcss += np.sum(np.linalg.norm(points_in_cluster - self.centroids[j], axis=1) ** 2)
        return wcss

    def step(self):
        self._update_assignments()
        self._update_centroids()
        self.iterations_used += 1
        current_wcss = self.calculate_wcss()
        self.wcss_history.append(current_wcss)
        return current_wcss


# ==========================================
# 2. budget allocation controller（successive halving）

def run_budget_allocation(data, seed_sets, total_budget=100):
    arms = [KMeansArm(i + 1, data, seeds) for i, seeds in enumerate(seed_sets)]

    # Phase 1 (Extensive Exploration): Assign 3 iterations to all 10 arms.
    print("--- Phase 1: 10 arms, 3 iterations each ---")
    for arm in arms:
        for _ in range(3):
            arm.step()

    # Sort by WCSS, keep the top 5
    arms.sort(key=lambda x: x.wcss_history[-1])
    active_arms = arms[:5]

    # Phase 2 (Focus on testing): Reassign 6 iterations to the remaining 5 arms.
    print("--- Phase 2: 5 arms, 6 iterations each ---")
    for arm in active_arms:
        for _ in range(6):
            arm.step()

    # Sort by WCSS, keep the top 2
    active_arms.sort(key=lambda x: x.wcss_history[-1])
    active_arms = active_arms[:2]

    # Phase 3 (Deep Utilization): Allocate 20 more iterations to the last two arms.
    print("--- Phase 3: 2 arms, 20 iterations each ---")
    for arm in active_arms:
        for _ in range(20):
            arm.step()

    # Final sorting, chose the best
    arms.sort(key=lambda x: x.wcss_history[-1])
    best_arm = arms[0]

    return arms, best_arm

def main():
    df = pd.read_excel('//Users//liduo//Desktop//Assignment_1_data.xlsx')
    data = df[['X1', 'X2', 'X3']].values
    np.random.seed(42)

    seed_sets = [
        [11, 37, 64], [1, 73, 84], [35, 13, 22], [90, 86, 27], [47, 81, 19],
        [42, 76, 34], [61, 54, 87], [88, 12, 62], [80, 37, 31], [91, 75, 93]
    ]

    all_arms, best_arm = run_budget_allocation(data, seed_sets, total_budget=100)

    print("\n" + "=" * 40)
    print(f"Selected Arm â: Seed Set {best_arm.arm_id}")
    print(f"Final WCSS for â: {best_arm.wcss_history[-1]:.4f}")
    print("\nFinal Centroids:")
    for i, centroid in enumerate(best_arm.centroids):
        print(f"  Cluster {i}: {centroid}")

    print("\nFinal Cluster Assignments for all data points:")
    print(best_arm.assignments)
    print("=" * 40)

    plt.figure(figsize=(10, 6))
    for arm in all_arms:
        plt.plot(range(len(arm.wcss_history)), arm.wcss_history,
                 marker='o', label=f'Arm {arm.arm_id} (Final WCSS: {arm.wcss_history[-1]:.1f})',
                 linewidth=2 if arm == best_arm else 1,
                 linestyle='-' if arm == best_arm else '--')

    plt.title('WCSS vs. Allocated Iterations')
    plt.xlabel('Allocated Iterations')
    plt.ylabel('WCSS')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()