import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import norm, multivariate_normal
import random
import time
from collections import defaultdict


class Node:
    """Represents a node in the federated learning network"""

    def __init__(self, node_id, position, reliability=0.6, data_quality=0.9):
        self.node_id = node_id
        self.position = position  # 2D position for calculating distances
        self.reliability = reliability  # Probability of participation
        self.data_quality = data_quality  # Quality of data (affects model performance)
        self.online = False  # Current participation status

    def __repr__(self):
        return f"Node({self.node_id}, reliability={self.reliability:.2f}, quality={self.data_quality:.2f})"


class FederatedNetwork:
    """Simulates a decentralized federated learning network with correlated participation patterns"""

    def __init__(self, num_nodes=20, correlation_strength=0.4, grid_size=100):
        self.num_nodes = num_nodes
        self.correlation_strength = correlation_strength
        self.grid_size = grid_size
        self.nodes = []
        self.correlation_matrix = None
        self.distance_matrix = None
        self.communication_costs = None
        self.current_path = None

        self._initialize_nodes()
        self._create_correlation_matrix()
        self._calculate_distance_matrix()
        self._calculate_communication_costs()

    def _initialize_nodes(self):
        """Create nodes with random positions and attributes"""
        for i in range(self.num_nodes):
            # Random position in 2D space
            pos = (random.uniform(0, self.grid_size), random.uniform(0, self.grid_size))

            # Random reliability between 0.6 and 0.95
            reliability = random.uniform(0.6, 0.95)

            # Random data quality between 0.7 and 0.99
            data_quality = random.uniform(0.7, 0.99)

            self.nodes.append(Node(i, pos, reliability, data_quality))

    def _create_correlation_matrix(self):
        """
        Create a correlation matrix for node participation
        Nodes that are closer to each other have more correlated participation patterns
        """
        n = self.num_nodes
        base_matrix = np.eye(n)

        # Calculate pairwise distances between nodes
        positions = np.array([node.position for node in self.nodes])
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Euclidean distance
                dist = np.linalg.norm(positions[i] - positions[j])
                # Normalize by max possible distance
                max_dist = np.sqrt(2) * self.grid_size
                norm_dist = dist / max_dist

                # Convert distance to correlation (closer = more correlated)
                # Correlation decays with distance
                correlation = self.correlation_strength * np.exp(-3 * norm_dist)

                # Make the matrix symmetric
                base_matrix[i, j] = correlation
                base_matrix[j, i] = correlation

        # Ensure the matrix is positive definite (required for valid correlation matrix)
        min_eig = np.min(np.linalg.eigvals(base_matrix))
        if min_eig < 0:
            base_matrix += (-min_eig + 0.01) * np.eye(n)

        # Normalize to ensure diagonal is 1
        for i in range(n):
            for j in range(n):
                if i != j:
                    base_matrix[i, j] = base_matrix[i, j] / np.sqrt(
                        base_matrix[i, i] * base_matrix[j, j]
                    )

        np.fill_diagonal(base_matrix, 1.0)
        self.correlation_matrix = base_matrix

    def _calculate_distance_matrix(self):
        """Calculate pairwise distances between nodes"""
        n = self.num_nodes
        self.distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Euclidean distance
                    self.distance_matrix[i, j] = np.linalg.norm(
                        np.array(self.nodes[i].position)
                        - np.array(self.nodes[j].position)
                    )

    def _calculate_communication_costs(self):
        """
        Calculate communication costs between nodes
        Cost depends on distance and data quality
        """
        n = self.num_nodes
        self.communication_costs = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Base cost is distance
                    base_cost = self.distance_matrix[i, j]

                    # Adjust cost based on data quality
                    # Lower quality data is less valuable, so cost per bit of useful info is higher
                    quality_factor = 1.0 / (
                        0.5 * (self.nodes[i].data_quality + self.nodes[j].data_quality)
                    )

                    self.communication_costs[i, j] = base_cost * quality_factor

    def simulate_node_participation(self):
        """
        Simulate which nodes are online in this round
        Uses the correlation matrix to generate correlated participation
        """
        # Generate correlated random variables using multivariate normal distribution
        mean = np.zeros(self.num_nodes)
        samples = np.random.multivariate_normal(mean, self.correlation_matrix)

        # Convert to binary participation based on node reliability
        thresholds = norm.ppf(1 - np.array([node.reliability for node in self.nodes]))
        participation = (samples > thresholds).astype(int)

        # Update node status
        for i, node in enumerate(self.nodes):
            node.online = bool(participation[i])

        return participation

    def calculate_path_cost(self, path, online_status):
        """
        Calculate the cost of a path considering only online nodes
        path: list of node indices
        online_status: binary array indicating if nodes are online
        """
        online_path = [node_idx for node_idx in path if online_status[node_idx]]

        if len(online_path) <= 1:
            return 0  # No communication needed

        cost = 0
        for i in range(len(online_path) - 1):
            src, dest = online_path[i], online_path[i + 1]
            cost += self.communication_costs[src, dest]

        # Add cost to return to first node (if any)
        if len(online_path) > 1:
            cost += self.communication_costs[online_path[-1], online_path[0]]

        return cost

    def expected_path_cost(self, path, num_samples=1000):
        """
        Calculate the expected cost of a path considering stochastic participation
        Uses Monte Carlo simulation
        """
        total_cost = 0

        for _ in range(num_samples):
            online_status = self.simulate_node_participation()
            total_cost += self.calculate_path_cost(path, online_status)

        return total_cost / num_samples

    def nearest_neighbor_path(self):
        """Generate a path using the nearest neighbor heuristic"""
        n = self.num_nodes
        path = [0]  # Start with node 0
        unvisited = set(range(1, n))

        while unvisited:
            current = path[-1]
            next_node = min(
                unvisited, key=lambda i: self.communication_costs[current, i]
            )
            path.append(next_node)
            unvisited.remove(next_node)

        return path

    def two_opt_swap(self, path, i, k):
        """Perform a 2-opt swap on the path"""
        new_path = path.copy()
        new_path[i : k + 1] = reversed(new_path[i : k + 1])
        return new_path

    def two_opt_path_optimization(self, initial_path, max_iterations=100):
        """Optimize a path using 2-opt local search"""
        best_path = initial_path.copy()
        best_cost = self.expected_path_cost(best_path)
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(1, len(best_path) - 1):
                for k in range(i + 1, len(best_path)):
                    new_path = self.two_opt_swap(best_path, i, k)
                    new_cost = self.expected_path_cost(new_path)

                    if new_cost < best_cost:
                        best_path = new_path
                        best_cost = new_cost
                        improved = True
                        print(
                            f"Iteration {iteration}: Found better path with cost {best_cost:.2f}"
                        )
                        break

                if improved:
                    break

        return best_path, best_cost

    def simulate_correlated_ant_colony(
        self, num_ants=10, num_iterations=20, alpha=1.0, beta=2.0, evaporation=0.5
    ):
        """
        Implement a version of Ant Colony Optimization that considers
        correlation between nodes (similar to the CPTSP approach)
        """
        n = self.num_nodes

        # Initialize pheromone matrix
        pheromone = np.ones((n, n))

        # Probability of being online
        reliability = np.array([node.reliability for node in self.nodes])

        best_path = None
        best_cost = float("inf")

        for iteration in range(num_iterations):
            paths = []
            costs = []

            # Each ant constructs a tour
            for _ in range(num_ants):
                # Start at a random node
                current = random.randrange(n)
                path = [current]
                unvisited = set(range(n))
                unvisited.remove(current)

                while unvisited:
                    # Calculate transition probabilities
                    probabilities = []

                    for next_node in unvisited:
                        # Calculate correlation-adjusted transition probability
                        # Higher pheromone and lower cost make a node more attractive
                        # Also consider correlation with already visited nodes and reliability

                        # Base attractiveness from pheromone and distance
                        tau = pheromone[current, next_node] ** alpha
                        eta = (
                            1.0 / self.communication_costs[current, next_node]
                        ) ** beta

                        # Correlation factor - favor nodes that are likely to be online together
                        # with nodes already in the path
                        correlation_factor = 1.0
                        for visited in path:
                            if visited != current:
                                # Higher correlation means these nodes tend to be online together
                                correlation = self.correlation_matrix[
                                    visited, next_node
                                ]
                                correlation_factor *= 1.0 + correlation

                        # Reliability factor - prefer more reliable nodes earlier in the path
                        reliability_factor = reliability[next_node] ** (
                            1.0 - len(path) / n
                        )

                        # Combined probability
                        prob = tau * eta * correlation_factor * reliability_factor
                        probabilities.append((next_node, prob))

                    # Choose next node
                    total = sum(p[1] for p in probabilities)
                    if total == 0:  # Handle division by zero
                        next_node = random.choice(list(unvisited))
                    else:
                        r = random.uniform(0, total)
                        cum_prob = 0
                        next_node = None

                        for node, prob in probabilities:
                            cum_prob += prob
                            if cum_prob >= r:
                                next_node = node
                                break

                        if next_node is None:  # Just in case
                            next_node = probabilities[-1][0]

                    path.append(next_node)
                    unvisited.remove(next_node)
                    current = next_node

                # Calculate the expected cost of this path
                cost = self.expected_path_cost(
                    path, num_samples=100
                )  # Fewer samples for speed
                paths.append(path)
                costs.append(cost)

                # Update best solution
                if cost < best_cost:
                    best_path = path.copy()
                    best_cost = cost
                    print(
                        f"Iteration {iteration+1}: Found better path with cost {best_cost:.2f}"
                    )

            # Update pheromone trails
            pheromone *= evaporation  # Evaporation

            # Add new pheromone based on path costs
            for path, cost in zip(paths, costs):
                if cost > 0:  # Avoid division by zero
                    deposit = 1.0 / cost
                    for i in range(len(path) - 1):
                        pheromone[path[i], path[i + 1]] += deposit
                        pheromone[path[i + 1], path[i]] += deposit  # Symmetric

                    # Close the loop
                    pheromone[path[-1], path[0]] += deposit
                    pheromone[path[0], path[-1]] += deposit

        return best_path, best_cost

    def visualize_network(self, path=None, highlight_online=False):
        """
        Visualize the network, optionally showing a path and highlighting online nodes
        """
        plt.figure(figsize=(10, 8))
        G = nx.Graph()

        # Add nodes
        online_status = self.simulate_node_participation() if highlight_online else None

        for i, node in enumerate(self.nodes):
            online = online_status[i] if highlight_online else True
            color = "green" if online else "red"
            G.add_node(i, pos=node.position, color=color, reliability=node.reliability)

        pos = nx.get_node_attributes(G, "pos")

        # Draw nodes
        node_colors = (
            [G.nodes[i]["color"] for i in G.nodes] if highlight_online else "skyblue"
        )
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)

        # Draw node labels
        labels = {i: f"{i}\n({self.nodes[i].reliability:.2f})" for i in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

        # Draw edges (path)
        if path:
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            edges.append((path[-1], path[0]))  # Close the loop

            # Create a continuous path that skips offline nodes
            if highlight_online:
                continuous_edges = []
                online_nodes_in_path = [node for node in path if online_status[node]]

                if len(online_nodes_in_path) > 1:
                    # Create edges between consecutive online nodes in the path
                    for i in range(len(online_nodes_in_path) - 1):
                        continuous_edges.append(
                            (online_nodes_in_path[i], online_nodes_in_path[i + 1])
                        )

                    # Close the loop if there are at least 2 online nodes
                    if len(online_nodes_in_path) > 1:
                        continuous_edges.append(
                            (online_nodes_in_path[-1], online_nodes_in_path[0])
                        )

                edges = continuous_edges

                # Draw dashed lines for the original path segments
                original_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                original_edges.append((path[-1], path[0]))  # Close the loop

                # Filter to only show edges where at least one node is online
                visible_original_edges = [
                    (u, v)
                    for (u, v) in original_edges
                    if online_status[u] or online_status[v]
                ]

                # Draw original path as dashed lines
                if visible_original_edges:
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=visible_original_edges,
                        width=1,
                        alpha=0.3,
                        edge_color="gray",
                        style="dashed",
                    )

            G.add_edges_from(edges)
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, width=2, alpha=0.7, edge_color="blue"
            )

        plt.title("Decentralized Federated Learning Network")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # Return the visualization data for potential further use
        return G, pos


# Example usage
def run_simulation():
    # Create a network with moderate correlation
    print("Initializing network...")
    network = FederatedNetwork(num_nodes=20, correlation_strength=0.4)

    # Visualize the initial network
    print("Visualizing initial network...")
    network.visualize_network()

    # Generate initial path using nearest neighbor
    print("Generating initial path...")
    initial_path = network.nearest_neighbor_path()
    initial_cost = network.expected_path_cost(initial_path)
    print(f"Initial path cost: {initial_cost:.2f}")

    # Visualize initial path
    print("Visualizing initial path...")
    network.visualize_network(path=initial_path)

    # Optimize using ACO
    print("Optimizing path using Correlated Ant Colony Optimization...")
    start_time = time.time()
    best_path, best_cost = network.simulate_correlated_ant_colony(
        num_ants=20, num_iterations=30, alpha=1.0, beta=2.5
    )
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(
        f"Best path cost: {best_cost:.2f} (improved by {(initial_cost - best_cost) / initial_cost * 100:.2f}%)"
    )

    # Visualize optimized path
    print("Visualizing optimized path...")
    network.visualize_network(path=best_path)

    # Show how the optimized path performs with online/offline nodes
    print("Visualizing path with online/offline nodes...")
    network.visualize_network(path=best_path, highlight_online=True)

    return network, best_path, best_cost


if __name__ == "__main__":
    network, best_path, best_cost = run_simulation()

    # Compare with pure random participation (no correlation)
    original_correlation = network.correlation_matrix.copy()
    network.correlation_matrix = np.eye(network.num_nodes)  # No correlation
    uncorrelated_cost = network.expected_path_cost(best_path)
    network.correlation_matrix = original_correlation  # Restore

    print(f"\nExpected cost with correlation: {best_cost:.2f}")
    print(f"Expected cost without correlation: {uncorrelated_cost:.2f}")
    print(
        f"Difference: {abs(best_cost - uncorrelated_cost) / uncorrelated_cost * 100:.2f}%"
    )

    # Print path details
    print("\nOptimized path:")
    for idx, node_id in enumerate(best_path):
        node = network.nodes[node_id]
        print(
            f"{idx+1}. Node {node_id} (reliability: {node.reliability:.2f}, quality: {node.data_quality:.2f})"
        )
