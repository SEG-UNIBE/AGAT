import time
from typing import Dict, List, Tuple
from src.models import *
import numpy as np

# Global cache for distances between clusters
distance_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}


### PUBLIC METHODS

def generate_grouping(pool: Pool, group_size: int, linkage_method: str, repair_method: str) -> List[List[User]]:
    """
    Creates groups of the specified size from a pool of users. Matches users based on their attributes
    and the given preferences.
    First generates an initial grouping using modified hierarchical clustering,
    then repairs the grouping to ensure all groups meet the target size.

    Args:
        pool (Pool): Pool of users.
        group_size (int): Desired group size.

        linkage_method (str): Linkage method for merging clusters.
            - "single": Single linkage.
            - "complete": Complete linkage.
            - "UPGMA": Unweighted Pair Group Method with Arithmetic Mean.
            - "WPGMA": Weighted Pair Group Method with Arithmetic Mean.
            - "total": Linkage based on using the sum of distances between all pairs of objects in the two clusters.

        repair_method (str): Method for repairing groups.
            - "merge": Merge two small groups to one that's too big, then redistribute users from the 
                merged group to smaller groups.
            - "break": Break up the worst group completely and redistribute its users to smaller groups.

    Returns:
        List[List[User]]: Final grouping with all groups.
    """
    if group_size < 2 or group_size > len(pool.users):
        raise ValueError("Invalid group size. Group size must be at least 2 and less than the number of users.")
    
    if linkage_method not in ["single", "complete", "UPGMA", "WPGMA", "total"]:
        raise ValueError("Invalid linkage method. Available methods: 'single', 'complete', 'UPGMA', 'WPGMA', 'total'")
    
    if repair_method not in ["merge", "break"]:
        raise ValueError("Invalid repair method. Available methods: 'merge', 'break'")

    # Resets Cache
    global distance_cache
    distance_cache = {}

    # Pad pool with dummy users to ensure all groups are of the target size
    remainder = len(pool.users) % group_size
    if remainder != 0:
        dummy_count = group_size - remainder
        _pad_pool_with_dummy_users(pool, dummy_count)

    # Create grouping with initial phase and repair phase
    initial_grouping = _generate_initial_grouping(pool, group_size, linkage_method)
    repaired_grouping = _repair_grouping(pool, initial_grouping, group_size, linkage_method, repair_method)

    # Remove dummy users from final grouping and remove them from the pool
    final_grouping = [[user for user in group if not isinstance(user, DummyUser)] for group in repaired_grouping]
    _remove_dummy_users(pool)

    distance_cache = {}

    return final_grouping


def generate_grouping_for_eval(pool: Pool, group_size: int, linkage_method: str, repair_method: str) -> Tuple[List[List[User]], float, float, int]:
    """
    Like 'generate_grouping' but measures and returns the time needed for the initial phase
    and repair phase additionally, as well as the number of users repaired.

    Returns:
        List[List[User]]: Final grouping with all groups.
        float: Time taken for the initial grouping.
        float: Time taken for the repair step.
        int: Number of users repaired
    """
    if group_size < 2 or group_size > len(pool.users):
        raise ValueError("Invalid group size. Group size must be at least 2 and less than the number of users.")
    
    if linkage_method not in ["single", "complete", "UPGMA", "WPGMA", "total"]:
        raise ValueError("Invalid linkage method. Available methods: 'single', 'complete', 'UPGMA', 'WPGMA', 'total'")
    
    if repair_method not in ["merge", "break"]:
        raise ValueError("Invalid repair method. Available methods: 'merge', 'break'")

    global distance_cache
    distance_cache = {}

    # Pad pool with dummy users to ensure all groups are of the target size
    remainder = len(pool.users) % group_size
    if remainder != 0:
        dummy_count = group_size - remainder
        _pad_pool_with_dummy_users(pool, dummy_count)

    time_start = time.perf_counter()
    initial_grouping = _generate_initial_grouping(pool, group_size, linkage_method)
    time_initial = time.perf_counter() - time_start

    users_repaired = sum(len(group) for group in initial_grouping if len(group) < group_size)
    repaired_grouping = _repair_grouping(pool, initial_grouping, group_size, linkage_method, repair_method)
    time_repair = time.perf_counter() - time_initial - time_start

    # Remove dummy users from final grouping and remove them from the pool
    final_grouping = [[user for user in group if not isinstance(user, DummyUser)] for group in repaired_grouping]
    _remove_dummy_users(pool)

    distance_cache = {}

    return final_grouping, time_initial, time_repair, users_repaired



def calculate_group_distance(group: List[User], pool: Pool) -> float:
    """
    Calculates the group distance for a given group.
    Sum of all pairwise distances between users.
    
    Args:
        group (List[User]): Group containing users
        pool (Pool): Pool object containing distance matrix.

    Returns:
        float: Group distance
    """
    if len(group) <= 1:
        return 0
    
    group_dist = 0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            idx1 = pool.get_index(group[i].id)
            idx2 = pool.get_index(group[j].id)
            group_dist += pool.dist_matrix[idx1][idx2]
            
    return group_dist


def calculate_avg_user_distance(group: List[User], pool: Pool) -> float:
    """
    Calculates the average distance between two users in a given group
    
    Args:
        group (List[User]): Group with users
        pool (Pool): Pool object containing distance matrix.
        
    Returns:
        float: Average distance between two users
    """
    if len(group) <= 1:
        return 0
    
    total_dist = calculate_group_distance(group, pool)
    # Divide by number of possible pairs
    return total_dist / (len(group) * (len(group) - 1) / 2)



### GROUPING METHODS

def _generate_initial_grouping(pool: Pool, group_size: int, linkage_method: str) -> List[List[User]]:
    """
    Generates an initial grouping using agglomerative hierarchical clustering. 
    The resulting groups are either of the target size or smaller.

    Args:
        pool (Pool): Pool of users
        group_size (int): Desired group size
        linkage_method (str): Linkage method used, to calculate distance between clusters

    Returns:
        List[List[User]]: Initial grouping with clusters of the target size or smaller.
    """    
    # Initialize each user as its own cluster
    clusters = [[user] for user in pool.users]
    
    # Merge clusters iteratively until no more merges are valid (group size constraint)
    while len(clusters) > (len(pool.users) // group_size):
        min_dist = float('inf')
        merge_i, merge_j = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if len(clusters[i]) + len(clusters[j]) > group_size:
                    continue
                dist = _get_cluster_distance(clusters[i], clusters[j], pool)
                
                if dist < min_dist:
                    min_dist = dist
                    merge_i = i
                    merge_j = j
        
        # No more merge possible
        if merge_i == -1:
            break
        
        # Create merged cluster and update cache
        merged_cluster = clusters[merge_i] + clusters[merge_j]
        _update_cache_after_merge(clusters, clusters[merge_i], clusters[merge_j], pool, linkage_method)
        
        # Update clusters list
        clusters[merge_i] = merged_cluster
        clusters.pop(merge_j)
    
    return clusters


def _repair_grouping(pool: Pool, initial_grouping: List[List[User]], group_size: int, linkage_method: str, 
                    repair_method: str) -> List[List[User]]:
    """
    Repairs the initial grouping to ensure all groups are of the target size.
    Only transforms the incomplete groups (= of size < group_size)

    Args:
        pool (Pool): Pool of users.
        initial_grouping (List[List[User]]): Initial grouping with clusters of the target size or smaller.
        group_size (int): Desired group size.
        linkage_method (str): Linkage method for merging clusters.
        repair_method (str): Method for repairing groups.
            - "merge": Merge two small groups to one that's too big, then redistribute users from the merged group 
                to smaller groups.
            - "break": Break up the worst group completely and redistribute its users to smaller groups.
    
    Returns:
        List[List[User]]: Repaired grouping with all groups of the target size.
    """
    if repair_method == "merge":
        return _repair_grouping_merge(pool, initial_grouping, group_size, linkage_method)
    else:
        return _repair_grouping_break(pool, initial_grouping, group_size, linkage_method)


def _repair_grouping_merge(pool: Pool, initial_grouping: List[List[User]], group_size: int, 
                           linkage_method: str) -> List[List[User]]:
    """
    Repairs the initial grouping by merging small groups to form larger groups, then redistributing 
    users from the merged groups.
    
    Args:
        pool (Pool): Pool of users.
        initial_grouping (List[List[User]]): Initial grouping with groups of target size or smaller.
        group_size (int): Desired group size.
        linkage_method (str): Linkage method for merging clusters.

    Returns:
        List[List[User]]: Repaired grouping with all groups of the target size.
    """
    complete_groups = [group for group in initial_grouping if len(group) == group_size]
    incomplete_groups = [group for group in initial_grouping if len(group) < group_size]

    unassigned_users = []

    # Merge groups until there are enough users to fill all remaining incomplete groups

    while len(incomplete_groups) > 1:
        min_dist = float('inf')
        merge_i, merge_j = -1, -1

        # Find the two groups with the smallest distance between them and merge them
        for i in range(len(incomplete_groups)):
            for j in range(i + 1, len(incomplete_groups)):
                dist = _get_cluster_distance(incomplete_groups[i], incomplete_groups[j], pool)
                if dist < min_dist:
                    min_dist = dist
                    merge_i = i
                    merge_j = j

        merged_group = incomplete_groups[merge_i] + incomplete_groups[merge_j]
        incomplete_groups.pop(max(merge_i, merge_j))
        incomplete_groups.pop(min(merge_i, merge_j))

        # Pop users from the merged group until it's of the target size
        while len(merged_group) > group_size:
            min_group_dist = float('inf')
            best_removal_idx = -1

            # Find the user to remove that minimizes the group distance of the remaining group
            for i in range(len(merged_group)):
                remaining_group = merged_group[:i] + merged_group[i + 1:]
                dist = calculate_group_distance(remaining_group, pool)
                if dist < min_group_dist:
                    min_group_dist = dist
                    best_removal_idx = i

            unassigned_users.append(merged_group.pop(best_removal_idx))
        
        complete_groups.append(merged_group)
        
        total_users_needed = sum(group_size - len(group) for group in incomplete_groups)
        if len(unassigned_users) > total_users_needed:
            raise ValueError("Not supposed to happen")
        if len(unassigned_users) == total_users_needed:
            break
    
    return complete_groups + _redistribute_users(pool, unassigned_users, incomplete_groups, group_size, linkage_method)


def _repair_grouping_break(pool: Pool, initial_grouping: List[List[User]], group_size: int,
                            linkage_method: str) -> List[List[User]]:
    """
    Repairs the initial grouping by breaking up the worst group completely and redistributing its users to smaller groups.
    
    Args:
        pool (Pool): Pool of users.
        initial_grouping (List[List[User]]): Initial grouping with clusters of the target size or smaller.
        group_size (int): Desired group size.
        linkage_method (str): Linkage method for merging clusters.
    
    Returns:
        List[List[User]]: Repaired grouping with all groups of the target size.
    """
    complete_groups = [group for group in initial_grouping if len(group) == group_size]
    incomplete_groups = [group for group in initial_grouping if len(group) < group_size]
    
    unassigned_users = []

    # Break up the worst groups until there are enough users to fill the remaining incomplete groups
    while True:
        total_needed_users = sum(group_size - len(group) for group in incomplete_groups)

        if len(unassigned_users) > total_needed_users:
            raise ValueError("Not supposed to happen")
        if len(unassigned_users) == total_needed_users:
            break

        # Find and break the group with the worst average user-pair distance
        worst_group_idx = max(
            range(len(incomplete_groups)),
            key=lambda idx: calculate_avg_user_distance(incomplete_groups[idx], pool)
        )
        worst_group = incomplete_groups.pop(worst_group_idx)
        unassigned_users.extend(worst_group)

    return complete_groups + _redistribute_users(pool, unassigned_users, incomplete_groups, group_size, linkage_method)


def _redistribute_users(pool: Pool, unassigned_users: List[User], incomplete_groups: List[List[User]],
                        group_size: int, linkage_method: str) -> List[List[User]]:
    """
    Redistributes users to groups that are too small based on the distance between the users and the groups.
    """
    complete_groups: List[List[User]] = []

    while unassigned_users and incomplete_groups:
        best_user = None
        best_group_idx = -1
        best_dist = float('inf')

        # Find the user and group pair with the smallest distance
        for user in unassigned_users:
            for i, group in enumerate(incomplete_groups):
                if len(group) < group_size:
                    dist = _get_cluster_distance([user], group, pool)
                    if dist < best_dist:
                        best_dist = dist
                        best_user = user
                        best_group_idx = i

        # Fuse the best user with the corresponding group
        if best_user is not None and best_group_idx != -1:
            clusters = incomplete_groups + [[user] for user in unassigned_users]
            _update_cache_after_merge(clusters, incomplete_groups[best_group_idx], [best_user], pool, linkage_method)
            incomplete_groups[best_group_idx].append(best_user)
            unassigned_users.remove(best_user)

        if len(incomplete_groups[best_group_idx]) == group_size:
            complete_groups.append(incomplete_groups.pop(best_group_idx))

    return complete_groups



### ClUSTER DISTANCE CACHE FUNCTIONS ###

def _get_cluster_key(cluster: List[User]) -> tuple:
    """Creates a unique, sorted tuple key for a cluster."""
    return tuple(sorted(user.id for user in cluster))


def _update_cache_after_merge(clusters: List[List[User]], cluster1: List[User], cluster2: List[User],
                              pool: Pool, linkage_method: str):
    """
    Updates the distance cache by removing unneeded entries and adding new distances for the merged cluster.
    
    Args:
        clusters: List of all clusters before the merge
        cluster1: cluster that merges with cluster2
        cluster2: cluster that merges with cluster1
        pool: Pool object containing distance matrix
        linkage_method: Method used for calculating distances
    """
    global distance_cache
    
    # Get keys for merged clusters
    key_i = _get_cluster_key(cluster1)
    key_j = _get_cluster_key(cluster2)
    new_key = _get_cluster_key(cluster1 + cluster2)
    
    # Remove unneeded entries from cache.
    # Keeps entries that don't involve the merged clusters, or involve a single-user cluster.
    # Single-user cluster entries are kept because they are needed for the repair step.
    new_cache = {}
    for (c1, c2), dist in distance_cache.items():
        if ((key_i not in (c1, c2) and key_j not in (c1, c2)) # If clusters are not involved
            or len(c1) == 1 or len(c2) == 1 # or if one of the clusters is a single-user cluster
            or (key_i, key_j) == (c1, c2) or (key_j, key_i) == (c1, c2)): # or if the entry is for the merged clusters

            new_cache[(c1, c2)] = dist
    
    remaining_clusters = [cluster for cluster in clusters if cluster != cluster1 and cluster != cluster2]
    # Calculate and cache distances between new merged cluster and all other clusters
    for cluster in remaining_clusters:
        cluster_key = _get_cluster_key(cluster)
        dist = _calculate_new_cluster_distance(cluster, cluster1, cluster2, pool, linkage_method)
        cache_key = tuple(sorted([new_key, cluster_key]))
        new_cache[cache_key] = dist

    # Calculate and cache distances between new merged cluster and all individual users. Needed for repair phase.
    for user in pool.users:
        dist = _calculate_new_cluster_distance([user], cluster1, cluster2, pool, linkage_method)
        user_key = _get_cluster_key([user])
        cache_key = tuple(sorted([new_key, user_key]))
        new_cache[cache_key] = dist
    
    distance_cache = new_cache


def _calculate_new_cluster_distance(cluster: List[User], cluster_i: List[User], cluster_j: List[User],
                                pool: Pool, linkage_method: str) -> float:
    """
    Calculates the distance between an existing cluster and a new merged cluster (cluster_i + cluster_j).
    Uses the specified linkage method and uses the cache for quicker calculations.

    Args:
        cluster: the existing cluster
        cluster_i: the first cluster that was merged
        cluster_j: the second cluster that was merged
        pool: Pool object containing distance matrix
        linkage_method: Method used for calculating distances
    
    Returns:
        float: Distance between the two clusters 'cluster' and 'cluster_i + cluster_j'
    """
    dist_1 = _get_cluster_distance(cluster, cluster_i, pool)
    dist_2 = _get_cluster_distance(cluster, cluster_j, pool)
    
    # This way avoid going through all the individual user pair distances for optimization
    if linkage_method == "single":
        return min(dist_1, dist_2)
    elif linkage_method == "complete":
        return max(dist_1, dist_2)
    elif linkage_method == "WPGMA":
        return (dist_1 + dist_2) / 2
    elif linkage_method == "UPGMA":
        return (dist_1 * len(cluster_i) + dist_2 * len(cluster_j)) / (len(cluster_i) + len(cluster_j))
    elif linkage_method == "total":
        return dist_1 + dist_2
    else:
        raise ValueError(f"Invalid linkage method: {linkage_method}")


def _get_cluster_distance(cluster1: List[User], cluster2: List[User], 
                               pool: Pool) -> float:
    """
    Gets the distance between two clusters. If the distance is not in the cache, it raises an error.
    
    Args:
        cluster1 (List[User]): List of User objects in the first cluster.
        cluster2 (List[User]): List of User objects in the second cluster.
        pool (Pool): Pool object containing distance matrix.
    
    Returns:
        float: Distance between the two clusters.
    """
    # Handle single-user clusters directly
    if len(cluster1) == 1 and len(cluster2) == 1:
        return pool.dist_matrix[pool.get_index(cluster1[0].id)][pool.get_index(cluster2[0].id)]
    
    # For larger clusters, the distance must be in the cache
    distance_cache_key = (_get_cluster_key(cluster1), _get_cluster_key(cluster2))
    
    if distance_cache_key in distance_cache:
        return distance_cache[distance_cache_key]
    
    if distance_cache_key[::-1] in distance_cache:
        return distance_cache[distance_cache_key[::-1]]
    
    raise ValueError("Cluster distance not found in cache.")


### DUMMY USER FUNCTIONS ###
 
def _pad_pool_with_dummy_users(pool: Pool, dummy_count: int) -> None:
    """
    Adds dummy users to the pool. Dummy users have an average distance to real users and an infinite distance to other dummies.
    This reduces the chance of dummys being grouped together (depends on linkage-method!)
    """
    original_pool_size = len(pool.users)
    
    # Extend existing distance matrix
    pool.dist_matrix = np.pad(pool.dist_matrix, 
                            ((0, dummy_count), (0, dummy_count)), 
                            mode='constant', 
                            constant_values=100000000000000000) # Basically infinity

    # Set dummy user distances: 0.5 to real users, 100000000000000000 to other dummies
    for i in range(original_pool_size, original_pool_size + dummy_count):
        pool.dist_matrix[i, :original_pool_size] = 0.5
        pool.dist_matrix[:original_pool_size, i] = 0.5
    
    # Create dummy users and update pool's mappings
    for i in range(dummy_count):
        dummy_id = max(pool.user_id_to_index.keys()) + 1
        dummy_user = DummyUser(dummy_id)
        pool.users.append(dummy_user)
        
        # Update the mappings
        new_index = original_pool_size + i
        pool.user_id_to_index[dummy_id] = new_index
        pool.index_to_user[new_index] = dummy_user



def _remove_dummy_users(pool: Pool) -> None:
    """
    Removes dummy users from the pool and updates the distance matrix and mappings.
    """
    pool.users = [user for user in pool.users if not isinstance(user, DummyUser)]
    pool.user_id_to_index = {user.id: i for i, user in enumerate(pool.users)}
    pool.index_to_user = {i: user for i, user in enumerate(pool.users)}
    pool.dist_matrix = pool.dist_matrix[:len(pool.users), :len(pool.users)]

