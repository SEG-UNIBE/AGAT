import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import time
from typing import Tuple
import pandas as pd
from src.grouping import *
from src.models import *
import random
import numpy as np
from itertools import combinations
from typing import List, Tuple


def generate_csv():
    """
    Generate a CSV file result.csv with the results

    Contain per row:
    - pool_size
    - group_size
    - type (one of the 5 linkage types or ground_truth, random_best, random_median)
    - repair_method (merge or break)
    - rank (when considering a specific ranking of groupings, where does this one rank?)
    - ranking_size (how many groupings was this one compared to?)
    - total_users_repaired
    - total_distance
    - best_group_distance
    - median_group_distance
    - worst_group_distance
    - time_matrix
    - time_initial
    - time_repair
    - time_total
    - seed
    - group_distances (list of all group distances in the grouping)
    - generated_grouping
    """
    df = pd.DataFrame(columns=["pool_size", "group_size", "type", "repair_methode", 
                               "rank", "ranking_size", "total_users_repaired", "total_distance", 
                               "best_group_distance", "median_group_distance", "worst_group_distance",
                                "time_matrix", "time_initial", "time_repair", "time_total", "seed",
                                "group_distances", "generated_grouping"])
    
    pool_sizes = [4, 6, 8, 10, 12, 16, 20, 24, 32, 50, 64, 100, 200, 500]
    seeds = [i for i in range(1, 11)]
    linkage_methods = ["single", "complete", "UPGMA", "WPGMA", "total"]
    repair_methods = ["merge", "break"]

    # Iterate through all seeds
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        print(f"Processing seed {seed}")

        # Iterate through all pool sizes
        for pool_size in pool_sizes:
            print(f"Processing pool size {pool_size} for seed {seed}")

            pool = create_sample_pool(pool_size)
            group_sizes = [i for i in range(2, pool_size) if pool_size % i == 0]
            _, time_matrix = generate_matrix(pool) # Measure time for distanz here already

            # Iterate through all group sizes with n % k = 0
            for group_size in group_sizes:
                print(f"Processing group size {group_size}")

                # Generate a ranking for ground truth or random-best
                time_ranking_start = time.perf_counter()
                if pool_size > 16:
                    ranking_size = 100000
                    ranking = generate_random_ranking(pool, group_size, ranking_size)
                else:
                    ranking = generate_complete_ranking(pool, group_size)
                    ranking_size = len(ranking)
                time_ranking = time.perf_counter() - time_ranking_start

                # Add Ground Truth if n <= 16, else Random-Best
                add_best_grouping_to_df(df, pool, group_size, ranking, time_matrix, time_ranking, seed)

                # Random-Median should be based on a random ranking, even if ground truth was used
                if pool_size <= 16:
                    ranking_for_median = generate_random_ranking(pool, group_size, 100000)
                else:
                    ranking_for_median = ranking

                # Add Random Median grouping
                add_random_median_grouping_to_df(df, pool, group_size, ranking_for_median, seed)

                # Calculate groupings with all 10 AGAT-Variants
                for linkage_method in linkage_methods:
                    for repair_method in repair_methods:
                        add_grouping_to_df(df, pool, group_size, ranking, linkage_method, repair_method, 
                                           time_matrix, seed)

        # Save DF after every seed-iteration                
        save_df_as_csv(df)
                        


def add_grouping_to_df(df: pd.DataFrame, pool: Pool, group_size: int, ranking: List[Tuple[List[List[User]], float]],
                          linkage_method: str, repair_method: str, time_matrix, seed: int):
    
    grouping, time_initial, time_repair, users_repaired = generate_grouping_for_eval(pool, group_size, linkage_method, repair_method)
    time_total = time_matrix + time_initial + time_repair
    total_distance, best_group_distance, median_group_distance, worst_group_distance = get_distances(pool, grouping)
    
    df.loc[len(df)] = {
        "pool_size": len(pool.users),
        "group_size": group_size,
        "type": linkage_method,
        "repair_method": repair_method,
        "generated_grouping": grouping,
        "group_distances": sorted([float(calculate_group_distance(group, pool)) for group in grouping]),
        "total_distance": total_distance,
        "best_group_distance": best_group_distance,
        "median_group_distance": median_group_distance,
        "worst_group_distance": worst_group_distance,
        "rank": get_rank(ranking, grouping, pool),
        "ranking_size": len(ranking),
        "total_users_repaired": users_repaired,
        "time_matrix": time_matrix,
        "time_initial": time_initial,
        "time_repair": time_repair,
        "time_total": time_total,
        "seed": seed
    }

                        
def add_best_grouping_to_df(df: pd.DataFrame, pool: Pool, group_size: int, ranking: List[Tuple[List[List[User]], float]], 
                      time_matrix, time_initial, seed: int):
    best_grouping = ranking[0][0]
    total_distance, best_group_distance, median_group_distance, worst_group_distance = get_distances(pool, best_grouping)
    time_total = time_matrix + time_initial
    
    df.loc[len(df)] = {
        "pool_size": len(pool.users),
        "group_size": group_size,
        "type": "ground_truth" if len(pool.users) <= 16 else "random_best",
        "repair_method": None,
        "generated_grouping": best_grouping,
        "group_distances": sorted([float(calculate_group_distance(group, pool)) for group in best_grouping]),
        "total_distance": total_distance,
        "best_group_distance": best_group_distance,
        "median_group_distance": median_group_distance,
        "worst_group_distance": worst_group_distance,
        "rank": 1,
        "ranking_size": len(ranking),
        "total_users_repaired": 0,
        "time_matrix": time_matrix,
        "time_initial": time_initial,
        "time_repair": 0,
        "time_total": time_total,
        "seed": seed
    }


def add_random_median_grouping_to_df(df: pd.DataFrame, pool: Pool, group_size: int, ranking: List[Tuple[List[List[User]], float]],
                                seed: int):
        
        # Add median of the ranking to the dataframe
        median_rank = len(ranking) // 2
        median_grouping = ranking[median_rank][0]
        total_distance, best_group_distance, median_group_distance, worst_group_distance = get_distances(pool, median_grouping)
        
        time_start = time.perf_counter()
        generate_random_grouping(pool, group_size)
        time_initial = time.perf_counter() - time_start

        df.loc[len(df)] = {
            "pool_size": len(pool.users),
            "group_size": group_size,
            "type": "random_median",
            "repair_method": None,
            "generated_grouping": median_grouping,
            "group_distances": sorted([float(calculate_group_distance(group, pool)) for group in median_grouping]),
            "total_distance": total_distance,
            "best_group_distance": best_group_distance,
            "median_group_distance": median_group_distance,
            "worst_group_distance": worst_group_distance,
            "rank": median_rank + 1,
            "ranking_size": len(ranking),
            "total_users_repaired": 0,
            "time_matrix": 0,
            "time_initial": time_initial,
            "time_repair": 0,
            "time_total": time_initial,
            "seed": seed
        }
    



def save_df_as_csv(df: pd.DataFrame):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "results.csv")
    df.to_csv(file_path, index=True)


def get_distances(pool: Pool, grouping: List[List[User]]) -> tuple:
    total_distance = 0
    best_group_distance = float("inf")
    worst_group_distance = 0
    all_group_distances = []

    for group in grouping:
        group_distance = calculate_group_distance(group, pool)
        all_group_distances.append(group_distance)

        best_group_distance = min(best_group_distance, group_distance)
        worst_group_distance = max(worst_group_distance, group_distance)

        total_distance += group_distance
    
    median_group_distance = np.median(all_group_distances)

    return total_distance, best_group_distance, median_group_distance, worst_group_distance


def generate_matrix(pool: Pool) -> tuple:
    """Generate distance matrix with time measurement"""
    start_time = time.perf_counter()
    matrix = pool.generate_distance_matrix()
    end_time = time.perf_counter()
    duration = end_time - start_time
    return (matrix, duration)


def create_sample_pool(pool_size: int) -> Pool:
    categories = ["A", "B", "C"]
    users = []
    preferences = []

    for i in range(1, pool_size + 1):
        user_id = i
        name = f"User {i}"
        attributes = {
            "criteria1": random.randint(1, 10),
            "criteria2": random.randint(1, 10),
            "criteria3": random.choice(categories),
            "criteria4": random.choice(categories)
        }
        users.append(User(id=user_id, name=name, attributes=attributes))

    weights = np.random.dirichlet(np.ones(4)).tolist()
    preferences.append(Preference(
        name="criteria1",
        matching_type="homogeneous",
        weight=weights[0],
        value_type="numerical",
        min_value=1,
        max_value=10
    ))
    preferences.append(Preference(
        name="criteria2",
        matching_type="heterogeneous",
        weight=weights[1],
        value_type="numerical",
        min_value=1,
        max_value=10
    ))
    preferences.append(Preference(
        name="criteria3",
        matching_type="homogeneous",
        weight=weights[2],
        value_type="categorical",
        categories=categories
    ))
    preferences.append(Preference(
        name="criteria4",
        matching_type="heterogeneous",
        weight=weights[3],
        value_type="categorical",
        categories=categories
    ))

    return Pool(users=users, preferences=preferences)






#generate a ranklist where x random groupings get created and sorted by total group distance. saves both the groupings and their distances
def generate_random_ranking(pool: Pool, group_size: int, entries: int) -> List[Tuple[List[List[User]], float]]:
    ranking = []
    for i in range(entries):
        grouping = generate_random_grouping(pool, group_size)
        total_distance = get_distances(pool, grouping)[0]
        ranking.append((grouping, total_distance))
    ranking.sort(key=lambda x: x[1])
    return ranking


def generate_random_grouping(pool: Pool, group_size: int) -> List[List[User]]:
    users = pool.users.copy()
    random.shuffle(users)
    return [users[i:i+group_size] for i in range(0, len(users), group_size)]


def get_rank(ranking: List[Tuple[List[List[User]], float]], grouping: List[List[User]], pool: Pool) -> int:
    """
    Add the grouping to the ranklist and return its rank.
    """
    entry = (grouping, get_distances(pool, grouping)[0])
    ranking.append(entry)
    ranking.sort(key=lambda x: x[1])
    rank = ranking.index(entry) + 1
    ranking.remove(entry)
    return rank




def generate_complete_ranking(pool: Pool, group_size: int) -> List[Tuple[List[List[User]], float]]:
    """
    Erzeugt alle möglichen eindeutigen Gruppierungen von pool.users in Gruppen der Grösse group_size.
    Berechnet deren Gesamtdistanz und gibt sie als sortierte Liste zurück (aufsteigend nach Distanz).
    Es werden keine Duplikate erzeugt (Weder Reihenfolge innerhalb einer Gruppe noch zwischen Gruppen spielt eine Rolle).

    Raises:
        ValueError: Falls die Anzahl Benutzer nicht durch group_size teilbar ist.
    """
    users = sorted(pool.users, key=lambda u: u.id)
    n_users = len(users)

    if n_users % group_size != 0:
        raise ValueError("Die Anzahl Benutzer muss durch group_size teilbar sein.")

    def partition_users(remaining: List[User]) -> List[List[List[User]]]:
        # Falls keine Benutzer mehr übrig sind, ist eine vollständige Aufteilung erreicht
        if not remaining:
            return [[]]

        first = remaining[0]
        rest = remaining[1:]
        result = []

        # Wähle group_size - 1 weitere Benutzer aus, um eine Gruppe zu bilden
        for combo in combinations(rest, group_size - 1):
            group = [first] + list(combo)
            group_set = set(group)
            # Entferne gewählte Gruppe aus den verbleibenden Benutzern
            new_remaining = [u for u in rest if u not in group_set]

            # Rekursiver Aufruf für die restlichen Benutzer
            for sub_part in partition_users(new_remaining):
                result.append([group] + sub_part)

        return result

    # Alle eindeutigen Gruppierungen generieren
    all_groupings = partition_users(users)

    # Gesamtdistanzen berechnen und sortieren
    ranklist = []
    for grouping in all_groupings:
        dist = get_distances(pool, grouping)[0]
        ranklist.append((grouping, dist))

    ranklist.sort(key=lambda x: x[1])

    
    return ranklist


generate_csv()