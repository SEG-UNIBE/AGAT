import csv
import json
from typing import Tuple
from src.models import *
from src.grouping import calculate_group_distance


def get_algorithm_params(config_path: str) -> Tuple[int, str, str]:
    """
    Get the algorithm parameters from the config file.

    Args:
        config_path (str): The path to the config

    Returns:
        Tuple[int, str, str]: The group size, linkage method, and repair method.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    return config["group_size"], config["linkage_method"], config["repair_method"]


def create_pool_from_files(config_path: str, users_path: str) -> Pool:
    """
    Create a Pool object from the config and users files.
    
    Args:
        config_path (str): The path to config JSON file
        users_path (str): The path to the CSV file containing users
        
    Returns:
        Pool: A Pool object containing the users and specified preferences.
    """
    # Load preferences from the config file
    with open(config_path, "r") as f:
        config = json.load(f)

    preferences = []
    for preference in config["preferences"]:
        if preference["value_type"] == "numerical":
            preferences.append(Preference(
                name=preference["name"],
                matching_type=preference["matching_type"],
                weight=preference["weight"],
                value_type=preference["value_type"],
                min_value=preference["min_value"],
                max_value=preference["max_value"]
            ))
        elif preference["value_type"] == "categorical":
            preferences.append(Preference(
                name=preference["name"],
                matching_type=preference["matching_type"],
                weight=preference["weight"],
                value_type=preference["value_type"],
                categories=preference["categories"]
            ))

    # Load users from the users file
    users = []
    with open(users_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)

        for row in reader:
            user_id = int(row[0])
            user_name = row[1].strip()
            attributes = {}
            for i in range(2, len(row)):
                criteria = headers[i].strip()
                attribute_value = row[i].strip()

                # Automatically cast numerical attributes to float
                if criteria in [pref.name for pref in preferences if pref.value_type == "numerical"]:
                    attribute_value = float(attribute_value)

                attributes[criteria] = attribute_value

            users.append(User(id=user_id, name=user_name, attributes=attributes))

    return Pool(users=users, preferences=preferences)


def create_output_file(grouping: List[List[User]], pool: Pool, output_path: str):
    """
    Create an output file with the generated groups and their group distances and the total distance.
    
    Args:
        grouping (List[List[User]]): A list of groups, where each group is a list of User objects.
        pool (Pool): The Pool object containing the distance matrix.
        output_path (str): The path to the output JSON file.
    """
    output_dict = {}
    total_distance = 0
    for i, group in enumerate(grouping):
        group_distance = calculate_group_distance(group, pool)
        total_distance += group_distance
        output_dict[f"group{i+1}"] = {
            "members": [{"user_id": user.id, "name": user.name} for user in group],
            "group_distance": group_distance
        }
    output_dict["total_distance"] = total_distance

    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=4)