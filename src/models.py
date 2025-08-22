from typing import List
import numpy as np


class User():
    """
    Represents a user with an ID, name and attributes.
    
    Attributes:
        id (int): The user's ID.
        name (str): The user's name.
        attributes (dict): A dictionary containing the user's attributes.
            Should be in the form 'preference_name: users_value'
    """
    def __init__(self, id: int, name: str, attributes: dict):
        self.id = id
        self.name = name
        self.attributes = attributes

    def __repr__(self):
        return str(self.id)


class DummyUser(User):
    """
    A dummy user to represent a fake user with no attributes. Only contains an ID
    
    Attributes:
        id (int): The Dummy user's ID.
    """
    def __init__(self, id: int):
        super().__init__(id, "Dummy User", {})


class Preference():
    """
    Defines how a specific criteria should be matched and weighted and what values it can take.

    Attributes:
        name (str): The name of the preference.
        matching_type (str): The type of matching to use. Can be "homogeneous" or "heterogeneous".
        weight (float): The weight of the preference. Should be a value between 0 and 1.
        value_type (str): The type of value the preference can take. Can be "numerical" or "categorical".
        min_value (int, optional): The minimum value the preference can take when value_type is "numerical"
        max_value (int, optional): The maximum value the preference can take when value_type is "numerical"
        categories (List[str], optional): The categories the preference can take when value_type is "categorical".
    """
    def __init__(self, name: str, matching_type: str, weight: float, value_type: str, 
                 min_value: int = None, max_value: int = None, categories: List[str] = []):
        self.name = name
        self.matching_type = matching_type
        self.weight = weight
        self.value_type = value_type
        self.min_value = min_value
        self.max_value = max_value
        self.categories = categories


class Pool():
    """
    Represents a pool of users and the preferences used for grouping.
        
    Attributes:
        users (List[User]): A list of User objects.
        preferences (List[Preference]): A list of Preference objects.
        user_id_to_index (dict): A dictionary mapping user IDs to their index in the users list.
        index_to_user (dict): A dictionary mapping user indices to their User object.
        dist_matrix (np.ndarray): A matrix containing the calculated distances between each pair of users. 
            The i-th row and j-th column contains the distance between the i-th and j-th users.
    """
    def __init__(self, users: List[User], preferences: List[Preference]):
        self.users = users
        self.preferences = preferences
        self.user_id_to_index = {user.id: i for i, user in enumerate(users)}
        self.index_to_user = {i: user for i, user in enumerate(users)}
        self.dist_matrix = self.generate_distance_matrix()


    def get_index(self, user_id: int) -> int:
        return self.user_id_to_index.get(user_id)
    
    def get_user(self, index: int) -> User:
        return self.index_to_user.get(index)

    def generate_distance_matrix(self) -> np.ndarray:
        n = len(self.users)
        dist_matrix = np.zeros((n, n))
            
        # Go through each pair of users and calculate their distance
        for i, user1 in enumerate(self.users):
            for j, user2 in enumerate(self.users):
                if i == j:
                    continue
                    
                sum_distances = 0
                sum_weights = 0
                
                # Go through each preference and calculate distance
                for preference in self.preferences:
                    user1_value = user1.attributes.get(preference.name)
                    user2_value = user2.attributes.get(preference.name)
                        
                    # Calculate distance based on preference type and matching type
                    if preference.value_type == "numerical":
                        val1 = float(user1_value)
                        val2 = float(user2_value)
                        range_size = preference.max_value - preference.min_value
                        
                        if preference.matching_type == "homogeneous":
                            pref_distance = abs((val1 - val2) / range_size)  # More similiar => closer to 0
                        else: # HETEROGENEOUS
                            pref_distance = 1 - abs((val1 - val2) / range_size)  # More similar => closer to 1
                        
                    else:  # CATEGORICAL
                        if preference.matching_type == "homogeneous":
                            pref_distance = 0 if user1_value == user2_value else 1
                        else:  # HETEROGENEOUS
                            pref_distance = 1 if user1_value == user2_value else 0
                        
                    # Add weighted score to total
                    sum_distances += pref_distance * preference.weight
                    sum_weights += preference.weight

                # Normalize final score
                if sum_weights > 0:
                    dist_matrix[i][j] = sum_distances / sum_weights

        return dist_matrix
    
