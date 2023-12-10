# Swapnil_PatilMapUp-_Data-_Analyst
MapUp Data Analyst: Python and Excel technical assessment
## Question 1: Car Matrix Generation
import pandas as pd

def generate_car_matrix(dataframe):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataframe)

    # Pivot the DataFrame to create the desired matrix
    matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    for i in range(min(matrix.shape)):
        matrix.iloc[i, i] = 0

    return matrix
result_matrix = generate_car_matrix('dataset-1.csv')
print(result_matrix)


## Question 2: Car Type Count Calculation
import pandas as pd
import numpy as np

def get_type_count(dataframe):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataframe)

    # Create a new column 'car_type' based on 'car' column values
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(
        np.select(conditions, choices, default='Undefined'), index=df.index
    )

    # Calculate count of occurrences for each 'car_type'
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_count_sorted = dict(sorted(type_count.items()))

    return type_count_sorted

result = get_type_count('dataset-1.csv')
print(result)


## Question 3: Bus Count Index Retrieval
import pandas as pd

def get_bus_indexes(dataframe):
    # Calculate the mean of the 'bus' column
    bus_mean = dataframe['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = dataframe[dataframe['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes


csv_file_path = 'dataset-1.csv'
df = pd.read_csv(csv_file_path)
result_bus_indexes = get_bus_indexes(df)

print("Indices where 'bus' values are greater than twice the mean:")
print(result_bus_indexes)

## Question 4: Route Filtering
import pandas as pd

def filter_routes(dataframe):
    # Calculate the average value of the 'truck' column
    truck_mean = dataframe['truck'].mean()

    # Filter the 'route' column based on the average 'truck' value
    filtered_routes = dataframe[dataframe['truck'] > 7]['route'].tolist()

    return filtered_routes


csv_file_path = 'dataset-1.csv'
df = pd.read_csv(csv_file_path)
result_filtered_routes = filter_routes(df)

print("Routes where the average 'truck' value is greater than 7:")
print(result_filtered_routes)

## Question 5: Matrix Value Modification
import pandas as pd

def multiply_matrix(result_matrix):
    # Create a copy of the DataFrame to avoid modifying the original
    modified_matrix = result_matrix.copy()

    # Multiply values greater than 20 by 0.75 and values 20 or less by 1.25
    modified_matrix[modified_matrix > 20] *= 0.75
    modified_matrix[modified_matrix <= 20] *= 1.25

    # Round the values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix



# Call the function
modified_result_matrix = multiply_matrix(result_matrix)

# Display the modified DataFrame
print("Modified Matrix:")
print(modified_result_matrix)



## Question 6: Time Check
import pandas as pd

def check_timestamp_completeness(dataframe):
    try:
        # Convert 'startDay' and 'endDay' to datetime objects
        dataframe['start_timestamp'] = pd.to_datetime(dataframe['startDay'] + ' ' + dataframe['startTime'])
        dataframe['end_timestamp'] = pd.to_datetime(dataframe['endDay'] + ' ' + dataframe['endTime'])

    except pd.errors.OutOfBoundsDatetime as e:
        # Print the error message and affected rows
        print(f"Error: {e}")
        problematic_rows = dataframe[pd.to_datetime(dataframe['startDay'], errors='coerce').isna()]
        print("Problematic Rows:")
        print(problematic_rows)

        return None

    # Extract day of the week and hour of the day
    dataframe['day_of_week'] = dataframe['start_timestamp'].dt.day_name()
    dataframe['hour_of_day'] = dataframe['start_timestamp'].dt.hour

    # Check completeness for each unique ('id', 'id_2') pair
    completeness_check = dataframe.groupby(['id', 'id_2']).apply(
        lambda group: (
            (group['day_of_week'].nunique() == 7) and
            (group['hour_of_day'].nunique() == 24) and
            (group['start_timestamp'].min().hour == 0) and
            (group['end_timestamp'].max().hour == 23)
        )
    )

    return completeness_check

csv_file_path = 'dataset-2.csv'
df = pd.read_csv(csv_file_path)

# Call the function
completeness_result = check_timestamp_completeness(df)

# Display the result
print("Timestamp Completeness Check:")
print(completeness_result)




#Python_Task_2
## Question 1: Distance Matrix Calculation

import pandas as pd
import networkx as nx

def calculate_distance_matrix(dataframe):
    print("Columns in the DataFrame:")
    print(dataframe.columns)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges and distances from the DataFrame
    for index, row in dataframe.iterrows():
        G.add_edge(row['id_start'], row['id_end'], distance=row['distance'])
    
    # Create a DataFrame to store distances
    nodes = sorted(G.nodes())
    distance_matrix = pd.DataFrame(index=nodes, columns=nodes, dtype=float)

    # Calculate cumulative distances
    for node_from in nodes:
        for node_to in nodes:
            if node_from == node_to:
                distance_matrix.at[node_from, node_to] = 0
            else:
                try:
                    distance_matrix.at[node_from, node_to] = nx.shortest_path_length(G, node_from, node_to, weight='distance')
                except nx.NetworkXNoPath:
                    # Handle cases where there is no path between nodes
                    distance_matrix.at[node_from, node_to] = float('inf')

    return distance_matrix

csv_file_path = 'dataset-3.csv'
df = pd.read_csv(csv_file_path)

# Call the function
result_distance_matrix = calculate_distance_matrix(df)

# Display the result
print("Distance Matrix:")
print(result_distance_matrix)


## Question 2: Unroll Distance Matrix
import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Initialize an empty list to store rows
    unrolled_rows = []

    # Iterate over the rows of the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Skip rows where id_start is equal to id_end
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                # Append a dictionary with the row information to the list
                unrolled_rows.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a DataFrame from the list of rows
    unrolled_df = pd.DataFrame(unrolled_rows)

    return unrolled_df


# Assuming 'result_distance_matrix' is the DataFrame from Question 1
unrolled_distance_df = unroll_distance_matrix(result_distance_matrix)

# Display the resulting DataFrame
print("Unrolled Distance DataFrame:")
print(unrolled_distance_df)

## Question 3: Finding IDs within Percentage Threshold
import pandas as pd

def find_ids_within_ten_percentage_threshold(df, reference_id):
    # Filter the DataFrame for rows with the given reference_id
    reference_df = df[df['id_start'] == reference_id]

    # Calculate the average distance for the reference_id
    reference_avg_distance = reference_df['distance'].mean()

    # Calculate the lower and upper bounds for the 10% threshold
    lower_bound = reference_avg_distance - 0.1 * reference_avg_distance
    upper_bound = reference_avg_distance + 0.1 * reference_avg_distance

    # Filter rows where the distance is within the 10% threshold
    within_threshold_df = df[(df['id_start'] != reference_id) & (df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]

    # Get unique values from the 'id_start' column and sort them
    result_ids = sorted(within_threshold_df['id_start'].unique())

    return result_ids

# Assuming 'unrolled_distance_df' is the DataFrame from the previous question
reference_value = 1  # Replace with the desired reference value
result_ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_distance_df, reference_value)

# Display the result
print(f"IDs within 10% threshold of the average distance for {reference_value}: {result_ids_within_threshold}")


## Question 4: Calculate Toll Rate

import pandas as pd

def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Add columns for toll rates based on vehicle types
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

# Example usage
# Assuming 'unrolled_distance_df' is the DataFrame from the previous question
toll_rate_df = calculate_toll_rate(unrolled_distance_df)

# Display the resulting DataFrame
print(toll_rate_df)



