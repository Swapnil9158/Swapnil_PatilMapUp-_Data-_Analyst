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
