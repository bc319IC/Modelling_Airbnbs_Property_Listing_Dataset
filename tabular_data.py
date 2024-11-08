import pandas as pd
import ast

def remove_rows_with_missing_ratings(df):
    '''
    Remove rows that have missing values in the rating columns.

    Parameters
    ----------
    df

    Returns
    -------
    df
    '''
    df = df.copy()
    rating_columns = ['Cleanliness_rating', 'Accuracy_rating', 'Location_rating', 'Check-in_rating', 'Value_rating']
    # Drop rows with missing values in any of the rating columns
    df = df.dropna(subset=rating_columns)
    return df

def combine_description_strings(df):
    '''
    Combine the list of description strings into a single string and clean up the description column.

    Parameters
    ----------
    df

    Returns
    -------
    df
    '''
    df = df.copy()
    # Remove rows where description is missing
    df = df.dropna(subset=['Description'])
    # Apply the cleaning function to the "Description" column
    df['Description'] = df['Description'].apply(clean_description)
    return df

def clean_description(desc_str):
    '''
    Helper function to convert description string to list and combine the elements.

    Parameters
    ----------
    desc_str

    Returns
    -------
    combined_desc
    '''
    try:
        # Convert the string representation of a list into an actual list using ast.literal_eval
        desc_list = ast.literal_eval(desc_str)
        # Ensure it's a list
        if not isinstance(desc_list, list):
            return desc_str  # Return as is if it's not a list
    except (ValueError, SyntaxError):
        # If the description is not a valid list representation, return it unchanged
        return desc_str
    # Remove empty strings from the list
    desc_list = [item for item in desc_list if item]
    # Combine the list elements into a single string with whitespace
    combined_desc = ' '.join(desc_list)
    # Remove the prefix "About this space"
    if combined_desc.startswith("About this space"):
        combined_desc = combined_desc.replace("About this space", "", 1).strip()
    return combined_desc

def set_default_feature_values(df):
    '''
    Replace empty values in the 'guests', 'beds', 'bathrooms', and 'bedrooms' columns with 1.

    Parameters
    ----------
    df

    Returns
    -------
    df
    '''
    df = df.copy()
    feature_columns = ['guests', 'beds', 'bathrooms', 'bedrooms']
    # Replace missing or empty values with 1 in these columns
    df[feature_columns] = df[feature_columns].fillna(1)
    return df

def clean_tabular_data(df):
    '''
    Clean the raw tabular data by applying all the preprocessing steps.

    Parameters
    ----------
    df

    Returns
    -------
    df
    '''
    df = df.copy()
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df

def set_column_types(df):
    '''
    Convert column types.

    Parameters
    ----------
    df

    Returns
    -------
    df
    '''
    # Choose columns
    float_cols = ['guests', 'bedrooms']
    cat_cols = ['Category']
    # Perform conversions
    df[float_cols] = df[float_cols].apply(pd.to_numeric, errors='coerce')
    df[cat_cols] = df[cat_cols].astype('category')
    df = df.dropna()
    return df

def load_airbnb(label):
    '''
    Loads the Airbnb dataset and returns features and labels as a tuple (features, labels).
    
    Parameters
    ----------
    label

    Returns
    -------
    features, label
    '''
    # Load the dataset
    df = pd.read_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')
    # Set column types
    df = set_column_types(df)
    # Separate features and labels
    labels = df[label]
    df = df.drop(columns=[label])
    # Convert all categorical columns to category codes
    for col in df.select_dtypes(include='category').columns:
        df[col + '_encoded'] = df[col].cat.codes
    features = df.select_dtypes(include=['number'])
    return features, labels


if __name__ == "__main__":
    # Load the raw data
    df = pd.read_csv('airbnb-property-listings/tabular_data/listing.csv')
    # Clean the dataset
    clean_df = clean_tabular_data(df)
    # Save the cleaned data to a new CSV file
    clean_df.to_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv', index=False)