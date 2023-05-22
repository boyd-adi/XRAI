def handle_outliers(data, threshold=1.5):
    # Calculate the IQR
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    
    # Define the lower and upper bounds
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    # Identify the outliers
    outliers = (data < lower_bound) | (data > upper_bound)
    
    # Remove the outliers from the dataset
    cleaned_data = data[~outliers]
    
    return cleaned_data
