
# Normalization (Min-Max Scaling) on Wine Dataset

## Overview

This notebook demonstrates how to apply Min-Max Scaling to normalize the features of the Wine Quality dataset. Normalization is crucial for many machine learning algorithms that are sensitive to the scale of features.


## Min-Max Scaling

Min-Max Scaling, or normalization, transforms features to a common scale, typically [0, 1]. This is essential when features have different ranges, as it ensures that all features contribute equally to the model.


### Why Use Min-Max Scaling?

- **Uniform Scale**: Ensures all features are on a similar scale, which is important for algorithms sensitive to feature scales, like gradient descent.
- **Improved Convergence**: Helps in faster convergence of optimization algorithms.
- **Feature Comparability**: Makes it easier to compare and visualize the features.


## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('WineQT.csv')

# Display the first few rows and basic information
df.head(10)
df.info()
df.describe()

# Binarize the 'quality' feature
df['quality'] = np.where(df['quality'] <= 6, 0, 1)

# Initialize the Min-Max Scaler
mms = MinMaxScaler()

# Normalize all features between 0 and 1
columns = df.columns
df[columns] = mms.fit_transform(df[columns])

# Display the transformed data
df.head(10)
df.describe()
```

## Conclusion

Applying Min-Max Scaling transforms all feature values to a range between 0 and 1, making them comparable and ready for machine learning algorithms that require normalized input.
