
# Normalization (Min-Max Scaling) on Wine Dataset

## Overview

This notebook demonstrates how to apply Min-Max Scaling to normalize the features of the Wine Quality dataset. Normalization is crucial for many machine learning algorithms that are sensitive to the scale of features.


## Min-Max Scaling

Min-Max Scaling, or normalization, transforms features to a common scale, typically [0, 1]. This is essential when features have different ranges, as it ensures that all features contribute equally to the model.

### Mathematical Formula

The Min-Max Scaling formula is:
![Min-Max Scaling Formula]([https://example.com/path/to/your/formula.png](data:image/png;base64,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)



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

```

This README provides a brief explanation of Min-Max Scaling, its implementation, and why it is important, along with the relevant mathematical aspects.
