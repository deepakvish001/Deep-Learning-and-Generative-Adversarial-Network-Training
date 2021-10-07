# Week 2 lecture notebook

## Outline

[Missing values](#missing-values)

[Decision tree classifier](#decision-tree)

[Apply a mask](#mask)

[Imputation](#imputation)

<a name="missing-values"></a>
## Missing values


```python
import numpy as np
import pandas as pd
```


```python
df = pd.DataFrame({"feature_1": [0.1,np.NaN,np.NaN,0.4],
                   "feature_2": [1.1,2.2,np.NaN,np.NaN]
                  })
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>feature_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.1</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.4</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Check if each value is missing


```python
df.isnull()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>feature_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



### Check if any values in a row are true



```python
df_booleans = pd.DataFrame({"col_1": [True,True,False],
                            "col_2": [True,False,False]
                           })
df_booleans
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_1</th>
      <th>col_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



- If we use pandas.DataFrame.any(), it checks if at least one value in a column is `True`, and if so, returns `True`.
- If all rows are `False`, then it returns `False` for that column


```python
df_booleans.any()
```




    col_1    True
    col_2    True
    dtype: bool



- Setting the axis to zero also checks if any item in a column is `True`


```python
df_booleans.any(axis=0)
```




    col_1    True
    col_2    True
    dtype: bool



- Setting the axis to `1` checks if any item in a **row** is `True`, and if so, returns true
- Similarily only when all values in a row are `False`, the function returns `False`.


```python
df_booleans.any(axis=1)
```




    0     True
    1     True
    2    False
    dtype: bool



### Sum booleans


```python
series_booleans = pd.Series([True,True,False])
series_booleans
```




    0     True
    1     True
    2    False
    dtype: bool



- When applying `sum` to a series (or list) of booleans, the `sum` function treats `True` as 1 and `False` as zero.


```python
sum(series_booleans)
```




    2



You will make use of these functions in this week's assignment!

### This is the end of this practice section.

Please continue on with the lecture videos!

---

<a name="decision-tree"></a>
## Decision Tree Classifier



```python
import pandas as pd
```


```python
X = pd.DataFrame({"feature_1":[0,1,2,3]})
y = pd.Series([0,0,1,1])
```


```python
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
y
```




    0    0
    1    0
    2    1
    3    1
    dtype: int64




```python
from sklearn.tree import DecisionTreeClassifier
```


```python
dt = DecisionTreeClassifier()
dt
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')




```python
dt.fit(X,y)
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')



### Set tree parameters


```python
dt = DecisionTreeClassifier(criterion='entropy',
                            max_depth=10,
                            min_samples_split=2
                           )
dt
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                           max_depth=10, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')



### Set parameters using a dictionary

- In Python, we can use a dictionary to set parameters of a function.
- We can define the name of the parameter as the 'key', and the value of that parameter as the 'value' for each key-value pair of the dictionary.


```python
tree_parameters = {'criterion': 'entropy',
                   'max_depth': 10,
                   'min_samples_split': 2
                  }
```

- We can pass in the dictionary and use `**` to 'unpack' that dictionary's key-value pairs as parameter values for the function.


```python
dt = DecisionTreeClassifier(**tree_parameters)
dt
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                           max_depth=10, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')



### This is the end of this practice section.

Please continue on with the lecture videos!

---

<a name="mask"></a>
## Apply a mask

Use a 'mask' to filter data of a dataframe


```python
import pandas as pd
```


```python
df = pd.DataFrame({"feature_1": [0,1,2,3,4]})
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
mask = df["feature_1"] >= 3
mask
```




    0    False
    1    False
    2    False
    3     True
    4     True
    Name: feature_1, dtype: bool




```python
df[mask]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



### Combining comparison operators

You'll want to be careful when combining more than one comparison operator, to avoid errors.
- Using the `and` operator on a series will result in a `ValueError`, because it's 


```python
df["feature_1"] >=2
```




    0    False
    1    False
    2     True
    3     True
    4     True
    Name: feature_1, dtype: bool




```python
df["feature_1" ] <=3
```




    0     True
    1     True
    2     True
    3     True
    4    False
    Name: feature_1, dtype: bool




```python
# NOTE: This will result in a ValueError
df["feature_1"] >=2 and df["feature_1" ] <=3
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-26-4feb82af6b46> in <module>
          1 # NOTE: This will result in a ValueError
    ----> 2 df["feature_1"] >=2 and df["feature_1" ] <=3
    

    /opt/conda/lib/python3.7/site-packages/pandas/core/generic.py in __nonzero__(self)
       1553             "The truth value of a {0} is ambiguous. "
       1554             "Use a.empty, a.bool(), a.item(), a.any() or a.all().".format(
    -> 1555                 self.__class__.__name__
       1556             )
       1557         )


    ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().


### How to combine two logical operators for Series
What we want is to look at the same row of each of the two series, and compare each pair of items, one row at a time. To do this, use:
- the `&` operator instead of `and`
- the `|` operator instead of `or`.
- Also, you'll need to surround each comparison with parenthese `(...)`


```python
# This will compare the series, one row at a time
(df["feature_1"] >=2) & (df["feature_1" ] <=3)
```




    0    False
    1    False
    2     True
    3     True
    4    False
    Name: feature_1, dtype: bool



### This is the end of this practice section.

Please continue on with the lecture videos!

---

<a name="imputation"></a>
## Imputation

We will use imputation functions provided by scikit-learn.  See the scikit-learn [documentation on imputation](https://scikit-learn.org/stable/modules/impute.html#iterative-imputer)


```python
import pandas as pd
import numpy as np
```


```python
df = pd.DataFrame({"feature_1": [0,1,2,3,4,5,6,7,8,9,10],
                   "feature_2": [0,np.NaN,20,30,40,50,60,70,80,np.NaN,100],
                  })
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature_1</th>
      <th>feature_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>



### Mean imputation


```python
from sklearn.impute import SimpleImputer
```


```python
mean_imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
mean_imputer
```




    SimpleImputer(add_indicator=False, copy=True, fill_value=None,
                  missing_values=nan, strategy='mean', verbose=0)




```python
mean_imputer.fit(df)
```




    SimpleImputer(add_indicator=False, copy=True, fill_value=None,
                  missing_values=nan, strategy='mean', verbose=0)




```python
nparray_imputed_mean = mean_imputer.transform(df)
nparray_imputed_mean
```




    array([[  0.,   0.],
           [  1.,  50.],
           [  2.,  20.],
           [  3.,  30.],
           [  4.,  40.],
           [  5.,  50.],
           [  6.,  60.],
           [  7.,  70.],
           [  8.,  80.],
           [  9.,  50.],
           [ 10., 100.]])



Notice how the missing values are replaced with `50` in both cases.

### Regression Imputation


```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```


```python
reg_imputer = IterativeImputer()
reg_imputer
```




    IterativeImputer(add_indicator=False, estimator=None,
                     imputation_order='ascending', initial_strategy='mean',
                     max_iter=10, max_value=None, min_value=None,
                     missing_values=nan, n_nearest_features=None, random_state=None,
                     sample_posterior=False, skip_complete=False, tol=0.001,
                     verbose=0)




```python
reg_imputer.fit(df)
```




    IterativeImputer(add_indicator=False, estimator=None,
                     imputation_order='ascending', initial_strategy='mean',
                     max_iter=10, max_value=None, min_value=None,
                     missing_values=nan, n_nearest_features=None, random_state=None,
                     sample_posterior=False, skip_complete=False, tol=0.001,
                     verbose=0)




```python
nparray_imputed_reg = reg_imputer.transform(df)
nparray_imputed_reg
```




    array([[  0.,   0.],
           [  1.,  10.],
           [  2.,  20.],
           [  3.,  30.],
           [  4.,  40.],
           [  5.,  50.],
           [  6.,  60.],
           [  7.,  70.],
           [  8.,  80.],
           [  9.,  90.],
           [ 10., 100.]])



Notice how the filled in values are replaced with `10` and `90` when using regression imputation. The imputation assumed a linear relationship between feature 1 and feature 2.

### This is the end of this practice section.

Please continue on with the lecture videos!

---


```python

```
