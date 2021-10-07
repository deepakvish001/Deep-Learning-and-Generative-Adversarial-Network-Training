# Course 2 week 1 lecture notebook Exercise 03


<a name="combine-features"></a>
## Combine features

In this exercise, you will practice how to combine features in a pandas dataframe.  This will help you in the graded assignment at the end of the week.  

In addition, you will explore why it makes more sense to multiply two features rather than add them in order to create interaction terms.

First, you will generate some data to work with.


```python
# Import pandas
import pandas as pd

# Import a pre-defined function that generates data
from utils import load_data
```


```python
# Generate features and labels
X, y = load_data(100)
```


```python
X.head()
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
      <th>Age</th>
      <th>Systolic_BP</th>
      <th>Diastolic_BP</th>
      <th>Cholesterol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>77.196340</td>
      <td>78.784208</td>
      <td>87.026569</td>
      <td>82.760275</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63.529850</td>
      <td>105.171676</td>
      <td>83.396113</td>
      <td>80.923284</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69.003986</td>
      <td>117.582259</td>
      <td>91.161966</td>
      <td>92.915422</td>
    </tr>
    <tr>
      <th>3</th>
      <td>82.638210</td>
      <td>94.131208</td>
      <td>69.470423</td>
      <td>95.766098</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78.346286</td>
      <td>105.385186</td>
      <td>87.250583</td>
      <td>120.868124</td>
    </tr>
  </tbody>
</table>
</div>




```python
feature_names = X.columns
feature_names
```




    Index(['Age', 'Systolic_BP', 'Diastolic_BP', 'Cholesterol'], dtype='object')



### Combine strings
Even though you can visually see feature names and type the name of the combined feature, you can programmatically create interaction features so that you can apply this to any dataframe.

Use f-strings to combine two strings.  There are other ways to do this, but Python's f-strings are quite useful.


```python
name1 = feature_names[0]
name2 = feature_names[1]

print(f"name1: {name1}")
print(f"name2: {name2}")
```

    name1: Age
    name2: Systolic_BP



```python
# Combine the names of two features into a single string, separated by '_&_' for clarity
combined_names = f"{name1}_&_{name2}"
combined_names
```




    'Age_&_Systolic_BP'



### Add two columns
- Add the values from two columns and put them into a new column.
- You'll do something similar in this week's assignment.


```python
X[combined_names] = X['Age'] + X['Systolic_BP']
X.head(2)
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
      <th>Age</th>
      <th>Systolic_BP</th>
      <th>Diastolic_BP</th>
      <th>Cholesterol</th>
      <th>Age_&amp;_Systolic_BP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>77.19634</td>
      <td>78.784208</td>
      <td>87.026569</td>
      <td>82.760275</td>
      <td>155.980548</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63.52985</td>
      <td>105.171676</td>
      <td>83.396113</td>
      <td>80.923284</td>
      <td>168.701526</td>
    </tr>
  </tbody>
</table>
</div>



### Why we multiply two features instead of adding

Why do you think it makes more sense to multiply two features together rather than adding them together?

Please take a look at two features, and compare what you get when you add them, versus when you multiply them together.


```python
# Generate a small dataset with two features
df = pd.DataFrame({'v1': [1,1,1,2,2,2,3,3,3],
                   'v2': [100,200,300,100,200,300,100,200,300]
                  })

# add the two features together
df['v1 + v2'] = df['v1'] + df['v2']

# multiply the two features together
df['v1 x v2'] = df['v1'] * df['v2']
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
      <th>v1</th>
      <th>v2</th>
      <th>v1 + v2</th>
      <th>v1 x v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>100</td>
      <td>101</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>200</td>
      <td>201</td>
      <td>200</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>300</td>
      <td>301</td>
      <td>300</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>100</td>
      <td>102</td>
      <td>200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>200</td>
      <td>202</td>
      <td>400</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>300</td>
      <td>302</td>
      <td>600</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>100</td>
      <td>103</td>
      <td>300</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>200</td>
      <td>203</td>
      <td>600</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>300</td>
      <td>303</td>
      <td>900</td>
    </tr>
  </tbody>
</table>
</div>



It may not be immediately apparent how adding or multiplying makes a difference; either way you get unique values for each of these operations.

To view the data in a more helpful way, rearrange the data (pivot it) so that:
- feature 1 is the row index 
- feature 2 is the column name.  
- Then set the sum of the two features as the value. 

Display the resulting data in a heatmap.


```python
# Import seaborn in order to use a heatmap plot
import seaborn as sns
```


```python
# Pivot the data so that v1 + v2 is the value

df_add = df.pivot(index='v1',
                  columns='v2',
                  values='v1 + v2'
                 )
print("v1 + v2\n")
display(df_add)
print()
sns.heatmap(df_add);
```

    v1 + v2
    



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
      <th>v2</th>
      <th>100</th>
      <th>200</th>
      <th>300</th>
    </tr>
    <tr>
      <th>v1</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>101</td>
      <td>201</td>
      <td>301</td>
    </tr>
    <tr>
      <th>2</th>
      <td>102</td>
      <td>202</td>
      <td>302</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103</td>
      <td>203</td>
      <td>303</td>
    </tr>
  </tbody>
</table>
</div>


    



![png](output_15_3.png)


Notice that it doesn't seem like you can easily distinguish clearly when you vary feature 1 (which ranges from 1 to 3), since feature 2 is so much larger in magnitude (100 to 300).  This is because you added the two features together.

#### View the 'multiply' interaction

Now pivot the data so that:
- feature 1 is the row index 
- feature 2 is the column name.  
- The values are 'v1 x v2' 

Use a heatmap to visualize the table.


```python
df_mult = df.pivot(index='v1',
                  columns='v2',
                  values='v1 x v2'
                 )
print('v1 x v2')
display(df_mult)
print()
sns.heatmap(df_mult);
```

    v1 x v2



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
      <th>v2</th>
      <th>100</th>
      <th>200</th>
      <th>300</th>
    </tr>
    <tr>
      <th>v1</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>200</td>
      <td>300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>200</td>
      <td>400</td>
      <td>600</td>
    </tr>
    <tr>
      <th>3</th>
      <td>300</td>
      <td>600</td>
      <td>900</td>
    </tr>
  </tbody>
</table>
</div>


    



![png](output_18_3.png)


Notice how when you multiply the features, the heatmap looks more like a 'grid' shape instead of three vertical bars.  

This means that you are more clearly able to make a distinction as feature 1 varies from 1 to 2 to 3.

### Discussion

When you find the interaction between two features, you ideally hope to see how varying one feature makes an impact on the interaction term.  This is better achieved by multiplying the two features together rather than adding them together.  

Another way to think of this is that you want to separate the feature space into a "grid", which you can do by multiplying the features together.

In this week's assignment, you will create interaction terms!

### This is the end of this practice section.

Please continue on with the lecture videos!

---
