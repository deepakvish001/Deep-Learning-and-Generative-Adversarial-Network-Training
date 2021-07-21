# AI4M Course 2 Week 3 lecture notebook

## Outline

[Count patients](#count-patients)

[Kaplan-Meier](#kaplan-meier)

<a name="count-patients"></a>
## Count patients


```python
import numpy as np
import pandas as pd
```

We'll work with data where:
- Time: days after a disease is diagnosed and the patient either dies or left the hospital's supervision.
- Event: 
    - 1 if the patient died
    - 0 if the patient was not observed to die beyond the given 'Time' (their data is censored)
    
Notice that these are the same numbers that you see in the lecture video about estimating survival.


```python
df = pd.DataFrame({'Time': [10,8,60,20,12,30,15],
                   'Event': [1,0,1,1,0,1,0]
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
      <th>Time</th>
      <th>Event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Count patients 

### Count number of censored patients


```python
df['Event'] == 0
```




    0    False
    1     True
    2    False
    3    False
    4     True
    5    False
    6     True
    Name: Event, dtype: bool



Patient 1, 4 and 6 were censored.

- Count how many patient records were censored

When we sum a series of booleans, `True` is treated as 1 and `False` is treated as 0.


```python
sum(df['Event'] == 0)
```




    3



### Count number of patients who definitely survived past time t

This assumes that any patient who was censored died at the time of being censored ( **died immediately**).

If a patient survived past time `t`:
- Their `Time` of event should be greater than `t`.  
- Notice that they can have an `Event` of either 1 or 0.  What matters is their `Time` value.


```python
t = 25
df['Time'] > t
```




    0    False
    1    False
    2     True
    3    False
    4    False
    5     True
    6    False
    Name: Time, dtype: bool




```python
sum(df['Time'] > t)
```




    2



### Count the number of patients who may have survived past t

This assumes that censored patients **never die**.
- The patient is censored at any time and we assume that they live forever.
- The patient died (`Event` is 1) but after time `t`


```python
t = 25
(df['Time'] > t) | (df['Event'] == 0)
```




    0    False
    1     True
    2     True
    3    False
    4     True
    5     True
    6     True
    dtype: bool




```python
sum( (df['Time'] > t) | (df['Event'] == 0) )
```




    5



### Count number of patients who were not censored before time t

If patient was not censored before time `t`:
- They either had an event (death) before `t`, at `t`, or after `t` (any time)
- Or, their `Time` occurs after time `t` (they may have either died or been censored at a later time after `t`)


```python
t = 25
(df['Event'] == 1) | (df['Time'] > t)
```




    0     True
    1    False
    2     True
    3     True
    4    False
    5     True
    6    False
    dtype: bool




```python
sum( (df['Event'] == 1) | (df['Time'] > t) )
```




    4



<a name="kaplan-meier"></a>
## Kaplan-Meier

The Kaplan Meier estimate of survival probability is:

$$
S(t) = \prod_{t_i \leq t} (1 - \frac{d_i}{n_i})
$$

- $t_i$ are the events observed in the dataset 
- $d_i$ is the number of deaths at time $t_i$
- $n_i$ is the number of people who we know have survived up to time $t_i$.



```python
import numpy as np
import pandas as pd
```


```python
df = pd.DataFrame({'Time': [3,3,2,2],
                   'Event': [0,1,0,1]
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
      <th>Time</th>
      <th>Event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Find those who survived up to time $t_i$

If they survived up to time $t_i$, 
- Their `Time` is either greater than $t_i$
- Or, their `Time` can be equal to $t_i$


```python
t_i = 2
df['Time'] >= t_i
```




    0    True
    1    True
    2    True
    3    True
    Name: Time, dtype: bool



You can use this to help you calculate $n_i$

### Find those who died at time $t_i$

- If they died at $t_i$:
- Their `Event` value is 1.  
- Also, their `Time` should be equal to $t_i$


```python
t_i = 2
(df['Event'] == 1) & (df['Time'] == t_i)
```




    0    False
    1    False
    2    False
    3     True
    dtype: bool



You can use this to help you calculate $d_i$

You'll implement Kaplan Meier in this week's assignment!
