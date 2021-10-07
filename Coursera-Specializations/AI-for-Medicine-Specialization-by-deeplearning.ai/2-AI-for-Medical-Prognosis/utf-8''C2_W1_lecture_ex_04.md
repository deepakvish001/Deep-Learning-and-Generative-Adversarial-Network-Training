# Course 2 week 1 lecture notebook Exercise 04
# Concordance index

In this week's graded assignment, you will implement the concordance index (c-index).  To get some practice with what you've seen in lecture, and to prepare for this week's assignment, you will write code to find permissible pairs, concordant pairs, and risk ties.

First start by importing packages and generating a small dataset.  The data is small enough that you can visually check the pairs of patients.


```python
# import packages
import pandas as pd
```

### Define the outcome `y`

- You will let `y` refer to the actual health outcome of the patient.
- 1 indicates disease, 0 indicates health (normal)


```python
# define 'y', the outcome of the patient
y = pd.Series([0,0,1,0])
y.name="health"
y
```




    0    0
    1    0
    2    1
    3    0
    Name: health, dtype: int64



### Define the risk scores
Define some risk scores that some model might produce for each patient.  Normally, you would run the patient features through a risk model to create these risk scores.  For practice, you will use the following values in the next cell.


```python
# Define the risk scores for each patient
risk_score = pd.Series([2.2, 3.3, 4.4, 4.4])
risk_score.name='risk score'
risk_score
```




    0    2.2
    1    3.3
    2    4.4
    3    4.4
    Name: risk score, dtype: float64



### Identify a permissible pair
A pair of patients is permissible if their outcomes are different. Use code to compare the labels.


```python
# Check patients 0 and 1 make a permissible pair.
if y[0] != y[1]:
    print(f"y[0]={y[0]} and y[1]={y[1]} is a permissible pair")
else:
    print(f"y[0]={y[0]} and y[1]={y[1]} is not a permissible pair")
```

    y[0]=0 and y[1]=0 is not a permissible pair



```python
# Check if patients 0 and 2 make a permissible pair
if y[0] != y[2]:
    print(f"y[0]={y[0]} and y[2]={y[2]} is a permissible pair")
else:
    print(f"y[0]={y[0]} and y[2]={y[2]} is NOT permissible pair")
```

    y[0]=0 and y[2]=1 is a permissible pair


### Check for risk ties
- For permissible pairs, check if they have the same risk score


```python
# Check if patients 2 and 3 make a risk tie
if risk_score[2] == risk_score[3]:
    print(f"patient 2 ({risk_score[2]}) and patient 3 ({risk_score[3]}) have a risk tie")
else:
    print(f"patient 2 ({risk_score[2]}) and patient 3 ({risk_score[3]}) DO NOT have a risk tie")
```

    patient 2 (4.4) and patient 3 (4.4) have a risk tie


### Concordant pairs
- Check if a permissible pair is also a concordant pair
- You'll check one case, where the first patient is healthy and the second has the disease.


```python
# Check if patient 1 and 2 make a concordant pair
if y[1] == 0 and y[2] == 1:
    if risk_score[1] < risk_score[2]:
        print(f"patient 1 and 2 is a concordant pair")
```

    patient 1 and 2 is a concordant pair


- Note that you checked the situation where patient 1 is healthy and patient 2 has the disease.
- You should also check the other situation where patient 1 has the disease and patient 2 is healthy.

You'll practice implementing the complete algorithm for c-index in this week's assignment!

### This is the end of this practice section.

Please continue on with the lecture videos!

---
