
# GradCAM: Continuation (Part 2) - Lecture Notebook

In the previous lecture notebook (GradCAM Part 1) we explored what Grad-Cam is and why it is useful. We also looked at how we can compute the activations of a particular layer using Keras API. In this notebook we will check the other element that Grad-CAM requires, the gradients of the model's output with respect to our desired layer's output. This is the "Grad" portion of Grad-CAM. 

Let's dive into it!


```python
import keras
from keras import backend as K
from util import *
```

    Using TensorFlow backend.


The `load_C3M3_model()` function has been taken care of and as last time, its internals are out of the scope of this notebook.


```python
# Load the model we used last time
model = load_C3M3_model()
```

    Got loss weights
    Loaded DenseNet
    Added layers
    Compiled Model
    Loaded Weights


Kindly recall from the previous notebook (GradCAM Part 1) that our model has 428 layers. 

We are now interested in getting the gradients when the model outputs a specific class. For this we will use Keras backend's `gradients(..)` function. This function requires two arguments: 

  - Loss (scalar tensor)
  - List of variables
  
Since we want the gradients with respect to the output, we can use our model's output tensor:


```python
# Save model's output in a variable
y = model.output

# Print model's output
y
```




    <tf.Tensor 'dense_1/Sigmoid:0' shape=(?, 14) dtype=float32>



However this is not a scalar (aka rank-0) tensor because it has axes. To transform this tensor into a scalar we can slice it like this:


```python
y = y[0]
y
```




    <tf.Tensor 'strided_slice:0' shape=(14,) dtype=float32>



It is still *not* a scalar tensor so we will have to slice it again:


```python
y = y[0]
y
```




    <tf.Tensor 'strided_slice_1:0' shape=() dtype=float32>



Now it is a scalar tensor!

The above slicing could be done in a single statement like this:

```python
y = y[0,0]
```

But the explicit version of it was shown for visibility purposes.

The first argument required by `gradients(..)` function is the loss, which we will like to get the gradient of, and the second is a list of parameters to compute the gradient with respect to. Since we are interested in getting the gradient of the output of the model with respect to the output of the last convolutional layer we need to specify the layer as we did  in the previous notebook:


```python
# Save the desired layer in a variable
layer = model.get_layer("conv5_block16_concat")

# Compute gradient of model's output with respect to last conv layer's output
gradients = K.gradients(y, layer.output)

# Print gradients list
gradients
```




    [<tf.Tensor 'gradients/AddN:0' shape=(?, ?, ?, 1024) dtype=float32>]



Notice that the gradients function returns a list of placeholder tensors. To get the actual placeholder we will get the first element of this list:


```python
# Get first (and only) element in the list
gradients = gradients[0]

# Print tensor placeholder
gradients
```




    <tf.Tensor 'gradients/AddN:0' shape=(?, ?, ?, 1024) dtype=float32>



As with the activations of the last convolutional layer in the previous notebook, we still need a function that uses this placeholder to compute the actual values for an input image. This can be done in the same manner as before. Remember this **function expects its arguments as lists or tuples**:


```python
# Instantiate the function to compute the gradients
gradients_function = K.function([model.input], [gradients])

# Print the gradients function
gradients_function
```




    <keras.backend.tensorflow_backend.Function at 0x7f124fc5f6d8>



Now that we have the function for computing the gradients, let's test it out on a particular image. Don't worry about the code to load the image, this has been taken care of for you, you should only care that an image ready to be processed will be saved in the x variable:


```python
# Load dataframe that contains information about the dataset of images
df = pd.read_csv("nih_new/train-small.csv")

# Path to the actual image
im_path = 'nih_new/images-small/00000599_000.png'

# Load the image and save it to a variable
x = load_image(im_path, df, preprocess=False)

# Display the image
plt.imshow(x, cmap = 'gray')
plt.show()
```


![png](output_19_0.png)


We should normalize this image before going forward, this has also been taken care of:


```python
# Calculate mean and standard deviation of a batch of images
mean, std = get_mean_std_per_batch(df)

# Normalize image
x = load_image_normalize(im_path, mean, std)
```

Now we have everything we need to compute the actual values of the gradients. In this case we should also **provide the input as a list or tuple**:


```python
# Run the function on the image and save it in a variable
actual_gradients = gradients_function([x])
```

An important intermediary step is to trim the batch dimension which can be done like this:


```python
# Remove batch dimension
actual_gradients = actual_gradients[0][0, :]
```


```python
# Print shape of the gradients array
print(f"Gradients of model's output with respect to output of last convolutional layer have shape: {actual_gradients.shape}")

# Print gradients array
actual_gradients
```

    Gradients of model's output with respect to output of last convolutional layer have shape: (10, 10, 1024)





    array([[[-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            ...,
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05]],
    
           [[-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            ...,
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05]],
    
           [[-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            ...,
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05]],
    
           ...,
    
           [[-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            ...,
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05]],
    
           [[-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            ...,
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05]],
    
           [[-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            ...,
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05],
            [-1.5733794e-09,  3.1699741e-09,  3.7147962e-07, ...,
              1.0372728e-04, -6.9426351e-05,  7.2338538e-05]]], dtype=float32)



Looks like everything worked out nicely! You will still have to wait for the assignment to see how these elements are used by Grad-CAM to get visual interpretations. Before you go you should know that there is a shortcut for these calculations by getting both elements from a single Keras function:


```python
# Save multi-input Keras function in a variable
activations_and_gradients_function = K.function([model.input], [layer.output, gradients])

# Run the function on our image
act_x, grad_x = activations_and_gradients_function([x])

# Remove batch dimension for both arrays
act_x = act_x[0, :]
grad_x = grad_x[0, :]
```


```python
# Print actual activations
print(act_x)

# Print actual gradients
print(grad_x)
```

    [[[-1.5343845e-01  1.4123724e-01 -1.9757609e-01 ...  2.0824453e-01
       -7.9827271e-02  2.3894583e-01]
      [-3.2100359e-01 -4.6550843e-01 -9.3461025e-01 ...  3.0469650e-01
       -9.0377383e-02  3.6720729e-01]
      [-2.3515984e-01 -2.2721815e-01 -9.4373369e-01 ...  2.6460707e-01
       -8.5206583e-02  3.1297126e-01]
      ...
      [-3.3458027e-01 -2.5393182e-01 -9.2875558e-01 ...  1.6176105e-01
       -1.0739267e-01  2.0712090e-01]
      [-1.7775959e-01  4.3856468e-02 -5.7920355e-01 ...  3.1388843e-01
       -1.3093445e-01  3.5897553e-01]
      [-1.6681334e-01  2.7054691e-01 -1.2726814e-01 ...  1.7067626e-01
       -8.0558173e-02  2.4668938e-01]]
    
     [[-3.1736961e-01  9.2963520e-03 -5.5861640e-01 ...  2.7064693e-01
       -1.2780805e-01  3.4593034e-01]
      [-1.3096830e-01 -3.5097411e-01 -2.8282785e-01 ...  4.5269516e-01
       -1.5534332e-01  5.4136336e-01]
      [-4.3571860e-01 -3.9957955e-01 -4.0801200e-01 ...  2.8063765e-01
       -1.2495937e-01  3.7415195e-01]
      ...
      [-2.7761659e-01 -5.3064013e-01 -5.4842830e-01 ...  1.9253999e-01
       -1.2812634e-01  2.3069727e-01]
      [-4.5409942e-01 -8.7442327e-01 -6.1426580e-01 ...  3.5173634e-01
       -1.5159036e-01  3.6092681e-01]
      [-4.8969400e-01 -1.7766926e-01 -1.4213458e-02 ...  2.0750809e-01
       -7.8806914e-02  2.4626185e-01]]
    
     [[-4.1035774e-01 -7.3956996e-03 -5.2129787e-01 ...  2.8508952e-01
       -1.2093155e-01  2.6571202e-01]
      [-1.1688797e+00 -1.0113001e+00 -2.7134556e-01 ...  4.7315854e-01
       -1.5717565e-01  3.8242674e-01]
      [-8.5275990e-01 -5.7765257e-01 -5.0458306e-01 ...  3.0554503e-01
       -9.9981576e-02  2.1944799e-01]
      ...
      [-1.3202590e+00  1.8371610e-01  1.7796636e-01 ...  7.2969705e-02
       -7.0173159e-02  1.0466556e-01]
      [-8.1227481e-01 -2.3811579e-02  1.2082573e+00 ...  2.0304865e-01
       -6.8502977e-02  1.6338965e-01]
      [-6.2042004e-01 -1.1466664e-01  1.0254879e+00 ...  9.4786108e-02
       -3.2457259e-02  9.3998685e-02]]
    
     ...
    
     [[-5.4403234e-01  1.5946254e-03  1.2096699e-01 ...  3.7806445e-01
       -1.5726990e-01  4.7968930e-01]
      [-3.2716766e-01  7.1170056e-01 -1.5058562e-02 ...  4.2537880e-01
       -1.3083649e-01  7.0594627e-01]
      [-1.9212844e+00 -1.2158600e+00 -7.6348162e-01 ...  1.9092430e-01
        1.5823427e-01  8.4129322e-01]
      ...
      [-1.9315977e+00 -7.7834153e-01 -6.1056775e-01 ...  1.9913573e-02
        1.3026717e-01  1.4355150e+00]
      [-1.6371340e+00  1.0755100e-01 -5.8090627e-01 ...  1.8748377e-01
       -7.3755167e-02  1.5150814e+00]
      [-7.9230893e-01  1.0160093e-01  3.2947856e-01 ... -1.0888398e-01
        1.0178839e-01  1.0603111e+00]]
    
     [[-5.6421053e-01 -2.9788837e-02 -5.3786653e-01 ...  3.1028652e-01
       -1.6652226e-01  4.8758659e-01]
      [-4.6102655e-01  1.4638299e-01 -5.1664060e-01 ...  4.7565815e-01
       -2.3842089e-01  6.7264563e-01]
      [-4.3581948e-01  5.6140935e-01 -1.4730456e+00 ...  3.5020867e-01
       -1.4836651e-01  5.3494191e-01]
      ...
      [-6.7289293e-01 -5.9881389e-02 -1.0087191e+00 ...  4.0596545e-02
       -6.6350162e-02  7.9924983e-01]
      [-2.2149906e-01  4.3203378e-01 -9.3042427e-01 ...  2.8882784e-01
       -2.2700994e-01  9.4611388e-01]
      [-4.6021724e-01 -1.4296138e-01 -5.9910685e-01 ...  2.6999030e-02
       -9.0014026e-02  7.4868810e-01]]
    
     [[-7.9662073e-01  2.7352807e-01  3.7429011e-01 ...  2.8498980e-01
       -1.2595838e-01  2.7435812e-01]
      [-8.9114231e-01 -1.2828174e-01  2.7379268e-01 ...  3.7110820e-01
       -1.3945863e-01  3.7343580e-01]
      [-5.5114996e-01  1.9286081e-02 -2.6363981e-01 ...  2.8650278e-01
       -1.0426319e-01  3.0358070e-01]
      ...
      [-4.4737351e-01  6.8707055e-01 -5.7893410e-02 ...  1.7416744e-01
       -8.9197822e-02  2.5765702e-01]
      [-4.8404604e-01  3.4624988e-01  1.6770643e-01 ...  2.7567405e-01
       -1.3088544e-01  3.3200648e-01]
      [-5.3579938e-01  2.3595692e-01  4.7545445e-01 ...  1.9573885e-01
       -6.8545997e-02  2.7971375e-01]]]
    [[[-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      ...
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]]
    
     [[-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      ...
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]]
    
     [[-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      ...
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]]
    
     ...
    
     [[-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      ...
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]]
    
     [[-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      ...
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]]
    
     [[-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      ...
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]
      [-1.5733794e-09  3.1699741e-09  3.7147962e-07 ...  1.0372728e-04
       -6.9426351e-05  7.2338538e-05]]]


**Congratulations on finishing this lecture notebook!** Hopefully you will now have a better understanding of how to leverage Keras's API power for computing gradients. Keep it up!
