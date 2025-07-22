import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#! -------- Introduction to Tensors --------
#TODO: Create Tensors

#? Scalar Tensor || https://docs.pytorch.org/docs/stable/tensors.html
scalar = torch.tensor(7)
scalar.ndim # Shows the number of dimensions of a tensor. Scalars have 0
scalar.item() # Gives back the integer or value.

#? Vector Tensor
vector = torch.tensor([7,7])
vector.ndim # Vectors are one dimensional tensors
vector.shape # Returns a tuple that represents the size of each dimension. For a one-dimensional vector.

#? MATRIX Tensor
MATRIX = torch.tensor([[7, 8], [9, 10]]) 
MATRIX.ndim # Matrix tensors are 2 dimensional.
MATRIX[0] # Would return [7,8]
MATRIX[1] # Would return [9, 10]
MATRIX.shape #TODO: I have to understand the .shape property more, as of right now this would return a 2x2 shape.

#? TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
#! Tensors are large, some have millions maybe even billions of numbers, so they'll start to become automatically generated.

TENSOR.ndim # Tensors are 3 dimensional.
TENSOR.shape #? This will return torch.Size([1,3,3]), since we have 3 numbers in the second dimension, and 3 numbers in the third.
TENSOR[0] #! This will return the entire tensor value, because in the first dimension (or 0th), the entire array exists there. (very confusing)


#! -------- Random Tensors --------
#  Random tensors are important because the way many neural networks
#  learn is that they start with tensors full of random numbers and then
#  adjust those numbers to better represent the data

#TODO: Create a random tensor of size (3,4)
random_tensor = torch.rand(3,4) # Two dimensional tensor with 4 rows and 3 columns of randomly generated values.

#TODO: Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(244,244,3)) #? Height, Width, Color channels (R, G, B)
random_image_size_tensor.shape
random_image_size_tensor.ndim #! Its 3 dimensional

#! -------- Zeros & Ones --------
# Create a tensor of all zeros or ones, useful in creating masks.

#TODO: Create a tensor of all zeros
zeros = torch.zeros(size=(3,4)) #& A tensor but all the values are zero.


#TODO: Create a tensor of all ones
ones = torch.ones(size=(3,4)) #& A tensor but all the values are one.


#? The default data type in pytorch is a float 32, unless stated otherwise.
ones.dtype # Returns the default type of the tensor, this will return float 32.


#! -------- Creating Tensors in a Range --------
# torch.range() is deprecated, we will use torch.arange() instead || https://docs.pytorch.org/docs/stable/generated/torch.arange.html#torch-arange

one_to_nine = torch.arange(start=0, end=10, step=1) # Generators a tensor from 1->9, because index starts at 0. 
#! The gap between each pair of adjacent points. Default: 1.


#! -------- Creating Tensors Like --------
# Returns a tensor filled with the scalar value 0, or 1, with the same size as input. 

#TODO: Create a tensor range, then turn it into all zeros with the same size.
one_to_hundred = torch.arange(start=0, end=100, step=5) # Adds 5 incrementally, so its a tensor with only 20 values.

twenty_zeros = torch.zeros_like(input=one_to_hundred) # generates a tensor filled with twenty zeros with the same size as one_to_hundred
twenty_ones = torch.ones_like(input=one_to_hundred) #  generates a tensor filled with twenty ones with the same size as one_to_hundred


#! -------- Tensor Datatypes --------

#NOTE: Tensor datatypes is one of the 3 big errors encountered in PyTorch & deep learning

#TODO: Research the documentation about the tensor datatypes, there are many. https://docs.pytorch.org/docs/stable/tensors.html#data-types


#? Float 32 Tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                                dtype=None, #? Refers to what datatype is the tensor (float32, float16, int16, etc.)
                                device=None, #? By default it's cpu, but using a TPU or GPU is recommended, use "cuda" for gpu.
                                requires_grad=False) #? If you want pytorch to track the gradients of a tensor.

float_32_tensor.dtype # Returns a float 32, even though it's specified as none, because the default type is float 32.
#! More bits, usually means more detail / precision, but slower computing.

int16_tensor = float_32_tensor.type(torch.int16) # Converting the float 32 tensor into a int16.
int16_tensor # Returns the same tensor as previously, but with no decimals & a dtype=torch.int16.

