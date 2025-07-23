import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#! ======== Introduction to Tensors ========
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
#* Tensors are large, some have millions maybe even billions of numbers, so they'll start to become automatically generated.

TENSOR.ndim # Tensors are 3 dimensional.
TENSOR.shape #? This will return torch.Size([1,3,3]), since we have 3 numbers in the second dimension, and 3 numbers in the third.
TENSOR[0] #! This will return the entire tensor value, because in the first dimension (or 0th), the entire array exists there. (very confusing)


#! ======== Random Tensors ========
#  Random tensors are important because the way many neural networks
#  learn is that they start with tensors full of random numbers and then
#  adjust those numbers to better represent the data

#TODO: Create a random tensor of size (3,4)
random_tensor = torch.rand(3,4) # Two dimensional tensor with 4 rows and 3 columns of randomly generated values.

#TODO: Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(244,244,3)) #? Height, Width, Color channels (R, G, B)
random_image_size_tensor.shape
random_image_size_tensor.ndim #! Its 3 dimensional

#! ======== Zeros & Ones ========
# Create a tensor of all zeros or ones, useful in creating masks.

#TODO: Create a tensor of all zeros
zeros = torch.zeros(size=(3,4)) #& A tensor but all the values are zero.


#TODO: Create a tensor of all ones
ones = torch.ones(size=(3,4)) #& A tensor but all the values are one.


#? The default data type in pytorch is a float 32, unless stated otherwise.
ones.dtype # Returns the default type of the tensor, this will return float 32.


#! ======== Creating Tensors in a Range ========
# torch.range() is deprecated, we will use torch.arange() instead || https://docs.pytorch.org/docs/stable/generated/torch.arange.html#torch-arange

one_to_nine = torch.arange(start=0, end=10, step=1) # Generators a tensor from 1->9, because index starts at 0. 
#! The gap between each pair of adjacent points. Default: 1.


#! ======== Creating Tensors Like ========
# Returns a tensor filled with the scalar value 0, or 1, with the same size as input. 

#TODO: Create a tensor range, then turn it into all zeros with the same size.
one_to_hundred = torch.arange(start=0, end=100, step=5) # Adds 5 incrementally, so its a tensor with only 20 values.

twenty_zeros = torch.zeros_like(input=one_to_hundred) # generates a tensor filled with twenty zeros with the same size as one_to_hundred
twenty_ones = torch.ones_like(input=one_to_hundred) #  generates a tensor filled with twenty ones with the same size as one_to_hundred


#! ======== Tensor Datatypes ========

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


#! ======== Getting Information From Tensor Attributes ========

#& Tensor data type is obtainable by using `tensor.dtype`
#& Tensor shape is obtainable by using `tensor.shape`
#& Tensor device is obtainable by using `tensor.device`

#? These cover the 3 major errors encountered.

some_tensor = torch.rand(3,4)

#TODO: Find out details about the tensor
some_tensor
f"Datatype of tensor: {some_tensor.dtype}" 
f"Shape of tensor: {some_tensor.shape}"  #! You can also use `some_tensor.size()`
f"Device  of tensor: {some_tensor.device}" 


#! ======== Manipulating Tensors | Tensor Operations ========

#* Tensor operations include: Addition, Subtraction, Multiplication, Division, Matrix multiplication

#TODO: Addition
tensor = torch.tensor([1,2,3])
tensor + 10 # this will result in tensor([11, 12, 13]). It adds to each value in the tensor 10.

#TODO: Multiplication
tensor * 10 # This will reutrn tensor([10, 20, 30]). It multiplies each value in the tensor by 10.

#TODO: Subtraction
tensor - 10 # This will return tensor([-9, -8, -7]). It subtracts each value in the tensor by 10.

#TODO: Division
tensor / 10 # This will return tensor([0.1, 0.2, 0.3]). It divides each value in the tensor by 10.

#* PyTorch in-bulit functions
torch.add(tensor, 10) # Addition
torch.mul(tensor, 10) # Multiplcation; First argument is the tensor, and the second arugment is the value.
torch.sub(tensor, 10) # Subtraction
torch.div(tensor, 10) # Division


#? What is matrix multiplcation? || https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
#* The product of two tensors, the behavior depends on the dimensionality of the tensors.

# Two main methods of performing multiplication in neural networks and deep learning.
# Element-wise multiplication (torch.mul) & Matrix Multiplcation (dot product).

#TODO: Element wise multiplcation
tensor * tensor # -> tensor([1,2,3]) * tensor([1,2,3]) = tensor([1,4,9]), 1*1, 2*2, 3*3

#TODO: Matrix multiplication 
torch.matmul(tensor, tensor) # -> tensor(14) | https://www.mathsisfun.com/algebra/matrix-multiplying.html

# OR

tensor @ tensor # @ another symbol for matmul.

#TODO: Matrix multiplication manually
1 * 1 + 2 * 2 + 3 * 3 # -> This will give 14

value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i] # - > This will give 14

#! Matrix multiplication manually is around x10 slower than using `torch.matmul()`

    
#? What are the matrix multiplication rules
#* A common error in deep learning, is a shape error. It's because the rules weren't followed.

#! 1. Inner dimensions must match.

# `(3, 2) @ (3, 2)` this will not work.
# `(2 ,3) @ (3, 2)` this will work.
# `(3, 2) @ (2, 3)` this will work.

# `(x, y) @ (y, x)` y is regarded as the inner dimension, and if y = 3, then this will work, since the inner dimensions are the same.

#! 2. The resulting matrix has the shape of the outer dimensions

# `(2 ,3) @ (3, 2)` -> `(2, 2)`, so it'll be a 2x2 shape
# `(3, 2) @ (2, 3)` -> `(3, 3)`, so it'll be a 3x3 shape

#* Shapes for matrix multiplication
tensor_A  = torch.tensor([[1,2], 
                          [3,4], 
                          [5, 6]])


tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])


# torch.mm(tensor_A, tensor_B) #! This stands for matrix multiplication, `torch.mm` -> `torch.matmul`
# This will not work, beecause the inner dimensions are not the same.

tensor_A.shape, tensor_B.shape #* Check the shapes.

#TODO: Fix the tensor shapes issues using transpose to manipulate one of the tensors.
# A transpose switches the axes or dimensions of a tensor.

tensor_B.T # This will switch the axes.

f"\n \nCompare tensors, Original tensor B: {tensor_B} \n New tensor: {tensor_B.T}"

f"\n \nCompare tensor shapes, Original tensor B: {tensor_B.shape} \n New tensor: {tensor_B.T.shape}"

f"\n \n Compare tensor shapes, Original tensor B: {tensor_B.shape} \n Tensor A: {tensor_A.shape}" # These will not be the same

f"\n \nCompare tensor shapes, New Tensor B: {tensor_B.T.shape} \n Tensor A: {tensor_A.shape}" # These will be the same, thus they can be matmuled.

torch.mm(tensor_A, tensor_B.T) # This will work.
# The resulting matrix will have the shape 3x3, because 3 is the number of the outside dimensions.


#! ======== Finding the min, max, mean, sum, etc (tensor aggregation) ========

#TODO: Create a tensor
x = torch.arange(0, 100, 10)
x.dtype #? Returns long, int64.

#TODO: Find the min
x.min()
# OR
torch.min(x)

#* Returns 0

#TODO: Find the max
torch.max(x)
# OR
x.max()

#* Returns 90

#TODO: Find the mean - this function requires a tensor of float32 datatype to work
# torch.mean(x) #! This will run into an error, because wrong datatype.

torch.mean(x.type(torch.float32)) # Convert it into a float 32.
x.type(torch.float32).mean() # This works as well.

#* Returns 45


#TODO: Find the sum
torch.sum(x)
# OR
x.sum()

#* Returns 450


#TODO: FIND THE POSITIONAL MAX AND MIN.
x.argmin() # Finding the position in the tensor that has the minimum value with argmin() -> returns index position of target tensor where minimum value occurs

#* Let's make a better example by having a new tensor
y = torch.arange(10, 100, 10)

y.argmin() # Returns tensor(0), because it's where the lowest value (10) is stored.
y[0] # Returns 10, the min


y.argmax() # Finding the position in the tensor that has the maximum value with argmax() -> returns index position of target tensor where maximum value occurs
# This function will return tensor(9)

y[9] # Returns 90, the maximum