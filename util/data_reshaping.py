import torch 

def stack_to_side(stacked_tensors):
    '''takes a stack of 3d tensors (so overall a 4d tensor) and reshapes them into a 
    grid, by putting them side by side. This is done to use convolutions over 
    multiple kernels. 
    Args:
        stacked_tensors(4d torch.tensor): the stacked tensor
    
    Returns(3d tensor): the reshaped tensor
    
    Example: for the case of stacked 2d tensors (where each 2x2 matrix respresents
         a 3d kernel in the use case)
    [[[1,1],
      [2,2]],
      
     [[3,3],
      [4,4]],

     [[5,5],
      [6,6]],

     [[7,7],
      [8,8]]]
    
    would be transformed into 
    [[1,1,3,3],
     [2,2,4,4],
     [5,5,7,7],
     [6,6,8,8]]
      '''
      
    shape = stacked_tensors.shape
    assert shape[2]==shape[3] 
    depth = shape[1]
    k_size = shape[2]

    if shape[0] == 4:
        first_dim = 2
        second_dim = 2
    if shape[0] == 16:
        first_dim = 4
        second_dim = 4  
    if shape[0] == 32:
        first_dim = 8
        second_dim = 4  
    if shape[0] == 64:
        first_dim = 8
        second_dim = 8 
    new_tensor = torch.zeros(depth,k_size*first_dim,k_size*second_dim)
    if torch.cuda.is_available():
        new_tensor = new_tensor.cuda()
    for i in range(first_dim):
        for j in range(second_dim):
            new_tensor[:,i*k_size:(i+1)*k_size,j*k_size:(j+1)*k_size] = stacked_tensors[i*second_dim+j] 
    return new_tensor

def pad_layer(params,depth=64,number=64):
    '''takes the params of a layer and padds them using zeros to 
    number x depth x k_size x k_size '''
    p_shape = params.shape
    k_size = p_shape[2]
    # add 0 - kernels for unit input size
    if torch.cuda.is_available():
        pad_vec = torch.zeros(number-p_shape[0],p_shape[1],k_size,k_size).cuda()
    else:
        pad_vec = torch.zeros(number-p_shape[0],p_shape[1],k_size,k_size)
    params = torch.cat((params,pad_vec),dim=0)
    p_shape = params.shape
    # pad zeros to ending of param vector
    if torch.cuda.is_available():
        pad_vec = torch.zeros(p_shape[0],depth-p_shape[1],k_size,k_size).cuda()
    else:
        pad_vec = torch.zeros(p_shape[0],depth-p_shape[1],k_size,k_size)
    params = torch.cat((params,pad_vec),dim=1)

def side_to_stack(sided_tensors):
    pass