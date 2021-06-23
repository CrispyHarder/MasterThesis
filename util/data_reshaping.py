from pickle import decode_long
import torch 
import math

def tensor_to_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

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
    device = stacked_tensors.device
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
    new_tensor = torch.zeros(depth, k_size*first_dim, k_size*second_dim, device=device)
    if torch.cuda.is_available():
        new_tensor = new_tensor.cuda()
    for i in range(first_dim):
        for j in range(second_dim):
            new_tensor[:,i*k_size:(i+1)*k_size, j*k_size:(j+1)*k_size] = stacked_tensors[i*second_dim+j] 
    return new_tensor

def pad_layer(params, depth=64, number=64):
    '''takes the params of a layer and padds them using zeros to 
    number x depth x k_size x k_size '''
    device = params.device
    p_shape = params.shape
    k_size = p_shape[2]
    # add 0 - kernels for unit input size
    pad_vec = torch.zeros(number-p_shape[0],p_shape[1],k_size,k_size,device=device)
    params = torch.cat((params, pad_vec), dim=0)
    p_shape = params.shape
    # pad zeros to ending of param vector
    pad_vec = torch.zeros(p_shape[0],depth-p_shape[1], k_size, k_size, device=device)
    params = torch.cat((params, pad_vec), dim=1)
    return params

def append_label_to_stacked(tensor, label, number_labels):
    device = tensor.device
    shape = tensor.shape
    k_size = shape[-1]
    number_kernels = shape[0]
    number_slices_to_label = math.ceil(number_labels/(k_size**2))
    index_position_slice = math.floor(label/(k_size**2))
    index_position_width = label % k_size
    index_position_height = (label % k_size**2) // k_size
    one_cath_slice = torch.zeros(number_slices_to_label, k_size, k_size)
    one_cath_slice[index_position_slice, index_position_width, index_position_height] = 1.0 
    cath_slices = torch.stack([one_cath_slice for _ in range(number_kernels)])
    cath_slices = cath_slices.to(device)
    tensor = torch.cat((tensor, cath_slices), dim=1)
    return tensor

def append_label_to_sided(tensor,k_size,label,number_labels):
    '''UNUSED 
    appends a label to a 3-dim tensor of sided kernels (i.e. a layer)
    (see above stack to sided). This is done by adding a "one-hot-matrix" 
    with a possible dim of 3. So to every kernel, that is one the "grid" a 2/3 dim
    kernel is appended
    Args:
        tensor(tensor): the tensor of sided kernels in a layer
        k_size(int): the original size of the kernels 
        label(int): the label to append
        number_labels(int): the total amount of labels
    
    Returns(tensor): the tensor with additonal depth for labels 
    e.g.
    for a matrix 
    [[2,2],
     [1,2]]
    
    [[0,1],
     [0,0]] is appended, where this is for label = 1 '''
    shape = tensor.shape
    number_kernels = int((shape[-1]/k_size)**2)
    number_slices_to_label = math.ceil(number_labels/(k_size**2))
    index_position_slice = math.floor(label/(k_size**2))
    index_position_width = label % k_size
    index_position_height = (label % k_size**2) // k_size
    one_cath_slice = torch.zeros(number_slices_to_label,k_size,k_size)
    one_cath_slice[index_position_slice,index_position_width,index_position_height] = 1.0 
    cath_slices = torch.stack([one_cath_slice for _ in range(number_kernels)])
    if torch.cuda.is_available():
        cath_slices = cath_slices.cuda()
    cath_slices = stack_to_side(cath_slices)
    return torch.cat((tensor,cath_slices),dim=0)

def append_label_to_vec(b_vec,labels,number_labels):
    '''appends labels to a batch of latent samples
    Args:
        b_vec(tensor): a batch of sampled latent vectors
        labels(int): the labels to append
        number_labels(int): the total amount of labels
    
    Returns(tensor):the batch of latents with labels appended
    '''
    eye = torch.eye(number_labels)
    eye = tensor_to_cuda(eye)
    one_hots = torch.stack([eye[label] for label in labels])
    print(one_hots)
    b_vec = torch.cat((b_vec,one_hots),dim=1)
    return b_vec

def side_to_stack(sided_tensors):
    pass