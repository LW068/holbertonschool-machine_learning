import numpy as np

def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    
    # Retrieve dimensions from A_prev's shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (kh, kw, c_prev, c_new) = W.shape
    
    # Retrieve information from "stride"
    (sh, sw) = stride
    
    # Retrieve dimensions from dZ's shape
    (m, h_new, w_new, c_new) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))                           
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.zeros((1, 1, 1, c_new))

    # Pad A_prev and dA_prev
    if padding == "same":
        A_prev = # your padding method here
        dA_prev = # your padding method here
    
    # Loop over the training examples
    for i in range(m):                       
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev[i]
        da_prev_pad = dA_prev[i]
        
        for h in range(h_new): # loop over vertical axis of the output volume
            for w in range(w_new): # loop over horizontal axis of the output volume
                for c in range(c_new): # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
        # Set the ith training example's dA_prev to the unpaded da_prev_pad (if pad > 0)
        if padding == "same":
            dA_prev[i, :, :, :] = da_prev_pad[ph:-ph, pw:-pw, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad
            
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, h_prev, w_prev, c_prev))
    
    return dA_prev, dW, db
