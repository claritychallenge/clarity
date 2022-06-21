import numpy as np


def GetNeuralNet():
    """
    Function to provide the weights derived for the ensemble of ten neural
    networks used for the HASPI_v2 intelligibility model. The neural networks
    have ten inputs, 4 neurons in the hidden layer, and one output neuron.
    The logsig activation function is used.

    Calling arguments: None

    Returned values:
    NNparam  vector of parameters defining the neural network
    Whid     cell array 10 x 1 for the weights linking the input to the hidden
             layer. Each cell is a 11 x 4 matrix of weights
    Wout     call array 5 x 1 for the weights linking the hidden to the output
             layer. Each cell is a 5 x 1 vector of weights.
    b        normalization so that the maximum neural net output is exactly 1.

    James M. Kates, 8 October 2019.
    Version for new neural network using actual TFS scores, 24 October 2019.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Set up the neural network parameters
    NNparam = np.zeros(6)
    NNparam[0] = 10  # Number of neurons in the input layer
    NNparam[1] = 4  # Number of neurons in the hidden layer
    NNparam[2] = 1  # Number of neurons in the output layer
    NNparam[3] = 1  # Activation function is logsig
    NNparam[4] = 0  # No offset for the activation function
    NNparam[5] = 1  # Maximum activation function value

    # Input to hidden layer weights
    Whid = [
        np.array(  # 1
            [
                [4.9980, -13.0590, 9.5478, -11.6760],
                [18.9793, -8.5842, -6.6974, 8.0382],
                [-37.8234, 26.9420, -6.6279, 2.6069],
                [4.1423, 5.2106, 10.3397, 9.4643],
                [-13.8839, 3.1211, -5.7794, -1.9207],
                [-17.0784, -8.5065, -16.7409, -1.6916],
                [-0.0696, -19.9487, -13.9734, -20.3328],
                [-10.5754, 15.5461, -3.9137, -2.0726],
                [-4.7145, 5.0427, 10.5728, 28.7840],
                [21.0595, -3.8171, 2.2084, 2.1929],
                [17.2857, 16.7562, -27.3290, 1.1543],
            ]
        ),
        np.array(  # 2
            [
                [-11.8283, -12.3466, 8.8198, 5.6027],
                [-8.3142, 6.2553, -4.1575, 13.7958],
                [27.6080, 3.3801, -7.9607, -33.7865],
                [1.3185, 5.7276, 8.3761, 0.8153],
                [4.0206, 3.4737, -7.0282, -9.8338],
                [-7.3265, -4.0271, -12.3923, -12.5861],
                [-17.9111, -23.1330, -16.2176, 0.2218],
                [15.0623, -3.9181, -2.3266, -21.2808],
                [1.0537, 34.5512, 8.7196, -8.7648],
                [-5.0357, -2.3610, -0.3678, 31.4586],
                [20.8312, 7.8687, -28.9087, 19.4417],
            ]
        ),
        np.array(  # 3
            [
                [9.5379, 4.4994, -13.1308, 0.9024],
                [3.9544, -2.4002, 2.6777, 22.9810],
                [-30.9452, -2.2645, 15.2613, -23.8526],
                [3.1327, 18.3449, 7.4923, -2.3167],
                [-4.3189, 6.5696, 2.5123, -15.6430],
                [-4.3704, -10.0506, 2.0855, -19.4876],
                [-9.6746, -9.9613, -30.5541, 3.4877],
                [-5.7179, -14.4015, 9.3838, -14.9651],
                [5.0717, -6.2980, 26.6210, -6.7466],
                [8.5857, -8.5345, -16.3236, 18.1852],
                [3.1709, -41.1078, 6.7127, 11.5747],
            ]
        ),
        np.array(  # 4
            [
                [9.3527, -13.3654, -2.1263, 5.1205],
                [9.4885, 1.9584, 21.8489, -8.0495],
                [-32.0886, 16.0934, -13.0956, -0.9466],
                [-4.9347, 6.1343, -0.7237, 21.6024],
                [-7.2456, 6.2478, -16.2268, 8.1160],
                [-5.9809, 0.7872, -20.7517, -9.8755],
                [-7.6038, -32.4284, -0.3817, -10.7850],
                [-5.5069, 11.0813, -14.9053, -18.0625],
                [8.9225, 27.1473, -10.8270, -7.0454],
                [7.4362, -19.8990, 12.3480, -6.7305],
                [6.3910, 7.1670, 11.7919, -38.1848],
            ]
        ),
        np.array(  # 5
            [
                [-12.0509, 8.7151, 12.9841, -12.7563],
                [-8.0669, 18.9493, -9.1899, 7.8737],
                [20.6577, -35.4767, -18.5397, 2.8544],
                [6.0629, -6.5786, 10.9516, 9.3709],
                [5.0354, -18.6275, -0.5501, 1.3219],
                [21.0090, -21.7111, 5.1285, -0.5481],
                [8.3379, -5.0779, 8.1280, -29.8252],
                [19.6124, -5.0156, -0.1799, -5.3723],
                [6.8287, 4.5828, 16.1024, 40.0935],
                [-30.5649, 10.5307, -11.8234, 0.4014],
                [-9.4186, 15.6892, -44.0505, 1.4371],
            ]
        ),
        np.array(  # 6
            [
                [8.9905, -16.4000, 13.3395, 8.9068],
                [11.0010, 11.3797, 14.8502, -14.2547],
                [-23.8174, 4.4221, -34.6896, -9.9423],
                [-8.1285, 4.0386, -5.7528, 7.6275],
                [-17.7683, 3.2188, -0.4409, 3.8280],
                [-14.2883, 2.4917, -16.7262, 13.1258],
                [-5.8409, -13.2882, -4.2047, 22.9793],
                [1.7396, 4.2947, -13.9206, 4.2493],
                [7.8760, 21.4827, -14.9673, -8.3899],
                [6.7850, -4.3356, 18.5928, -12.0981],
                [7.4116, -2.0622, 4.7621, -40.2684],
            ]
        ),
        np.array(  # 7
            [
                [-13.2736, 9.9119, 3.4659, 2.8783],
                [0.4675, -0.8187, 0.3497, 20.7397],
                [17.4133, -27.7575, -1.4997, -23.8363],
                [3.9760, 4.8989, 15.8285, -6.6393],
                [7.6936, 1.1009, 5.0979, -15.8340],
                [-0.2380, -4.6432, -8.9580, -17.8548],
                [-31.1510, -14.2219, -11.0122, 3.0247],
                [9.6552, -7.9702, -14.6836, -12.9456],
                [25.9963, 6.3569, -5.0912, -5.4249],
                [-15.9809, 9.4330, -10.4158, 15.9834],
                [6.1126, 0.1713, -43.7492, 14.7425],
            ]
        ),
        np.array(  # 8
            [
                [-11.6727, -15.7084, 9.9095, -7.3946],
                [4.4142, -4.4821, 10.9888, 0.0966],
                [6.4298, 25.5445, -32.7311, 4.1951],
                [8.4468, 16.3594, 7.0755, 7.2817],
                [-2.5481, 15.7296, -12.2159, -2.5490],
                [-3.2812, -0.6972, -13.1754, -0.7216],
                [-19.5254, -25.2440, -7.6636, -15.0124],
                [2.3548, 8.5716, -6.7492, 3.8422],
                [26.9615, 6.6441, 3.1680, 15.6611],
                [6.6129, -15.7791, 9.3453, 2.7809],
                [-3.6429, -0.8727, 0.2410, -0.7045],
            ]
        ),
        np.array(  # 9
            [
                [-13.9106, 3.1943, 8.7525, 7.8378],
                [4.1210, 0.4603, -7.2471, 16.2216],
                [9.3064, -3.8093, -14.4721, -34.2848],
                [11.6147, 17.6926, -1.5339, 2.6700],
                [5.3305, 4.0299, -13.0022, -15.3827],
                [-3.5035, -7.2305, 6.8711, -12.6676],
                [-25.5936, -9.8940, 10.5552, 2.4690],
                [7.7159, -17.8905, 6.5517, -17.6486],
                [26.7162, -5.0092, -3.5613, -0.0383],
                [-11.7304, -6.5251, -4.2616, 19.8528],
                [3.2551, -35.4889, -2.2133, 6.7308],
            ]
        ),
        np.array(  # 10
            [
                [13.5754, -13.4585, 2.5816, 7.5809],
                [-9.7189, 7.6225, -3.0220, 17.7773],
                [-25.6273, 4.1225, 4.2090, -35.4511],
                [5.3909, 11.0694, 15.5337, -1.3336],
                [-1.2964, 5.5829, 6.9950, -9.9642],
                [10.1510, 2.2819, -9.6950, -14.6332],
                [12.5032, -31.1403, -13.2782, 0.1385],
                [-2.6178, 6.8453, -20.5308, -16.9705],
                [-2.5462, 30.2576, -3.5750, 1.3910],
                [-6.2286, -14.7841, -7.3953, 17.8740],
                [-15.8615, 3.6023, -40.9104, 7.7481],
            ]
        ),
    ]

    # Hidden to output layer weights
    Wout = [
        np.array([[-0.1316, -2.5182, 1.6401, -3.2093, 1.7924]]).T,  # 1
        np.array([[-0.1653, 1.7375, 1.5526, -3.2349, -2.2877]]).T,  # 2
        np.array([[0.1847, -3.1987, -2.4941, 2.7106, -1.8048]]).T,  # 3
        np.array([[0.3962, -3.2952, 3.0003, -2.2602, -2.3269]]).T,  # 4
        np.array([[-0.0646, 1.3288, -3.4087, -2.0046, 1.8565]]).T,  # 5
        np.array([[1.3676, -3.4129, 1.6895, -1.8913, -1.5595]]).T,  # 6
        np.array([[0.8124, 2.7171, -3.0867, -2.3310, -2.3657]]).T,  # 7
        np.array([[-0.2743, 1.4949, 0.7896, -4.0589, 1.1257]]).T,  # 8
        np.array([[0.1307, 2.2788, -2.3633, -1.5073, -2.9985]]).T,  # 9
        np.array([[0.1024, -0.9517, 2.2123, -2.4008, -3.1655]]).T,  # 10
    ]

    # Normalization factor
    b = 0.9508

    return NNparam, Whid, Wout, b


def NNfeedfwdEns(data, NNparam, Whid, Wout):
    """
    Function to compute the neural network ensemble response to a set of
    inputs. The neural network is defined in NNfeedforwardZ.

    Calling arguments:
    data      array of features input to the neural network
    NNparam   vector of neural network paramters
    Whid      cell array of hidden layer weights for each network
    Wout      cell array of output layer weights for each network

    Returned value:
    model     neural network output vector averaged over the ensemble

    James M. Kates, 20 September 2011.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Data and network parameters
    if data.ndim == 1:
        data = np.expand_dims(data, 0)

    ncond = data.shape[0]  # Number of conditions in the input data
    K = len(Whid)  # Number of networks in the ensemble

    # Ensemble average of the predictions over the set of neural networks used for training
    predict = np.zeros((ncond, K))
    for k in range(K):
        for n in range(ncond):
            d = data[n, :]
            _, output = NNfeedforward(d, NNparam, Whid[k], Wout[k])
            predict[n, k] = output[1]

    model = np.mean(predict, 1)

    return model


def NNfeedforward(data, NNparam, Whid, Wout):
    """
    Function to compute the outputs at each layer of a neural network given
    the input to the network and the weights. The activiation function is an
    offset logistic function that gives either a logsig or hyperbolic
    tangent; the outputs from each layer have been reduced by the offset. The
    structure of the network is an input layer, one hidden layer, and an
    output layer. The first values in vectors hidden and output are set to 1
    by the function, and the remaining values correspond to the outputs at
    each neuron in the layer.

    Calling arguments:
    data       feature vector input to the neural network
    NNparam    network parameters from NNinit
    Whid       matrix of weights for the hidden layer
    Wout       matrix of weights for the output layer

    Returned values:
    hidden     vector of outputs from the hidden layer
    output     vector of outputs from the output layer

    James M. Kates, 26 October 2010.
    Translated from MATLAB to Python by Zuzanna Podwinska, March 2022.
    """
    # Correct Wout shape
    if len(Wout.shape) == 1:
        Wout = np.expand_dims(Wout, 1)

    # Extract parameters from the parameter array
    nx = int(NNparam[0])
    nhid = int(NNparam[1])
    nout = int(NNparam[2])
    beta = float(NNparam[3])
    offset = float(NNparam[4])
    again = float(NNparam[5])

    # Initialize the array storage
    x = np.zeros(nx + 1)
    hidden = np.zeros(nhid + 1)
    output = np.zeros(nout + 1)

    # Initialize the nodes used for constants
    x[0] = 1
    hidden[0] = 1
    output[0] = 1

    # Input layer
    for i in range(1, nx + 1):
        x[i] = data[i - 1]

    # Response of the hidden layer
    for j in range(1, nhid + 1):
        sumhid = np.sum(Whid[:, j - 1] * x)
        hidden[j] = (again / (1 + np.exp(-beta * sumhid))) - offset

    # Response of the output layer
    for k in range(1, nout + 1):
        sumout = np.sum(Wout[:, k - 1] * hidden)
        output[k] = (again / (1 + np.exp(-beta * sumout))) - offset

    return hidden, output
