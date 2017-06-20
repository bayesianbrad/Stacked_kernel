# Stacked_kernel_GP
*Only tested on Mac El Captain OS Python 3*

To run, you will need the following:

- Python3
- GPy
- Numpy
- Matpolotlib
- Scipy

And if you would like to run on a GPU pycuda 
      *Note: Only works with RBF (SE) Kernels. 
             Functionality may be disabled
            
Note2: The comments in the code may not be up-to-date with what is actually happening in the script. Will clear up later. 

To add more kernels from the GPy framework, go to dataprocessing.py , 
                                            edit *kernel_list* add additional kernels to this list or remove them>

Note: 
- If using an IDE with tab completion typing Gpy.kern.<press_tab> will show all available kernels 
- I've only tested with RBF, StdPeriodic, Matern32, Matern52 and MLP. 
- Some kernels may not work with the current framework as is, as some kernels do not have a lengthscale. 
                                           
To add more datasets, first add your desired dataset to the outer *data* folder:
- Then go to either importdata.py or import2d.py (if using 2D starting input) and add an "else if option == <you_define_an_option_name> and add the following:
    - data      = <import_YOUR_data_function>
    - data      = data[:,None] % transposes data to a column array
    - size_data = size_data(data.shape[0])
    - time      = np.linspace(0,1,size_data)
- For import2d.py add something like this:
    - elif option == 'stock2d':
    - stock_no            = 13
    - data,input2         = sd.get_stock(stock_no,if_2d = True)
    - data                = data[0:1000][:,None]
    - size_data           = data.shape[0]
    - inputs1             = np.linspace(0,1,size_data)
    - inputs2             = input2[0:1000][:,None]
    - inputs              = np.column_stack((inputs1,inputs2))
                                                     
                                                     
- Go to main.py or mainNd.py and add to the *Options* list the name that you defined.
- Typically all observed inputs are times, where times is a linearly spaced array between [0,1] and is spaced depending on the size of your data set. 
- All things are put into to column vector form so that they can be used throughout the script


To run: 
- Go to main.py or mainNd.py
- Go to the main function at the bottom of the script
- Edit how many *runs* you want (no of layers) , the *split* of your training to test data and edit the *options[<insert_Option_list_number>]*
and *Models[<insert_0_or_1>]* depending on whehther you want the augmented or single input model

The scrip automatically generates plots for each layer and saves them in ../Plots/<choosen_model>/<option>
it also saves all data that has been gathered and save .csv files for each layer ../Data/<choosen_model>/data_layer_/<choosen_option>
