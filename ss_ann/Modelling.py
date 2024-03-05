import numpy as np
from ss_ann.PINN import PINN_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import load_model
from preprocessing import *
from postprocessing import *
from Plots import *
import pickle as pk


def import_data(file):

    ij_link_path = '/home/combustion/Desktop/labfm/lucas/neigh/ij_link' + str(file) + '.csv'
    coor_path = '/home/combustion/Desktop/labfm/lucas/coor/coor' + str(file) + '.csv'
    weights_path = '/home/combustion/Desktop/labfm/lucas/weights/laplace/w_' + str(file) + '.csv'
    dx_path = '/home/combustion/Desktop/labfm/lucas/dx/dx' + str(file) + '.csv'

    ij_link = np.genfromtxt(ij_link_path, delimiter=',', skip_header=0)
    coor = np.genfromtxt(coor_path, delimiter=',', skip_header=0)
    coor = coor[:, :-1]
    weights = np.genfromtxt(weights_path, delimiter=',', skip_header=0)
    weights = trim_zero_columns(weights[:, 1:])
    dx = np.genfromtxt(dx_path, delimiter=',', skip_header=0)
    dx = dx[0]
    return ij_link, coor, weights, dx


def import_stored_data(file, order, noise):
    ij_link_path = '/home/combustion/Desktop/PhD/Shape Function Surrogate/Order_'+str(order)+'/Noise_'+str(noise)+'/Data/neigh/ij_link' \
                   + str(file) + '.csv'
    coor_path = '/home/combustion/Desktop/PhD/Shape Function Surrogate/Order_'+str(order)+'/Noise_'+str(noise)+'/Data/coor/coor' \
                + str(file) + '.csv'
    weights_path = '/home/combustion/Desktop/PhD/Shape Function Surrogate/Order_'+str(order)+'/Noise_'+str(noise)+'/Data/weights/laplace/w_'\
                   + str(file) + '.csv'
    dx_path = '/home/combustion/Desktop/PhD/Shape Function Surrogate/Order_'+str(order)+'/Noise_'+str(noise)+'/Data/dx/dx' \
              + str(file) + '.csv'

    ij_link = np.genfromtxt(ij_link_path, delimiter=',', skip_header=0)
    coor = np.genfromtxt(coor_path, delimiter=',', skip_header=0)
    coor = coor[:, :-1]
    weights = np.genfromtxt(weights_path, delimiter=',', skip_header=0)
    weights = trim_zero_columns(weights[:, 1:])
    dx = np.genfromtxt(dx_path, delimiter=',', skip_header=0)
    dx = dx[0]
    return ij_link, coor, weights, dx


file = 8
#ij_link1, coor1, weights1, dx1 = import_stored_data(file, order=2, noise=0.5)

#features1 = feat_extract(coor1, ij_link1)
#features1 = features1[:, 1:, :]  # This removes the first item of features which is always 0


ij_link2, coor2, weights2, dx2 = import_stored_data(file, order=2, noise=0.3)
features2 = feat_extract(coor2, ij_link2)
features2 = features2[:, 1:, :]


features = features2#np.concatenate((features1, features2), axis=0)
weights = weights2#np.concatenate((weights1, weights2), axis=0)
coor = coor2#np.concatenate((coor1, coor2), axis=0)
dx = dx2

stand = 4 # stand determines the type of normalization that will be applied
if stand == 1:
    stand_feature, stand_label, f_mean, f_stdv, l_mean, l_stdv = standardize_comp_stencil(features, weights)
elif stand == 2:
    stand_feature, stand_label, f_mean, l_mean, h_scale_xy, h_scale_w = non_dimension(features, weights, dx, dtype='laplace')
elif stand == 3:
    stand_feature, stand_label, f_mean, f_stdv, l_mean, l_stdv = global_standard(features, weights)
elif stand == 4:
    stand_feature, stand_label, f_mean, f_stdv, l_stdv = std_dev_norm(features, weights, 'laplace')


train_f, train_l, test_f, test_l, train_index, test_index = create_train_test(stand_feature, stand_label,
                                                                              tt_split=0.9, seed=1) # This generates the test data

train_f, val_f, train_l, val_l = train_test_split(train_f, train_l, test_size=0.2, random_state=1) # This generates the validation data


N = train_l.shape[1]

model_instance = PINN_model(input_size=2*N, num_neurons=32, num_layers=2, output_size=N)

model = model_instance.get_model()

model.compile(optimizer='adam', loss='mean_absolute_error')

early_stop = EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True)


history = model.fit(train_f, train_l, epochs=100, batch_size=32, validation_data=(val_f, val_l))

# To save ANN trained
# model.save('my_model.keras')

# To save training history
# with open('history.pk', 'wb') as f:
#     pickle.dump(history.history, f)

plot_training(history)

pred_l = model.predict(test_f)

'''The two functions below rescale the data to their original magnitude'''
if stand == 1:
    scaled_actual_l, scaled_pred_l, scaled_feat = rescale_stencil(test_l, pred_l, test_f, f_mean, f_stdv, l_mean,
                                                                  l_stdv, test_index)
elif stand == 2:
    scaled_actual_l, scaled_pred_l, scaled_feat = rescale_h(test_l, pred_l, test_f, f_mean, l_mean, h_scale_xy,
                                                        h_scale_w, test_index)
elif stand == 3:
    scaled_actual_l, scaled_pred_l, scaled_feat = rescale_global_stand(test_l, pred_l, test_f, f_mean, f_stdv, l_mean,
                                                                       l_stdv, test_index)
elif stand == 4:
    scaled_actual_l, scaled_pred_l, scaled_feat = rescale_std(test_l, pred_l, test_f, f_mean, f_stdv, l_stdv, test_index)


test_neigh_coor = d_2_c(coor, test_index, scaled_feat)

plot_node_prediction_error(scaled_pred_l, scaled_actual_l, test_neigh_coor, node='random', size=20, option=3)


pred = error_test_func(scaled_feat, scaled_pred_l)
act = error_test_func(scaled_feat, scaled_actual_l)
err = act - pred
err_mean = np.mean(err)
err_std = np.std(err)

point_v_actual = error_point_v(scaled_feat, scaled_actual_l, polynomial=2)
point_v_pred = error_point_v(scaled_feat, scaled_pred_l, polynomial=2)



# To open saved model
# restored_model = load_model('my_model.keras')

# To open saved training history
# with open('history.pk', 'rb') as file_pi:
#     loaded_history = pickle.load(file_pi)
