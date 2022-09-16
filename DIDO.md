# DIDO

## dataset

* Each folder contains a ‘’**data.hdf5**‘’ containing：

​	（1）time **ts**

​	（2）Ground truth value of IMU frame in the gravity frame：acceleration **gt_acc**，position **gt_p**，linear velocity **gt_v**，attitude **gt_q** (in w,x,y,z form)

​	（3）Ground truth value of IMU frame in the IMU frame：angular velocity **gt_gyr**，

​	（4）IMU measurements：acceleration **acc**，angular velocity **gyr**

* gen_list.txt：contains all the name of data
* train.txtx：training data
* val.txt：validation data
* test：testing data

NOTE: The content in the existed train.txt, val.txt and test.txt are just examples. You can choose your own training, validation and testing data.

## De_bias_acc

The neural network to reduce bias of the accelerator measurments

Execute De_bias_acc_main.py for **training** 

```shell
 
cd De_bias_acc/src
python3 De_bias_acc_main.py \
--mode train \
--root_dir ../../dataset \
--train_list ../../dataset/train.txt \
--val_list ../../dataset/val.txt \
--out_dir ../train_outputs
```

--root_dir ../../dataset \ # root directory of data
--train_list ../../dataset/train.txt \ # the list of training data
--val_list ../../dataset/val.txt \  # the list of validation data
--out_dir ../train_outputs \  # output directory

Execute De_bias_acc_main.py for **testing** 

```
cd De_bias_acc/src
python3 De_bias_acc_main.py \
--mode test \
--root_dir ../../dataset \
--test_list ../../dataset/test.txt \
--model_path ../train_outputs/checkpoints/checkpoint_*.pt \
--out_dir ../test_outputs
```

--root_dir ../../dataset \ # root directory of data
--test_list ../../dataset/test.txt \ # the list of testing data
--model_path ../train_outputs/checkpoints/checkpoint_*.pt \ # saving path of de_acc_bias_net
--out_dir test_outputs  # output directory (there are the velocity figures obtained by acceleration integration using the true attitude in this directory)

# De_bias_gyr

The neural network to reduce bias of the gyroscope measurments

Execute the De_bias_gyr_main.py for **training**

```
cd De_bias_gyr/src
python3 De_bias_gyr_main.py \
--mode train \
--root_dir ../../dataset \
--train_list ../../dataset/train.txt \
--val_list ../../dataset/val.txt \
--out_dir ../train_outputs
```

--root_dir ../../dataset \ # root directory of data
--train_list ../../dataset/train.txt \ # the list of training data
--val_list ../../dataset/val.txt \  # the list of validation data
--out_dir ../train_outputs \  # output directory

# Rotation_ekf

First, using the De_bias_net to estimate the acceleration and angular velocity.

```
cd Rotation_ekf/src
python3 generate_net_acc_net_gyr.py \
--root_dir ../../dataset \
--network_acc_path ../../De_bias_acc/train_outputs/checkpoints/checkpoint_*.pt \
--network_gyr_path ../../De_bias_gyr/train_outputs/checkpoints/checkpoint_*.pt \
--test_list ../../dataset/gen_list.txt \
--out_dir ../output
```

--root_dir ../../dataset \ # root directory of data
--network_acc_path ../../De_bias_acc/train_outputs/checkpoints/checkpoint_*.pt # saving path of de_acc_bias_net
--network_gyr_path ../../De_bias_acc/train_outputs/checkpoints/checkpoint_*.pt # saving path of de_gyr_bias_net
--test_list ../../dataset/gen_list.txt # the list of all the data in the dataset
--out_dir ../output # output directory



NOTE: The produced acceleration and angular velocity will be saved in directory “**../output/net_acc**” and “**../output/net_gyr**”



Second, obtain the attitude after gravity alignment update.

```
cd Rotation_ekf/src
python3 Rotation_stage.py \
--root_dir ../../dataset \
--network_acc_out_path ../output/net_acc/ \
--network_gyr_out_path ../output/net_gyr/ \
--test_list ../../dataset/test.txt \
--out_dir ../output
```

--root_dir ../../dataset \ # root directory of data
--network_acc_out_path ../output/net_acc/ \ # saving path of acceleration after using de_bias_net
--network_gyr_out_path ../output/net_gyr/ \# saving path of angular velocity after using de_bias_net
--test_list ../../dataset/test.txt \ # the testing data
--out_dir ../output   # output directory



NOTE: There are some results in the output directory

* **ekf_q**:  attitude estimated by rotation ekf
* **euler**: figures of the attitude estimation (**euler_pred**: the euler angle estimated by directly new angular velocity integration;  **euler_gt_euler**: the ground truth euler angle; **ekf_euler**: euler angle estimated by  rotation ekf)
* **euler_pred_error**: figures of the attitude estimation error (**euler_pred_error**: error of euler_pred; **ekf_euler_error**: error of ekf_euler)

# V_P_net

The attitude estimated from rotation_ekf is used in validation and testing

1、Training

```
# training x axis
cd V_P_net/src
python3 V_P_net_main.py \
--mode train \
--root_dir ../../dataset \
--train_list ../../dataset/train.txt \
--val_list ../../dataset/val.txt \
--out_dir ../train_outputs_x \
--train_axis x_axis
```

```
# training y axis
cd V_P_net/src
python3 V_P_net_main.py \
--mode train \
--root_dir ../../dataset \
--train_list ../../dataset/train.txt \
--val_list ../../dataset/val.txt \
--out_dir ../train_outputs_y \
--train_axis y_axis
```

```
# training z axis
cd V_P_net/src
python3 V_P_net_main.py \
--mode train \
--root_dir ../../dataset \
--train_list ../../dataset/train.txt \
--val_list ../../dataset/val.txt \
--out_dir ../train_outputs_z \
--train_axis z_axis
```

--root_dir ../../dataset \ # root directory of data
--train_list ../../dataset/train.txt \ # the list of training data
--val_list ../../dataset/val.txt \ # the list of validation data
--out_dir ../train_outputs_x  # output directory
--train_axis x_axis  # the axis of velocity and position are trained

2、Testing

```
cd V_P_net/src
python3 V_P_net_main.py \
--mode test \
--root_dir ../../dataset \
--test_list ../../dataset/train.txt \
--out_dir ../test_outputs \
--x_model ../train_outputs_x/checkpoints/checkpoint_*.pt \
--y_model ../train_outputs_y/checkpoints/checkpoint_*.pt \
--z_model ../train_outputs_z/checkpoints/checkpoint_*.pt
```

--root_dir ../../dataset \ # root directory of data
--test_list ../../dataset/train.txt \ # the list of testing data
--out_dir ../test_outputs  # output directory
--x_model ../train_outputs_x/checkpoints/checkpoint*.pt \ # the saving path of v_p_net about x axis
--y_model ../train_outputs_y/checkpoints/checkpoint*.pt \ # the saving path of v_p_net about y axis
--z_model ../train_outputs_z/checkpoints/checkpoint_*.pt \ # the saving path of v_p_net about z axis



NOTE: There are some results in the output directory

* **vp**：the mean value of estimated linear velocity and position

* **vp_cov**：the covariance value of estimated linear velocity and position

* **imu_p_cov_in_world_frame**：figure of position covariance

* **imu_p_error_in_world_frame**：figure of position estimation error

* **imu_p_in_world_frame**：figure of position estimation

* **imu_v_cov_in_world_frame**：figure of linear velocity covariance

* **imu_v_error_in_world_frame**：figure of linear velocity estimation error

* **imu_v_in_world_frame**：figure of velocity estimation estimation

# Res_dynamic

Training

```

cd Res_dynamic/src
python3 Res_dynamic_main.py \
--mode train \
--root_dir ../../dataset \
--train_list ../../dataset/train.txt \
--val_list ../../dataset/val.txt \
--out_dir ../train_outputs
```

--root_dir ../../dataset \ # root directory of data
--train_list ../../dataset/train.txt \ # the list of training data
--val_list ../../dataset/val.txt \  # the list of validation data
--out_dir train_outputs \  # output directory

# Translation_ekf

```
cd Translation_ekf/src
python3 Translation_stage.py \
--root_dir ../../dataset \
--test_list ../../dataset/test.txt \
--out_dir ../output \
--network_dyn_path ../../Res_dynamic/train_outputs/checkpoints/checkpoint_*.pt \
--network_v_p_path ../../V_P_net/src/test_outputs/vp/ \
--network_acc_path ../../Rotation_ekf/output/net_acc/ \
--network_gyr_path ../../Rotation_ekf/output/net_gyr/ \
--network_q_path ../../Rotation_ekf/output/ekf_q/ \
```

--root_dir ../../dataset \ # root directory of data
--test_list ../../dataset/test.txt \  # the list of testing data
--out_dir ../output \  # output directory
--network_dyn_path ../../Res_dynamic/train_outputs/checkpoints/checkpoint_*.pt # the path of res_dynamics_net
--network_v_p_path ../../V_P_net/src/test_outputs/vp/ # the path of the result of v_p_net
--network_acc_path ../../Rotation_ekf/output/net_acc/ # the path of acceleration modified by de_bias net
--network_gyr_path ../../Rotation_ekf/output/net_gyr/ # the path of angular velocity modified by de_bias net
--network_q_path ../../Rotation_ekf/output/ekf_q/ # the path of attitude got from rotation ekf