#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

import rclpy
import numpy as np
import pandas as pd
import tensorflow as tf
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleRatesSetpoint
from px4_msgs.msg import VehicleAttitudeSetpoint


class OffboardControl(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

        self.NN1 = tf.keras.models.load_model('/home/thariqh15/ws_sensor_combined/src/ciis_drone/models/NN_IV_1_sim1.h5')
        self.NN2 = tf.keras.models.load_model('/home/thariqh15/ws_sensor_combined/src/ciis_drone/models/NN_IV_2_sim1.h5')
        self.NN3 = tf.keras.models.load_model('/home/thariqh15/ws_sensor_combined/src/ciis_drone/models/NN_IV_3_sim1.h5')
        self.NN4 = tf.keras.models.load_model('/home/thariqh15/ws_sensor_combined/src/ciis_drone/models/NN_IV_4_sim1.h5')
        self.NN5 = tf.keras.models.load_model('/home/thariqh15/ws_sensor_combined/src/ciis_drone/models/NN_IV_5_sim1.h5')
        self.NN6 = tf.keras.models.load_model('/home/thariqh15/ws_sensor_combined/src/ciis_drone/models/NN_IV_6_sim1.h5')

        dataAttitude = pd.read_csv("/home/thariqh15/ulog_file/2023-6-20/log_1_2023-6-2-01-55-11_vehicle_attitude_0.csv")
        # dataPosition = pd.read_csv("/home/thariqh15/ulog_file/2023-6-20/log_1_2023-6-2-01-55-11_vehicle_local_position_0.csv")
        dataPosition = pd.read_csv("/home/thariqh15/ulog_file/2023-6-20/log_1_2023-6-2-01-55-11_trajectory_setpoint_0.csv")
        len_dataPosition = len(dataPosition)
        len_dataAtt = len(dataAttitude)
        # self.x = np.array(dataPosition['x'])
        # self.y = np.array(dataPosition['y'])
        # self.z = np.array(dataPosition['z'])
        self.x = np.array(dataPosition['position[0]'])
        self.y = np.array(dataPosition['position[1]'])
        self.z = np.array(dataPosition['position[2]'])
        self.yaw = np.array(dataPosition['yaw'])
        error_percentage = 50
        error = np.random.uniform(-error_percentage, error_percentage, size=len(self.x)) / 100
        self.x = self.x + error
        error = np.random.uniform(-error_percentage, error_percentage, size=len(self.y)) / 100
        self.y = self.y + error
        error = np.random.uniform(-error_percentage, error_percentage, size=len(self.z)) / 100
        self.z = self.z + error
        error = np.random.uniform(-error_percentage, error_percentage, size=len(self.yaw)) / 100
        self.yaw = self.yaw + error


        # q0 = dataAttitude['q[0]']
        # q1 = dataAttitude['q[1]']
        # q2 = dataAttitude['q[2]']
        # q3 = dataAttitude['q[3]']

        # roll_input_data,pitch_input_data,yaw_input_data = self.quaternion_to_euler_angle_vectorized(q0,q1,q2,q3)
    
        # scaler = MinMaxScaler(feature_range=(-1,1))

        # avg_data = math.ceil(len_dataAtt/len_dataPosition)

        # data_roll = 0;data_roll_arr = []
        # data_pitch = 0;data_pitch_arr = []
        # data_yaw = 0;data_yawSin_arr = [];data_yawCos_arr = [];data_yaw_arr=[]

        # ## ROLL DATA
        # for i,v in roll_input_data.items():
        #     if i % avg_data != 0:
        #         data_roll += v
        #     else:
        #         data_roll = data_roll/avg_data
        #         data_roll_arr.append(data_roll)
        #         data_roll = 0

        # ## YAW DATA
        # for i,v in yaw_input_data.items():
        #     if i % avg_data != 0:
        #         data_yaw += v
        #     else:
        #         data_yaw = data_yaw/avg_data
        #         data_yaw_arr.append(data_yaw)
        #         data_yaw = 0

        # ## PITCH DATA
        # for i,v in pitch_input_data.items():
        #     if i % avg_data != 0:
        #         data_pitch += v
        #     else:
        #         data_pitch = data_pitch/avg_data
        #         data_pitch_arr.append(data_pitch)
        #         data_pitch = 0

        # data_roll_arr = np.array(data_roll_arr)
        # data_pitch_arr = np.array(data_pitch_arr)
        # data_yaw_arr = np.array(data_yaw_arr)

        # # roll = data_roll_arr
        # # pitch = data_pitch_arr
        # self.roll = scaler.fit_transform(data_roll_arr.reshape(-1,1))
        # self.pitch = scaler.fit_transform(data_pitch_arr.reshape(-1,1))
        # self.yawCos = np.cos((data_yaw_arr)*np.pi/180)
        # self.yawSin = np.sin((data_yaw_arr)*np.pi/180)
        # self.z1 = z[:39498]
        # yaw = scaler.fit_transform(data_yaw_arr.reshape(-1,1))
        # self.maxz = [np.max(x),np.max(y),np.max(z),np.max(yaw)]
        # self.minz = [np.min(x),np.min(y),np.min(z),np.min(yaw)]

        # self.x_NN1 = np.zeros(len(x))
        # self.y_NN1 = np.zeros(len(y))
        # self.z_NN1 = np.zeros(len(z))
        # self.yaw_NN1 = np.zeros(len(yaw))
        # self.x_NN2 = np.zeros(len(x))
        # self.y_NN2 = np.zeros(len(y))
        # self.z_NN2 = np.zeros(len(z))
        # self.yaw_NN2 = np.zeros(len(yaw))
        # self.x_NN3 = np.zeros(len(x))
        # self.y_NN3 = np.zeros(len(y))
        # self.z_NN3 = np.zeros(len(z))
        # self.yaw_NN3 = np.zeros(len(yaw))
        # self.x_NN4 = np.zeros(len(x))
        # self.y_NN4 = np.zeros(len(y))
        # self.z_NN4 = np.zeros(len(z))
        # self.yaw_NN4 = np.zeros(len(yaw))
        # self.x_NN5 = np.zeros(len(x))
        # self.y_NN5 = np.zeros(len(y))
        # self.z_NN5 = np.zeros(len(z))
        # self.yaw_NN5 = np.zeros(len(yaw))
        # self.roll_NN1 = np.zeros(len(self.roll))
        # self.pitch_NN2 = np.zeros(len(self.pitch))
        # self.yawCos_NN3 = np.zeros(len(self.yawCos))
        # self.yawSin_NN4 = np.zeros(len(self.yawSin))
        # self.z1_NN5 = np.zeros(len(z))

        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos_profile
        )
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.publisher_attitude = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile)
        # timer_period = 0.00725  # seconds
        timer_period = 0.2
        # timer_period = 1
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.dt = timer_period
        self.offboard_setpoint_counter = 0
        self.theta = 0.0
        self.radius = 10.0
        self.omega = 0.5
        self.added = 50
        self.data_counter = 0
        self.counter = 0
    
    def quaternion_to_euler_angle_vectorized(self, w, x, y, z):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.degrees(np.arctan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)

        t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
        Y = np.degrees(np.arcsin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.degrees(np.arctan2(t3, t4))

        return X, Y, Z
    
    def publish_vehicle_command(self,command,param1=0.0,param2=0.0):
        comm = VehicleCommand()
        comm.param1 = param1
        comm.param2 = param2
        comm.command = command
        comm.target_system = 1
        comm.target_component = 1
        comm.source_system = 1
        comm.source_component = 1
        comm.from_external = True
        comm.timestamp = int(Clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(comm)
    
    def motor_control(self,command,param1=0.0,param2=0.0,param3=0.0,param4=0.0):
        comm = VehicleCommand()
        comm.param1 = param1
        comm.param2 = param2
        comm.param3 = param3
        comm.param4 = param4
        comm.command = command
        comm.target_system = 1
        comm.target_component = 1
        comm.source_system = 1
        comm.source_component = 1
        comm.from_external = True
        comm.timestamp = int(Clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(comm)

    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        # print("NAV_STATUS: ", msg.nav_state)
        # print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state

    def publish_offboard_control_mode(self):
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position=True
        offboard_msg.velocity=False
        offboard_msg.acceleration=False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        self.publisher_offboard_mode.publish(offboard_msg)

    def cmdloop_callback(self):
        # Publish offboard control modes
        # self.get_logger().info("a")
        self.publish_offboard_control_mode()
        if self.offboard_setpoint_counter == 10:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,1.0)
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,1.0,6.0)

        
        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            # self.motor_control(VehicleCommand.VEHICLE_CMD_DO_REPEAT_SERVO,1.0,1500.0,2.0,10.0)
            # self.motor_control(VehicleCommand.VEHICLE_CMD_DO_REPEAT_SERVO,2.0,1500.0,2.0,10.0)
            # self.motor_control(VehicleCommand.VEHICLE_CMD_DO_REPEAT_SERVO,3.0,1500.0,2.0,10.0)
            # self.motor_control(VehicleCommand.VEHICLE_CMD_DO_REPEAT_SERVO,4.0,1500.0,2.0,10.0)
            trajectory_msg = TrajectorySetpoint()
            trajectory_msg.position[0] = self.x[self.data_counter]
            trajectory_msg.position[1] = self.y[self.data_counter]
            trajectory_msg.position[2] = self.z[self.data_counter]
            trajectory_msg.yaw = self.yaw[self.data_counter]
            trajectory_msg.timestamp = int(Clock().now().nanoseconds/1000)
            self.publisher_trajectory.publish(trajectory_msg)

            # if self.data_counter != 39498:
            #     if self.data_counter > 2:
            #         x_IV1 = np.zeros((3,5))
            #         x_IV2 = np.zeros((3,5))
            #         x_IV3 = np.zeros((3,5))
            #         x_IV4 = np.zeros((3,5))
            #         x_IV5 = np.zeros((3,5))

            #         x_IV1[:,0] = self.roll[self.counter:self.counter+3].flatten()
            #         x_IV1[:,1] = self.x_NN1[self.counter:self.counter+3].flatten()
            #         x_IV1[:,2] = self.y_NN1[self.counter:self.counter+3].flatten()
            #         x_IV1[:,3] = self.z_NN1[self.counter:self.counter+3].flatten()
            #         x_IV1[:,4] = self.yaw_NN1[self.counter:self.counter+3].flatten()

            #         x_IV2[:,0] = self.pitch[self.counter:self.counter+3].flatten()
            #         x_IV2[:,1] = self.x_NN2[self.counter:self.counter+3].flatten()
            #         x_IV2[:,2] = self.y_NN2[self.counter:self.counter+3].flatten()
            #         x_IV2[:,3] = self.z_NN2[self.counter:self.counter+3].flatten()
            #         x_IV2[:,4] = self.yaw_NN2[self.counter:self.counter+3].flatten()

            #         x_IV3[:,0] = self.yawCos[self.counter:self.counter+3].flatten()
            #         x_IV3[:,1] = self.x_NN3[self.counter:self.counter+3].flatten()
            #         x_IV3[:,2] = self.y_NN3[self.counter:self.counter+3].flatten()
            #         x_IV3[:,3] = self.z_NN3[self.counter:self.counter+3].flatten()
            #         x_IV3[:,4] = self.yaw_NN3[self.counter:self.counter+3].flatten()

            #         x_IV4[:,0] = self.yawSin[self.counter:self.counter+3].flatten()
            #         x_IV4[:,1] = self.x_NN4[self.counter:self.counter+3].flatten()
            #         x_IV4[:,2] = self.y_NN4[self.counter:self.counter+3].flatten()
            #         x_IV4[:,3] = self.z_NN4[self.counter:self.counter+3].flatten()
            #         x_IV4[:,4] = self.yaw_NN4[self.counter:self.counter+3].flatten()

            #         x_IV5[:,0] = self.z1[self.counter:self.counter+3].flatten()
            #         x_IV5[:,1] = self.x_NN5[self.counter:self.counter+3].flatten()
            #         x_IV5[:,2] = self.y_NN5[self.counter:self.counter+3].flatten()
            #         x_IV5[:,3] = self.z_NN5[self.counter:self.counter+3].flatten()
            #         x_IV5[:,4] = self.yaw_NN5[self.counter:self.counter+3].flatten()
                    
            #         x_IV1 = [x_IV1]
            #         x_IV2 = [x_IV2]
            #         x_IV3 = [x_IV3]
            #         x_IV4 = [x_IV4]
            #         x_IV5 = [x_IV5]

            #         x_IV1 = np.asarray(x_IV1,dtype='float64')
            #         x_IV2 = np.asarray(x_IV2,dtype='float64')
            #         x_IV3 = np.asarray(x_IV3,dtype='float64')
            #         x_IV4 = np.asarray(x_IV4,dtype='float64')
            #         x_IV5 = np.asarray(x_IV5,dtype='float64')

            #         pred1 = self.NN1.predict(x=[x_IV1])
            #         pred2 = self.NN2.predict(x=[x_IV2])
            #         pred3 = self.NN3.predict(x=[x_IV3])
            #         pred4 = self.NN4.predict(x=[x_IV4])
            #         pred5 = self.NN5.predict(x=[x_IV5])

            #         self.x_NN1[self.counter+3] = pred1[:,0][0]
            #         self.y_NN1[self.counter+3] = pred1[:,1][0]
            #         self.z_NN1[self.counter+3] = pred1[:,2][0]
            #         self.yaw_NN1[self.counter+3] = pred1[:,3][0]

            #         self.x_NN2[self.counter+3] = pred2[:,0][0]
            #         self.y_NN2[self.counter+3] = pred2[:,1][0]
            #         self.z_NN2[self.counter+3] = pred2[:,2][0]
            #         self.yaw_NN2[self.counter+3] = pred2[:,3][0]

            #         self.x_NN3[self.counter+3] = pred3[:,0][0]
            #         self.y_NN3[self.counter+3] = pred3[:,1][0]
            #         self.z_NN3[self.counter+3] = pred3[:,2][0]
            #         self.yaw_NN3[self.counter+3] = pred3[:,3][0]

            #         self.x_NN4[self.counter+3] = pred4[:,0][0]
            #         self.y_NN4[self.counter+3] = pred4[:,1][0]
            #         self.z_NN4[self.counter+3] = pred4[:,2][0]
            #         self.yaw_NN4[self.counter+3] = pred4[:,3][0]

            #         self.x_NN5[self.counter+3] = pred5[:,0][0]
            #         self.y_NN5[self.counter+3] = pred5[:,1][0]
            #         self.z_NN5[self.counter+3] = pred5[:,2][0]
            #         self.yaw_NN5[self.counter+3] = pred5[:,3][0]

            #         x_NN = np.zeros((39498,20))
            #         x_NN[:,0] = self.x_NN1[self.counter]
            #         x_NN[:,1] = self.y_NN1[self.counter]
            #         x_NN[:,2] = self.z_NN1[self.counter]
            #         x_NN[:,3] = self.yaw_NN1[self.counter]
            #         x_NN[:,4] = self.x_NN2[self.counter]
            #         x_NN[:,5] = self.y_NN2[self.counter]
            #         x_NN[:,6] = self.z_NN2[self.counter]
            #         x_NN[:,7] = self.yaw_NN2[self.counter]
            #         x_NN[:,8] = self.x_NN3[self.counter]
            #         x_NN[:,9] = self.y_NN3[self.counter]
            #         x_NN[:,10] = self.z_NN3[self.counter]
            #         x_NN[:,11] = self.yaw_NN3[self.counter]
            #         x_NN[:,12] = self.x_NN4[self.counter]
            #         x_NN[:,13] = self.y_NN4[self.counter]
            #         x_NN[:,14] = self.z_NN4[self.counter]
            #         x_NN[:,15] = self.yaw_NN4[self.counter]
            #         x_NN[:,16] = self.x_NN5[self.counter]
            #         x_NN[:,17] = self.y_NN5[self.counter]
            #         x_NN[:,18] = self.z_NN5[self.counter]
            #         x_NN[:,19] = self.yaw_NN5[self.counter]
            #         pred_NN = self.NN6.predict(x = [x_NN])
            #         self.hasil_x = (pred_NN[:,0][0] + 1)*(self.maxz[0] - self.minz[0])/2 + self.minz[0]
            #         self.hasil_y = (pred_NN[:,1][0] + 1)*(self.maxz[1] - self.minz[1])/2 + self.minz[1]
            #         self.hasil_z = (pred_NN[:,2][0] + 1)*(self.maxz[2] - self.minz[2])/2 + self.minz[2]
            #         self.hasil_yaw = (pred_NN[:,3][0] + 1)*(self.maxz[3] - self.minz[3])/2 + self.minz[3]
            #         self.counter += 1

            # self.theta = self.theta + self.omega * self.dt
            # rates_msg = VehicleAttitudeSetpoint()
            # rates_msg.roll_body = self.roll[self.data_counter]
            # rates_msg.pitch_body = self.pitch[self.data_counter]
            # rates_msg.yaw_body = self.yaw[self.data_counter]
            # # rates_msg.thrust_body[2] = self.thrust[self.data_counter]
            # rates_msg.timestamp = int(Clock().now().nanoseconds/1000)
            # self.publisher_attitude.publish(rates_msg)
            # print(self.z_s[self.data_counter])
            self.data_counter+=1
            print("Data Counter " + str(self.data_counter))
        
        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter+=1
            print(self.offboard_setpoint_counter)


def main(args=None):
    rclpy.init(args=args)
    
    offboard_control = OffboardControl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()