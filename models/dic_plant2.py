import socket
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import json

def Main():

    # tf.config.set_visible_devices([], 'GPU')
   
    host = '192.168.0.1' #Server ip
    port = 4000

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))

    NN_PWM1 = tf.keras.models.load_model('saved_model/NN_ID_1_new5.h5')
    NN_PWM2 = tf.keras.models.load_model('saved_model/NN_ID_2_new5.h5')
    NN_PWM3 = tf.keras.models.load_model('saved_model/NN_ID_3_new5.h5')
    NN_PWM4 = tf.keras.models.load_model('saved_model/NN_ID_4_new5.h5')
    NN_ALL_PWM = tf.keras.models.load_model('saved_model/NN_ID_6_new5.h5')
    
    len_data = 570
    pwm1 = np.zeros(len_data)
    pwm2 = np.zeros(len_data)
    pwm3 = np.zeros(len_data)
    pwm4 = np.zeros(len_data)

    roll_pred = np.zeros(len_data)
    pitch_pred = np.zeros(len_data)
    yawCos_pred = np.zeros(len_data)
    yawSin_pred = np.zeros(len_data)
    speed_pred = np.zeros(len_data)

    n = 0
    counter = 0
    ### PWM HAPUS
    processing_time = []
    data_attitude_input = pd.read_csv("./data/04_40_31_vehicle_rates_setpoint_0.csv")
    data_position_input = pd.read_csv("./data/04_40_31_vehicle_local_position_0.csv")
    # pwm_0 = np.array(data_pwm['output[0]'])
    # pwm_1 = np.array(data_pwm['output[1]'])
    # pwm_2 = np.array(data_pwm['output[2]'])
    # pwm_3 = np.array(data_pwm['output[3]'])
    pos_x = np.array(data_position_input['x'])
    pos_y = np.array(data_position_input['y'])
    pos_z = np.array(data_position_input['z'])
    pos_x[:len_data]
    pos_y[:len_data]
    pos_z[:len_data]
    speed = np.sqrt(np.square(pos_x)+np.square(pos_y)+np.square(pos_z))
    # zscore(pwm_0);zscore(pwm_1);zscore(pwm_2);zscore(pwm_3)
    zscore(speed)
    scaler = MinMaxScaler(feature_range=(-1,1))

    # pwm1 = scaler.fit_transform(pwm_0.reshape(-1,1))
    # pwm2 = scaler.fit_transform(pwm_1.reshape(-1,1))
    # pwm3 = scaler.fit_transform(pwm_2.reshape(-1,1))
    # pwm4 = scaler.fit_transform(pwm_3.reshape(-1,1))
    speed = scaler.fit_transform(speed.reshape(-1,1))

    ## DATA ROLL, PITCH, YAW
    avg_data = int(len(data_attitude_input)/len_data)
    roll_input_data = data_attitude_input['roll']
    pitch_input_data = data_attitude_input['pitch']
    yaw_input_data = data_attitude_input['yaw']

    data_roll = 0;data_roll_arr = []
    data_pitch = 0;data_pitch_arr = []
    data_yaw = 0;data_yaw_arr=[]

    ## ROLL DATA
    for i,v in roll_input_data.items():
        if i % 5 != 0:
            data_roll += v
        else:
            data_roll = data_roll/avg_data
            data_roll_arr.append(data_roll)
            data_roll = 0

    ## YAW DATA
    for i,v in yaw_input_data.items():
        if i % 5 != 0:
            data_yaw += v
        else:
            data_yaw = data_yaw/avg_data
            data_yaw_arr.append(data_yaw)
            data_yaw = 0

    ## PITCH DATA
    for i,v in pitch_input_data.items():
        if i % 5 != 0:
            data_pitch += v
        else:
            data_pitch = data_pitch/avg_data
            data_pitch_arr.append(data_pitch)
            data_pitch = 0

    data_roll_arr = np.array(data_roll_arr)
    data_pitch_arr = np.array(data_pitch_arr)
    data_yaw_arr = np.array(data_yaw_arr)
    zscore(data_roll_arr);zscore(data_pitch_arr);zscore(data_yaw_arr)

    roll = scaler.fit_transform(data_roll_arr.reshape(-1,1))
    pitch = scaler.fit_transform(data_pitch_arr.reshape(-1,1))
    yawCos = np.cos(np.rad2deg(data_yaw_arr))
    yawSin = np.sin(np.rad2deg(data_yaw_arr))

    pitch_NN1 = np.zeros(len(pitch))
    yawCos_NN1 = np.zeros(len(yawCos))
    yawSin_NN1 = np.zeros(len(yawSin))
    speed_NN1 = np.zeros(len(speed))
    roll_NN1 = np.zeros(len(roll))

    pitch_NN2 = np.zeros(len(pitch))
    yawCos_NN2 = np.zeros(len(yawCos))
    yawSin_NN2 = np.zeros(len(yawSin))
    speed_NN2 = np.zeros(len(speed))
    roll_NN2 = np.zeros(len(roll))

    pitch_NN3 = np.zeros(len(pitch))
    yawCos_NN3 = np.zeros(len(yawCos))
    yawSin_NN3 = np.zeros(len(yawSin))
    speed_NN3 = np.zeros(len(speed))
    roll_NN3 = np.zeros(len(roll))

    pitch_NN4 = np.zeros(len(pitch))
    yawCos_NN4 = np.zeros(len(yawCos))
    yawSin_NN4 = np.zeros(len(yawSin))
    speed_NN4 = np.zeros(len(speed))
    roll_NN4 = np.zeros(len(roll))

    print("Server Started")
    prev_time = int(round(time.time()*1000))
        
    while True:
        curr_time = int(round(time.time()*1000))
        delta = curr_time - prev_time
        # print("Sampling Time: ", delta," ms")
        processing_time.append(delta)        
        data, addr = s.recvfrom(1024)
        data = data.decode('utf-8')

        if data == "end" or n == len_data:
            print("STOP!")
            break
        dataReceived = json.loads(data)
        if n < len_data:
            pwm1[n] = dataReceived['output'][0]
            pwm2[n] = dataReceived['output'][1]
            pwm3[n] = dataReceived['output'][2]
            pwm4[n] = dataReceived['output'][3]
        if n > 2 and n < len_data - 3:
            x_ID1 = np.zeros((3,6))
            x_ID2 = np.zeros((3,6))
            x_ID3 = np.zeros((3,6))
            x_ID4 = np.zeros((3,6))      
            x_ID1[:,0] = roll_NN1[counter:counter+3].flatten()
            x_ID1[:,1] = pitch_NN1[counter:counter+3].flatten()
            x_ID1[:,2] = yawCos_NN1[counter:counter+3].flatten()
            x_ID1[:,3] = yawSin_NN1[counter:counter+3].flatten()
            x_ID1[:,4] = speed_NN1[counter:counter+3].flatten()
            x_ID1[:,5] = pwm1[counter:counter+3].flatten()

            x_ID2[:,0] = roll_NN2[counter:counter+3].flatten()
            x_ID2[:,1] = pitch_NN2[counter:counter+3].flatten()
            x_ID2[:,2] = yawCos_NN2[counter:counter+3].flatten()
            x_ID2[:,3] = yawSin_NN2[counter:counter+3].flatten()
            x_ID2[:,4] = speed_NN2[counter:counter+3].flatten()
            x_ID2[:,5] = pwm2[counter:counter+3].flatten()

            x_ID3[:,0] = roll_NN3[counter:counter+3].flatten()
            x_ID3[:,1] = pitch_NN3[counter:counter+3].flatten()
            x_ID3[:,2] = yawCos_NN3[counter:counter+3].flatten()
            x_ID3[:,3] = yawSin_NN3[counter:counter+3].flatten()
            x_ID3[:,4] = speed_NN3[counter:counter+3].flatten()
            x_ID3[:,5] = pwm3[counter:counter+3].flatten()

            x_ID4[:,0] = roll_NN4[counter:counter+3].flatten()
            x_ID4[:,1] = pitch_NN4[counter:counter+3].flatten()
            x_ID4[:,2] = yawCos_NN4[counter:counter+3].flatten()
            x_ID4[:,3] = yawSin_NN4[counter:counter+3].flatten()
            x_ID4[:,4] = speed_NN4[counter:counter+3].flatten()
            x_ID4[:,5] = pwm4[counter:counter+3].flatten()
            
            x_ID1 = [x_ID1]
            x_ID2 = [x_ID2]
            x_ID3 = [x_ID3]
            x_ID4 = [x_ID4]

            x_ID1 = np.asarray(x_ID1,dtype='float64')
            x_ID2 = np.asarray(x_ID2,dtype='float64')
            x_ID3 = np.asarray(x_ID3,dtype='float64')
            x_ID4 = np.asarray(x_ID4,dtype='float64')
            hasilPWM1 = NN_PWM1.predict(x=[x_ID1])
            hasilPWM2 = NN_PWM2.predict(x=[x_ID2])
            hasilPWM3 = NN_PWM3.predict(x=[x_ID3])
            hasilPWM4 = NN_PWM4.predict(x=[x_ID4])
            
            roll_NN1[counter+3] = hasilPWM1[:,0][0]
            pitch_NN1[counter+3] = hasilPWM1[:,1][0]
            yawCos_NN1[counter+3] = hasilPWM1[:,2][0]
            yawSin_NN1[counter+3] = hasilPWM1[:,3][0]
            speed_NN1[counter+3] = hasilPWM1[:,4][0]

            roll_NN2[counter+3] = hasilPWM2[:,0][0]
            pitch_NN2[counter+3] = hasilPWM2[:,1][0]
            yawCos_NN2[counter+3] = hasilPWM2[:,2][0]
            yawSin_NN2[counter+3] = hasilPWM2[:,3][0]
            speed_NN2[counter+3] = hasilPWM2[:,4][0]

            roll_NN3[counter+3] = hasilPWM3[:,0][0]
            pitch_NN3[counter+3] = hasilPWM3[:,1][0]
            yawCos_NN3[counter+3] = hasilPWM3[:,2][0]
            yawSin_NN3[counter+3] = hasilPWM3[:,3][0]
            speed_NN3[counter+3] = hasilPWM3[:,4][0]

            roll_NN4[counter+3] = hasilPWM4[:,0][0]
            pitch_NN4[counter+3] = hasilPWM4[:,1][0]
            yawCos_NN4[counter+3] = hasilPWM4[:,2][0]
            yawSin_NN4[counter+3] = hasilPWM4[:,3][0]
            speed_NN4[counter+3] = hasilPWM4[:,4][0]

            x_NN = np.zeros((1,20))
            x_NN[:,0] = roll_NN1[counter]
            x_NN[:,1] = pitch_NN1[counter]
            x_NN[:,2] = yawCos_NN1[counter]
            x_NN[:,3] = yawSin_NN1[counter]
            x_NN[:,4] = speed_NN1[counter]
            x_NN[:,5] = roll_NN2[counter]
            x_NN[:,6] = pitch_NN2[counter]
            x_NN[:,7] = yawCos_NN2[counter]
            x_NN[:,8] = yawSin_NN2[counter]
            x_NN[:,9] = speed_NN2[counter]
            x_NN[:,10] = roll_NN3[counter]
            x_NN[:,11] = pitch_NN3[counter]
            x_NN[:,12] = yawCos_NN3[counter]
            x_NN[:,13] = yawSin_NN3[counter]
            x_NN[:,14] = speed_NN3[counter]
            x_NN[:,15] = roll_NN4[counter]
            x_NN[:,16] = pitch_NN4[counter]
            x_NN[:,17] = yawCos_NN4[counter]
            x_NN[:,18] = yawSin_NN4[counter]
            x_NN[:,19] = speed_NN4[counter]

            hasilAll = NN_ALL_PWM.predict(x=[x_NN])
            # print(roll_in)
            roll_pred[counter] = hasilAll[:,0][0]
            pitch_pred[counter] = hasilAll[:,1][0]
            yawCos_pred[counter] = hasilAll[:,2][0]
            yawSin_pred[counter] = hasilAll[:,3][0]
            speed_pred[counter] = hasilAll[:,4][0]
            counter = counter + 1
            # dataKirim = json.dumps({"output":hasilAll[0].tolist()})
            # s.sendto(dataKirim.encode('utf-8'), addr)
        elif n > len_data - 3 and counter < len_data:
            x_NN = np.zeros((1,20))
            x_NN[:,0] = roll_NN1[counter]
            x_NN[:,1] = pitch_NN1[counter]
            x_NN[:,2] = yawCos_NN1[counter]
            x_NN[:,3] = yawSin_NN1[counter]
            x_NN[:,4] = speed_NN1[counter]
            x_NN[:,5] = roll_NN2[counter]
            x_NN[:,6] = pitch_NN2[counter]
            x_NN[:,7] = yawCos_NN2[counter]
            x_NN[:,8] = yawSin_NN2[counter]
            x_NN[:,9] = speed_NN2[counter]
            x_NN[:,10] = roll_NN3[counter]
            x_NN[:,11] = pitch_NN3[counter]
            x_NN[:,12] = yawCos_NN3[counter]
            x_NN[:,13] = yawSin_NN3[counter]
            x_NN[:,14] = speed_NN3[counter]
            x_NN[:,15] = roll_NN4[counter]
            x_NN[:,16] = pitch_NN4[counter]
            x_NN[:,17] = yawCos_NN4[counter]
            x_NN[:,18] = yawSin_NN4[counter]
            x_NN[:,19] = speed_NN4[counter]

            hasilAll = NN_ALL_PWM.predict(x=[x_NN])
            roll_pred[counter] = hasilAll[:,0][0]
            pitch_pred[counter] = hasilAll[:,1][0]
            yawCos_pred[counter] = hasilAll[:,2][0]
            yawSin_pred[counter] = hasilAll[:,3][0]
            speed_pred[counter] = hasilAll[:,4][0]
            counter = counter + 1
            # dataKirim = json.dumps({"output":hasilAll[0].tolist()})
            # s.sendto(dataKirim.encode('utf-8'), addr)
        n = n+1
        prev_time = curr_time
        
    s.close()
    
    plt.figure()
    plt.subplot(511)
    plt.plot(roll_NN1,label = 'prediction')
    plt.plot(roll,label = 'real_data')
    plt.subplot(512)
    plt.plot(pitch_NN1,label = 'prediction')
    plt.plot(pitch,label = 'real_data')
    plt.subplot(513)
    plt.plot(yawCos_NN1,label = 'prediction')
    plt.plot(yawCos,label = 'real_data')
    plt.subplot(514)
    plt.plot(yawSin_NN1,label = 'prediction')
    plt.plot(yawSin,label = 'real_data')
    plt.subplot(515)
    plt.plot(speed_NN1,label = 'prediction')
    plt.plot(speed,label = 'real_data')
    plt.tight_layout()
    plt.legend()
    plt.show()

    plt.figure()
    plt.subplot(511)
    plt.plot(roll_NN2,label = 'prediction')
    plt.plot(roll,label = 'real_data')
    plt.subplot(512)
    plt.plot(pitch_NN2,label = 'prediction')
    plt.plot(pitch,label = 'real_data')
    plt.subplot(513)
    plt.plot(yawCos_NN2,label = 'prediction')
    plt.plot(yawCos,label = 'real_data')
    plt.subplot(514)
    plt.plot(yawSin_NN2,label = 'prediction')
    plt.plot(yawSin,label = 'real_data')
    plt.subplot(515)
    plt.plot(speed_NN2,label = 'prediction')
    plt.plot(speed,label = 'real_data')
    plt.tight_layout()
    plt.legend()
    plt.show()

    plt.figure()
    plt.subplot(511)
    plt.plot(roll_NN3,label = 'prediction')
    plt.plot(roll,label = 'real_data')
    plt.subplot(512)
    plt.plot(pitch_NN3,label = 'prediction')
    plt.plot(pitch,label = 'real_data')
    plt.subplot(513)
    plt.plot(yawCos_NN3,label = 'prediction')
    plt.plot(yawCos,label = 'real_data')
    plt.subplot(514)
    plt.plot(yawSin_NN3,label = 'prediction')
    plt.plot(yawSin,label = 'real_data')
    plt.subplot(515)
    plt.plot(speed_NN3,label = 'prediction')
    plt.plot(speed,label = 'real_data')
    plt.tight_layout()
    plt.legend()
    plt.show()

    plt.figure()
    plt.subplot(511)
    plt.plot(roll_NN4,label = 'prediction')
    plt.plot(roll,label = 'real_data')
    plt.subplot(512)
    plt.plot(pitch_NN4,label = 'prediction')
    plt.plot(pitch,label = 'real_data')
    plt.subplot(513)
    plt.plot(yawCos_NN4,label = 'prediction')
    plt.plot(yawCos,label = 'real_data')
    plt.subplot(514)
    plt.plot(yawSin_NN4,label = 'prediction')
    plt.plot(yawSin,label = 'real_data')
    plt.subplot(515)
    plt.plot(speed_NN4,label = 'prediction')
    plt.plot(speed,label = 'real_data')
    plt.tight_layout()
    plt.legend()
    plt.show()

    plt.figure()
    plt.subplot(511)
    plt.plot(roll_pred,label = 'prediction')
    plt.plot(roll,label = 'real_data')
    plt.subplot(512)
    plt.plot(pitch_pred,label = 'prediction')
    plt.plot(pitch,label = 'real_data')
    plt.subplot(513)
    plt.plot(yawCos_pred,label = 'prediction')
    plt.plot(yawCos,label = 'real_data')
    plt.subplot(514)
    plt.plot(yawSin_pred,label = 'prediction')
    plt.plot(yawSin,label = 'real_data')
    plt.subplot(515)
    plt.plot(speed_pred,label = 'prediction')
    plt.plot(speed,label = 'real_data')
    plt.tight_layout()
    plt.legend()
    plt.show()

    # ## convert your array into a dataframe
    # df1 = pd.DataFrame(r)
    # df2 = pd.DataFrame(y)
    # df3 = pd.DataFrame(processing_time_np)

    # ## save to xlsx file

    # filepath1 = 'y_ref_DIC_hil.xlsx'
    # filepath2 = 'y_actual_DIC_hil.xlsx'
    # filepath3 = 'processing_time_tensorflow.xlsx'

    # df1.to_excel(filepath1, index=False)
    # df2.to_excel(filepath2, index=False)
    # df3.to_excel(filepath3, index=False)
    

if __name__=='__main__':
    Main()
