import socket
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore


def Main():

    # tf.config.set_visible_devices([], 'GPU')
   
    host = '192.168.0.1' #Server ip
    port = 4000

    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # s.bind((host, port))

    NN_PWM1 = tf.keras.models.load_model('saved_model/NN_ID_1_new2')
    NN_PWM2 = tf.keras.models.load_model('saved_model/NN_ID_2_new2')
    NN_PWM3 = tf.keras.models.load_model('saved_model/NN_ID_3_new2')
    NN_PWM4 = tf.keras.models.load_model('saved_model/NN_ID_4_new2')
    NN_ALL_PWM = tf.keras.models.load_model('saved_model/NN_ID_6_new2')
    
    len_data = 570
    # pwm1 = np.zeros(len_data)
    # pwm2 = np.zeros(len_data)
    # pwm3 = np.zeros(len_data)
    # pwm4 = np.zeros(len_data)
    roll_in = np.zeros((4,len_data))
    pitch_in = np.zeros((4,len_data))
    yawCos_in = np.zeros((4,len_data))
    yawSin_in = np.zeros((4,len_data))
    speed_in = np.zeros((4,len_data))

    roll_pred = np.zeros(len_data)
    pitch_pred = np.zeros(len_data)
    yawCos_pred = np.zeros(len_data)
    yawSin_pred = np.zeros(len_data)
    speed_pred = np.zeros(len_data)

    n = 0
    mn = 0
    counter = 0
    ### PWM HAPUS
    processing_time = []
    data_pwm = pd.read_csv("./data/04_40_31_actuator_outputs_0.csv")
    data_attitude_input = pd.read_csv("./data/04_40_31_vehicle_rates_setpoint_0.csv")
    data_position_input = pd.read_csv("./data/04_40_31_vehicle_local_position_0.csv")
    pwm_0 = np.array(data_pwm['output[0]'])
    pwm_1 = np.array(data_pwm['output[1]'])
    pwm_2 = np.array(data_pwm['output[2]'])
    pwm_3 = np.array(data_pwm['output[3]'])
    pos_x = np.array(data_position_input['x'])
    pos_y = np.array(data_position_input['y'])
    pos_z = np.array(data_position_input['z'])
    pos_x[:len_data]
    pos_y[:len_data]
    pos_z[:len_data]
    speed = np.sqrt(np.square(pos_x)+np.square(pos_y)+np.square(pos_z))
    zscore(pwm_0);zscore(pwm_1);zscore(pwm_2);zscore(pwm_3);zscore(speed)
    scaler = MinMaxScaler(feature_range=(-1,1))

    pwm1 = scaler.fit_transform(pwm_0.reshape(-1,1))
    pwm2 = scaler.fit_transform(pwm_1.reshape(-1,1))
    pwm3 = scaler.fit_transform(pwm_2.reshape(-1,1))
    pwm4 = scaler.fit_transform(pwm_3.reshape(-1,1))
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

    print("Server Started")
    prev_time = int(round(time.time()*1000))
        
    while True:
        curr_time = int(round(time.time()*1000))
        delta = curr_time - prev_time
        print("Sampling Time: ", delta," ms")
        processing_time.append(delta)        
        # data, addr = s.recvfrom(10)
        # data = data.decode('utf-8')

        if n == len_data:
            break
        # if data == "end":
        #     break
        # pwm1[n] = float(data[0])
        # pwm2[n] = float(data[1])
        # pwm3[n] = float(data[2])
        # pwm4[n] = float(data[3])
        if n > 2:
            in_id1 = [[[roll_in[n],pitch_in[n],yawCos_in[n],yawSin_in[n],speed_in[n],pwm1[n]],[roll_in[n-1],pitch_in[n-1],yawCos_in[n-1],yawSin_in[n-1],speed_in[n-1],pwm1[n-1]],[roll_in[n-2],pitch_in[n-2],yawCos_in[n-2],yawSin_in[n-2],speed_in[n-2],pwm1[n-2]]]]
            in_id2 = [[[roll_in[n],pitch_in[n],yawCos_in[n],yawSin_in[n],speed_in[n],pwm2[n]],[roll_in[n-1],pitch_in[n-1],yawCos_in[n-1],yawSin_in[n-1],speed_in[n-1],pwm2[n-1]],[roll_in[n-2],pitch_in[n-2],yawCos_in[n-2],yawSin_in[n-2],speed_in[n-2],pwm2[n-2]]]]
            in_id3 = [[[roll_in[n],pitch_in[n],yawCos_in[n],yawSin_in[n],speed_in[n],pwm3[n]],[roll_in[n-1],pitch_in[n-1],yawCos_in[n-1],yawSin_in[n-1],speed_in[n-1],pwm3[n-1]],[roll_in[n-2],pitch_in[n-2],yawCos_in[n-2],yawSin_in[n-2],speed_in[n-2],pwm3[n-2]]]]
            in_id4 = [[[roll_in[n],pitch_in[n],yawCos_in[n],yawSin_in[n],speed_in[n],pwm4[n]],[roll_in[n-1],pitch_in[n-1],yawCos_in[n-1],yawSin_in[n-1],speed_in[n-1],pwm4[n-1]],[roll_in[n-2],pitch_in[n-2],yawCos_in[n-2],yawSin_in[n-2],speed_in[n-2],pwm4[n-2]]]]
            in_id1 = np.asarray(in_id1,dtype='float64')
            in_id2 = np.asarray(in_id2,dtype='float64')
            in_id3 = np.asarray(in_id3,dtype='float64')
            in_id4 = np.asarray(in_id4,dtype='float64')
            hasilPWM1 = NN_PWM1.predict([in_id1])
            hasilPWM2 = NN_PWM2.predict([in_id2])
            hasilPWM3 = NN_PWM3.predict([in_id3])
            hasilPWM4 = NN_PWM4.predict([in_id4])

            hasil_hasil1 = hasilPWM1[0].tolist()
            hasil_hasil2 = hasilPWM2[0].tolist()
            hasil_hasil3 = hasilPWM3[0].tolist()
            hasil_hasil4 = hasilPWM4[0].tolist()

            hasil_hasil = [hasil_hasil1,hasil_hasil2,hasil_hasil3,hasil_hasil4]

            for mb in range(5):
                for ma in range(4):
                    if mb == 0:
                        roll_in[ma,n] = hasil_hasil[ma][mb]
                    elif mb == 1:
                        pitch_in[ma,n] = hasil_hasil[ma][mb]
                    elif mb == 2:
                        yawCos_in[ma,n] = hasil_hasil[ma][mb]
                    elif mb == 3:
                        yawSin_in[ma,n] = hasil_hasil[ma][mb]
                    elif mb == 4:
                        speed_in[ma,n] = hasil_hasil[ma][mb]

            all_hasil = hasilPWM1[0].tolist() + hasilPWM2[0].tolist() + hasilPWM3[0].tolist() + hasilPWM4[0].tolist()
            all_hasil = np.asarray([all_hasil],dtype='float64')
            hasilAll = NN_ALL_PWM.predict([all_hasil])
            print(hasilAll[0])
            roll_in[n-3] = hasilAll[0].tolist()[0]
            pitch_in[n-3] = hasilAll[0].tolist()[1]
            yawCos_in[n-3] = hasilAll[0].tolist()[2]
            yawSin_in[n-3] = hasilAll[0].tolist()[3]
            speed_in[n-3] = hasilAll[0].tolist()[4]
        n = n+1
        # hasil = hasil.tolist()[0]
        # hasil = hasil[0]
        # bulat = round(hasil,4)
        # y.append(bulat)
        # counter = counter + 1
        # s.sendto(str(bulat).encode('utf-8'), addr)
        prev_time = curr_time
        
    # s.close()
    plt.subplot(511)
    plt.plot(roll_in,label = 'prediction')
    plt.plot(roll,label = 'real_data')
    plt.subplot(512)
    plt.plot(pitch_in,label = 'prediction')
    plt.plot(pitch,label = 'real_data')
    plt.subplot(513)
    plt.plot(yawCos_in,label = 'prediction')
    plt.plot(yawCos,label = 'real_data')
    plt.subplot(514)
    plt.plot(yawSin_in,label = 'prediction')
    plt.plot(yawSin,label = 'real_data')
    plt.subplot(515)
    plt.plot(speed_in,label = 'prediction')
    plt.plot(speed,label = 'real_data')
    plt.tight_layout()
    plt.legend()
    plt.show()
    # #setpoint dari data ppr
    # r = []
    # for i in range(len(y_data)):
    #     #for j in range(5):
    #     r.append([y_data[i]])

    # r = np.array(r)
    # processing_time_np = np.array(processing_time)
    # y = np.array(y)
    # y = y.reshape(y.shape[0],1)

    # print("\n\nMSE: ", mean_squared_error(y, r))
    
    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # ax.plot(y, linewidth = 1)
    # ax.plot(r, linewidth = 1)
    # ax.legend(["y_act_DIC", "y_ref"], prop = {'size' : 10})
    # ax.set_frame_on(True)
    # plt.show()

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
