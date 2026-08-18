import socketio
import numpy as np
import time
import pypolar as plr

# ip_var = "192.168.1.122:5000"
# sio = socketio.Client()
# i = 0
# while True:
#     try:
#         i += 1
#         sio.connect(f"ws://{ip_var}")
#         break
#     except socketio.exceptions.ConnectionError:
#         if i > 5:
#             print("Exceeded maximum number of retries. Exiting...")
#             exit()
#         print("Connection failed. Retrying...")
#         time.sleep(1)
        
# print('Connected')


# # DELAYED BT
# parameters = {
#     "hip_delay_idx": 25,
#     "h_flex_torque_scale": 0.2,
#     "h_ext_torque_scale": 0.2,
#     "h_flex_ang_scale": 0.0,
#     "h_ext_ang_scale": 0.0,
#     "h_flex_power_scale": 0.0,
#     "h_ext_power_scale": 0.0,
#     "h_flex_grf_scale": 0.0,
#     "h_ext_grf_scale": 0.0,
#     "knee_delay_idx": 0,
#     "k_ext_torque_scale": 0.15,
#     "k_flex_torque_scale": 0.15,
#     "k_ext_ang_scale": 0.0,
#     "k_flex_ang_scale": 0.0,
#     "k_ext_power_scale": 0.0,
#     "k_flex_power_scale": 0.0,
#     "k_ext_grf_scale": 0.0,
#     "k_flex_grf_scale": 0.0
# }
# sio.emit("update_inputs", parameters)
# input()


class Exo(plr.Device):
    
    def __init__(self, exo_ip, connect=True):
        self.exo_ip = None
        if connect:
            self.sio = socketio.Client()
            i = 0
            while True:
                try:
                    i += 1
                    self.sio.connect(f"ws://{self.exo_ip}")
                    break
                except socketio.exceptions.ConnectionError:
                    if i > 5:
                        print("Exceeded maximum number of retries. Exiting...")
                        exit()
                    print("Connection failed. Retrying...")
                    time.sleep(1)
        else:
            self.sio = None

Exo(exo_ip="192.168.1.122:5000")
input()