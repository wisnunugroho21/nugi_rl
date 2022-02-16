import os
import sys
import optparse
import random
from gym import spaces

import numpy as np

sys.path.append('/usr/share/sumo/tools')
from sumolib import checkBinary
import traci

class SumoEnv:  
    def __init__(self, new_route: bool = True) -> None:        
        self.time = 0
        self.run = False
        
        self.last_action = -1
        
        self.observation_space  = spaces.Box(-100, 100, (2, ))
        self.action_space       = spaces.Discrete(4)

    def get_obs_dim(self):
        return 2
            
    def get_action_dim(self):
        return 4
        
    def _generate_routefile(self, route_files: str) -> None:
        if os.path.exists(route_files):
            os.remove(route_files)

        # demand per second from different directions
        probs = self._generate_probs_route()

        with open(route_files, "w") as routes:
            print("""<routes>
            <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>

            <route id="kiri_kanan" edges="kiri_ke_tengah tengah_ke_kanan"/>
            <route id="kanan_kiri" edges="kanan_ke_tengah tengah_ke_kiri"/>
            <route id="kiri_bawah" edges="kiri_ke_tengah tengah_ke_bawah"/>
            <route id="kanan_bawah" edges="kanan_ke_tengah tengah_ke_bawah"/>
            <route id="bawah_kiri" edges="bawah_ke_tengah tengah_ke_kiri"/>
            <route id="bawah_kanan" edges="bawah_ke_tengah tengah_ke_kanan"/>
            
            <route id="atas_bawah" edges="atas_ke_tengah tengah_ke_bawah"/>
            <route id="bawah_atas" edges="bawah_ke_tengah tengah_ke_atas"/>
            <route id="kiri_atas" edges="kiri_ke_tengah tengah_ke_atas"/>
            <route id="kanan_atas" edges="kanan_ke_tengah tengah_ke_atas"/>
            <route id="atas_kiri" edges="atas_ke_tengah tengah_ke_kiri"/>
            <route id="atas_kanan" edges="atas_ke_tengah tengah_ke_kanan"/>            
            """, file=routes)
            
            for i in range(0, 400, 2):
                sc = np.random.choice(12, p = probs)                
                route_name = ''
                
                if sc == 0:
                    route_name = 'kiri_kanan'                  

                elif sc == 1:
                    route_name = 'kanan_kiri'                  

                elif sc == 2:
                    route_name = 'kiri_bawah'                  

                elif sc == 3:
                    route_name = 'kanan_bawah'

                elif sc == 4:
                    route_name = 'bawah_kiri'                  

                elif sc == 5:
                    route_name = 'bawah_kanan'

                elif sc == 6:
                    route_name = 'atas_bawah'

                elif sc == 7:
                    route_name = 'bawah_atas'

                elif sc == 8:
                    route_name = 'kiri_atas'

                elif sc == 9:
                    route_name = 'kanan_atas'

                elif sc == 10:
                    route_name = 'atas_kiri'

                elif sc == 11:
                    route_name = 'atas_kanan'
                        
                for idx in range(4):
                    print('    <vehicle id="%s_%i_%i" type="car" route="%s" depart="%i" />' % (route_name, i, idx, route_name, i), file = routes)
                    
            print("</routes>", file = routes)

    def _generate_probs_route(self) -> list:
        level = np.random.choice(4)

        if level == 0:
            sc = np.random.choice(4)
            
            if sc == 0:
                return [1.0 / 3, 0, 1.0 / 3, 0, 0, 0, 
                    0, 0, 1.0 / 3, 0, 0, 0]

            elif sc == 1:
                return [0, 1.0 / 3, 0, 1.0 / 3, 0, 0, 
                    0, 0, 0, 1.0 / 3, 0, 0]

            elif sc == 2:
                return [0, 0, 0, 0, 1.0 / 3, 1.0 / 3, 
                    0, 1.0 / 3, 0, 0, 0, 0]

            elif sc == 3:
                return [0, 0, 0, 0, 0, 0, 
                    1.0 / 3, 0, 0, 0, 1.0 / 3, 1.0 / 3]

        elif level == 1:
            sc = np.random.choice(6)

            if sc == 0:
                return [1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 0, 0, 
                    0, 0, 1.0 / 6, 1.0 / 6, 0, 0]

            elif sc == 1:
                return [1.0/ 6, 0, 1.0/ 6, 0, 1.0/ 6, 1.0/ 6,
                    0, 1.0/ 6, 1.0/ 6, 0, 0, 0]

            elif sc == 2:
                return [1.0/ 6, 0, 1.0/ 6, 0, 0, 0,
                    1.0/ 6, 0, 1.0/ 6, 0, 1.0/ 6, 1.0/ 6]

            elif sc == 3:
                return [0, 1.0/ 6, 0, 1.0/ 6, 1.0/ 6, 1.0/ 6,
                    0, 1.0/ 6, 0, 1.0/ 6, 0, 0]

            elif sc == 4:
                return [0, 1.0/ 6, 0, 1.0/ 6, 0, 0,
                    1.0/ 6, 0, 0, 1.0/ 6, 1.0/ 6, 1.0/ 6]

            elif sc == 5:
                return [0, 0, 0, 0, 1.0/ 6, 1.0/ 6,
                    1.0/ 6, 1.0/ 6, 0, 0, 1.0/ 6, 1.0/ 6]

        elif level == 2:
            sc = np.random.choice(4)

            if sc == 0:
                return [1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9,
                    0, 1.0 / 9, 1.0 / 9, 1.0 / 9, 0, 0]

            elif sc == 1:
                return [1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 0, 0, 
                    1.0 / 9, 0, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9]

            elif sc == 2:
                return [1.0 / 9, 0, 1.0 / 9, 0, 1.0 / 9, 1.0 / 9, 
                    1.0 / 9, 1.0 / 9, 1.0 / 9, 0, 1.0 / 9, 1.0 / 9]

            elif sc == 3:
                return [0, 1.0 / 9, 0, 1.0 / 9, 1.0 / 9, 1.0 / 9, 
                    1.0 / 9, 1.0 / 9, 0, 1.0 / 9, 1.0 / 9, 1.0 / 9]

        # elif level == 3:
        #     return [1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12, 
        #         1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12]

        elif level == 3:
            sc = np.random.choice(4)

            if sc == 0:
                return [2.0 / 15, 1.0 / 15, 2.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 
                    1.0 / 15, 1.0 / 15, 2.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15]

            elif sc == 1:
                return [1.0 / 15, 2.0 / 15, 1.0 / 15, 2.0 / 15, 1.0 / 15, 1.0 / 15, 
                    1.0 / 15, 1.0 / 15, 1.0 / 15, 2.0 / 15, 1.0 / 15, 1.0 / 15]

            elif sc == 2:
                return [1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 2.0 / 15, 2.0 / 15, 
                    1.0 / 15, 2.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15]

            elif sc == 3:
                return [1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 
                    2.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 2.0 / 15, 2.0 / 15]

        elif level == 4:
            sc = np.random.choice(6)

            if sc == 0:
                return [2.0 / 18, 2.0 / 18, 2.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18, 
                    1.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18]

            elif sc == 1:
                return [2.0 / 18, 1.0 / 18, 2.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18,
                    1.0 / 18, 2.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18]

            elif sc == 2:
                return [2.0 / 18, 1.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18,
                    2.0 / 18, 1.0 / 18, 2.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18]

            elif sc == 3:
                return [1.0 / 18, 2.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18, 2.0 / 18,
                    1.0 / 18, 2.0 / 18, 1.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18]

            elif sc == 4:
                return [1.0 / 18, 2.0 / 18, 1.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18,
                    2.0 / 18, 1.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18, 2.0 / 18]

            elif sc == 5:
                return [1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18,
                    2.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18]

    def _get_data_kendaraan(self, kendaraan_ids: list) -> list:
        return list(map(lambda id: [traci.vehicle.getLanePosition(id), traci.vehicle.getSpeed(id)], kendaraan_ids))
    
    def reset(self) -> np.ndarray:
        if self.run:
            traci.close()

        sumoBinary = checkBinary('sumo')

        self._generate_routefile("nugi_rl/environment/sumo/test1.rou.xml") # first, generate the route file for this simulation          
            
        traci.start([sumoBinary, "-c", "nugi_rl/environment/sumo/test1.sumocfg", 
        "--tripinfo-output", "nugi_rl/environment/sumo/test1.xml",
        "--no-step-log",
        "--no-warnings",
        "--duration-log.disable"])
        self.run = True
        
        self.time = 0
        self.last_action = -1

        self.waktu_merah_atas = 0
        self.waktu_merah_bawah = 0
        self.waktu_merah_kanan = 0
        self.waktu_merah_kiri = 0

        kendaraan_array = [[0], [0], [0], [0]]
        for idx in range(len(kendaraan_array)):
            start = np.full((1, 2), -1)
            zeros = np.full((50, 2), -100)
            kendaraan_array[idx] = np.concatenate([start, zeros], 0)

        return np.stack(kendaraan_array)
    
    def step(self, action) -> tuple: 
        phase_changed = 0
        if action != self.last_action and self.last_action != -1:
            traci.trafficlight.setPhase("lampu_lalu_lintas", self.last_action * 2 + 1)
            traci.simulationStep()
            
            phase_changed = 1
            self.last_action = action

        traci.trafficlight.setPhase("lampu_lalu_lintas", action * 2)
        traci.simulationStep()

        """ print(traci.trafficlight.getControlledLinks("lampu_lalu_lintas")) 
        print('----')  """
              
        linkIndexes = traci.trafficlight.getControlledLinks("lampu_lalu_lintas")

        kendaraan_array = [
            self._get_data_kendaraan(traci.lane.getLastStepVehicleIDs('bawah_ke_tengah_0')) + self._get_data_kendaraan(traci.lane.getLastStepVehicleIDs('bawah_ke_tengah_1')),
            self._get_data_kendaraan(traci.lane.getLastStepVehicleIDs('atas_ke_tengah_0')) + self._get_data_kendaraan(traci.lane.getLastStepVehicleIDs('atas_ke_tengah_1')),
            self._get_data_kendaraan(traci.lane.getLastStepVehicleIDs('kiri_ke_tengah_0')) + self._get_data_kendaraan(traci.lane.getLastStepVehicleIDs('kiri_ke_tengah_1')),
            self._get_data_kendaraan(traci.lane.getLastStepVehicleIDs('kanan_ke_tengah_0')) + self._get_data_kendaraan(traci.lane.getLastStepVehicleIDs('kanan_ke_tengah_1'))
        ]
        
        panjang_antrian_bawah   = traci.lane.getLastStepHaltingNumber('bawah_ke_tengah_0') + traci.lane.getLastStepHaltingNumber('bawah_ke_tengah_1')
        panjang_antrian_kanan   = traci.lane.getLastStepHaltingNumber('kanan_ke_tengah_0') + traci.lane.getLastStepHaltingNumber('kanan_ke_tengah_1')
        panjang_antrian_kiri    = traci.lane.getLastStepHaltingNumber('kiri_ke_tengah_0') + traci.lane.getLastStepHaltingNumber('kiri_ke_tengah_1')
        panjang_antrian_atas    = traci.lane.getLastStepHaltingNumber('atas_ke_tengah_0') + traci.lane.getLastStepHaltingNumber('atas_ke_tengah_1')

        kecepatan_kendaraan_bawah   = (traci.lane.getLastStepMeanSpeed('bawah_ke_tengah_0') + traci.lane.getLastStepMeanSpeed('bawah_ke_tengah_1')) / 2
        kecepatan_kendaraan_kanan   = (traci.lane.getLastStepMeanSpeed('kanan_ke_tengah_0') + traci.lane.getLastStepMeanSpeed('kanan_ke_tengah_1')) / 2
        kecepatan_kendaraan_kiri    = (traci.lane.getLastStepMeanSpeed('kiri_ke_tengah_0') + traci.lane.getLastStepMeanSpeed('kiri_ke_tengah_1')) / 2
        kecepatan_kendaraan_atas    = (traci.lane.getLastStepMeanSpeed('atas_ke_tengah_0') + traci.lane.getLastStepMeanSpeed('atas_ke_tengah_1')) / 2

        waktu_menunggu_bawah    = (traci.lane.getWaitingTime('bawah_ke_tengah_0') + traci.lane.getWaitingTime('bawah_ke_tengah_1')) / 2
        waktu_menunggu_kanan    = (traci.lane.getWaitingTime('kanan_ke_tengah_0') + traci.lane.getWaitingTime('kanan_ke_tengah_1')) / 2
        waktu_menunggu_kiri     = (traci.lane.getWaitingTime('kiri_ke_tengah_0') + traci.lane.getWaitingTime('kiri_ke_tengah_1')) / 2
        waktu_menunggu_atas     = (traci.lane.getWaitingTime('atas_ke_tengah_0') + traci.lane.getWaitingTime('atas_ke_tengah_1')) / 2

        kecepatan_diperbolehkan_bawah   = (traci.lane.getMaxSpeed('bawah_ke_tengah_0') + traci.lane.getMaxSpeed('bawah_ke_tengah_1')) / 2
        kecepatan_diperbolehkan_kanan   = (traci.lane.getMaxSpeed('kanan_ke_tengah_0') + traci.lane.getMaxSpeed('kanan_ke_tengah_1')) / 2
        kecepatan_diperbolehkan_kiri    = (traci.lane.getMaxSpeed('kiri_ke_tengah_0') + traci.lane.getMaxSpeed('kiri_ke_tengah_1')) / 2
        kecepatan_diperbolehkan_atas    = (traci.lane.getMaxSpeed('atas_ke_tengah_0') + traci.lane.getMaxSpeed('atas_ke_tengah_1')) / 2

        # banyak_lolos_perempatan     = traci.edge.getLastStepVehicleNumber('titik_tengah')
        # banyak_antrian_perempatan   = traci.edge.getLastStepHaltingNumber('titik_tengah')
            
        reward = 0
        banyak_kendaraan_tabrakan = traci.simulation.getCollidingVehiclesNumber()

        banyak_kendaraan_tersangkut = 0
        for linkIndex in range(len(linkIndexes)):
            blockingVehicles = traci.trafficlight.getBlockingVehicles("lampu_lalu_lintas", linkIndex)
            banyak_kendaraan_tersangkut += len(blockingVehicles)

        waktu_delay_bawah   = 1 - (kecepatan_kendaraan_bawah / kecepatan_diperbolehkan_bawah)
        waktu_delay_kanan   = 1 - (kecepatan_kendaraan_kanan / kecepatan_diperbolehkan_kanan)
        waktu_delay_kiri    = 1 - (kecepatan_kendaraan_kiri / kecepatan_diperbolehkan_kiri)        
        waktu_delay_atas    = 1 - (kecepatan_kendaraan_atas / kecepatan_diperbolehkan_atas)

        reward += (panjang_antrian_bawah * -0.25 + waktu_delay_bawah * -0.25 + waktu_menunggu_bawah * -0.25) 
        reward += (panjang_antrian_kanan * -0.25 + waktu_delay_kanan * -0.25 + waktu_menunggu_kanan * -0.25)
        reward += (panjang_antrian_kiri * -0.25 + waktu_delay_kiri * -0.25 + waktu_menunggu_kiri * -0.25)
        reward += (panjang_antrian_atas * -0.25 + waktu_delay_atas * -0.25 + waktu_menunggu_atas * -0.25)
        reward += (phase_changed * -5.0)
        # reward += (banyak_lolos_perempatan - banyak_antrian_perempatan)
        reward += (banyak_kendaraan_tabrakan * -20.0)
        reward += (banyak_kendaraan_tersangkut * -5.0)
        
        self.time += 1
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        if not done:
            done = self.time > 30000        

        out_arr = []
        for arr in kendaraan_array:  
            start = np.full((1, 2), -1)

            if len(arr) > 0:      
                zeros   = np.full((50 - len(arr), 2), -100)
                obs     = np.array(arr)
                obs     = np.concatenate([start, obs, zeros], 0)
            else:
                zeros   = np.full((50, 2), -100) 
                obs     = np.concatenate([start, zeros], 0)
            out_arr.append(obs)

        """ if len(kendaraan_array) > 0:       
            zeros   = np.full((50 - len(arr), 3), -100)
            obs     = np.array(arr)
            obs     = np.concatenate([start, obs, zeros], 0)

        else:
            zeros   = np.full((150, 3), -100) 
            obs     = np.concatenate([start, zeros], 0) """

        info = {
            "colliding": banyak_kendaraan_tabrakan,
            "blocking": banyak_kendaraan_tersangkut
        }
            
        return np.stack(out_arr), reward, done, info