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
    def __init__(self, new_route = True):        
        self.time = 0
        self.run = False
        
        self.waktu_merah_kanan = 1
        self.waktu_merah_bawah = 1
        self.waktu_merah_kiri = 1
        
        if new_route:
            self.__generate_routefile("nugi_rl/environment/sumo/test1.rou.xml") # first, generate the route file for this simulation

        self.observation_space  = spaces.Box(-100, 100, (8, ))
        self.action_space       = spaces.Discrete(4)
        
    def __generate_routefile(self, route_files):
        random.seed(10)  # make tests reproducible
        N = 100  # number of time steps
        # demand per second from different directions
        pLR = 1.0 / 6
        pRL = 1.0 / 24
        pLB = 1.0 / 6
        pRB = 1.0 / 24
        pBL = 1.0 / 24
        pBR = 1.0 / 6

        pUB = 1.0 / 24
        pBU = 1.0 / 6
        pLU = 1.0 / 24
        pRU = 1.0 / 24
        pUL = 1.0 / 24
        pUR = 1.0 / 24

        probs = [pLR, pRL, pLB, pRB, pBL, pBR, pUB, pBU, pLU, pRU, pUL, pUR]

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

            vehNr = 0            
            for i in range(N):
                sc = np.random.choice(12, p = probs)

                if sc == 0:
                    print('    <vehicle id="kiri_kanan_%i" type="car" route="kiri_kanan" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if sc == 1:
                    print('    <vehicle id="kanan_kiri_%i" type="car" route="kanan_kiri" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if sc == 2:
                    print('    <vehicle id="kiri_bawah_%i" type="car" route="kiri_bawah" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if sc == 3:
                    print('    <vehicle id="kanan_bawah_%i" type="car" route="kanan_bawah" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if sc == 4:
                    print('    <vehicle id="bawah_kiri_%i" type="car" route="bawah_kiri" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if sc == 5:
                    print('    <vehicle id="bawah_kanan_%i" type="car" route="bawah_kanan" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    

                if sc == 6:
                    print('    <vehicle id="atas_bawah_%i" type="car" route="atas_bawah" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if sc == 7:
                    print('    <vehicle id="bawah_atas_%i" type="car" route="bawah_atas" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if sc == 8:
                    print('    <vehicle id="kiri_atas_%i" type="car" route="kiri_atas" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if sc == 9:
                    print('    <vehicle id="kanan_atas_%i" type="car" route="kanan_atas" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if sc == 10:
                    print('    <vehicle id="atas_kiri_%i" type="car" route="atas_kiri" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if sc == 11:
                    print('    <vehicle id="atas_kanan_%i" type="car" route="atas_kanan" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    
            print("</routes>", file=routes)
    
    def reset(self):
        sumoBinary = checkBinary('sumo')
        
        if self.run:
            traci.close()  
            
        traci.start([sumoBinary, "-c", "nugi_rl/environment/sumo/test1.sumocfg", 
        "--tripinfo-output", "nugi_rl/environment/sumo/test1.xml",
        "--no-step-log",
        "--no-warnings",
        "--duration-log.disable"])
        self.run = True
        
        self.time = 0

        self.waktu_merah_atas = 1
        self.waktu_merah_bawah = 1
        self.waktu_merah_kanan = 1
        self.waktu_merah_kiri = 1

        return np.zeros(8)
    
    def step(self, action):
        reward = 0
        traci.trafficlight.setPhase("gneJ10", action * 2)
        traci.simulationStep()

        banyak_kendaraan_tabrakan = traci.simulation.getCollidingVehiclesNumber()

        persentase_kendaraan_bawah = (traci.lane.getLastStepOccupancy('bawah_ke_tengah_0') + traci.lane.getLastStepOccupancy('bawah_ke_tengah_1')) / 2        
        persentase_kendaraan_kanan = (traci.lane.getLastStepOccupancy('kanan_ke_tengah_0') + traci.lane.getLastStepOccupancy('kanan_ke_tengah_1')) / 2
        persentase_kendaraan_kiri = (traci.lane.getLastStepOccupancy('kiri_ke_tengah_0') + traci.lane.getLastStepOccupancy('kiri_ke_tengah_1')) / 2
        persentase_kendaraan_atas = (traci.lane.getLastStepOccupancy('atas_ke_tengah_0') + traci.lane.getLastStepOccupancy('atas_ke_tengah_1')) / 2
        
        panjang_antrian_bawah = traci.lane.getLastStepHaltingNumber('bawah_ke_tengah_0') + traci.lane.getLastStepHaltingNumber('bawah_ke_tengah_1')
        panjang_antrian_kanan = traci.lane.getLastStepHaltingNumber('kanan_ke_tengah_0') + traci.lane.getLastStepHaltingNumber('kanan_ke_tengah_1')
        panjang_antrian_kiri = traci.lane.getLastStepHaltingNumber('kiri_ke_tengah_0') + traci.lane.getLastStepHaltingNumber('kiri_ke_tengah_1')
        panjang_antrian_atas = traci.lane.getLastStepHaltingNumber('atas_ke_tengah_0') + traci.lane.getLastStepHaltingNumber('atas_ke_tengah_1')

        kecepatan_kendaraan_bawah = (traci.lane.getLastStepMeanSpeed('bawah_ke_tengah_0') + traci.lane.getLastStepMeanSpeed('bawah_ke_tengah_1')) / 2
        kecepatan_kendaraan_kanan = (traci.lane.getLastStepMeanSpeed('kanan_ke_tengah_0') + traci.lane.getLastStepMeanSpeed('kanan_ke_tengah_1')) / 2
        kecepatan_kendaraan_kiri = (traci.lane.getLastStepMeanSpeed('kiri_ke_tengah_0') + traci.lane.getLastStepMeanSpeed('kiri_ke_tengah_1')) / 2
        kecepatan_kendaraan_atas = (traci.lane.getLastStepMeanSpeed('atas_ke_tengah_0') + traci.lane.getLastStepMeanSpeed('atas_ke_tengah_1')) / 2
        
        if action == 0:
            self.waktu_merah_atas = 1
            self.waktu_merah_kanan += 0.1
            self.waktu_merah_kiri += 0.1
            self.waktu_merah_bawah += 0.1

        elif action == 1:
            self.waktu_merah_atas += 0.1
            self.waktu_merah_kanan = 1
            self.waktu_merah_kiri += 0.1
            self.waktu_merah_bawah += 0.1

        elif action == 2:
            self.waktu_merah_atas += 0.1
            self.waktu_merah_kanan += 0.1
            self.waktu_merah_kiri += 0.1
            self.waktu_merah_bawah = 1

        elif action == 3:
            self.waktu_merah_atas += 0.1
            self.waktu_merah_kanan += 0.1
            self.waktu_merah_kiri = 1
            self.waktu_merah_bawah += 0.1
            
        reward += (panjang_antrian_bawah * 0.5 * self.waktu_merah_bawah) 
        reward += (panjang_antrian_atas * 0.5 * self.waktu_merah_atas)
        reward += (panjang_antrian_kanan * 0.5 * self.waktu_merah_kanan)
        reward += (panjang_antrian_kiri * 0.5 * self.waktu_merah_kiri)
        reward += (banyak_kendaraan_tabrakan * 5.0)
        reward *= -1
        
        self.time += 1
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        if not done:
            done = self.time > 10000

        obs = np.array([persentase_kendaraan_bawah, persentase_kendaraan_kanan, persentase_kendaraan_kiri, persentase_kendaraan_atas,
            kecepatan_kendaraan_bawah, kecepatan_kendaraan_kanan, kecepatan_kendaraan_kiri, kecepatan_kendaraan_atas])

        info = {}
            
        return obs, reward, done, info