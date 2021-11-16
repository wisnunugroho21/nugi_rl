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
    def __init__(self):        
        self.time = 0
        self.run = False
        
        self.waktu_merah_kanan = 1
        self.waktu_merah_bawah = 1
        self.waktu_merah_kiri = 1
        
        self.__generate_routefile("environment/sumo/test1.rou.xml") # first, generate the route file for this simulation

        self.observation_space  = spaces.Box(-100, 100, (12, ))
        self.action_space       = spaces.Discrete(3)
        
    def __generate_routefile(self, route_files):
        random.seed(10)  # make tests reproducible
        N = 300  # number of time steps
        # demand per second from different directions
        pLR = 1. / 8
        pRL = 1. / 8
        pLB = 1. / 20
        pRB = 1. / 20
        pBL = 1. / 24
        pBR = 1. / 24

        pUB = 1. / 20
        pBU = 1. / 20
        pLU = 1. / 24
        pRU = 1. / 24
        pUL = 1. / 8
        pUR = 1. / 8

        with open(route_files, "w") as routes:
            print("""<routes>
            <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>

            <route id="kiri_kanan" edges="titik_kiri titik_tengah titik_kanan"/>
            <route id="kanan_kiri" edges="titik_kanan titik_tengah titik_kiri"/>
            <route id="kiri_bawah" edges="titik_kiri titik_tengah titik_bawah"/>
            <route id="kanan_bawah" edges="titik_kanan titik_tengah titik_bawah"/>
            <route id="bawah_kiri" edges="titik_bawah titik_tengah titik_kiri"/>
            <route id="bawah_kanan" edges="titik_bawah titik_tengah titik_kanan"/>
            
            <route id="atas_bawah" edges="titik_atas titik_tengah titik_bawah"/>
            <route id="bawah_atas" edges="titik_bawah titik_tengah titik_atas"/>
            <route id="kiri_atas" edges="titik_kiri titik_tengah titik_atas"/>
            <route id="kanan_atas" edges="titik_kiri titik_tengah titik_atas"/>
            <route id="atas_kiri" edges="titik_atas titik_tengah titik_kiri"/>
            <route id="atas_kanan" edges="titik_atas titik_tengah titik_kanan"/>            
            """, file=routes)

            vehNr = 0            
            for i in range(N):
                if random.uniform(0, 1) < pLR:
                    print('    <vehicle id="kiri_kanan_%i" type="car" route="kiri_kanan" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if random.uniform(0, 1) < pRL:
                    print('    <vehicle id="kanan_kiri_%i" type="car" route="kanan_kiri" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if random.uniform(0, 1) < pLB:
                    print('    <vehicle id="kiri_bawah_%i" type="car" route="kiri_bawah" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if random.uniform(0, 1) < pRB: 
                    print('    <vehicle id="kanan_bawah_%i" type="car" route="kanan_bawah" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if random.uniform(0, 1) < pBL:
                    print('    <vehicle id="bawah_kiri_%i" type="car" route="bawah_kiri" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if random.uniform(0, 1) < pBR:
                    print('    <vehicle id="bawah_kanan_%i" type="car" route="bawah_kanan" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    

                if random.uniform(0, 1) < pUB:
                    print('    <vehicle id="atas_bawah_%i" type="car" route="atas_bawah" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if random.uniform(0, 1) < pBU:
                    print('    <vehicle id="bawah_atas_%i" type="car" route="bawah_atas" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if random.uniform(0, 1) < pLU:
                    print('    <vehicle id="kiri_atas_%i" type="car" route="kiri_atas" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if random.uniform(0, 1) < pRU:
                    print('    <vehicle id="kanan_atas_%i" type="car" route="kanan_atas" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if random.uniform(0, 1) < pUL:
                    print('    <vehicle id="atas_kiri_%i" type="car" route="atas_kiri" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1

                if random.uniform(0, 1) < pUR:
                    print('    <vehicle id="atas_kanan_%i" type="car" route="atas_kanan" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    
            print("</routes>", file=routes)
    
    def reset(self):
        sumoBinary = checkBinary('sumo-gui')
        
        if self.run:
            traci.close()  
            
        traci.start([sumoBinary, "-c", "environment/sumo/test1.sumocfg", 
        "--tripinfo-output", "environment/sumo/test1.xml",
        "--no-step-log",
        # "--no-warnings",
        "--duration-log.disable"])
        self.run = True
        
        self.time = 0

        self.waktu_merah_atas = 0
        self.waktu_merah_bawah = 0
        self.waktu_merah_kanan = 0
        self.waktu_merah_kiri = 0

        return np.zeros(8)
    
    def step(self, action):
        reward = 0
        traci.trafficlight.setPhase("tl_1", action * 2)
        traci.simulationStep()

        banyak_kendaraan_tabrakan = traci.simulation.getCollidingVehiclesNumber()

        persentase_kendaraan_bawah = (traci.lanearea.getLastStepOccupancy('detektor_bawah_1') + traci.lanearea.getLastStepOccupancy('detektor_bawah_2')) / 2        
        persentase_kendaraan_kanan = (traci.lanearea.getLastStepOccupancy('detektor_kanan_1') + traci.lanearea.getLastStepOccupancy('detektor_kanan_2')) / 2
        persentase_kendaraan_kiri = (traci.lanearea.getLastStepOccupancy('detektor_kiri_1') + traci.lanearea.getLastStepOccupancy('detektor_kiri_2')) / 2
        persentase_kendaraan_atas = (traci.lanearea.getLastStepOccupancy('detektor_atas_1') + traci.lanearea.getLastStepOccupancy('detektor_atas_2')) / 2
        
        panjang_antrian_bawah = traci.lanearea.getLastStepHaltingNumber('detektor_bawah_1') + traci.lanearea.getLastStepHaltingNumber('detektor_bawah_2')
        panjang_antrian_kanan = traci.lanearea.getLastStepHaltingNumber('detektor_kanan_1') + traci.lanearea.getLastStepHaltingNumber('detektor_kanan_2')
        panjang_antrian_kiri = traci.lanearea.getLastStepHaltingNumber('detektor_kiri_1') + traci.lanearea.getLastStepHaltingNumber('detektor_kiri_2')
        panjang_antrian_atas = traci.lanearea.getLastStepHaltingNumber('detektor_atas_1') + traci.lanearea.getLastStepHaltingNumber('detektor_atas_2')

        kecepatan_kendaraan_bawah = (traci.lanearea.getLastStepMeanSpeed('detektor_bawah_1') + traci.lanearea.getLastStepMeanSpeed('detektor_bawah_2')) / 2
        kecepatan_kendaraan_kanan = (traci.lanearea.getLastStepMeanSpeed('detektor_kanan_1') + traci.lanearea.getLastStepMeanSpeed('detektor_kanan_2')) / 2
        kecepatan_kendaraan_kiri = (traci.lanearea.getLastStepMeanSpeed('detektor_kiri_1') + traci.lanearea.getLastStepMeanSpeed('detektor_kiri_2')) / 2
        kecepatan_kendaraan_atas = (traci.lanearea.getLastStepMeanSpeed('detektor_atas_1') + traci.lanearea.getLastStepMeanSpeed('detektor_atas_2')) / 2
        
        if action == 0:
            self.waktu_merah_atas = 0
            self.waktu_merah_kanan += 1
            self.waktu_merah_kiri += 1
            self.waktu_merah_bawah += 1

        elif action == 1:
            self.waktu_merah_atas += 1
            self.waktu_merah_kanan = 0
            self.waktu_merah_kiri += 1
            self.waktu_merah_bawah += 1

        elif action == 2:
            self.waktu_merah_atas += 1
            self.waktu_merah_kanan += 1
            self.waktu_merah_kiri += 1
            self.waktu_merah_bawah = 0

        elif action == 3:
            self.waktu_merah_atas += 1
            self.waktu_merah_kanan += 1
            self.waktu_merah_kiri = 0
            self.waktu_merah_bawah += 1
            
        reward += (self.waktu_merah_bawah * 0.1) + ((panjang_antrian_bawah) * 0.5)
        reward += (self.waktu_merah_atas * 0.1) + ((panjang_antrian_atas) * 0.5)
        reward += (self.waktu_merah_kanan * 0.1) + ((panjang_antrian_kanan) * 0.5)
        reward += (self.waktu_merah_kiri * 0.1) + ((panjang_antrian_kiri) * 0.5)        
        reward += (banyak_kendaraan_tabrakan * 5.0)
        reward *= -1
        
        self.time += 1
        done = traci.simulation.getMinExpectedNumber() <= 0
        
        if not done:
            done = self.time > 100000

        obs = np.array([persentase_kendaraan_bawah, persentase_kendaraan_kanan, persentase_kendaraan_kiri, persentase_kendaraan_atas,
            kecepatan_kendaraan_bawah, kecepatan_kendaraan_kanan, kecepatan_kendaraan_kiri, kecepatan_kendaraan_atas])

        info = {}
            
        return obs, reward, done, info