import os
import sys

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import Tensor

sys.path.append("/usr/share/sumo/tools")
import traci
from sumolib import checkBinary

from nugi_rl.environment.base import Environment


class SumoEnv(Environment):
    def __init__(self, new_route: bool = True) -> None:
        self.time = 0
        self.run = False

        self.last_action = -1

        self.observation_space = Box(-1, 100, (3,))
        self.action_space = Discrete(4)

    def _generate_routefile(self, route_files: str) -> None:
        if os.path.exists(route_files):
            os.remove(route_files)

        # demand per second from different directions
        probs = self._generate_probs_route()

        with open(route_files, "w") as routes:
            print(
                """<routes>
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
            """,
                file=routes,
            )

            for i in range(0, 800, 4):
                sc = np.random.choice(12, p=probs)
                route_name = ""

                if sc == 0:
                    route_name = "kiri_kanan"

                elif sc == 1:
                    route_name = "kanan_kiri"

                elif sc == 2:
                    route_name = "kiri_bawah"

                elif sc == 3:
                    route_name = "kanan_bawah"

                elif sc == 4:
                    route_name = "bawah_kiri"

                elif sc == 5:
                    route_name = "bawah_kanan"

                elif sc == 6:
                    route_name = "atas_bawah"

                elif sc == 7:
                    route_name = "bawah_atas"

                elif sc == 8:
                    route_name = "kiri_atas"

                elif sc == 9:
                    route_name = "kanan_atas"

                elif sc == 10:
                    route_name = "atas_kiri"

                elif sc == 11:
                    route_name = "atas_kanan"

                for idx in range(1):
                    print(
                        '    <vehicle id="%s_%i_%i" type="car" route="%s" depart="%i" />'
                        % (route_name, i, idx, route_name, i),
                        file=routes,
                    )

            print("</routes>", file=routes)

    def _generate_probs_route(self) -> list[float]:
        level = np.random.choice(3)

        if level == 0:
            sc = np.random.choice(4)

            if sc == 0:
                return [1.0 / 3, 0, 1.0 / 3, 0, 0, 0, 0, 0, 1.0 / 3, 0, 0, 0]

            elif sc == 1:
                return [0, 1.0 / 3, 0, 1.0 / 3, 0, 0, 0, 0, 0, 1.0 / 3, 0, 0]

            elif sc == 2:
                return [0, 0, 0, 0, 1.0 / 3, 1.0 / 3, 0, 1.0 / 3, 0, 0, 0, 0]

            elif sc == 3:
                return [0, 0, 0, 0, 0, 0, 1.0 / 3, 0, 0, 0, 1.0 / 3, 1.0 / 3]

        elif level == 1:
            sc = np.random.choice(6)

            if sc == 0:
                return [
                    1.0 / 6,
                    1.0 / 6,
                    1.0 / 6,
                    1.0 / 6,
                    0,
                    0,
                    0,
                    0,
                    1.0 / 6,
                    1.0 / 6,
                    0,
                    0,
                ]

            elif sc == 1:
                return [
                    1.0 / 6,
                    0,
                    1.0 / 6,
                    0,
                    1.0 / 6,
                    1.0 / 6,
                    0,
                    1.0 / 6,
                    1.0 / 6,
                    0,
                    0,
                    0,
                ]

            elif sc == 2:
                return [
                    1.0 / 6,
                    0,
                    1.0 / 6,
                    0,
                    0,
                    0,
                    1.0 / 6,
                    0,
                    1.0 / 6,
                    0,
                    1.0 / 6,
                    1.0 / 6,
                ]

            elif sc == 3:
                return [
                    0,
                    1.0 / 6,
                    0,
                    1.0 / 6,
                    1.0 / 6,
                    1.0 / 6,
                    0,
                    1.0 / 6,
                    0,
                    1.0 / 6,
                    0,
                    0,
                ]

            elif sc == 4:
                return [
                    0,
                    1.0 / 6,
                    0,
                    1.0 / 6,
                    0,
                    0,
                    1.0 / 6,
                    0,
                    0,
                    1.0 / 6,
                    1.0 / 6,
                    1.0 / 6,
                ]

            elif sc == 5:
                return [
                    0,
                    0,
                    0,
                    0,
                    1.0 / 6,
                    1.0 / 6,
                    1.0 / 6,
                    1.0 / 6,
                    0,
                    0,
                    1.0 / 6,
                    1.0 / 6,
                ]

        elif level == 2:
            sc = np.random.choice(4)

            if sc == 0:
                return [
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    0,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    0,
                    0,
                ]

            elif sc == 1:
                return [
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    0,
                    0,
                    1.0 / 9,
                    0,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                ]

            elif sc == 2:
                return [
                    1.0 / 9,
                    0,
                    1.0 / 9,
                    0,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    0,
                    1.0 / 9,
                    1.0 / 9,
                ]

            elif sc == 3:
                return [
                    0,
                    1.0 / 9,
                    0,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                    0,
                    1.0 / 9,
                    1.0 / 9,
                    1.0 / 9,
                ]

        return []

        # elif level == 3:
        #     return [1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12,
        #         1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12, 1.0 / 12]

        # elif level == 3:
        #     sc = np.random.choice(4)

        #     if sc == 0:
        #         return [2.0 / 15, 1.0 / 15, 2.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15,
        #             1.0 / 15, 1.0 / 15, 2.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15]

        #     elif sc == 1:
        #         return [1.0 / 15, 2.0 / 15, 1.0 / 15, 2.0 / 15, 1.0 / 15, 1.0 / 15,
        #             1.0 / 15, 1.0 / 15, 1.0 / 15, 2.0 / 15, 1.0 / 15, 1.0 / 15]

        #     elif sc == 2:
        #         return [1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 2.0 / 15, 2.0 / 15,
        #             1.0 / 15, 2.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15]

        #     elif sc == 3:
        #         return [1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15,
        #             2.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 2.0 / 15, 2.0 / 15]

        # elif level == 4:
        #     sc = np.random.choice(6)

        #     if sc == 0:
        #         return [2.0 / 18, 2.0 / 18, 2.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18,
        #             1.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18]

        #     elif sc == 1:
        #         return [2.0 / 18, 1.0 / 18, 2.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18,
        #             1.0 / 18, 2.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18]

        #     elif sc == 2:
        #         return [2.0 / 18, 1.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18,
        #             2.0 / 18, 1.0 / 18, 2.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18]

        #     elif sc == 3:
        #         return [1.0 / 18, 2.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18, 2.0 / 18,
        #             1.0 / 18, 2.0 / 18, 1.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18]

        #     elif sc == 4:
        #         return [1.0 / 18, 2.0 / 18, 1.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18,
        #             2.0 / 18, 1.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18, 2.0 / 18]

        #     elif sc == 5:
        #         return [1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18,
        #             2.0 / 18, 2.0 / 18, 1.0 / 18, 1.0 / 18, 2.0 / 18, 2.0 / 18]

    def _get_data_kendaraan(self, kendaraan_ids: list, position: int) -> list:
        return [
            map(
                lambda id: [
                    traci.vehicle.getLanePosition(id) / 200,
                    traci.vehicle.getSpeed(id),
                    position,
                ],
                kendaraan_ids,
            )
        ]

    def is_discrete(self) -> bool:
        return True

    def get_obs_dim(self) -> int:
        return 3

    def get_action_dim(self) -> int:
        return 4

    def reset(self) -> Tensor:
        if self.run:
            traci.close()

        sumoBinary = checkBinary("sumo")

        self._generate_routefile(
            "nugi_rl/environment/sumo/test1.rou.xml"
        )  # first, generate the route file for this simulation

        traci.start(
            [
                sumoBinary,
                "-c",
                "nugi_rl/environment/sumo/test1.sumocfg",
                "--tripinfo-output",
                "nugi_rl/environment/sumo/test1.xml",
                "--no-step-log",
                "--no-warnings",
                "--duration-log.disable",
            ]
        )
        self.run = True

        self.time = 0
        self.last_action = -1

        self.waktu_merah_atas = 0
        self.waktu_merah_bawah = 0
        self.waktu_merah_kanan = 0
        self.waktu_merah_kiri = 0

        panjang_antrian_bawah = traci.lane.getLastStepHaltingNumber(
            "bawah_ke_tengah_0"
        ) + traci.lane.getLastStepHaltingNumber("bawah_ke_tengah_1")
        panjang_antrian_kanan = traci.lane.getLastStepHaltingNumber(
            "kanan_ke_tengah_0"
        ) + traci.lane.getLastStepHaltingNumber("kanan_ke_tengah_1")
        panjang_antrian_kiri = traci.lane.getLastStepHaltingNumber(
            "kiri_ke_tengah_0"
        ) + traci.lane.getLastStepHaltingNumber("kiri_ke_tengah_1")
        panjang_antrian_atas = traci.lane.getLastStepHaltingNumber(
            "atas_ke_tengah_0"
        ) + traci.lane.getLastStepHaltingNumber("atas_ke_tengah_1")

        waktu_menunggu_bawah = traci.lane.getWaitingTime(
            "bawah_ke_tengah_0"
        ) + traci.lane.getWaitingTime("bawah_ke_tengah_1")
        waktu_menunggu_kanan = traci.lane.getWaitingTime(
            "kanan_ke_tengah_0"
        ) + traci.lane.getWaitingTime("kanan_ke_tengah_1")
        waktu_menunggu_kiri = traci.lane.getWaitingTime(
            "kiri_ke_tengah_0"
        ) + traci.lane.getWaitingTime("kiri_ke_tengah_1")
        waktu_menunggu_atas = traci.lane.getWaitingTime(
            "atas_ke_tengah_0"
        ) + traci.lane.getWaitingTime("atas_ke_tengah_1")

        obs2 = [
            panjang_antrian_bawah,
            panjang_antrian_kanan,
            panjang_antrian_kiri,
            panjang_antrian_atas,
            waktu_menunggu_bawah / (panjang_antrian_bawah + 1e-3),
            waktu_menunggu_kanan / (panjang_antrian_kanan + 1e-3),
            waktu_menunggu_kiri / (panjang_antrian_kiri + 1e-3),
            panjang_antrian_atas / (waktu_menunggu_atas + 1e-3),
        ]

        start = np.full((1, 3), -0.5)
        zeros = np.full((150, 3), -1)
        kendaraan_array = np.concatenate([start, zeros], 0)

        obs = np.stack([kendaraan_array])
        obs2 = np.stack(obs2)

        return torch.Tensor([obs, obs2])

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        phase_changed = 0
        if action != self.last_action and self.last_action != -1:
            traci.trafficlight.setPhase("lampu_lalu_lintas", self.last_action * 2 + 1)
            traci.simulationStep()

            phase_changed = 1
            self.last_action = action

        traci.trafficlight.setPhase("lampu_lalu_lintas", action * 2)
        traci.simulationStep()

        kendaraan_array = (
            self._get_data_kendaraan(
                traci.lane.getLastStepVehicleIDs("bawah_ke_tengah_0"), 1
            )
            + self._get_data_kendaraan(
                traci.lane.getLastStepVehicleIDs("bawah_ke_tengah_1"), 2
            )
            + self._get_data_kendaraan(
                traci.lane.getLastStepVehicleIDs("atas_ke_tengah_0"), 3
            )
            + self._get_data_kendaraan(
                traci.lane.getLastStepVehicleIDs("atas_ke_tengah_1"), 4
            )
            + self._get_data_kendaraan(
                traci.lane.getLastStepVehicleIDs("kiri_ke_tengah_0"), 5
            )
            + self._get_data_kendaraan(
                traci.lane.getLastStepVehicleIDs("kiri_ke_tengah_1"), 6
            )
            + self._get_data_kendaraan(
                traci.lane.getLastStepVehicleIDs("kanan_ke_tengah_0"), 7
            )
            + self._get_data_kendaraan(
                traci.lane.getLastStepVehicleIDs("kanan_ke_tengah_1"), 8
            )
        )

        start = np.full((1, 3), -0.5)
        if len(kendaraan_array) > 0:
            zeros = np.full((150 - len(kendaraan_array), 3), -1)
            obs = np.array(kendaraan_array)
            obs = np.concatenate([start, obs, zeros], 0)

        else:
            zeros = np.full((150, 3), -1)
            obs = np.concatenate([start, zeros], 0)

        obs = np.stack([obs])

        panjang_antrian_bawah = traci.lane.getLastStepHaltingNumber(
            "bawah_ke_tengah_0"
        ) + traci.lane.getLastStepHaltingNumber("bawah_ke_tengah_1")
        panjang_antrian_kanan = traci.lane.getLastStepHaltingNumber(
            "kanan_ke_tengah_0"
        ) + traci.lane.getLastStepHaltingNumber("kanan_ke_tengah_1")
        panjang_antrian_kiri = traci.lane.getLastStepHaltingNumber(
            "kiri_ke_tengah_0"
        ) + traci.lane.getLastStepHaltingNumber("kiri_ke_tengah_1")
        panjang_antrian_atas = traci.lane.getLastStepHaltingNumber(
            "atas_ke_tengah_0"
        ) + traci.lane.getLastStepHaltingNumber("atas_ke_tengah_1")

        waktu_menunggu_bawah = traci.lane.getWaitingTime(
            "bawah_ke_tengah_0"
        ) + traci.lane.getWaitingTime("bawah_ke_tengah_1")
        waktu_menunggu_kanan = traci.lane.getWaitingTime(
            "kanan_ke_tengah_0"
        ) + traci.lane.getWaitingTime("kanan_ke_tengah_1")
        waktu_menunggu_kiri = traci.lane.getWaitingTime(
            "kiri_ke_tengah_0"
        ) + traci.lane.getWaitingTime("kiri_ke_tengah_1")
        waktu_menunggu_atas = traci.lane.getWaitingTime(
            "atas_ke_tengah_0"
        ) + traci.lane.getWaitingTime("atas_ke_tengah_1")

        kecepatan_kendaraan_bawah = traci.lane.getLastStepMeanSpeed(
            "bawah_ke_tengah_0"
        ) + traci.lane.getLastStepMeanSpeed("bawah_ke_tengah_1")
        kecepatan_kendaraan_kanan = traci.lane.getLastStepMeanSpeed(
            "kanan_ke_tengah_0"
        ) + traci.lane.getLastStepMeanSpeed("kanan_ke_tengah_1")
        kecepatan_kendaraan_kiri = traci.lane.getLastStepMeanSpeed(
            "kiri_ke_tengah_0"
        ) + traci.lane.getLastStepMeanSpeed("kiri_ke_tengah_1")
        kecepatan_kendaraan_atas = traci.lane.getLastStepMeanSpeed(
            "atas_ke_tengah_0"
        ) + traci.lane.getLastStepMeanSpeed("atas_ke_tengah_1")

        kecepatan_diperbolehkan_bawah = traci.lane.getMaxSpeed(
            "bawah_ke_tengah_0"
        ) + traci.lane.getMaxSpeed("bawah_ke_tengah_1")
        kecepatan_diperbolehkan_kanan = traci.lane.getMaxSpeed(
            "kanan_ke_tengah_0"
        ) + traci.lane.getMaxSpeed("kanan_ke_tengah_1")
        kecepatan_diperbolehkan_kiri = traci.lane.getMaxSpeed(
            "kiri_ke_tengah_0"
        ) + traci.lane.getMaxSpeed("kiri_ke_tengah_1")
        kecepatan_diperbolehkan_atas = traci.lane.getMaxSpeed(
            "atas_ke_tengah_0"
        ) + traci.lane.getMaxSpeed("atas_ke_tengah_1")

        obs2 = [
            panjang_antrian_bawah,
            panjang_antrian_kanan,
            panjang_antrian_kiri,
            panjang_antrian_atas,
            waktu_menunggu_bawah / (panjang_antrian_bawah + 1e-3),
            waktu_menunggu_kanan / (panjang_antrian_kanan + 1e-3),
            waktu_menunggu_kiri / (panjang_antrian_kiri + 1e-3),
            panjang_antrian_atas / (waktu_menunggu_atas + 1e-3),
        ]

        # banyak_lolos_perempatan     = traci.edge.getLastStepVehicleNumber('titik_tengah')
        # banyak_antrian_perempatan   = traci.edge.getLastStepHaltingNumber('titik_tengah')

        reward = 0
        banyak_kendaraan_tabrakan = traci.simulation.getCollidingVehiclesNumber()

        banyak_kendaraan_tersangkut = 0
        linkIndexes = traci.trafficlight.getControlledLinks("lampu_lalu_lintas")

        for linkIndex in range(len(linkIndexes)):
            blockingVehicles = traci.trafficlight.getBlockingVehicles(
                "lampu_lalu_lintas", linkIndex
            )
            banyak_kendaraan_tersangkut += len(blockingVehicles)

        waktu_travel_bawah = 1 - (
            kecepatan_kendaraan_bawah / kecepatan_diperbolehkan_bawah
        )
        waktu_travel_kanan = 1 - (
            kecepatan_kendaraan_kanan / kecepatan_diperbolehkan_kanan
        )
        waktu_travel_kiri = 1 - (
            kecepatan_kendaraan_kiri / kecepatan_diperbolehkan_kiri
        )
        waktu_travel_atas = 1 - (
            kecepatan_kendaraan_atas / kecepatan_diperbolehkan_atas
        )

        reward += waktu_travel_bawah * -0.25 + waktu_menunggu_bawah * -0.25
        reward += waktu_travel_kanan * -0.25 + waktu_menunggu_kanan * -0.25
        reward += waktu_travel_kiri * -0.25 + waktu_menunggu_kiri * -0.25
        reward += waktu_travel_atas * -0.25 + waktu_menunggu_atas * -0.25
        reward += phase_changed * -0.5
        # reward += (banyak_lolos_perempatan - banyak_antrian_perempatan)
        reward += banyak_kendaraan_tabrakan * -2.0
        reward += banyak_kendaraan_tersangkut * -0.5

        self.time += 1
        done = traci.simulation.getMinExpectedNumber() <= 0

        if not done:
            done = self.time > 30000

        """ out_arr = []
        for arr in kendaraan_array:
            start = np.full((1, 3), -0.5)

            if len(arr) > 0:
                zeros   = np.full((50 - len(arr), 3), -1)
                obs     = np.array(arr)
                obs     = np.concatenate([start, obs, zeros], 0)
            else:
                zeros   = np.full((50, 3), -1)
                obs     = np.concatenate([start, zeros], 0)
            out_arr.append(obs) """

        """ if len(kendaraan_array) > 0:
            zeros   = np.full((50 - len(arr), 3), -1)
            obs     = np.array(arr)
            obs     = np.concatenate([start, obs, zeros], 0)

        else:
            zeros   = np.full((150, 3), -1)
            obs     = np.concatenate([start, zeros], 0) """

        obs2 = np.stack(obs2)

        return torch.Tensor([obs, obs2]), reward, torch.Tensor(done)
