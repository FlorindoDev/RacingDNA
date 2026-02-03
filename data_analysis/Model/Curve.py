import math
from typing import List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt


class Curve:
         
    def __init__(
        self,
        corner_id: int,
        current_corner_dist: float,
        lower_bound: float,
        upper_bound: float,
        compound: str,
        life: int,
        stint: int,
        time: List[float],
        rpm: List[float],
        speed: List[float],
        throttle: List[float],
        brake: List[float],
        distance: List[float],
        acc_x: List[float],
        acc_y: List[float],
        acc_z: List[float],
        x: List[float],
        y: List[float],
        z: List[float]
    ):
        self.corner_id = corner_id
        self.current_corner_dist = current_corner_dist
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.time = time
        self.rpm = rpm
        self.speed = speed
        self.throttle = throttle
        self.brake = brake
        self.distance = distance
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.acc_z = acc_z
        self.x = x
        self.y = y
        self.z = z
        self.compound = compound
        self.life = life
        self.stint = stint
        self.latent_variable = None
        self.num_cluster = None

    @classmethod
    def from_sample(
        cls,
        sample: np.ndarray,
        mask: np.ndarray,
        latent_variable: Optional[List[float]] = None,
        num_cluster: Optional[int] = None,
    ):
        """
        sample: vettore 1D con layout:
          0: life
          1:51 speed
          51:101 rpm
          101:151 throttle
          151:201 brake
          201:251 acc_x
          251:301 acc_y
          301:351 acc_z
          352,353: compound one-hot (hard/medium) altrimenti soft
        mask: boolean mask compatibile (stesse slice)
        """
        compound = ""
        # compound
        if sample[352] != 0:
            compound = "HARD"
        elif sample[353] != 0:
            compound = "INTERMEDIATE"
        elif sample[354] != 0:
            compound = "MEDIUM"
        else:
            compound = "SOFT"

        bool_mask = mask.astype(bool)

        life = int(sample[0])  # se life non è mascherato, meglio così

        speed = sample[1:51][bool_mask[1:51]].tolist()
        rpm = sample[51:101][bool_mask[51:101]].tolist()
        throttle = sample[101:151][bool_mask[101:151]].tolist()
        brake = sample[151:201][bool_mask[151:201]].tolist()
        acc_x = sample[201:251][bool_mask[201:251]].tolist()
        acc_y = sample[251:301][bool_mask[251:301]].tolist()
        acc_z = sample[301:351][bool_mask[301:351]].tolist()

        # Create the instance
        instance = cls(
            corner_id=-1,
            current_corner_dist=-1,
            lower_bound=-1,
            upper_bound=-1,
            compound=compound,
            life=life,
            stint=-1,
            time=list(),
            rpm=rpm,
            speed=speed,
            throttle=throttle,
            brake=brake,
            distance=list(),
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            x=list(),
            y=list(),
            z=list()
        )
        
        # Set optional attributes
        instance.latent_variable = latent_variable
        instance.num_cluster = num_cluster
        
        return instance


    #########################################################
    #                   Metodi privati                      #
    #########################################################


    def print(self):
        print(f"[!] apex:{self.apex_dist}; start: {self.distance[0]} -> end: {self.distance[-1]}")

    
    #########################################################
    #                   Metrice semplici                    #
    #########################################################


    #########################################################
    #                Metrice più complesse                  #
    #########################################################


    #########################################################
    #                       Grafici                         #
    #########################################################
