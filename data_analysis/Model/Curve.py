import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_for_array import find_closest_value

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
        z: List[float],
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

