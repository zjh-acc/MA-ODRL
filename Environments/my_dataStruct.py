import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class timeSlots(object):
    """The set of discrete time slots of the system"""
    def __init__(
        self, 
        start: int, 
        end: int, 
        slot_length: int) -> None:
        """method to initialize the time slots
        Args:
            start: the start time of the system
            end: the end time of the system
            slot_length: the length of each time slot"""   
        self._start = start
        self._end = end
        self._slot_length = slot_length

        self._number = int((end - start + 1) / slot_length)
        self._now = start
        self.reset()

    def __str__(self) -> str:
        return f"now time: {self._now}, [{self._start} , {self._end}] with {self._slot_length} = {self._number} slots"
    
    def add_time(self) -> None:
        """method to add time to the system"""
        self._now += 1

    def is_end(self) -> bool:
        """method to check if the system is at the end of the time slots"""
        return self._now >= self._end

    def get_slot_length(self) -> int:
        """method to get the length of each time slot"""
        return int(self._slot_length)

    def get_number(self) -> int:
        return int(self._number)

    def now(self) -> int:
        return int(self._now)
    
    def get_start(self) -> int:
        return int(self._start)

    def get_end(self) -> int:
        return int(self._end)

    def reset(self) -> None:
        self._now = self._start

class location(object):
    """ the location of the node. """
    def __init__(self, x: float, y: float) -> None:
        """ initialize the location.
        Args:
            x: the x coordinate.
            y: the y coordinate.
        """
        self._x = x
        self._y = y

    def __str__(self) -> str:
        return f"x: {self._x}, y: {self._y}"

    def get_x(self) -> float:
        return self._x

    def get_y(self) -> float:
        return self._y

    def get_distance(self, location: "location") -> float:
        """ get the distance between two locations.
        Args:
            location: the location.
        Returns:
            the distance.
        """
        return np.math.sqrt(
            (self._x - location.get_x())**2 + 
            (self._y - location.get_y())**2
        )

class trajectory(object):
    """ the trajectory of the node. """
    def __init__(self, 
                timeSlots: timeSlots, 
                locations: List[location]) -> None:
        """ initialize the trajectory.
        Args:
            max_time_slots: the maximum number of time slots.
            locations: the location list.
        """
        self._locations = locations

        if len(self._locations) != timeSlots.get_number():
            raise ValueError("The number of locations must be equal to the max_timestampes.")

# __str__()，用于返回对象的字符串表示形式。通常在打印对象或将对象转换为字符串时被调用。
    def __str__(self) -> str:
        # 将列表推导式生成的位置对象字符串列表转换为字符串。外部的 str() 函数将整个列表转换为字符串表示形式，包括方括号。
        return str([str(location) for location in self._locations])

    def get_location(self, nowTimeSlot: int) -> location:
        """ get the location of the timestamp.
        Args:
            timestamp: the timestamp.
        Returns:
            the location.
        """
        return self._locations[nowTimeSlot]

    def get_locations(self) -> List[location]:
        """ get the locations.
        Returns:
            the locations.
        """
        return self._locations

    def __str__(self) -> str:
        """ print the trajectory.
        Returns:
            the string of the trajectory.
        """
        print_result= ""
        # enumerate() 函数用于将一个可迭代对象（例如列表）转换为一个由  索引  和  对应元素组成的枚举对象。
        for index, location in enumerate(self._locations):
            if index % 10 == 0:
                print_result += "\n"
            print_result += "(" + str(index) + ", "
            print_result += str(location.get_x()) + ", "
            print_result += str(location.get_y()) + ")"
        return print_result
    

class vehicle(object):
    """" the vehicle. """
    def __init__(
        self, 
        vehicle_index: int,
        vehicle_trajectory: trajectory,

        transmission_power: float,
        # seed: int: 这个参数是一个整数类型，表示随机数生成的种子值。
        # seed 参数允许我们在随机数生成器中设置一个初始值，以获得可控制的随机数序列。
        seed: int) -> None:
        """ initialize the vehicle.
        Args:
            vehicle_index: the index of vehicle. e.g. 0, 1, 2, ...
            vehicle_trajectory: the trajectory of the vehicle.
            information_number: the number of information list.
            sensed_information_number: the maximum number of information, which can be sensed by the vehicle.
            min_sensing_cost: the minimum sensing cost.
            max_sensing_cost: the maximum sensing cost.
            transmission_power: the transmission power.
            seed: the random seed.
        """
        self._vehicle_index = vehicle_index
        self._vehicle_trajectory = vehicle_trajectory

        self._transmission_power = transmission_power
        self._seed = seed

        if self._sensed_information_number > self._information_number:
            raise ValueError("The max information number must be less than the information number.")
        
        self._information_canbe_sensed = self.information_types_can_be_sensed()

        self._sensing_cost = self.sensing_cost_of_information()

    def __str__(self) -> str:
        return f"vehicle_index: {self._vehicle_index}\n vehicle_trajectory: {self._vehicle_trajectory}\n information_number: {self._information_number}\n sensed_information_number: {self._sensed_information_number}\n min_sensing_cost: {self._min_sensing_cost}\n max_sensing_cost: {self._max_sensing_cost}\n transmission_power: {self._transmission_power}\n seed: {self._seed}\n information_canbe_sensed: {self._information_canbe_sensed}\n sensing_cost: {self._sensing_cost}"

    def get_vehicle_index(self) -> int:
        return int(self._vehicle_index)

    def get_transmission_power(self) -> float:
        return self._transmission_power
        
        # sensed_information: Optional[List[int]]：这个参数是一个可选的整数列表，表示被感知到的信息。
        # Optional 表示这个参数可以是 None 或一个整数列表。如果 None，则表示没有被感知到的信息。
        # np.ndarray 表示返回一个 NumPy 数组。
    def get_sensed_information_type(self, sensed_information: Optional[List[int]]) -> np.ndarray:
        # 全零数组一维数组，其长度为 self._sensed_information_number
        sensed_information_type = np.zeros((self._sensed_information_number,))
        for i in range(self._sensed_information_number):
            if sensed_information[i] == 1:
                sensed_information_type[i] = self.get_information_canbe_sensed()[i]
            else:
                sensed_information_type[i] = -1
        return sensed_information_type

    def information_types_can_be_sensed(self) -> List[int]:
        # 设置 NumPy 随机数生成器的种子值为 self._seed
        np.random.seed(self._seed)
        return list(np.random.choice(
            a=self._information_number,
            size=self._sensed_information_number,
            replace=False))

    def sensing_cost_of_information(self) -> List[float]:
        np.random.seed(self._seed)
        # 用于生成服从均匀分布的随机数。
        # 其中 low 是均匀分布的下界，high 是均匀分布的上界，size 是生成的随机数的形状。
        return list(np.random.uniform(
            low=self._min_sensing_cost,
            high=self._max_sensing_cost,
            size=self._sensed_information_number
        ))

    def get_sensing_cost(self) -> List[float]:
        return self._sensing_cost
    
    def get_sensing_cost_by_type(self, type: int) -> float:
        for _ in range(self._sensed_information_number):
            if self._information_canbe_sensed[_] == type:
                return self._sensing_cost[_]
        raise ValueError("The type is not in the sensing cost list. type: " + str(type))
    
    def get_vehicle_location(self, nowTimeSlot: int) -> location:
        return self._vehicle_trajectory.get_location(nowTimeSlot)

    def get_distance_between_edge(self, nowTimeSlot: int, edge_location: location) -> float:
        return self._vehicle_trajectory.get_location(nowTimeSlot).get_distance(edge_location)

    def get_sensed_information_number(self) -> int:
        return self._sensed_information_number
    
    def get_information_canbe_sensed(self) -> List[int]:
        return self._information_canbe_sensed

    def get_information_type_canbe_sensed(self, index: int) -> int:
        return self._information_canbe_sensed[index]
    
    def get_vehicle_trajectory(self) -> trajectory:
        return self._vehicle_trajectory

class vehicleList(object):
    """ the vehicle list. """
    def __init__(
        self, 
        number: int, 
        time_slots: timeSlots,
        trajectories_file_name: str,
        information_number: int,
        sensed_information_number: int,
        min_sensing_cost: float,
        max_sensing_cost: float,
        transmission_power: float,
        # 生成多个随机数序列，您可以为每个序列指定一个种子值。
        # seeds: List[int] 表示一个整数类型的列表，其中包含了多个种子值。
        seeds: List[int]) -> None:
        """ initialize the vehicle list.
        Args:
            number: the number of vehicles.
            trajectories_file_name: the file name of the vehicle trajectories.
            information_number: the number of information list.
            sensed_information_number: the maximum number of information, which can be sensed by the vehicle.
            min_sensing_cost: the minimum sensing cost.
            max_sensing_cost: the maximum sensing cost.
            transmission_power: the transmission power.
            seeds: the random seed list.
        """
        self._number = number
        self._trajectories_file_name = trajectories_file_name
        self._information_number = information_number
        self._sensed_information_number = sensed_information_number
        self._min_sensing_cost = min_sensing_cost
        self._max_sensing_cost = max_sensing_cost
        self._transmission_power = transmission_power
        self._seeds = seeds

        self._vehicle_trajectories = self.read_vehicle_trajectories(time_slots)
        self._vehicle_list: List[vehicle] = []

        for i in range(self._number):
            self._vehicle_list.append(
                vehicle(
                    vehicle_index=i,
                    vehicle_trajectory=self._vehicle_trajectories[i],
                    information_number=self._information_number,
                    sensed_information_number=self._sensed_information_number,
                    min_sensing_cost=self._min_sensing_cost,
                    max_sensing_cost=self._max_sensing_cost,
                    transmission_power=self._transmission_power,
                    seed=self._seeds[i]
                )
            )

    def __str__(self) -> str:
        return f"number: {self._number}\n information_number: {self._information_number}\n sensed_information_number: {self._sensed_information_number}\n min_sensing_cost: {self._min_sensing_cost}\n max_sensing_cost: {self._max_sensing_cost}\n transmission_power: {self._transmission_power}\n seeds: {self._seeds}\n vehicle_list: {self._vehicle_list}" + "\n" + str([str(vehicle) for vehicle in self._vehicle_list])

    def get_number(self) -> int:
        return int(self._number)
    
    def get_vehicle_list(self) -> List[vehicle]:
        return self._vehicle_list

    def get_vehicle(self, vehicle_index: int) -> vehicle:
        return self._vehicle_list[vehicle_index]
    
    def get_sensed_information_number(self) -> int:
        return int(self._sensed_information_number)

    def get_vehicle_trajectories(self) -> List[trajectory]:
        return self._vehicle_trajectories

    def read_vehicle_trajectories(self, timeSlots: timeSlots) -> List[trajectory]:
# 使用了 Pandas 库中的 read_csv 函数，用于从一个 CSV 文件中读取数据并创建一个 DataFrame 对象。
        df = pd.read_csv(
            # 文件名，表示要读取的 CSV 文件的路径。
            self._trajectories_file_name, 
            # 指定列名，将 CSV 文件中的列映射到 DataFrame 中的列。header指定了 CSV 文件中包含   列名的行数。
            names=['vehicle_id', 'time', 'longitude', 'latitude'], header=0)

        max_vehicle_id = df['vehicle_id'].max()

        selected_vehicle_id = []
        for vehicle_id in range(int(max_vehicle_id)):
            # 选择指定车辆 ID 的所有行，创建一个新的 DataFrame new_df
            new_df = df[df['vehicle_id'] == vehicle_id]
            # 计算新 DataFrame 中 longitude 和 latitude 列的最大值和最小值，得到车辆轨迹的边界框。
            max_x = new_df['longitude'].max()
            max_y = new_df['latitude'].max()
            min_x = new_df['longitude'].min()
            min_y = new_df['latitude'].min()
            distance = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
            selected_vehicle_id.append(
                {
                    "vehicle_id": vehicle_id,
                    "distance": distance
                })
# 匿名函数 lambda，它接受一个参数 x，表示列表中的每个元素，然后返回 x["distance"]，即每个元素的距离值，作为排序的依据。
        selected_vehicle_id.sort(key=lambda x : x["distance"], reverse=True)
        new_vehicle_id = 0
        vehicle_trajectories: List[trajectory] = []
        # 遍历排序后的列表中的前 self._number 个元素
        for vehicle_id in selected_vehicle_id[ : self._number]:
            '''
            在 Pandas 中，使用布尔索引是一种常见的操作，可以用来选择 DataFrame 中满足特定条件的行。
            当您使用布尔表达式 df['vehicle_id'] == vehicle_id["vehicle_id"] 对 DataFrame 进行索引时，
            实际上是在告诉 Pandas 去筛选出满足条件的行，并将其组成一个新的 DataFrame 返回给您。
            Pandas 使用这种布尔索引的方式来提供灵活的数据选择和过滤功能，让用户能够根据特定条件快速获取所需的数据。
            '''
            new_df = df[df['vehicle_id'] == vehicle_id["vehicle_id"]]
            loc_list: List[location] = []
            # 遍历 new_df 中的每一行，并从中提取出经度和纬度信息。
            for row in new_df.itertuples():
                # time = getattr(row, 'time')
                x = getattr(row, 'longitude')
                y = getattr(row, 'latitude')
                loc = location(x, y)
                loc_list.append(loc)

            new_vehicle_trajectory: trajectory = trajectory(
                timeSlots=timeSlots,
                locations=loc_list
            )
            new_vehicle_id += 1
            vehicle_trajectories.append(new_vehicle_trajectory)

        return vehicle_trajectories


class edge(object):
    """ the edge. """
    def __init__(
        self, 
        edge_index: int,
        edge_location: location,
        communication_range: float,
        bandwidth: float) -> None:
        """ initialize the edge.
        Args:
            edge_index: the index of edge. e.g. 0, 1, 2, ...
            information_number: the number of information list.
            edge_location: the location of the edge.
            communication_range: the range of V2I communications.
            bandwidth: the bandwidth of edge.
        """
        self._edge_index = edge_index
        self._edge_location = edge_location
        self._communication_range = communication_range
        self._bandwidth = bandwidth

    def get_edge_index(self) -> int:
        return int(self._edge_index)

    def get_edge_location(self) -> location:
        return self._edge_location

    def get_communication_range(self) -> float:
        return self._communication_range
    
    def get_bandwidth(self) -> float:
        return self._bandwidth