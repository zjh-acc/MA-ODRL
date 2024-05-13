import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

# pass
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

# pass
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

# pass
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

# pass
class PFMs(object):
    def __init__(
        self, 
        type: int, 
        model_runtime : float,
        model_size: float,
        
        zero_shot : float,
        one_shot : float,
        PFMj_context_win_size : float) -> None:
        """ initialize the information.
        Args:
            type: the type of the information.
            data_size: the data size of the information.
        """
        self._type = type
        self._model_runtime= model_runtime
        self._model_size = model_size
        self._zero_shot = zero_shot
        self._one_shot = one_shot
        self._PFMj_context_win_size = PFMj_context_win_size

    def __str__(self) -> str:
        return f"type: {self._type}\n model_runtime: {self._model_runtime}\n model_size: {self._model_size} \
        zero_shot: {self._zero_shot}\n one_shot: {self._one_shot}\n PFMj_context_win_size: {self._PFMj_context_win_size}"

    def get_type(self) -> int:
        return int(self._type)
    
    def get_model_runtime(self) -> float:
        return self._model_runtime
    
    def get_model_size(self) -> float:
        return self._model_size
    
    def get_zero_shot(self) -> float:
        return self._zero_shot
    
    def get_one_shote(self) -> float:
        return self._one_shot
    
    def get_PFMj_context_win_size(self) -> float:
        return self._PFMj_context_win_size

# pass
class service(object):
    def __init__(
        self, 
        type: int, 
        data_size: float,
        GPU_cycles : float,
        PFMs : List[PFMs],
        update_interval: float) -> None:
        """ initialize the service.
        Args:
            type: the type of the service.
            data_size: the data size of the service.
            GPU_cycles: 
            PFMs:
        """
        self._type = type
        self._data_size = data_size
        self._GPU_cycles = GPU_cycles
        self._PFMs = PFMs
        self._update_interval = update_interval

    def __str__(self) -> str:
        return f"type: {self._type}\n data_size: {self._data_size}\n 
        GPU_cycles: {self._GPU_cycle}\n PFMs: {self._PFMs}\n update_interval: {self._update_interval}"

    def get_type(self) -> int:
        return int(self._type)
    
    def get_data_size(self) -> float:
        return self._data_size
    
    def get_GPU_cycles(self) -> float:
        return self._GPU_cycles
    
    def get_PFMs(self) -> List[PFMs]:
        # 二维数组，对应关系
        self._PFMs = self._PFMs[type]
        return self._PFMs

    def get_update_interval(self) -> float:
        return self._update_interval


class vehicle(object):
    """" the vehicle. """
    def __init__(
        self, 
        vehicle_index: int,
        vehicle_trajectory: trajectory,

        vehicle_location : location,
        vehicle_service : service,

        transmission_power: float,

        # 车辆v产生服务s，服务需要由边缘节点上的模型执行，多个模型都能执行这个服务，两者有个一对多的关系，这个关系代码实现应该在哪个模块中实现
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

        seed._vehicle_location = vehicle_location
        seed._vehicle_service = vehicle_service

        if self._sensed_information_number > self._information_number:
            raise ValueError("The max information number must be less than the information number.")

    def __str__(self) -> str:
        return f"vehicle_index: {self._vehicle_index}\n vehicle_trajectory: {self._vehicle_trajectory}\n information_number: {self._information_number}\n sensed_information_number: {self._sensed_information_number}\n min_sensing_cost: {self._min_sensing_cost}\n max_sensing_cost: {self._max_sensing_cost}\n transmission_power: {self._transmission_power}\n seed: {self._seed}\n information_canbe_sensed: {self._information_canbe_sensed}\n sensing_cost: {self._sensing_cost}"

    def get_vehicle_index(self) -> int:
        return int(self._vehicle_index)

    def get_transmission_power(self) -> float:
        return self._transmission_power

    def get_vehicle_location(self) -> location:
        return self._vehicle_location
    
    def get_vehicle_service(self) -> service:
        return self._vehicle_service
    
    def get_vehicle_location(self, nowTimeSlot: int) -> location:
        return self._vehicle_trajectory.get_location(nowTimeSlot)

    def get_distance_between_edge(self, nowTimeSlot: int, edge_location: location) -> float:
        return self._vehicle_trajectory.get_location(nowTimeSlot).get_distance(edge_location)
    
    def get_vehicle_trajectory(self) -> trajectory:
        return self._vehicle_trajectory

class vehicleList(object):
    """ the vehicle list. """
    def __init__(
        self, 
        number: int, 
        
        time_slots: timeSlots,
        trajectories_file_name: str,


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

        self._transmission_power = transmission_power
        self._seeds = seeds

        self._vehicle_trajectories = self.read_vehicle_trajectories(time_slots)
        self._vehicle_list: List[vehicle] = []

        for i in range(self._number):
            self._vehicle_list.append(
                vehicle(
                    vehicle_index=i,
                    vehicle_trajectory=self._vehicle_trajectories[i],


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

class vehicleAction(object):
    """ the action of the vehicle. """
    def __init__(
        self, 
        vehicle_index: int,
        now_time: int,
        sensed_information: Optional[List[int]] = None,
        sensing_frequencies: Optional[List[float]] = None,
        uploading_priorities: Optional[List[float]] = None,
        transmission_power: Optional[float] = None, 
        action_time: Optional[int] = None) -> None:
        """ initialize the vehicle action.
        Args:
            vehicle_index: the index of vehicle. e.g. 0, 1, 2, ...
            now_time: the current time.
            vehicle_list: the vehicle list.
            sensed_information: the sensed information.
                e.g., 0 or 1, indicates whether the information is sensed or not.
                and the type of information is rocorded in vehicle.information_canbe_sensed .
            sensing_frequencies: the sensing frequencies.
            uploading_priorities: the uploading priorities.
            transmission_power: the transmission power.
            action_time: the time of the action.
        """
        self._vehicle_index = vehicle_index
        self._now_time = now_time
        self._sensed_information = sensed_information
        self._sensing_frequencies = sensing_frequencies
        self._uploading_priorities = uploading_priorities
        self._transmission_power = transmission_power
        self._action_time = action_time
    
    def __str__(self) -> str:
        return f"vehicle_index: {self._vehicle_index}, 
        now_time: {self._now_time}, 
        sensed_information: {self._sensed_information}, 
        sensing_frequencies: {self._sensing_frequencies}, 
        uploading_priorities: {self._uploading_priorities}, 
        transmission_power: {self._transmission_power}, 
        action_time: {self._action_time}"

    def check_action(self, nowTimeSlot: int, vehicle_list: vehicleList) -> bool:
        """ check the action.
        Args:
            nowTimeSlot: the time of the action.
        Returns:
            True if the action is valid.
        """
        if self._action_time != nowTimeSlot:
            return False
        if self._vehicle_index >= len(vehicle_list.get_vehicle_list()):
            return False
        vehicle = vehicle_list.get_vehicle(self._vehicle_index)
        if not (len(self._sensed_information) == len(self._sensing_frequencies) == len(self._uploading_priorities)):
            return False
        if self._transmission_power > vehicle.get_transmission_power():
            return False
        return True

    def get_sensed_information(self) -> List[int]:
        return self._sensed_information

    def get_sensing_frequencies(self) -> List[float]:
        return self._sensing_frequencies
    
    def get_uploading_priorities(self) -> List[float]:
        return self._uploading_priorities

    def get_transmission_power(self) -> float:
        return self._transmission_power

    def get_action_time(self) -> int:
        return self._action_time
    
# pass
class edge(object):
    """ the edge. """
    def __init__(
        self, 
        edge_index: int,
        edge_location: location,
        communication_range: float,
        small_range : float,
        PFMs : List[PFMs],
        
        MAX_power : float,

        Strage_capability : float,
        GPU_memory : float,
        GPU_cycles : float) -> None:
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
        self._small_range = small_range
        
        self._PFMs = PFMs
        self._MAX_power = MAX_power
        self._Strage_capability = Strage_capability
        self._GPU_memory = GPU_memory
        self._GPU_cycles = GPU_cycles


    def get_edge_index(self) -> int:
        return int(self._edge_index)

    def get_edge_location(self) -> location:
        return self._edge_location

    def get_communication_range(self) -> float:
        return self._communication_range
    
    def get_small_range(self) -> float:
        return self._small_range
    
    # def get_bandwidth(self) -> float:
    #     return self._bandwidth

    def get_PFMs(self) -> List[PFMs]:
        return self._PFMs

    def get_MAX_power(self) -> float:
        return self._MAX_power
    
    def get_Strage_capability(self) -> float:
        return self._Strage_capability
    
    def get_GPU_cycles(self) -> float:
        return self._GPU_cycles
    
    def get_GPU_memory(self) -> float:
        return self._GPU_memory

class edgeAction(object):
    """ the action of the edge. """
    def __init__(
        self, 
        edge: edge,
        now_time: int,
        vehicle_pair_number: int,
        bandwidth_allocation: np.ndarray,

        PFMs_number : int,


        action_time: int) -> None:
        """ initialize the edge action.
        Args:
            edge: the edge.
            now_time: the current time.
            vehicle_number: the number of vehicles.
            action_time: the time of the action.
        """
        self._edge_bandwidth = edge.get_bandwidth()
        self._now_time = now_time
        self._vehicle_pair_number = vehicle_pair_number
        self._action_time = action_time
        self._bandwidth_allocation = bandwidth_allocation

    def __str__(self) -> str:
        return f"edge_bandwidth: {self._edge_bandwidth}\n now_time: {self._now_time}\n vehicle_number: {self._vehicle_number}\n action_time: {self._action_time}\n bandwidth_allocation: {self._bandwidth_allocation}"

    def get_bandwidth_allocation(self) -> np.ndarray:
        return self._bandwidth_allocation

    def get_the_sum_of_bandwidth_allocation(self) -> float:
        return np.sum(self._bandwidth_allocation)

    def check_action(self, nowTimeSlot: int) -> bool:
        """ check the action.
        Args:
            nowTimeSlot: the time of the action.
        Returns:
            True if the action is valid.
        """
        if self._action_time != nowTimeSlot:
            print("the action time is not correct.")
            return False
        if self._vehicle_number != len(self._bandwidth_allocation):
            print("the number of vehicles is not correct.")
            return False
        if self._edge_bandwidth < self.get_the_sum_of_bandwidth_allocation():
            print("the allocated bandwidth exceeds its cability.")
            print("the allocated bandwidth:", self.get_the_sum_of_bandwidth_allocation())
            print("the edge bandwidth:", self._edge_bandwidth)
            return False
        return True


class serviceList(object):
    """
    This class is used to store the information list of the environment.
    to store the whole information list, 
    which randomly initialize the characteristics of each information,
    including the type, data size, update interval.
    """
    def __init__(
        self, 
        number: int, 
        data_size_low_bound: float,
        data_size_up_bound: float,
        data_types_number: int,

        update_interval_low_bound: float,
        update_interval_up_bound: float,

        vehicle_list: vehicleList,
        edge_node: edge,
        
        white_gaussian_noise: int,
        mean_channel_fading_gain: float,
        second_moment_channel_fading_gain: float,
        path_loss_exponent: int,
        seed: int, ) -> None:
        """ initialize the information list.
        Args:
            number: the number of information int the list.
            seed: the random seed.
            data_size_low_bound: the low bound of the data size.
            data_size_up_bound: the up bound of the data size.
            data_types_number: the number of data types.
            update_interval_low_bound: the low bound of the update interval.
            update_interval_up_bound: the up bound of the update interval.
        """
        self._number = number
        self._seed = seed
        self._data_size_low_bound = data_size_low_bound
        self._data_size_up_bound = data_size_up_bound
        self._data_types_number = data_types_number
        self._update_interval_low_bound = update_interval_low_bound
        self._update_interval_up_bound = update_interval_up_bound

        if self._data_types_number != self._number:
            self._data_types_number = self._number
        np.random.seed(self._seed)
        self._types_of_information: List[int] = np.random.permutation(
            list(range(self._data_types_number))
        )

        np.random.seed(self._seed)
        self._data_size_of_information: List[float] = np.random.uniform(
            low=self._data_size_low_bound,
            high=self._data_size_up_bound,
            size=self._number,
        )

        np.random.seed(self._seed)
        self._update_interval_of_information: List[float] = np.random.uniform(
            low=self._update_interval_low_bound, 
            high=self._update_interval_up_bound,
            size=self._number, 
        )

        self._information_list: List[information] = []
        for i in range(self._number):
            self._information_list.append(
                information(
                    type=self._types_of_information[i],
                    data_size=self._data_size_of_information[i],
                    update_interval=self._update_interval_of_information[i]
                )
            )
        
        self._mean_service_time_of_types, self._second_moment_service_time_of_types = \
            self.compute_mean_and_second_moment_service_time_of_types(
                vehicle_list=vehicle_list,
                edge_node=edge_node,
                white_gaussian_noise=white_gaussian_noise,
                mean_channel_fading_gain=mean_channel_fading_gain,
                second_moment_channel_fading_gain=second_moment_channel_fading_gain,
                path_loss_exponent=path_loss_exponent
            )

    def __str__(self) -> str:
        return f"number: {self._number}\n seed: {self._seed}\n data_size_low_bound: {self._data_size_low_bound}\n data_size_up_bound: {self._data_size_up_bound}\n data_types_number: {self._data_types_number}\n update_interval_low_bound: {self._update_interval_low_bound}\n update_interval_up_bound: {self._update_interval_up_bound}\n types_of_information: {self._types_of_information}\n data_size_of_information: {self._data_size_of_information}\n update_inter_of_information: {self._update_interval_of_information}"

    def get_number(self) -> int:
        """ get the number of information.
        Returns:
            the number of information.
        """
        return int(self._number)

    def get_information_list(self) -> List[information]:
        return self._information_list
    
    def get_information_type_by_index(self, index: int) -> int:
        if index >= self._number:
            raise ValueError("The index is out of range.")
        return self._information_list[index].get_type()

    def get_information_by_type(self, type: int) -> information:
        """method to get the information by type"""
        for infor in self._information_list:
            if infor.get_type() == type:
                return infor
        raise ValueError("The type is not in the list.")

    def get_information_siez_by_type(self, type: int) -> float:
        """method to get the information size by type"""
        for information in self._information_list:
            if information.get_type() == type:
                return information.get_data_size()
        raise ValueError("The type is not in the list.")

    def get_information_update_interval_by_type(self, type: int) -> float:
        """method to get the information update interval by type"""
        for information in self._information_list:
            if information.get_type() == type:
                return information.get_update_interval()
        raise ValueError("The type is not in the list.")

    def get_mean_service_time_of_types(self) -> np.ndarray:
        return self._mean_service_time_of_types
    
    def get_second_moment_service_time_of_types(self) -> np.ndarray:
        return self._second_moment_service_time_of_types

    def get_mean_service_time_by_vehicle_and_type(self, vehicle_index: int, data_type_index: int) -> float:
        return self._mean_service_time_of_types[vehicle_index][data_type_index]

    def get_second_moment_service_time_by_vehicle_and_type(self, vehicle_index: int, data_type_index: int) -> float:
        return self._second_moment_service_time_of_types[vehicle_index][data_type_index]

    def compute_mean_and_second_moment_service_time_of_types(
        self, 
        vehicle_list: vehicleList,
        edge_node: edge,
        white_gaussian_noise,
        mean_channel_fading_gain,
        second_moment_channel_fading_gain,
        path_loss_exponent) -> Tuple[np.ndarray, np.ndarray]:
        """
        method to get the mean and second moment service time of 
        each type of information at each vehile.
        Args:
            vehicle_list: the vehicle list.
            edge_node: the edge node.
            white_gaussian_noise: the additive white gaussian noise.
            mean_channel_fading_gain: the mean channel fadding gain.
            second_moment_channel_fading_gain: the second channel fadding gain.
            path_loss_exponent: the path loss exponent.
        Returns:
            the mean and second moment service time of each type of information.
        """
        from Environments.utilities import generate_channel_fading_gain, compute_SNR, compute_transmission_rate

        vehicle_number = vehicle_list.get_number()
        mean_service_time_of_types = np.zeros(shape=(vehicle_number, self._data_types_number))
        second_moment_service_time_of_types = np.zeros(shape=(vehicle_number, self._data_types_number))

        for vehicle_index in range(vehicle_number):
            vehicle = vehicle_list.get_vehicle(vehicle_index)
            for data_type_index in range(self._data_types_number):
                transmission_time = []
                for location in vehicle.get_vehicle_trajectory().get_locations():
                    distance = location.get_distance(edge_node.get_edge_location())
                    channel_fading_gain = generate_channel_fading_gain(
                        mean_channel_fading_gain=mean_channel_fading_gain,
                        second_moment_channel_fading_gain=second_moment_channel_fading_gain
                    )
                    SNR = compute_SNR(
                        white_gaussian_noise=white_gaussian_noise,
                        channel_fading_gain=channel_fading_gain,
                        distance=distance,
                        path_loss_exponent=path_loss_exponent,
                        transmission_power=vehicle.get_transmission_power()
                    )
                    bandwidth = edge_node.get_bandwidth() / vehicle_number
                    if self.get_information_siez_by_type(data_type_index) / compute_transmission_rate(SNR, bandwidth) != np.Inf:
                        transmission_time.append(self.get_information_siez_by_type(data_type_index) / compute_transmission_rate(SNR, bandwidth))
                mean_service_time = np.array(transmission_time).mean()
                second_moment_service_time = np.array(transmission_time).var()
                mean_service_time_of_types[vehicle_index][data_type_index] = mean_service_time
                second_moment_service_time_of_types[vehicle_index][data_type_index] = second_moment_service_time

        return mean_service_time_of_types, second_moment_service_time_of_types



