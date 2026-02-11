
# Baseline 스위칭

- 시작 좌표 변경!!

1. Proposed Method
`
python3 src/fly_locomotion/fly_locomotion/gesture_new.py
`

2. Gauntlet Method
`
python3 src/fly_locomotion/fly_locomotion/gesture_gauntlet.py
`

3. 이름 모름
`
python3 /home/min/7cmdehdrb/fuck_flight/src/fly_locomotion/fly_locomotion/gesture_finger.py
`


---

# 실험 세팅

1. Waypoint 실험
- Waypoint Reset

self.__reset = False -> False면 스폰, True면 디스폰
`
python3 /home/min/7cmdehdrb/fuck_flight/src/fly_locomotion/fly_locomotion/waypoint_pos_reset.py
`
`
ros2 topic pub /waypoint_reset std_msgs/msg/Bool data:\ false\ 
`

random_int_1 = 0
random_int_2 = 0 -> 활성화
`
python3 /home/min/7cmdehdrb/fuck_flight/src/fly_locomotion/fly_locomotion/spawn_mani.py
`


2. Manipulator 찾기 실험

self.__reset = True -> False면 스폰, True면 디스폰
`
python3 /home/min/7cmdehdrb/fuck_flight/src/fly_locomotion/fly_locomotion/waypoint_pos_reset.py
`
random_int_1 = 0
random_int_2 = 0 -> 비활성화
`
python3 /home/min/7cmdehdrb/fuck_flight/src/fly_locomotion/fly_locomotion/spawn_mani.py
`