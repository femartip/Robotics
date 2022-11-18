# Robotics

Model of the world we are working with:

![image](https://user-images.githubusercontent.com/99536660/202528461-c61f9254-3cf0-4c8e-8463-33859ee24ac4.png)

## KM&EKF - Kalman Filter & Extended Kalman Filter
  - This folder contians variuous implementations for a Turtlebot. This was simulated using the ROS environment. To try the programms, the "kalman_simulation" is needed, and should be placed as a ROS package. 
  
  -> KFBasicMotion.py; This program uses the kalman filter for a basic turtlebot, where the motion is based on a relative position between the beacon and the robot frame. The robot uses spherical wheels and only moves in a straight line. 
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/201900900-33fc0947-1843-4637-9936-a747b2d222b0.png)
  
  To try this, launch in ros exercise_1_spherical_wheels.launch
  
  ->KFWheelDiameter.py; Given the previous model, know we do not know the diameter of the wheels and we want to find it with a KF.
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/201904145-77c56236-2b68-41ff-acc7-a0bc2833b4a5.png)

  To try this, launch in ros exercise_2_wheel_diameter.launch
  
  ->EKF_For_Turtlebot.py; Given the first model, we change the wheels so instead of beeing spherical, they are circular. This enables us two type of motions, rotation and translation. This is a real model of the turtlebot.
  Result:  
  
  ![image](https://user-images.githubusercontent.com/99536660/201904651-cafa17ba-c6f7-4073-8f82-1086f5ad4e05.png)

  To try this, launch in ros exercise_4_normal_wheels.launch
  
  ->EKF-Distance&Direction.py; This model only uses the simple translation as in the first model. However know it is not based on beacons, but on distance and direction. This adds more noise.
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/201904945-eb1d8816-21ed-4413-9097-69447a14329e.png)

  To try this, launch in ros exercise_5_sensor_model.launch

## SLAM - Simultaneous Localization and Mapping
  - This folder contians variuous implementations for a Turtlebot. This was simulated using the ROS environment. To try the programms, the "slam_simulation" + "turtlebot3_costum" is needed, and should be placed as a ROS package. 
  
  ->EKF_dead_reckoning.py; EKF with dead reckoning, this is without sensor data. So it only estimates it state. This makes uncertainty increase over time. 
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/202521722-825560e9-7bec-4622-91f8-e7e67152062b.png)
  
  To try this, launch in ros demo_1_env.launch
  
  ->SLAM+KF.py; Robot estimates relative position to beacons, without knowing the beacons position. This is estimated using KF.
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/202522673-24c726e5-dad1-43c9-bf24-573b912932ff.png)
  
  To try this, launch in ros exercise_2_env.launch
  
  ->SLAM+PF.py; The same problem as in the  previous program, but resolved applying particle filter without resampling. 
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/202523262-2f6ec6f2-1915-4af4-8c93-7eb4147fee14.png)

  To try this, launch in ros exercise_3_env.launch
  
  ->SLAM+KF_loop_clousure.py; More realistic aproximation where robot does not see always all the beacons, here it only sees beacons that are in a 65º area in front of it. Implemented using KF. Noise increments over distance. 
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/202523986-97046b35-375c-4163-b996-6b4cad7b8a3e.png)

  To try this, launch in ros demo_2_env.launch
  
  ->SLAM+KF_landmark_association.py; Building up from previous model, know when robot sees and detects a beacon, it does not know which one is. This implementation adds a new beacon every measurement (using KF). Then beacons are merged by deciding they are the same one.
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/202525076-ecf5a731-a948-4c1e-967d-4a6a12be7961.png) *Without merging beacons.
  
  ![image](https://user-images.githubusercontent.com/99536660/202529009-0288c345-9935-4901-a659-aac0af3f8b0b.png)
  ![image](https://user-images.githubusercontent.com/99536660/202529097-b49e3735-4892-49d5-9918-eb9d1cb6eb12.png)
  *When merging
  
  To try this, launch in ros exercise_5_env.launch
  
  ->PF+EKF_Fast_SLAM.py; This implementation uses the same model as before, however know we use the PF ony to sample robots trajectory. Then for each samples trajectory, treates beacon position with KF.
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/202528283-c1993c4f-f4a4-4e17-bc94-3ed62f37f953.png)
  
  To try this, launch in ros exercise_6a_env.launch
  
  ->Fast_SLAM+l_a+resampling.py; Same implementation as before, but with the addition that we add landmark association and resampling to our model.
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/202527360-e8731496-427d-4b73-8653-6e6b165e7f0c.png)
  
  To try this, launch in ros exercise_6bc_env.launch
  
  ## DP - Dynamic Programming 
  
  Now we are working with a model of a maze where the robot needs to find a way towards the reward (star). The robot can only move N,S,E,W. All the solutions are implemented with dynamic programming.This was simulated using the ROS environment. To try the programms, the "dp_simulation" + "turtlebot3_costum" is needed, and should be placed as a ROS package.
  ![image](https://user-images.githubusercontent.com/99536660/202760181-95d9ccf6-8463-459f-a0b4-3e969c9ec72f.png)
  
  ->DP_Simple_Maze.py; Simplest model where robot can change in direction with any cost.
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/202760963-89155cca-4c2e-4f7f-bf96-798f483b5cb0.png)

  To try this, launch in ros exercise_1_env.launch

  ->DP_head&inertia.py; In this model the reward function changes, so it is more costly turning that going forwards, this is beacuse more force needs to be applied so that the robot turns. 
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/202761581-3d1e9a8b-5fd8-4721-af52-6def534cb40e.png)

  To try this, launch in ros exercise_2a_env.launch

  ->DP_Simple_Maze+Uncertainty.py; This model is based on the simple one, however uncertainty is added to the movement. Usually terrain is not perfect and can cause that one wheel spins faster than another, this could cause that eventough we think we are going in a straight line, we are tilting in another direction.
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/202762237-ca4e1f7a-aa7b-475d-a961-d0dafda4bab6.png)

  To try this, launch in ros exercise_2b_env.launch

  ->DP_Simple_Maze+Various_Rewards.py; In this case, instead of only considering there is one reward, there are more than one.
  Result:
  
  ![image](https://user-images.githubusercontent.com/99536660/202762551-0f092151-39a5-4ac3-943d-d84e6ac038e2.png)

  To try this, launch in ros exercise_2b_env.launch
