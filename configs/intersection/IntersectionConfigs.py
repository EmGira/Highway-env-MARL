def get_simple_multi_agent_config(num_agents=2, obs_type="Kinematics"):

    return {
              
                "observation": { 
                    "type": "MultiAgentObservation",
                    "observation_config": { 
                        "type": "Kinematics",
                        "vehicles_count": 15,  #15
                        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                        "features_range": {
                            "x": [-100, 100],
                            "y": [-100, 100],
                            "vx": [-20, 20],
                            "vy": [-20, 20],
                        },
                        "absolute": True, #changed
                        "flatten": False,
                        "observe_intentions": True, #changed
                    }
                },

                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                        "longitudinal": True,
                        "lateral": False,
                        "target_speeds": [0, 4.5, 9],
                    }
                    
                },

                "duration": 40,  # [s]

            
                "destination": "o1",
                "multi_destinations": None,
                "spawn_points": None,

                "controlled_vehicles": num_agents,
                "initial_vehicle_count": 1, #changed from 1 
                "spawn_probability": 0.6, #changed from 0.6 to 0.

                "screen_width": 1200,
                "screen_height": 1200,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,

                "collision_reward": -50, #-50
                "high_speed_reward": 1,
                "arrived_reward": 50, #+50
                "reward_speed_range": [7.0, 9.0],
                "normalize_reward": False, #try normalize 
                "offroad_terminal": False,

            }

def get_randomized_Simple_config(num_agents=2, obs_type="Kinematics"):

    config = get_simple_multi_agent_config(num_agents=num_agents, obs_type=obs_type)

    config["initial_vehicle_count"] = 4
    config["spawn_probability"] = 0.2

    config["destination"] = None
    config["multi_destinations"] = None
    config["spawn_points"] = None

    config["observation"]["observation_config"]["absolute"] = False
    config["action"]["action_config"]["target_speeds"] = [0, 1.5, 3, 4.5, 6]
    
    config["reward_speed_range"] = [0, 6]

    config["high_speed_reward"] = 0.5
    config["duration"] = 60

    return config

def get_busy_intersection_config(num_agents=2, obs_type="Kinematics"):

    return {
              
                "observation": { 
                    "type": "MultiAgentObservation",
                    "observation_config": { 
                        "type": "Kinematics",
                        "vehicles_count": 15,  
                        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                        "features_range": {
                            "x": [-100, 100],
                            "y": [-100, 100],
                            "vx": [-20, 20],
                            "vy": [-20, 20],
                        },
                        "absolute": True,
                        "flatten": False,
                        "observe_intentions": False,
                    }
                },

                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "DiscreteMetaAction",
                        "longitudinal": True,
                        "lateral": False,
                        "target_speeds": [0, 4.5, 9],
                    }
                    
                },

                "duration": 40,  # [s], longer periods so the agent has time to wait for traffic

                "destination": "o1",
                "multi_destinations": None,
                "spawn_points": None,

                "controlled_vehicles": num_agents,
                "initial_vehicle_count": 5, #higer traffic
                "spawn_probability": 0.5,
                
                "screen_width": 1200,
                "screen_height": 1200,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,

                "collision_reward": -5,
                "high_speed_reward": 1,
                "arrived_reward": 1,
                "reward_speed_range": [7.0, 9.0],
                "normalize_reward": False,
                "offroad_terminal": False,

            }



    return {
        "observation": { 
            "type": "MultiAgentObservation",
            "observation_config": { "type": "Kinematics",
                         
                                    "absolute": False,
                                    "order": "sorted",
                                    "observe_intentions": True #the destination of other vehicle is observed, this is True because it makes sense that cars would be using a turn signal
                                }
        },

        "action": { 
            "type": "MultiAgentAction",
            "action_config": { "type": "DiscreteMetaAction" }
        },

    
        "controlled_vehicles": num_agents,
        "vehicles_count": 10, 

        # Intersections need to reward slower speeds compared to highway.
        "reward_speed_range": [10, 30], 
        
        # heavy collision punishment so agents quickly learn NOT to crash
        "collision_reward": -50.0, # TODOO try -50 
        
        #reward for staying alive and exiting the intersection
        "high_speed_reward": 1, 
        "arrived_reward": 5.0, 

        "on_road_reward": 0.1,
        "offroad_terminal": True,
     
        
        "offscreen_rendering": False, 
        "render_mode": None,
        
        
        "normalize_reward": False, 
        
        #"duration": 10,  TODOO figure out if needed in heavy training 
    }