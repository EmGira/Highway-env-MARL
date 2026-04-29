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
                        "lateral": True,
                        "target_speeds": [0, 4.5, 9],
                    }
                    
                },

                "duration": 40,  # [s]

            
                "destination": None,
                "multi_destinations": None,
                "spawn_points": None,

                "controlled_vehicles": num_agents,
                "initial_vehicle_count": 1, 
                "spawn_probability": 0.6,

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

              
              
                "initial_simulation_steps": 3, 
                "ego_vehicle_speed_limit": 9, 
                "speeding_penalty": -0.5,
                "tailgating_penalty": -2

            }

def get_improved_Simple_config(num_agents=2, obs_type="Kinematics"):

    config = get_simple_multi_agent_config(num_agents=num_agents, obs_type=obs_type)

    config["initial_vehicle_count"] = 4#1
    config["spawn_probability"] = 0.2#0.6

    config["initial_simulation_steps"] = 7 # 3
    config["ego_vehicle_speed_limit"] = 9

    config["speeding_penalty"] = -2
    config["tailgating_penalty"] = -2
    config["stopped_penalty"] = -0.05


    config["destination"] = None
    config["multi_destinations"] = None
    config["spawn_points"] = None


    config["observation"]["observation_config"]["absolute"] = False 
    config["action"]["action_config"]["target_speeds"] = [0, 1.5, 3, 4.5, 6]
    
    config["reward_speed_range"] = [0, 6]
    config["high_speed_reward"] = 0.5

    config["normalize_reward"] = True

    
    config["duration"] = 60 

    return config


def experimental(num_agents = 2, obs_type="kinematics"):

    config = get_simple_multi_agent_config(num_agents=num_agents, obs_type=obs_type)

    config["collision_reward"] = -200
    config["arrived_reward"] = 100

    config["initial_vehicle_count"] = 4#1
    config["spawn_probability"] = 0.2#0.6

    config["initial_simulation_steps"] = 7 # 3
    config["ego_vehicle_speed_limit"] = 9

    config["speeding_penalty"] = -0.1
    config["tailgating_penalty"] = -0.1
    config["stopped_penalty"] = -0.1


    config["destination"] = None
    config["multi_destinations"] = None
    config["spawn_points"] = None


    config["observation"]["observation_config"]["absolute"] = False 
    config["action"]["action_config"]["target_speeds"] = [0, 1.5, 3, 4.5, 6]
    
    config["reward_speed_range"] = [0, 6]
    config["high_speed_reward"] = 0.2

    
    config["duration"] = 60 



    return config

