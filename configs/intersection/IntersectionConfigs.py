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
                        "absolute": False, #changed
                        "flatten": False,
                        "observe_intentions": False, #changed
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
    return {
        "observation": { 
            "type": "MultiAgentObservation",
            "observation_config": { 
                "type": obs_type,
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": False, 
                "flatten": False,
                "observe_intentions": False, 
            }
        },

        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
                "target_speeds": [0, 1.5, 3, 4.5, 6],
            }
        },

        "duration": 60,  # [s]
        

        # Rewards & Penalties
        "collision_reward": -50, 
        "arrived_reward": 50, 

        "high_speed_reward": 0.5,
        "reward_speed_range": [0, 6],
        
        
        "offroad_terminal": False,

        "speeding_penalty": -0.01,     #-2
        "tailgating_penalty": 0,   #-2
        "stopped_penalty": -0.05,

        "normalize_reward": True, 
        
        # Simulation specifics
        "initial_simulation_steps": 7, 
        "ego_vehicle_speed_limit": 9, 

        "controlled_vehicles": num_agents,
        "initial_vehicle_count": 4, 
        "spawn_probability": 0.2,



        #Visualization
        "screen_width": 1200,
        "screen_height": 1200,
        "centering_position": [0.5, 0.6],
        "scaling": 5.5 * 1.3,

        #Destinations and spawn points
        "destination": None,
        "multi_destinations": None,
        "spawn_points": None,
    }
