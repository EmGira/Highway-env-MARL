

def get_default_custom_env_config(num_agents=2, obs_type="Kinematics"):

    return {
            "observation": { 
                "type": "MultiAgentObservation",
                "observation_config": { 
                    "type": "Kinematics"
                }
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction"
                }
            },
            "controlled_vehicles": num_agents,

            #rewards
            "collision_reward": -5,
            "high_speed_reward": 0.4,
            "reward_speed_range": [7.0, 9.0],
            "arrived_reward": 1,
            "right_lane_reward": 0.1,  
            "lane_change_reward": 0,
            "merging_speed_reward": -0.5,

            #others
            
            "normalize_reward": False,
            "offroad_terminal": False,
            "destination": "o1",
            "other_vehicles_destinations": [
                "o1", "o2", "sxs", "sxr", "exs", "exr", "nxs", "nxr",
            ],

            "initial_vehicle_count": 4,
            "spawn_probability": 0.1,

            "duration": 60,

            #graphics
            "screen_width": 1200,
            "screen_height": 1200,
        }



