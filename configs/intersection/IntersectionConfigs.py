


def get_multi_agent_config(num_agents=2, obs_type="Kinematics"):

    return {
        "observation": { 
            "type": "MultiAgentObservation",
            "observation_config": { "type": "Kinematics" }
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
        "collision_reward": -5.0, 
        
        #reward for staying alive and exiting the intersection
        "high_speed_reward": 1, 
        "arrived_reward": 5.0, 

        "on_road_reward": 0.1,
        "offroad_terminal": True,
        "reward_config": {
            "collision_reward": -5.0, 
        },
        
        
        "normalize_reward": False, 
        
        #"duration": 10,  TODOO figure out if needed in heavy training 
    }