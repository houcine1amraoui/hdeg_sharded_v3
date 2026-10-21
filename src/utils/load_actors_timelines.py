from datetime import datetime

def load_actor_timelines(config):
    # Helper to parse datetime strings
    def parse_dt(dt_str):
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

    # Return as dictionary of tuples
    timelines = {
        "actor1_t1": (parse_dt(config["preprocessing"]["actors_timelines"]["actor1_t1_start"]), 
                      parse_dt(config["preprocessing"]["actors_timelines"]["actor1_t1_end"])),
        "actor2":    (parse_dt(config["preprocessing"]["actors_timelines"]["actor2_start"]),    
                      parse_dt(config["preprocessing"]["actors_timelines"]["actor2_end"])),
        "actor1_t2": (parse_dt(config["preprocessing"]["actors_timelines"]["actor1_t2_start"]), 
                      parse_dt(config["preprocessing"]["actors_timelines"]["actor1_t2_end"]))
    }
    return timelines