from models.vehicle_model import VehicleModel

class SingleTrack3Model(VehicleModel):
    def __init__(self):
        self.dim_state = 3
        self.dim_control = 2
        pass


