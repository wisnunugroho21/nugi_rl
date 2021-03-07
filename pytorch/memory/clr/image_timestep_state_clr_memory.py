from memory.clr.image_timestep_clr_memory import ImageTimestepClrMemory

class ImageTimestepStateClrMemory(ImageTimestepClrMemory):
    def save_eps(self, data_state):        
        image, _    = data_state
        return super().save_eps(image)

    def save_replace_all(self, data_states):
        images, _    = data_states
        super().save_replace_all(images)

    def save_all(self, data_states):
        images, _  = data_states
        super().save_all(images)
