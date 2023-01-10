import prior
import json
from ai2thor.controller import Controller
import copy
from PIL import Image
import pdb

def get_top_down_frame(controller):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)

if __name__ == '__main__':
    house_idx = 3
    dataset = prior.load_dataset("procthor-10k")
    house_data = dataset['train'].data[house_idx]
    house_dict = json.loads(house_data.decode('utf-8'))
    with open('demo.json', 'w+') as f:
        json.dump(house_dict, f, indent=2)

    house = dataset['train'][house_idx]
    controller = Controller(scene=house)

    img = get_top_down_frame(controller)
    img.show()


