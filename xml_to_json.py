import xml.etree.ElementTree as ET
import numpy as np
import json

def parse_metashape_transform(matrix_text):
    vals = list(map(float, matrix_text.split()))
    T = np.array(vals).reshape((4, 4))
    
    R_wc = T[:3, :3]       # world-to-camera rotation
    t_wc = T[:3, 3]        # world-to-camera translation
    
    # Convert to camera-to-world
    R_cw = R_wc.T
    C = -R_wc.T @ t_wc     # camera center in world coords
    
    return R_cw, C, R_wc, t_wc

def xml_to_json(xml_file, output_json):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    cameras_out = []
    
    for cam in root.findall(".//camera"):
        cam_id = cam.get("id")
        img_name = cam.attrib.get("label")

        transform_node = cam.find("transform")
        if transform_node is None:
            continue

        R_cw, C, R_wc, t_wc = parse_metashape_transform(transform_node.text)

        cameras_out.append({
            "id": cam_id,
            "image": img_name,
            "rotation": R_cw.tolist(),
            "translation": (-R_cw @ C).tolist(),  # store in SfM style (R, t)
            "center": C.tolist()
        })

    with open(output_json, "w") as f:
        json.dump({"cameras": cameras_out}, f, indent=2)

    print("Finished. Saved:", output_json)


xml_to_json("camera.xml", "cameras.json")
