import re
import numpy as np
from .bvh import Bvh
from .node import Node
from .motion_array import MotionArray

def get_line(f):
    return f.readline().strip()

def get_name(f):
    line = get_line(f)
    if line == "}":
        return None
    result = re.match(r"^(JOINT|ROOT|End) (\w+)$", line)
    return {"type": result[1], "name": result[2] }

def consume_open_brackets(f):
    line = get_line(f)
    if line != "{":
        raise ValueError(f"Expected open brackets get {line}")

def get_offset(f):
    line = get_line(f)
    result = re.match(r"^OFFSET ([\d|\.|-]+) ([\d|\.|-]+) ([\d|\.|-]+)$", line)
    return float(result[1]), float(result[2]), float(result[3])

def get_channels(f):
    line = get_line(f)
    result = re.match(r"^CHANNELS \d+ (.+)$", line)
    return result[1].split()

def create_node(f, joints, parent):
    name = get_name(f)
    if name is None:
        return None
    consume_open_brackets(f)
    offset = get_offset(f)
    channels = get_channels(f) if "End" != name["type"] else None
    joints.append(Node(name['name'], name['type'], parent, offset, channels))
    curr_node = joints[-1]
    while(True):
        child_node = create_node(f, joints, curr_node)
        if child_node is not None:
            curr_node.children.append(child_node)
        else:
            break
    return curr_node
    
def get_hierarchy(f) -> list:
    joints = []
    hierarchy_line = get_line(f)
    if(hierarchy_line == "HIERARCHY"):
        create_node(f, joints, None)
    else:
        raise ValueError(f"File should starts with HIERARCHY instead of {hierarchy_line}")
    return joints

def get_count_frames(f):
    line = get_line(f)
    result = re.match(r"^Frames: (\d+)$", line)
    return int(result[1])

def get_frame_time(f):
    line = get_line(f)
    result = re.match(r"^Frame Time: ([\d|\.|-]+)$", line)
    return float(result[1])

def get_motion_array(f, count_frames):
    motion_array = []

    for _ in range(count_frames):
        line = get_line(f)
        vertexes_array = [float(v) for v in line.split()]
        motion_array.append(vertexes_array) 
    return np.array(motion_array) 

def get_motion(f):
    motion_line = get_line(f)
    if(motion_line != "MOTION"):
        raise ValueError(f"Expected MOTION instead of {motion_line}")
    count_frames = get_count_frames(f)
    frame_time = get_frame_time(f)
    motion_array = get_motion_array(f, count_frames)
    return MotionArray(count_frames, frame_time, motion_array)

def get_bvh_from_file(file_name):
    with open(file_name) as f:
        joints = get_hierarchy(f)
        motion = get_motion(f)
        return Bvh(joints, motion)

def write_line(content, f, buffor = ""):
    f.write(f"{buffor}{content}\n")

def write_offset(offset, f, bufor):
    conv_offset = [format(o, 'f') for o in offset]
    write_line(f"OFFSET {' '.join(conv_offset)}", f, bufor)

def write_channels(channels, f, bufor):
    if channels is not None:
        write_line(f"CHANNELS {len(channels)} {' '.join(channels)}", f, bufor)

def write_childs(childs, f, bufor):
    for child in childs:
        write_node(f, child, bufor)

def write_node(f, curr_joint, buffor):
    write_line(f"{curr_joint.type} {curr_joint.name}", f, buffor)
    write_line('{', f, buffor)
    nw_buffor = "\t" + buffor
    write_offset(curr_joint.offset, f, nw_buffor)
    write_channels(curr_joint.channels, f, nw_buffor)
    write_childs(curr_joint.children, f, nw_buffor)
    write_line('}', f, buffor)

def write_motion_array(array, f):
    for row in array:
        row_as_str = [format(v, 'f') for v in row]
        write_line(' '.join(row_as_str), f)

def save_bvh_to_file(bvh_object, file_name):
    with open(file_name, "w") as f:
        write_line("HIERARCHY", f)
        write_node(f, bvh_object.joints[0], "")
        write_line("MOTION", f)
        write_line(f"Frames: {bvh_object.motion.count_frames}", f)
        write_line(f"Frame Time: {bvh_object.motion.frame_time}", f)
        write_motion_array(bvh_object.motion.motion_array, f)

def group_joints(node: Node):
    print(node.name)
    for child in node.children:
        if child.type == 'End':
            continue
        group_joints(child)
