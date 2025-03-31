from bvh_converter import get_bvh_from_file, save_bvh_to_file
import re
import random

from os import listdir
from os import path

augmentation_nr = 3
main_dir = "../datasets/lafan1_test"

org = None



def print_child(joint):
    print(joint.name + ": ", end="")
    for child in joint.children:
        print(child.name, end=" ")
    print()
    for child in joint.children:
        print_child(child)

for org_file_name in listdir(main_dir):
    file_name = path.join(main_dir, org_file_name)
    bvh = get_bvh_from_file(file_name)
    joint = bvh.get_joint("Hips")
    names = bvh.get_movable_joints_names()
    for p in enumerate(names):
        print(p)
    print_child(joint)
    print(len(names))
    
    break

#for org_file_name in listdir(main_dir):
#    file_name = path.join(main_dir, org_file_name)

#    for _ in range(augmentation_nr):
#        bvh = get_bvh_from_file(file_name)
#        joints = bvh.get_movable_joints_names()
#        ch_joint = random.choice(joints[1:])

#        bvh.disable_joint_with_children(ch_joint)

#        new_file_name = re.sub(r'(.*)(subject\d\.bvh$)', r'\g<1>no-' + ch_joint.lower() + r'_\g<2>', file_name)
#        save_bvh_to_file(bvh, new_file_name)
#        print(new_file_name)