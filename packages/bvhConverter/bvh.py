class Bvh:

    def __init__(self, joints, motion):
        self.joints = joints
        self.motion = motion

    def disable_motion_variable(self, idx):
        self.motion.motion_array[:,idx] = self.motion.motion_array[0, idx]

    def disable_joint(self, joint_name):
        for motion_variable in self.get_joints_motion_idxs(joint_name):
            self.disable_motion_variable(motion_variable["idx"])

    def disable_joint_with_children(self, joint):
        if isinstance(joint, str):
            joint = self.get_joint(joint)
        if joint.type == "End":
            return
        self.disable_joint(joint.name)
        for child in joint.children:
            self.disable_joint_with_children(child)

    def get_joints_motion_idxs(self, joint_name):
        start_idx = 0
        channels = None
        for j in self.joints:
            if j.name == joint_name:
                channels = j.channels
                break
            else:
                start_idx += len(j.channels) if j.channels else 0
        if channels is None:
            raise ValueError(f"Couldn't find joint {joint_name}")
        return [{"name": ch, "idx": start_idx + i} for i, ch in enumerate(channels)]
    
    def get_joint(self, joint_name):
        for j in self.joints:
            if j.name == joint_name:
                return j

    def get_joints_names(self):
        return [j.name for j in self.joints]
    
    def get_movable_joints_names(self):
        return list(filter(lambda joint_name: joint_name != "Site", self.get_joints_names()))