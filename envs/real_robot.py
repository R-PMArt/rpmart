import time
import numpy as np
import transformations as tf
from frankx import Affine, JointMotion, Robot, Waypoint, WaypointMotion, Gripper, LinearRelativeMotion, LinearMotion, ImpedanceMotion


class Panda():
    def __init__(self,host='172.16.0.2'):
        self.robot = Robot(host)
        self.gripper = Gripper(host)
        self.setGripper(20,0.1)
        self.max_gripper_width = 0.08
        self.robot.set_default_behavior()
        self.robot.recover_from_errors()
        # Reduce the acceleration and velocity dynamic
        self.robot.set_dynamic_rel(0.2)
        # self.robot.set_dynamic_rel(0.05)

        # self.robot.velocity_rel = 0.1
        # self.robot.acceleration_rel = 0.02
        # self.robot.jerk_rel = 0.01

        self.joint_tolerance = 0.01
        # state = self.robot.read_once()
        # print('\nPose: ', self.robot.current_pose())
        # print('O_TT_E: ', state.O_T_EE)
        # print('Joints: ', state.q)
        # print('Elbow: ', state.elbow)
        # pdb.set_trace()
        self.in_impedance_control = False
        
    def setGripper(self,force=20.0, speed=0.02):
        self.gripper.gripper_speed = speed # m/s
        self.gripper.gripper_force = force # N

    def gripper_close(self) -> bool:
        # can be used to grasp
        is_graspped = self.gripper.clamp()
        return is_graspped
    
    def gripper_open(self) -> None:
        self.gripper.open()
    
    def gripper_release(self, width:float) -> None:
        # can be used to release after grasping
        self.gripper.release(min(max(width, 0.0), self.max_gripper_width))
        
    def move_gripper(self, width:float) -> None:
        self.gripper.move(min(max(width, 0.0), self.max_gripper_width), self.gripper.gripper_speed)     # m

    def read_gripper(self) -> float:
        return self.gripper.width()

    def is_grasping(self) -> bool:
        return self.gripper.is_grasping()

    def moveJoint(self,joint,moveTarget=True):
        assert len(joint)==7, "panda DOF is 7"
        if not self.in_impedance_control:
            self.robot.move(JointMotion(joint))
        else:
            raise NotImplementedError
        # while moveTarget:
        #     current = self.robot.read_once().q
        #     if all([np.abs(current[j] - joint[j])<self.joint_tolerance for j in range(len(joint))]):
        #         break
        return True
    
    def readJoint(self):
        if not self.in_impedance_control:
            state = self.robot.read_once()
            joints = state.q
        else:
            raise NotImplementedError
        assert len(joints) == 7, "panda DOF is 7"
        return joints
    
    def homing(self) -> None:
        # joint = [-0.2918438928353856, -0.970780364569858, 0.10614118311070558, -1.3263296233896118, 0.28714199085241543, 1.4429661556967983, 0.8502855184922615]                # microwave: pad + safe (rotate)
        # joint = [-0.32146529861813555, -0.6174831717455548, 0.08796035485936884, -0.8542264670393085, 0.2846642250021548, 1.2692416279845777, 0.7918693021188179]               # refrigerator: storagefurniture
        # joint = [0.07228589984826875, -0.856545356798933, -0.005984785356738588, -1.446693722022207, -0.0739646362066269, 1.5132004619969288, 0.8178283093992272]               # safe: pad + microwave
        # joint = [-0.2249300478901436, -0.8004290411025673, 0.10279602963647609, -1.2284506426734476, 0.22189371273337696, 1.3787900806797873, 0.7783415511498849]               # storagefurniture: microwave
        # joint = [-0.2249300478901436, -0.8004290411025673, 0.10279602963647609, -1.2284506426734476, 0.22189371273337696, 1.3787900806797873, 0.7783415511498849]               # drawer: microwave
        joint = [-0.23090655681806208, -0.7368697004085705, 0.06469194421473157, -1.5633050220115945, 0.06594510726133981, 1.6454856337730452, 0.7169523042954654]              # washingmachine: pad + microwave
        self.moveJoint(joint)

    def readPose(self):
        if not self.in_impedance_control:
            pose = np.array(self.robot.read_once().O_T_EE).reshape(4, 4).T      # EE2robot, gripper pose
        else:
            pose = np.array(self.impedance_motion.get_robotstate().O_T_EE).reshape(4, 4).T
        return pose
    
    def movePose(self, pose):
        # gripper pose
        # tf.euler_from_matrix(pose, axes='rzyx')
        # R.from_euler('ZYX', [-1.560670, -0.745688, 1.922058]).as_matrix(), rpy->matrix
        if not self.in_impedance_control:
            tr = pose[:3, 3]
            rot = tf.euler_from_matrix(pose, axes='rzyx')
            motion = LinearMotion(Affine(tr[0], tr[1], tr[2], rot[0], rot[1], rot[2]))
            self.robot.move(motion)
        else:
            tr = pose[:3, 3]
            rot = tf.euler_from_matrix(pose, axes='rzyx')
            self.impedance_motion.target = Affine(tr[0], tr[1], tr[2], rot[0], rot[1], rot[2])
    
    def start_impedance_control(self, tr_stiffness=1000.0, rot_stiffness=20.0):
        print("you need rebuild frankx to support this")
        self.impedance_motion = ImpedanceMotion(tr_stiffness, rot_stiffness)
        self.robot_thread = self.robot.move_async(self.impedance_motion)
        time.sleep(0.5)
        self.in_impedance_control = True
    
    def end_impedance_control(self):
        self.impedance_motion.finish()
        self.robot_thread.join()
        self.impedance_motion = None
        self.robot_thread = None
        self.in_impedance_control = False
    
    def readWrench(self):
        if not self.in_impedance_control:
            wrench = np.array(self.robot.read_once().O_F_ext_hat_K)                                   # in base
        else:
            wrench = np.array(self.impedance_motion.get_robotstate().O_F_ext_hat_K)
        return wrench


if __name__ == "__main__":
    robot = Panda()
    # test gripper
    robot.gripper_close()
    robot.gripper_open()
    is_graspped = robot.gripper_close()
    is_graspped = is_graspped and robot.is_grasping()
    print("is_graspped:", is_graspped)
    robot.move_gripper(0.05)
    gripper_width = robot.read_gripper()
    print("gripper width:", gripper_width)
    # test arm
    robot.homing()
    joint = robot.readJoint()
    print("current joint:", joint)
    EE2robot = robot.readPose()
    print("current pose:", EE2robot)
    target_pose = EE2robot.copy()
    target_pose[:3, 3] += np.array([0.05, 0.05, 0.05])
    robot.movePose(target_pose)
    joint = robot.readJoint()
    print("current joint:", joint)
    EE2robot = robot.readPose()
    print("current pose:", EE2robot)
    robot.start_impedance_control()
    for i in range(10):
        current_pose = robot.readPose()
        print(i, current_pose)
        target_pose = current_pose.copy()
        target_pose[:3, 3] += np.array([0., 0.02, 0.])
        robot.movePose(target_pose)
        time.sleep(0.3)
    EE2robot = robot.readPose()
    print("current pose:", EE2robot)
    robot.end_impedance_control()
    robot.homing()
    # pose = robot.readPose()
    # np.save("pose_00.npy", pose)
    # joints = robot.readJoint()
    # np.save("joints_00.npy", joints)
