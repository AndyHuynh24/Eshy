"""Open eshy.xml in the MuJoCo viewer (no policy)."""
import mujoco
import mujoco.viewer


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("eshy.xml")
    data = mujoco.MjData(model)

    print("Robot model loaded!")
    print(f"  Bodies: {model.nbody}")
    print(f"  Joints: {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print(f"  Joint names: {[model.joint(i).name for i in range(model.njnt)]}")
    mujoco.viewer.launch(model, data)
