
//
// Created by qiayuan on 1/24/22.
//

#include "legged_unitree_hw/UnitreeHW.h"

#ifdef UNITREE_SDK_3_3_1
#include "unitree_legged_sdk_3_3_1/unitree_joystick.h"
#elif UNITREE_SDK_3_8_0
#include "unitree_legged_sdk_3_8_0/joystick.h"
#endif

#include <sensor_msgs/Joy.h>

namespace legged {
bool UnitreeHW::init(ros::NodeHandle& root_nh, ros::NodeHandle& robot_hw_nh) {
  if (!LeggedHW::init(root_nh, robot_hw_nh)) {
    return false;
  }

  robot_hw_nh.getParam("power_limit", powerLimit_);

  setupJoints();
  setupImu();
  setupContactSensor(robot_hw_nh);

#ifdef UNITREE_SDK_3_3_1
  udp_ = std::make_shared<UNITREE_LEGGED_SDK::UDP>(UNITREE_LEGGED_SDK::LOWLEVEL);
#elif UNITREE_SDK_3_8_0
  udp_ = std::make_shared<UNITREE_LEGGED_SDK::UDP>(UNITREE_LEGGED_SDK::LOWLEVEL, 8090, "192.168.123.10", 8007);
#endif

  udp_->InitCmdData(lowCmd_);

  std::string robot_type;
  root_nh.getParam("robot_type", robot_type);
#ifdef UNITREE_SDK_3_3_1
  if (robot_type == "a1") {
    safety_ = std::make_shared<UNITREE_LEGGED_SDK::Safety>(UNITREE_LEGGED_SDK::LeggedType::A1);
  } else if (robot_type == "aliengo") {
    safety_ = std::make_shared<UNITREE_LEGGED_SDK::Safety>(UNITREE_LEGGED_SDK::LeggedType::Aliengo);
  }
#elif UNITREE_SDK_3_8_0
  if (robot_type == "go1"){
    safety_ = std::make_shared<UNITREE_LEGGED_SDK::Safety>(UNITREE_LEGGED_SDK::LeggedType::Go1);
  }
#endif
  else {
    ROS_FATAL("Unknown robot type: %s", robot_type.c_str());
    return false;
  }

  joyPublisher_ = root_nh.advertise<sensor_msgs::Joy>("/joy", 10);

  imu_pub = robot_hw_nh.advertise<sensor_msgs::Imu>("/unitree_hardware/imu", 100);
  joint_foot_pub = robot_hw_nh.advertise<sensor_msgs::JointState>("/unitree_hardware/joint_foot", 100);

  swap_joint_indices = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
  swap_foot_indices = {1, 0, 3, 2};

  // to record contact bias
  first_contact_force_read = false;

  return true;
}

void UnitreeHW::read(const ros::Time& currTime /*time*/, const ros::Duration& /*period*/) {
  udp_->Recv();
  udp_->GetRecv(lowState_);

  for (int i = 0; i < 12; ++i) {
    jointData_[i].pos_ = lowState_.motorState[i].q;
    jointData_[i].vel_ = lowState_.motorState[i].dq;
    jointData_[i].tau_ = lowState_.motorState[i].tauEst;
  }

  imuData_.ori_[0] = lowState_.imu.quaternion[1];
  imuData_.ori_[1] = lowState_.imu.quaternion[2];
  imuData_.ori_[2] = lowState_.imu.quaternion[3];
  imuData_.ori_[3] = lowState_.imu.quaternion[0];
  imuData_.angularVel_[0] = lowState_.imu.gyroscope[0] - (imuCalibrated_ ? gyroBias_[0] : 0.0);
  imuData_.angularVel_[1] = lowState_.imu.gyroscope[1] - (imuCalibrated_ ? gyroBias_[1] : 0.0);
  imuData_.angularVel_[2] = lowState_.imu.gyroscope[2] - (imuCalibrated_ ? gyroBias_[2] : 0.0);

  imuData_.linearAcc_[0] = lowState_.imu.accelerometer[0] - (imuCalibrated_ ? accelBias_[0] : 0.0);
  imuData_.linearAcc_[1] = lowState_.imu.accelerometer[1] - (imuCalibrated_ ? accelBias_[0] : 0.0);
  imuData_.linearAcc_[2] = lowState_.imu.accelerometer[2] - (imuCalibrated_ ? accelBias_[0] : 0.0);

  // a temporary logic for Go1, may not work well if the robot stands up
  // initially. record the first contact force reading as contact force bias
  if (first_contact_force_read == false) {
    initTime = currTime;
    first_contact_force_read = true;
  } else {
    ros::Duration elapsedTime = (currTime - initTime);
    double timeSinceStart = elapsedTime.toSec();

    if (timeSinceStart < 0.5) {
      for (size_t i = 0; i < CONTACT_SENSOR_NAMES.size(); ++i) {
        contactBias_[i] = lowState_.footForce[i];
        contactState_[i] = 0;
      }
    } else {
      for (size_t i = 0; i < CONTACT_SENSOR_NAMES.size(); ++i) {
        // FR FL RR RL
        contactState_[i] = (lowState_.footForce[i] - contactBias_[i]) > contactThreshold_;
      }
    }
  }

  // Set feedforward and velocity cmd to zero to avoid for safety when not
  // controller setCommand
  std::vector<std::string> names = hybridJointInterface_.getNames();
  for (const auto& name : names) {
    HybridJointHandle handle = hybridJointInterface_.getHandle(name);
    handle.setFeedforward(0.);
    handle.setVelocityDesired(0.);
    handle.setKd(3.);
  }

  // publish ros topics
  ros::Time now = ros::Time::now();
  imu_msg.header.stamp = now;

  imu_msg.linear_acceleration.x = lowState_.imu.accelerometer[0];
  imu_msg.linear_acceleration.y = lowState_.imu.accelerometer[1];
  imu_msg.linear_acceleration.z = lowState_.imu.accelerometer[2];

  imu_msg.angular_velocity.x = lowState_.imu.gyroscope[0];
  imu_msg.angular_velocity.y = lowState_.imu.gyroscope[1];
  imu_msg.angular_velocity.z = lowState_.imu.gyroscope[2];

  imu_pub.publish(imu_msg);
  joint_foot_msg.header.stamp = now;

  joint_foot_msg.name = {"FL0", "FL1", "FL2", "FR0", "FR1",     "FR2",     "RL0",     "RL1",
                         "RL2", "RR0", "RR1", "RR2", "FL_foot", "FR_foot", "RL_foot", "RR_foot"};

  joint_foot_msg.position.resize(NUM_DOF + NUM_LEG);
  joint_foot_msg.velocity.resize(NUM_DOF + NUM_LEG);
  joint_foot_msg.effort.resize(NUM_DOF + NUM_LEG);

  for (int i = 0; i < NUM_DOF; i++) {
    int swap_i = swap_joint_indices[i];
    joint_foot_msg.position[i] = lowState_.motorState[swap_i].q;
    joint_foot_msg.velocity[i] = lowState_.motorState[swap_i].dq;
    joint_foot_msg.effort[i] = lowState_.motorState[swap_i].tauEst;
  }

  // read foot_force
  for (int i = 0; i < NUM_LEG; i++) {
    int swap_i = swap_foot_indices[i];  // 1 0 3 2 (index of footForce) (FL FR RL RR) SO
                                        // THE ORDER OF footForce is: FR FL RR RL
    joint_foot_msg.effort[NUM_DOF + i] = lowState_.footForce[swap_i] - contactBias_[swap_i];
  }
  joint_foot_pub.publish(joint_foot_msg);

  updateJoystick(currTime);
}

void UnitreeHW::write(const ros::Time& /*time*/, const ros::Duration& /*period*/) {
  for (int i = 0; i < 12; ++i) {
    lowCmd_.motorCmd[i].q = static_cast<float>(jointData_[i].posDes_);
    lowCmd_.motorCmd[i].dq = static_cast<float>(jointData_[i].velDes_);
    lowCmd_.motorCmd[i].Kp = static_cast<float>(jointData_[i].kp_);
    lowCmd_.motorCmd[i].Kd = static_cast<float>(jointData_[i].kd_);
    lowCmd_.motorCmd[i].tau = static_cast<float>(jointData_[i].ff_);
  }
  safety_->PositionLimit(lowCmd_);
  safety_->PowerProtect(lowCmd_, lowState_, powerLimit_);
  udp_->SetSend(lowCmd_);
  udp_->Send();
}

bool UnitreeHW::setupJoints() {
  for (const auto& joint : urdfModel_->joints_) {
    int leg_index = 0;
    int joint_index = 0;
    if (joint.first.find("RF") != std::string::npos) {
      leg_index = UNITREE_LEGGED_SDK::FR_;
    } else if (joint.first.find("LF") != std::string::npos) {
      leg_index = UNITREE_LEGGED_SDK::FL_;
    } else if (joint.first.find("RH") != std::string::npos) {
      leg_index = UNITREE_LEGGED_SDK::RR_;
    } else if (joint.first.find("LH") != std::string::npos) {
      leg_index = UNITREE_LEGGED_SDK::RL_;
    } else {
      continue;
    }

    if (joint.first.find("HAA") != std::string::npos) {
      joint_index = 0;
    } else if (joint.first.find("HFE") != std::string::npos) {
      joint_index = 1;
    } else if (joint.first.find("KFE") != std::string::npos) {
      joint_index = 2;
    } else {
      continue;
    }

    int index = leg_index * 3 + joint_index;
    hardware_interface::JointStateHandle state_handle(joint.first, &jointData_[index].pos_, &jointData_[index].vel_,
                                                      &jointData_[index].tau_);
    jointStateInterface_.registerHandle(state_handle);
    hybridJointInterface_.registerHandle(HybridJointHandle(state_handle, &jointData_[index].posDes_, &jointData_[index].velDes_,
                                                           &jointData_[index].kp_, &jointData_[index].kd_, &jointData_[index].ff_));
  }
  return true;
}

bool UnitreeHW::setupImu() {
  imuSensorInterface_.registerHandle(hardware_interface::ImuSensorHandle("unitree_imu", "unitree_imu", imuData_.ori_, imuData_.oriCov_,
                                                                         imuData_.angularVel_, imuData_.angularVelCov_, imuData_.linearAcc_,
                                                                         imuData_.linearAccCov_));
  imuData_.oriCov_[0] = 0.0012;
  imuData_.oriCov_[4] = 0.0012;
  imuData_.oriCov_[8] = 0.0012;

  imuData_.angularVelCov_[0] = 0.0004;
  imuData_.angularVelCov_[4] = 0.0004;
  imuData_.angularVelCov_[8] = 0.0004;

  return true;
}
/// New calibration function that accumulates both IMU and contact sensor data.
void UnitreeHW::calibrate(double duration_sec) {
  int sampleCount = 0;
  std::array<double, 3> gyroSum   = {0.0, 0.0, 0.0};
  std::array<double, 3> accelSum  = {0.0, 0.0, 0.0};
  // Prepare a container to sum contact sensor forces.
  std::vector<double> contactBiasSum(CONTACT_SENSOR_NAMES.size(), 0.0);

  ros::Time startTime = ros::Time::now();
  // Assume the sensor is level during calibration.
  const double g = 9.81;
  std::array<double, 3> gravityRef = {0.0, 0.0, g};

  ROS_INFO("Starting calibration for %.2f seconds...", duration_sec);
  while (ros::ok() && (ros::Time::now() - startTime).toSec() < duration_sec) {
    udp_->Recv();
    udp_->GetRecv(lowState_);

    // Accumulate IMU data.
    gyroSum[0]  += lowState_.imu.gyroscope[0];
    gyroSum[1]  += lowState_.imu.gyroscope[1];
    gyroSum[2]  += lowState_.imu.gyroscope[2];

    accelSum[0] += lowState_.imu.accelerometer[0];
    accelSum[1] += lowState_.imu.accelerometer[1];
    accelSum[2] += lowState_.imu.accelerometer[2];

    // Accumulate contact sensor force data.
    for (size_t i = 0; i < CONTACT_SENSOR_NAMES.size(); ++i) {
      contactBiasSum[i] += lowState_.footForce[i];
    }

    sampleCount++;
    ros::Duration(0.01).sleep();  // Adjust sample delay as needed.
  }

  if (sampleCount > 0) {
    // Compute average IMU values.
    double avg_gyro[3] = {gyroSum[0] / sampleCount,
                          gyroSum[1] / sampleCount,
                          gyroSum[2] / sampleCount};

    double avg_accel[3] = {accelSum[0] / sampleCount,
                           accelSum[1] / sampleCount,
                           accelSum[2] / sampleCount};

    // Under the assumption that when level and static:
    //   avg_accel = bias + gravityRef
    // Therefore, bias = avg_accel - gravityRef.
    accelBias_[0] = avg_accel[0] - gravityRef[0];
    accelBias_[1] = avg_accel[1] - gravityRef[1];
    accelBias_[2] = avg_accel[2] - gravityRef[2];

    // Set the gyroscope bias to the computed average.
    gyroBias_[0] = avg_gyro[0];
    gyroBias_[1] = avg_gyro[1];
    gyroBias_[2] = avg_gyro[2];
  }
  imuCalibrated_ = true;
  ROS_WARN("IMU calibration complete. Gyro bias: [%.4f, %.4f, %.4f]",
           gyroBias_[0], gyroBias_[1], gyroBias_[2]);
  ROS_WARN("Accel bias: [%.4f, %.4f, %.4f]",
           accelBias_[0], accelBias_[1], accelBias_[2]);

  // Compute and store the contact sensor biases.
  for (size_t i = 0; i < CONTACT_SENSOR_NAMES.size(); ++i) {
    contactBias_[i] = contactBiasSum[i] / sampleCount;
    ROS_WARN("Contact sensor %s bias: %.4f", CONTACT_SENSOR_NAMES[i].c_str(), contactBias_[i]);
  }
  contactCalibrated_ = true;
}

bool UnitreeHW::setupContactSensor(ros::NodeHandle& nh) {
  nh.getParam("contact_threshold", contactThreshold_);
  for (size_t i = 0; i < CONTACT_SENSOR_NAMES.size(); ++i) {
    contactSensorInterface_.registerHandle(ContactSensorHandle(CONTACT_SENSOR_NAMES[i], &contactState_[i]));
  }
  return true;
}

void UnitreeHW::updateJoystick(const ros::Time& time) {
  if ((time - lastJoyPub_).toSec() < 1 / 50.) {
    return;
  }
  lastJoyPub_ = time;
  xRockerBtnDataStruct keyData;
  memcpy(&keyData, &lowState_.wirelessRemote[0], 40);
  sensor_msgs::Joy joyMsg;  // Pack as same as Logitech F710
  joyMsg.axes.push_back(-keyData.lx);
  joyMsg.axes.push_back(keyData.ly);
  joyMsg.axes.push_back(-keyData.rx);
  joyMsg.axes.push_back(keyData.ry);
  joyMsg.buttons.push_back(keyData.btn.components.X);
  joyMsg.buttons.push_back(keyData.btn.components.A);
  joyMsg.buttons.push_back(keyData.btn.components.B);
  joyMsg.buttons.push_back(keyData.btn.components.Y);
  joyMsg.buttons.push_back(keyData.btn.components.L1);
  joyMsg.buttons.push_back(keyData.btn.components.R1);
  joyMsg.buttons.push_back(keyData.btn.components.L2);
  joyMsg.buttons.push_back(keyData.btn.components.R2);
  joyMsg.buttons.push_back(keyData.btn.components.select);
  joyMsg.buttons.push_back(keyData.btn.components.start);
  joyPublisher_.publish(joyMsg);
}

}  // namespace legged
