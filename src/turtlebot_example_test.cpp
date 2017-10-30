//  ///////////////////////////////////////////////////////////
//
// turtlebot_example.cpp
// This file contains example code for use with ME 597 lab 2
// It outlines the basic setup of a ros node and the various 
// inputs and outputs needed for this lab
// 
// Author: James Servos 
// Edited: Nima Mohajerin
//
// //////////////////////////////////////////////////////////

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_datatypes.h>
#include <gazebo_msgs/ModelStates.h>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>
#include <Eigen/Dense>
#include <random>
#include <limits>

#include "kdtree.h"
#define FRAND_TO(X) (static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/(X))))

typedef Eigen::Matrix<double, 3, 3> Matrix3d;

std::vector<Vector3d> last_scan_list;
kd_node *last_scan_tree;

float iterative_closest_point(std::vector<Vector3d>& new_scan_list,
                              Matrix3d TR, Vector3d TT, bool use_kd_tree) {
  float MMSE = 0;
  for (int j = 0; j < new_scan_list.size(); j++) {
    new_scan_list[j] = TR*new_scan_list[j] + TT;
    float cur_best;
    if (use_kd_tree) {
      cur_best = nearest_neighbour_squared_2d(last_scan_tree, new_scan_list[j], 0);
    } else {
      cur_best = std::numeric_limits<float>::infinity();
      for (int k = 0; k < last_scan_list.size(); k++) {
        float cur_dist = (new_scan_list[j][0] - last_scan_list[k][0]) *
                         (new_scan_list[j][0] - last_scan_list[k][0]) *
                         (new_scan_list[j][1] - last_scan_list[k][1]) *
                         (new_scan_list[j][1] - last_scan_list[k][1]);
        if (cur_dist < cur_best) {
          cur_best = cur_dist;
        }
      }
    }

    MMSE += cur_best;
  }
  MMSE /= new_scan_list.size();
}

Vector3d centroid_difference(std::vector<Vector3d> last_scan_list,
                             std::vector<Vector3d> new_scan_list) {
  Vector3d last_centroid = Vector3d::Zero();
  for (int i = 0; i < last_scan_list.size(); i++) {
    last_centroid[0] += last_scan_list[i][0];
    last_centroid[1] += last_scan_list[i][1];
  }
  last_centroid = 1.0/last_scan_list.size() * last_centroid;
  Vector3d new_centroid = Vector3d::Zero();
  for (int i = 0; i < new_scan_list.size(); i++) {
    new_centroid[0] += new_scan_list[i][0];
    new_centroid[1] += new_scan_list[i][1];
  }
  new_centroid = 1.0/new_scan_list.size() * new_centroid;
  return last_centroid - new_centroid;
}

int main(int argc, char **argv)
{
	//Initialize the ROS framework
  ros::init(argc,argv,"main_control");
  ros::NodeHandle n;

  //Set the loop rate
  ros::Rate loop_rate(20);    //20Hz update rate

  while (ros::ok())
  {
  	loop_rate.sleep(); //Maintain the loop rate
  	ros::spinOnce();   //Check for new messages

    std::vector<Vector3d> new_scan_list1;
    std::vector<Vector3d> new_scan_list2;
    Vector3d v;
    last_scan_list.clear();
    new_scan_list1.clear();
    for (int i = 0; i < 1000; i++) {
      Vector3d v;
      v << 1+FRAND_TO(10), -1+FRAND_TO(10), 2+FRAND_TO(10);
      last_scan_list.push_back(v);
      v << FRAND_TO(10), FRAND_TO(10), FRAND_TO(10);
      new_scan_list1.push_back(v);
    }
    new_scan_list2 = new_scan_list1;
    last_scan_tree = create_2d_tree(last_scan_list, 0);
    // Match centroids
    Vector3d TT = centroid_difference(last_scan_list, new_scan_list1);
    Matrix3d TR = Matrix3d::Ones();
    ROS_INFO("%f", iterative_closest_point(new_scan_list1, TR, TT, true) -
                   iterative_closest_point(new_scan_list2, TR, TT, false));
  	//Main loop code goes here:
  }
  return 0;
}
