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

#define ICP_ITERATIONS 20
#define FRAND_TO(X) (static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/(X))))

typedef Eigen::Matrix<double, 3, 1> Vector3d;

std::vector<Vector2d> last_scan_list;
kd_node *last_scan_tree;

float nearest_neighbour_squared_brute_force(std::vector<Vector2d>& last_scan_list,
                                            Vector2d new_scan_point) {
  float cur_best = std::numeric_limits<float>::infinity();
  for (int k = 0; k < last_scan_list.size(); k++) {
    float cur_dist = (new_scan_point[0] - last_scan_list[k][0]) *
                     (new_scan_point[0] - last_scan_list[k][0]) *
                     (new_scan_point[1] - last_scan_list[k][1]) *
                     (new_scan_point[1] - last_scan_list[k][1]);
    if (cur_dist < cur_best) {
      cur_best = cur_dist;
    }
  }
  return cur_best;
}

float calc_mmse(std::vector<Vector2d>& scan_list,
                Eigen::Rotation2D<double>& TR, Vector2d& TT) {
  float mmse = 0;
  for (int j = 0; j < scan_list.size(); j++) {
    Vector2d transformed_point = TR*scan_list[j] + TT;
#ifdef USE_KD_TREE
    float cur_best = nearest_neighbour_squared_2d(last_scan_tree, transformed_point, 0);
#else
    float cur_best = nearest_neighbour_squared_brute_force(last_scan_list, transformed_point);
#endif
    mmse += cur_best;
  }
  mmse /= scan_list.size();
  return mmse;
}

void iterative_closest_point(std::vector<Vector2d> scan_list,
                              Eigen::Rotation2D<double>& TR, Vector2d& TT) {
  static double eps = 0.001;
  Vector2d dx(eps, 0.0);
  Vector2d dy(0.0, eps);
  Eigen::Rotation2D<double> dr(eps);
  for (int i = 0; i < ICP_ITERATIONS; i++) {

    // numerical derivatives
    float mmse = calc_mmse(scan_list, TR, TT);
    Vector2d TTdx = TT+dx;
    Vector2d TTdy = TT+dy;
    Eigen::Rotation2D<double> TRdr = TR*dr;
    float dm_dx = (calc_mmse(scan_list, TR, TTdx) - mmse)/eps;
    float dm_dy = (calc_mmse(scan_list, TR, TTdy) - mmse)/eps;
    float dm_dr = (calc_mmse(scan_list, TRdr, TT) - mmse)/eps;

    Eigen::Rotation2D<double> R(-mmse / (3*dm_dr));
    Vector2d T(mmse / (3*dm_dx), mmse / (3*dm_dy));

    TR *= R;
    TT = R*TT + T;
  }
}

Vector2d centroid(std::vector<Vector2d> list) {
  Vector2d centroid = Vector2d::Zero();
  for (int i = 0; i < list.size(); i++) {
    centroid[0] += list[i][0];
    centroid[1] += list[i][1];
  }
  centroid = 1.0/list.size() * centroid;
  return centroid;
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

    std::vector<Vector2d> new_scan_list1;
    std::vector<Vector2d> new_scan_list2;
    Vector2d v;
    last_scan_list.clear();
    new_scan_list1.clear();
    for (int i = 0; i < 1000; i++) {
      Vector2d v;
      v << 10+FRAND_TO(10), -10+FRAND_TO(10), FRAND_TO(10);
      last_scan_list.push_back(v);
      v << FRAND_TO(10), FRAND_TO(10), FRAND_TO(10);
      new_scan_list1.push_back(v);
    }
    new_scan_list2 = new_scan_list1;
    last_scan_tree = create_2d_tree(last_scan_list, 0);
    // Match centroids
    Vector2d TT = centroid(last_scan_list) - centroid(new_scan_list1);
    Eigen::Rotation2D<double> TR(0.0);
    iterative_closest_point(new_scan_list1, TR, TT);
    ROS_INFO("%f, %f, %f", TT(0), TT(0), TR.angle());
    //Main loop code goes here:
  }
  return 0;
}
