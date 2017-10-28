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
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_datatypes.h>
#include <gazebo_msgs/ModelStates.h>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/LaserScan.h>

#define EPS 0.02

#define YAW_OFFSET 0
#define IMAGE_WIDTH 640

#define MAP_WIDTH 100
#define MAP_RESOLUTION 0.1
#define MAP_ORIGIN_X (-5)
#define MAP_ORIGIN_Y (-5)

#define P_SENSE 0.9
#define P_UNSENSE 0.4
#define MAX_SENSE 4.5

#define POINT_IN_MAP(X,Y) (X > -1 && X < MAP_WIDTH && Y > -1 && Y < MAP_WIDTH)

ros::Publisher pose_publisher;
ros::Publisher marker_pub;

typedef unsigned char uint8;
typedef char int8;

int8 occupancy_grid[MAP_WIDTH*MAP_WIDTH];
float logit_occupancy_grid[MAP_WIDTH*MAP_WIDTH];

bool pose_updated = false;
double ips_x;
double ips_y;
double ips_yaw;


void update_map();
float logit(float p);
float i_logit(float l);
void bresenham(int x0, int y0, int x1, int y1, std::vector<int>& x, std::vector<int>& y);

short sgn(int x) { return x >= 0 ? 1 : -1; }

//Callback function for the Position topic (SIMULATION)
void pose_callback(const gazebo_msgs::ModelStates& msg) {

  int i;
  for(i = 0; i < msg.name.size(); i++) if(msg.name[i] == "mobile_base") break;
  ips_x = msg.pose[i].position.x ;
  ips_y = msg.pose[i].position.y ;
  // ROS_INFO("POSE X: %f Y:%f", ips_x, ips_y);
  ips_yaw = tf::getYaw(msg.pose[i].orientation);
  pose_updated = true;
}

//Callback function for the Position topic (LIVE)
// void pose_callback(const geometry_msgs::PoseWithCovarianceStamped& msg)
// {

//   ips_x = msg.pose.pose.position.x; // Robot X psotition
//   ips_y = msg.pose.pose.position.y; // Robot Y psotition
//   ips_yaw = tf::getYaw(msg.pose.pose.orientation); // Robot Yaw
//   ROS_INFO("pose_callback X: %f Y: %f Yaw: %f", ips_x, ips_y, ips_yaw);
//   pose_updated = true;
// }


void update_map(const sensor_msgs::LaserScan msg) {
  int x_map_idx = round((ips_x - MAP_ORIGIN_X)/MAP_RESOLUTION);
  int y_map_idx = round((ips_y - MAP_ORIGIN_Y)/MAP_RESOLUTION);
  std::vector<int> line_x;
  std::vector<int> line_y;
  int prev_endpoint_x = 0;
  int prev_endpoint_y = 0;

  float fov = msg.angle_max - msg.angle_min;
  int num_points = abs(fov / msg.angle_increment);
  ROS_INFO("RANGE: %f", msg.ranges[num_points/2]);
  for (int i = 0; i < num_points; i += 2) {
    line_x.clear();
    line_y.clear();
    double theta = ips_yaw + msg.angle_min + i*msg.angle_increment + YAW_OFFSET;

    float range = msg.ranges[i];
    if (std::isnan(range)) {
      continue;
    }
    range = (range > msg.range_max) ? msg.range_max : range;
    int endpoint_x = x_map_idx + range / MAP_RESOLUTION * cos(theta);
    int endpoint_y = y_map_idx + range / MAP_RESOLUTION * sin(theta);

    // Do not draw the same line twice
    if (endpoint_x == prev_endpoint_x &&
        endpoint_y == prev_endpoint_y) {
      continue;
    }
    bresenham(x_map_idx, y_map_idx, endpoint_x, endpoint_y, line_x, line_y);
    while(!line_x.empty()) {
      int next_x = line_x.back();
      int next_y = line_y.back();
      line_x.pop_back();
      line_y.pop_back();
      // Do not update points off map
      if (POINT_IN_MAP(next_x, next_y)) {
        logit_occupancy_grid[next_x*MAP_WIDTH + next_y] += logit(P_UNSENSE);
      }

    }
    if (range < MAX_SENSE && POINT_IN_MAP(endpoint_x, endpoint_y)) {
      logit_occupancy_grid[endpoint_x*MAP_WIDTH + endpoint_y] += logit(P_SENSE);
    }
    prev_endpoint_x = endpoint_x;
    prev_endpoint_y = endpoint_y;
  }
}

float i_logit(float l) {
    float exp_l = exp(l);
    return exp_l/(1+exp_l);
}

float logit(float p) {
    return log(p/(1-p));
}


//Callback function for the map
void map_callback(const nav_msgs::OccupancyGrid& msg)
{
    //This function is called when a new map is received
    
    //you probably want to save the map into a form which is easy to work with
}

void image_callback(const sensor_msgs::LaserScan msg)
{
  if (pose_updated) {
    update_map(msg);
    pose_updated = false;
  }
}

//Bresenham line algorithm (pass empty vectors)
// Usage: (x0, y0) is the first point and (x1, y1) is the second point. The calculated
//        points (x, y) are stored in the x and y vector. x and y should be empty 
//	  vectors of integers and shold be defined where this function is called from.
void bresenham(int x0, int y0, int x1, int y1, std::vector<int>& x, std::vector<int>& y) {

  int dx = abs(x1 - x0);
  int dy = abs(y1 - y0);
  int dx2 = x1 - x0;
  int dy2 = y1 - y0;
  
  const bool s = abs(dy) > abs(dx);

  if (s) {
    int dx2 = dx;
    dx = dy;
    dy = dx2;
  }

  int inc1 = 2 * dy;
  int d = inc1 - dx;
  int inc2 = d - dx;

  x.push_back(x0);
  y.push_back(y0);

  while (x0 != x1 || y0 != y1) {
    if (s) y0+=sgn(dy2); else x0+=sgn(dx2);
    if (d < 0) d += inc1;
    else {
      d += inc2;
      if (s) x0+=sgn(dx2); else y0+=sgn(dy2);
    }

    //Add point to vector
    x.push_back(x0);
    y.push_back(y0);
  }
}

void update_occupancy_grid() {
  for (int i = 0; i < MAP_WIDTH; i++) {
    for (int j = 0; j < MAP_WIDTH; j++) {
      if (fabs(logit_occupancy_grid[i*MAP_WIDTH + j]) < EPS) {
        occupancy_grid[i*MAP_WIDTH + j] = 50;
      } else {
        occupancy_grid[i*MAP_WIDTH + j] = round(i_logit(logit_occupancy_grid[i*MAP_WIDTH + j]) * 100);
      }
    }
  }
}

int main(int argc, char **argv)
{
  //Initialize the ROS framework
  ros::init(argc,argv,"main_control");
  ros::NodeHandle n;

  //Subscribe to the desired topics and assign callbacks
  // ros::Subscriber pose_sub = n.subscribe("/indoor_pos", 1, pose_callback);
  ros::Subscriber pose_sub = n.subscribe("/gazebo/model_states", 1, pose_callback);
  ros::Subscriber map_sub = n.subscribe("/map", 1, map_callback);
  ros::Subscriber kinect_sub = n.subscribe("/scan", 1, image_callback);

  //Setup topics to Publish from this node
  ros::Publisher velocity_publisher = n.advertise<geometry_msgs::Twist>("/cmd_vel_mux/input/navi", 1);
  pose_publisher = n.advertise<geometry_msgs::PoseStamped>("/pose", 1, true);
  marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1, true);
  ros::Publisher grid_publisher = n.advertise<nav_msgs::OccupancyGrid>("/map", 2);

  //Velocity control variable
  geometry_msgs::Twist vel;


  nav_msgs::OccupancyGrid occ_grid_msg;
  
  occ_grid_msg.info.resolution = MAP_RESOLUTION;
  occ_grid_msg.info.width = MAP_WIDTH;
  occ_grid_msg.info.height = MAP_WIDTH;
  occ_grid_msg.info.origin.position.x = MAP_ORIGIN_X;
  occ_grid_msg.info.origin.position.y = MAP_ORIGIN_Y;

  //Set the loop rate
  ros::Rate loop_rate(20);    //20Hz update rate


  while (ros::ok())
  {
  	loop_rate.sleep(); //Maintain the loop rate
  	ros::spinOnce();   //Check for new messages

  	//Main loop code goes here:
  	vel.linear.x = 0; // set linear speed
  	vel.angular.z = 0; // set angular speed

  	velocity_publisher.publish(vel); // Publish the command velocity
    update_occupancy_grid();
    occ_grid_msg.data = std::vector<signed char>(occupancy_grid, occupancy_grid+MAP_WIDTH*MAP_WIDTH);

    ROS_INFO("UPDATED");
    grid_publisher.publish(occ_grid_msg);
  }

  return 0;
}
