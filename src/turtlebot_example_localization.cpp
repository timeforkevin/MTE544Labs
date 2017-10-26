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
#include <sensor_msgs/Image.h>

#define EPS 0.02

#define FOV (60.0/180*M_PI)
#define YAW_OFFSET 0
#define IMAGE_WIDTH 640

#define MAP_WIDTH 100
#define MAP_RESOLUTION 0.1
#define MAP_ORIGIN_X (-5)
#define MAP_ORIGIN_Y (-5)

#define P_SENSE 0.8
#define P_UNSENSE 0.4
#define MAX_SENSE 4.5

#define NUM_PARTICLES 100

#define POINT_IN_MAP(X,Y) (X > -1 && X < MAP_WIDTH && Y > -1 && Y < MAP_WIDTH)

ros::Publisher pose_publisher;
ros::Publisher marker_pub;

typedef unsigned char uint8;
typedef char int8;

typedef {
  double x;
  double y;
  double yaw;
  double weight;
} particle;

particle particle_set[NUM_PARTICLES];

geometry_msgs::twist odom_input;


float depth_section[IMAGE_WIDTH];

int8 occupancy_grid[MAP_WIDTH*MAP_WIDTH];
float logit_occupancy_grid[MAP_WIDTH*MAP_WIDTH];

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
}

float i_logit(float l) {
    float exp_l = exp(l);
    return exp_l/(1+exp_l);
}

float logit(float p) {
    return log(p/(1-p));
}

//Callback function for the Position topic (LIVE)
/*
void pose_callback(const geometry_msgs::PoseWithCovarianceStamped& msg)
{

	ips_x X = msg.pose.pose.position.x; // Robot X psotition
	ips_y Y = msg.pose.pose.position.y; // Robot Y psotition
	ips_yaw = tf::getYaw(msg.pose.pose.orientation); // Robot Yaw
	ROS_DEBUG("pose_callback X: %f Y: %f Yaw: %f", X, Y, Yaw);
}*/

//Callback function for the map
void map_callback(const nav_msgs::OccupancyGrid& msg)
{
    //This function is called when a new map is received
    
    //you probably want to save the map into a form which is easy to work with
}

void image_callback(const sensor_msgs::ImageConstPtr& img)
{
  int channels = img->step/img->width;
  int mid_idx = img->height * img->step/2;
  std::memcpy(depth_section, img->data.data()+mid_idx, img->step);
  ROS_INFO("depth:%f", depth_section[IMAGE_WIDTH/2]);
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

void set_weight(particle *p) {
  for (int i = 0; i < MAP_WIDTH; i++) {
    for (int j = 0; j < MAP_WIDTH; j++) {
      // If occupied
      if (occupancy_grid[i*MAP_WIDTH + j] > 80) {
        double theta
        = atan2(j*MAP_RESOLUTION + MAP_ORIGIN_Y - p->y,
                i*MAP_RESOLUTION + MAP_ORIGIN_X - p->x);
      }
    }
  }
}

/// Update with Motion Model
void prediction_update(float dt) {
  for (int i = 0; i < NUM_PARTICLES; i++) {
    particle_set[i].x += odom_input.linear.x*cos(yaw)*dt; // TODO Add random uncertainty
    particle_set[i].y += odom_input.linear.x*sin(yaw)*dt; // TODO Add random uncertainty
    particle_set[i].yaw += odom_input.angular.z*dt; // TODO Add random uncertainty
    if (!POINT_IN_MAP(particle_set[i].x, particle_set[i].y)) {
      // TODO Redistribute somewhere
    }
  }
}


void measurement_update() {

  for (int i = 0; i < NUM_PARTICLES; i++) {
    set_weight(&particle_set[i]);
  }

  for (int i = 0; i < NUM_PARTICLES; i++) {
    // Resample 
  }
}

int main(int argc, char **argv)
{
	//Initialize the ROS framework
  ros::init(argc,argv,"main_control");
  ros::NodeHandle n;

  //Subscribe to the desired topics and assign callbacks
  ros::Subscriber pose_sub = n.subscribe("/gazebo/model_states", 1, pose_callback);
  ros::Subscriber map_sub = n.subscribe("/map", 1, map_callback);
  ros::Subscriber kinect_sub = n.subscribe("/camera/depth/image_raw", 1, image_callback);

  //Setup topics to Publish from this node
  ros::Publisher velocity_publisher = n.advertise<geometry_msgs::Twist>("/cmd_vel_mux/input/navi", 1);
  pose_publisher = n.advertise<geometry_msgs::PoseStamped>("/pose", 1, true);
  marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1, true);

  //Velocity control variable
  geometry_msgs::Twist vel;

  //Set the loop rate
  ros::Rate loop_rate(20);    //20Hz update rate

  for (int i = 0; i < NUM_PARTICLES; i++) {
    // TODO: initial distribution of particles
  }

  while (ros::ok())
  {
  	loop_rate.sleep(); //Maintain the loop rate
  	ros::spinOnce();   //Check for new messages

  	//Main loop code goes here:
  	vel.linear.x = 0; // set linear speed
  	vel.angular.z = 0; // set angular speed

  	velocity_publisher.publish(vel); // Publish the command velocity

    ROS_INFO("UPDATED");
  }

  return 0;
}
