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
#include <Eigen/Dense>

#define EPS 0.02

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
typedef Eigen::Matrix<double, 3, 1> Vector3d;

typedef struct {
  Vector3d x;
  double weight;
} particle;

particle particle_set[NUM_PARTICLES];

geometry_msgs::Twist odom_input;


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

void set_weights() {
  for (int i = 0; i < NUM_PARTICLES; i++) {

  }
}

/// Update with Motion Model
void prediction_update(float dt) {
  for (int i = 0; i < NUM_PARTICLES; i++) {
    Vector3d Bu;
    Bu << odom_input.linear.x*cos(particle_set[i].x(2))*dt,
         odom_input.linear.x*sin(particle_set[i].x(2))*dt,
         odom_input.angular.z*dt;



    particle_set[i].x += Bu;

    if (!POINT_IN_MAP(particle_set[i].x[0], particle_set[i].x[1])) {
      // TODO Redistribute somewhere
    }
  }
}


void measurement_update() {

  set_weights();


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
