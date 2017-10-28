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
#include <random>

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
#define FRAND_TO(X) (static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/X)))

typedef unsigned char uint8;
typedef char int8;
typedef Eigen::Matrix<double, 3, 1> Vector3d;

typedef struct {
  Vector3d x;
  double weight;
} particle;

ros::Publisher pose_publisher;
ros::Publisher marker_pub;
geometry_msgs::Twist odom_input;

particle *particle_set;

std::random_device rd;
std::mt19937 e2(rd());


// Function Prototypes
void update_map();
float logit(float p);
float i_logit(float l);
void bresenham(int x0, int y0, int x1, int y1, std::vector<int>& x, std::vector<int>& y);

short sgn(int x) { return x >= 0 ? 1 : -1; }

//Callback function for the Position topic (SIMULATION)
void pose_callback(const gazebo_msgs::ModelStates& msg) {

  int i;
  for(i = 0; i < msg.name.size(); i++) if(msg.name[i] == "mobile_base") break;
  double ips_x = msg.pose[i].position.x ;
  double ips_y = msg.pose[i].position.y ;
  double ips_yaw = tf::getYaw(msg.pose[i].orientation);
  // ROS_INFO("POSE X: %f Y:%f", ips_x, ips_y);
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

void image_callback(const sensor_msgs::ImageConstPtr& img)
{
  int channels = img->step/img->width;
  int mid_idx = img->height * img->step/2;
  std::memcpy(depth_section, img->data.data()+mid_idx, img->step);
  ROS_INFO("depth:%f", depth_section[IMAGE_WIDTH/2]);
}

int map_to_cdf(double sample, double *cdf, int cdf_size) {
  // Binary Search
  int i_begin = 0;
  int i_end = cdf_size - 1;
  while (i_begin != i_end) {
    int i_test = (i_begin + i_end)/2;
    if (sample < cdf[i_test]) {
      i_end = i_test;
    } else {
      i_begin = i_test + 1;
    }
  }
  return i_begin;
}

/// Update with Motion Model
void prediction_update(float dt) {
  static std::normal_distribution<double> distx(0.0, 0.03);
  static std::normal_distribution<double> disty(0.0, 0.03);
  static std::normal_distribution<double> distyaw(0.0, 0.03);

  for (int i = 0; i < NUM_PARTICLES; i++) {
    particle *p = &particle_set[i];

    Vector3d Bu;
    Bu << odom_input.linear.x*cos(p->x(2)),
         odom_input.linear.x*sin(p->x(2)),
         odom_input.angular.z;

    // Add disturbance
    Vector3d eps;
    eps << distx(e2), disty(e2), distyaw(e2);

    p->x += Bu*dt + eps;

    // Maintain yaw between +-pi
    if (p->x(2) > M_PI) {
      p->x(2) -= 2*M_PI;
    } else if (p->x(2) < -M_PI) {
      p->x(2) += 2*M_PI;
    }
  }
}

void measurement_update(double ips_x, double ips_y, double ips_yaw) {
  static double varx = 0.01;
  static double vary = 0.01;
  static double varyaw = 0.01;

  // Set Weights
  for (int i = 0; i < NUM_PARTICLES; i++) {
    particle *p = &particle_set[i];
    double diffx = p->x(0) - ips_x;
    double diffy = p->x(1) - ips_y;
    double diffyaw = p->x(2) - ips_yaw;
    if (diffyaw > M_PI) {
      diffyaw -= 2*M_PI;
    } else if (diffyaw < -M_PI) {
      diffyaw += 2*M_PI;
    }

    p->weight = exp(-(diffx*diffx/varx +
                      diffy*diffy/vary +
                      diffyaw*diffyaw/varyaw)/2);
  }

  // Calculate CDF of particles
  double cdf[NUM_PARTICLES];
  double running_sum = 0;
  for (int i = 0; i < NUM_PARTICLES; i++) {
    particle *p = &particle_set[i];
    cdf[i] = running_sum += p->weight;
  }

  // Resample
  particle *new_particle_set = (particle*)malloc(sizeof(particle) * NUM_PARTICLES);
  for (int i = 0; i < NUM_PARTICLES; i++) {
    particle *sampled = particle_set[map_to_cdf(FRAND_TO(running_sum), cdf, NUM_PARTICLES)];
    memcpy(&new_particle_set[i], sampled);
  }
  free(particle_set);
  particle_set = new_particle_set;
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

  particle_set = (particle*)malloc(sizeof(particle) * NUM_PARTICLES);
  for (int i = 0; i < NUM_PARTICLES; i++) {
    particle *p = &particle_set[i];
    p->x << FRAND_TO(10) + MAP_ORIGIN_X,
            FRAND_TO(10) + MAP_ORIGIN_Y,
            FRAND_TO(2*M_PI) - M_PI;
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

  free(particle_set);
  return 0;
}
