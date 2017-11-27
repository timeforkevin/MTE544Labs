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
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <eigen3/Eigen/Dense>
#include <random>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

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
#define FRAND_TO(X) (static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/(X))))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

typedef unsigned char uint8;
typedef char int8;
typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<double, 3, 3> Matrix3d;

typedef struct {
  Vector3d x;
  double weight;
} particle;

ros::Publisher pose_publisher;
ros::Publisher marker_pub;
ros::Publisher path_publisher;

visualization_msgs::Marker points;
ros::Time last_pred;
ros::Time last_ips;
particle *particle_set;

std::random_device rd;
std::mt19937 e2(rd());

// path plotting
std::vector<geometry_msgs::PoseStamped> plan;
nav_msgs::Path gui_path;


// Function Prototypes
void prediction_update(geometry_msgs::Twist odom_input, Matrix3d odom_cov, ros::Duration dt);
void measurement_update(double ips_x, double ips_y, double ips_yaw);
void gui_path_update();

short sgn(int x) { return x >= 0 ? 1 : -1; }

//Callback function for the Position topic (SIMULATION)
void pose_callback(const gazebo_msgs::ModelStates& msg) {
  static std::normal_distribution<double> dist(0.0, 1);
  ros::Time now = ros::Time::now();
  if (now - last_ips > ros::Duration(0.25)) {
    int i;
    for(i = 0; i < msg.name.size(); i++) if(msg.name[i] == "mobile_base") break;
    double ips_x = msg.pose[i].position.x + dist(e2);
    double ips_y = msg.pose[i].position.y + dist(e2);
    double ips_yaw = tf::getYaw(msg.pose[i].orientation) + dist(e2);
    measurement_update(ips_x, ips_y, ips_yaw);
    // ROS_INFO("POSE X: %f Y:%f", ips_x, ips_y);
    last_ips = now;
  }
}

//Callback function for the Position topic (LIVE)

// void pose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
// {

//   double ips_x = msg->pose.pose.position.x; // Robot X psotition
//   double ips_y = msg->pose.pose.position.y; // Robot Y psotition
//   double ips_yaw = tf::getYaw(msg->pose.pose.orientation); // Robot Yaw
//   measurement_update(ips_x, ips_y, ips_yaw);
// }

void odom_callback(nav_msgs::Odometry msg) {
  if (last_pred.isZero()) {
    last_pred = ros::Time::now();
    return;
  }
  geometry_msgs::Twist odom_input = msg.twist.twist;
  Matrix3d odom_cov;
  odom_cov << msg.twist.covariance[0], msg.twist.covariance[1], msg.twist.covariance[5],
              msg.twist.covariance[6], msg.twist.covariance[7], msg.twist.covariance[11],
              msg.twist.covariance[30], msg.twist.covariance[31], msg.twist.covariance[35];
  odom_cov(0, 0) = MAX(odom_cov(0, 0), 0.05);
  odom_cov(1, 1) = MAX(odom_cov(1, 1), 0.05);
  odom_cov(2, 2) = MAX(odom_cov(2, 2), 0.05);


  ros::Time now = ros::Time::now();
  prediction_update(odom_input, odom_cov, now - last_pred);

  points.header.stamp = ros::Time::now();
  points.points.clear();
  for (int i = 0; i < NUM_PARTICLES; i++) {
    geometry_msgs::Point p;
    p.x = particle_set[i].x(0);
    p.y = particle_set[i].x(1);
    points.points.push_back(p);
  }
  marker_pub.publish(points);

  last_pred = now;
}

void image_callback(const sensor_msgs::ImageConstPtr& img)
{
}

//Callback function for the map
void map_callback(const nav_msgs::OccupancyGrid& msg)
{
    //This function is called when a new map is received
    //you probably want to save the map into a form which is easy to work with
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
void prediction_update(geometry_msgs::Twist odom_input, Matrix3d odom_cov, ros::Duration dt) {
  static std::normal_distribution<double> dist(0.0, 1);
  Vector3d randn;

  Eigen::EigenSolver<Matrix3d> es(odom_cov);
  Vector3d lambda = es.eigenvalues().cwiseAbs();
  Matrix3d E = es.eigenvectors().cwiseAbs();

  // std::cout << std::endl << odom_cov << std::endl;

  for (int i = 0; i < NUM_PARTICLES; i++) {
    particle *p = &particle_set[i];

    Vector3d Bu;
    Bu << odom_input.linear.x*cos(p->x(2)),
         odom_input.linear.x*sin(p->x(2)),
         odom_input.angular.z;

    // Add disturbance
    randn << dist(e2),
             dist(e2),
             dist(e2);
    Vector3d delta = E*lambda.cwiseSqrt();
    delta = (delta.array()*randn.array()).matrix();

    p->x += (Bu + delta)*dt.toSec();
    // ROS_INFO("delta: %f, %f, %f", delta(0), delta(1), delta(2));
    // ROS_INFO("E*lambda: %f, %f, %f", E*lambda(0), E*lambda(1), E*lambda(2));
    // Maintain yaw between +-pi
    if (p->x(2) > M_PI) {
      p->x(2) -= 2*M_PI;
    } else if (p->x(2) < -M_PI) {
      p->x(2) += 2*M_PI;
    }
  }
  ROS_INFO("MOTION MODEL");
}

void measurement_update(double ips_x, double ips_y, double ips_yaw) {
  static double varx = 1;
  static double vary = 1;
  static double varyaw = 1;

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
                      diffy*diffy/vary + diffyaw*diffyaw/varyaw)/2);
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
    int sample_index = map_to_cdf(FRAND_TO(running_sum), cdf, NUM_PARTICLES);
    particle *sampled = &particle_set[sample_index];
    // ROS_INFO("i:%d w:%f", sample_index, sampled->weight, sampled->x(0));
    memcpy(&new_particle_set[i], sampled, sizeof(particle));
  }
  free(particle_set);
  particle_set = new_particle_set;
  ROS_INFO("RESAMPLE w = %f", running_sum);
  gui_path_update();
}

void gui_path_update() {
  geometry_msgs::PoseStamped pose;
  for (int i = 0; i < NUM_PARTICLES; i++) {
    particle *sampled = &particle_set[i];
    pose.pose.position.x += sampled->x(0);
    pose.pose.position.y += sampled->x(1);
    pose.pose.position.z = 0.0;
    pose.pose.orientation.x = 0.0;
    pose.pose.orientation.y = 0.0;
    pose.pose.orientation.z = 0.0;
    pose.pose.orientation.w = sampled->x(2);
  }

  pose.pose.position.x /= NUM_PARTICLES;
  pose.pose.position.y /= NUM_PARTICLES;
  pose.pose.orientation.w /= NUM_PARTICLES;

  plan.push_back(pose);
  gui_path.poses.resize(plan.size());

  if(!plan.empty()) {
    gui_path.header.frame_id = "map";
    gui_path.header.stamp = plan[0].header.stamp;
  }

  for(unsigned int i=0; i < plan.size(); i++){
    gui_path.poses[i] = plan[i];
  }

  path_publisher.publish(gui_path);
}

int main(int argc, char **argv)
{
  particle_set = (particle*)malloc(sizeof(particle) * NUM_PARTICLES);
  for (int i = 0; i < NUM_PARTICLES; i++) {
    particle *p = &particle_set[i];
    p->x << FRAND_TO(10) + MAP_ORIGIN_X,
            FRAND_TO(10) + MAP_ORIGIN_Y,
            FRAND_TO(2*M_PI) - M_PI;
    ROS_INFO("yaw:%f", p->x(2));
  }

  points.header.frame_id = "map";
  points.id = 0;
  points.ns = "particles";
  points.type = visualization_msgs::Marker::POINTS;
  points.action = 0;
  points.color.g = 1;
  points.color.a = 1;
  points.lifetime = ros::Duration(0);
  points.frame_locked = true;
  points.pose.orientation.w = 1.0;
  points.scale.x = 0.05;
  points.scale.y = 0.05;

  //Initialize the ROS framework
  ros::init(argc,argv,"main_control");
  ros::NodeHandle n;

  //Subscribe to the desired topics and assign 
  ros::Subscriber pose_sub = n.subscribe("/gazebo/model_states", 1, pose_callback);
  // ros::Subscriber pose_sub = n.subscribe("/indoor_pos", 1, pose_callback);
  ros::Subscriber odom_sub = n.subscribe("/odom", 1, odom_callback);
  ros::Subscriber map_sub = n.subscribe("/map", 1, map_callback);
  ros::Subscriber kinect_sub = n.subscribe("/camera/depth/image_raw", 1, image_callback);

  //Setup topics to Publish from this node
  ros::Publisher velocity_publisher = n.advertise<geometry_msgs::Twist>("/cmd_vel_mux/input/navi", 1);
  pose_publisher = n.advertise<geometry_msgs::PoseStamped>("/pose", 1, true);
  marker_pub = n.advertise<visualization_msgs::Marker>("/visualization_marker", 1, true);
  path_publisher = n.advertise<nav_msgs::Path>("/visualization_path", 1, true);

  //Velocity control variable
  geometry_msgs::Twist vel;

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
  }

  free(particle_set);
  return 0;
}