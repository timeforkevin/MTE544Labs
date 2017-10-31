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
#define ICP_ITERATIONS 20

#define POINT_IN_MAP(X,Y) (X > -1 && X < MAP_WIDTH && Y > -1 && Y < MAP_WIDTH)
#define FRAND_TO(X) (static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/(X))))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

#define USE_KD_TREE

typedef unsigned char uint8;
typedef char int8;
typedef Eigen::Matrix<double, 3, 3> Matrix3d;
typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Rotation2D<double> Rotation2D;

typedef struct {
  Vector3d x;
  double weight;
} particle;

ros::Publisher pose_publisher;
ros::Publisher marker_pub;

visualization_msgs::Marker points;
ros::Time last_motion_model_pred;
ros::Time last_scan_reg_pred;
ros::Time last_ips;
particle *particle_set;

std::random_device rd;
std::mt19937 e2(rd());

#ifdef USE_KD_TREE
kd_node *last_scan_tree;
#endif
std::vector<Vector2d> last_scan_list;

// Function Prototypes
void scan_registration_prediction_update(const sensor_msgs::LaserScan msg, ros::Duration dt);
void motion_model_prediction_update(geometry_msgs::Twist odom_input, Matrix3d odom_cov, ros::Duration dt);
void measurement_update(double ips_x, double ips_y, double ips_yaw);

short sgn(int x) { return x >= 0 ? 1 : -1; }

//Callback function for the Position topic (SIMULATION)
void pose_callback(const gazebo_msgs::ModelStates& msg) {
  static std::normal_distribution<double> dist(0.0, 1);
  ros::Time now = ros::Time::now();
  if (now - last_ips > ros::Duration(2)) {
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
/*
void pose_callback(const geometry_msgs::PoseWithCovarianceStamped& msg)
{

	double ips_x X = msg.pose.pose.position.x; // Robot X psotition
	double ips_y Y = msg.pose.pose.position.y; // Robot Y psotition
	double ips_yaw = tf::getYaw(msg.pose.pose.orientation); // Robot Yaw
  measurement_update(ips_x, ips_y, ips_yaw);
	ROS_DEBUG("pose_callback X: %f Y: %f Yaw: %f", X, Y, Yaw);
}*/

void odom_callback(nav_msgs::Odometry msg) {
  if (last_motion_model_pred.isZero()) {
    last_motion_model_pred = ros::Time::now();
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
  // motion_model_prediction_update(odom_input, odom_cov, now - last_motion_model_pred);

  // points.header.stamp = ros::Time::now();
  // points.points.clear();
  // for (int i = 0; i < NUM_PARTICLES; i++) {
  //   geometry_msgs::Point p;
  //   p.x = particle_set[i].x(0);
  //   p.y = particle_set[i].x(1);
  //   points.points.push_back(p);
  // }
  // marker_pub.publish(points);

  last_motion_model_pred = now;
}

void image_callback(const sensor_msgs::LaserScan msg) {
  if (last_scan_reg_pred.isZero()) {
    last_scan_reg_pred = ros::Time::now();
    return;
  }
  ros::Time now = ros::Time::now();
  scan_registration_prediction_update(msg, now - last_scan_reg_pred);

  points.header.stamp = ros::Time::now();
  points.points.clear();
  for (int i = 0; i < NUM_PARTICLES; i++) {
    geometry_msgs::Point p;
    p.x = particle_set[i].x(0);
    p.y = particle_set[i].x(1);
    points.points.push_back(p);
  }
  marker_pub.publish(points);

  last_scan_reg_pred = now;
}

//Callback function for the map
void map_callback(const nav_msgs::OccupancyGrid& msg)
{
    //This function is called when a new map is received
    //you probably want to save the map into a form which is easy to work with
}

double nearest_neighbour_squared_brute_force(std::vector<Vector2d>& last_scan_list,
                                            Vector2d new_scan_point) {
  double cur_best = std::numeric_limits<double>::infinity();
  for (int k = 0; k < last_scan_list.size(); k++) {
    double cur_dist = (new_scan_point[0] - last_scan_list[k][0]) *
                     (new_scan_point[0] - last_scan_list[k][0]) *
                     (new_scan_point[1] - last_scan_list[k][1]) *
                     (new_scan_point[1] - last_scan_list[k][1]);
    if (cur_dist < cur_best) {
      cur_best = cur_dist;
    }
  }
  return cur_best;
}

double calc_mmse(std::vector<Vector2d>& scan_list,
                Rotation2D& TR, Vector2d& TT) {
  double mmse = 0;
  for (int j = 0; j < scan_list.size(); j++) {
    Vector2d transformed_point = TR*scan_list[j] + TT;
#ifdef USE_KD_TREE
    double cur_best = nearest_neighbour_squared_2d(last_scan_tree, transformed_point, 0);
#else
    double cur_best = nearest_neighbour_squared_brute_force(last_scan_list, transformed_point);
#endif
    mmse += cur_best;
  }
  mmse /= scan_list.size();
  return mmse;
}

double iterative_closest_point(std::vector<Vector2d> scan_list,
                               Rotation2D& TR, Vector2d& TT) {
  double eps = 0.0001;
  Vector2d dx(eps, 0.0);
  Vector2d dy(0.0, eps);
  Rotation2D dr(eps);
  double mmse = 0.0;
  double next_mmse = 0.0;
  double dm_dx = 0.0;
  double dm_dy = 0.0;
  double dm_dr = 0.0;
  double gamma_translate = 10;
  double gamma_rotate = 10;
  bool failed_translate = false;
  bool failed_rotate = false;
  for (int i = 0; i < ICP_ITERATIONS; i++) {
    // Gradient Descent Rotation
    if (!failed_rotate) {
      Rotation2D TRdr = TR*dr;
      mmse = calc_mmse(scan_list, TR, TT);
      dm_dr = (calc_mmse(scan_list, TRdr, TT) - mmse)/eps;
    }
    Rotation2D R(-gamma_rotate*dm_dr);
    R *= TR;

    next_mmse = calc_mmse(scan_list, R, TT);
    if (next_mmse < mmse) {
      mmse = next_mmse;
      TR = R;
      if (!failed_rotate) {
        gamma_rotate *= 2;
      }
      failed_rotate = false;
    } else {
      failed_rotate = true;
      gamma_rotate /= 2;
    }

    // Gradient Descent Translation
    if (!failed_translate) {
      Vector2d TTdx = TT+dx;
      Vector2d TTdy = TT+dy;
      dm_dx = (calc_mmse(scan_list, TR, TTdx) - mmse)/eps;
      dm_dy = (calc_mmse(scan_list, TR, TTdy) - mmse)/eps;
    }

    Vector2d T(-gamma_translate*dm_dx, -gamma_translate*dm_dy);
    T = R*TT + T;

    next_mmse = calc_mmse(scan_list, TR, T);
    if (next_mmse < mmse) {
      mmse = next_mmse;
      TT = T;
      if (!failed_translate) {
        gamma_translate *= 2;
      }
      failed_translate = false;
    } else {
      failed_translate = true;
      gamma_translate /= 2;
    }
    if (mmse < 1e-11) {
      break;
    }
  }
  // ROS_INFO("MMSE: %.4e X:%.4f Y:%.4f R:%.4e", mmse, TT(0), TT(1), TR.angle());
  return mmse;
}

Vector2d centroid(std::vector<Vector2d> list) {
  Vector2d centroid = Vector2d::Zero();
  for (int i = 0; i < list.size(); i++) {
    centroid += list[i];
  }
  centroid /= list.size();
  return centroid;
}

void scan_registration_prediction_update(const sensor_msgs::LaserScan msg, ros::Duration dt) {
  std::vector<Vector2d> new_scan_list;
  std::vector<Vector2d> icp_scan_list;
  float fov = msg.angle_max - msg.angle_min;
  int num_points = abs(fov / msg.angle_increment);
  for (int i = 0; i < num_points; i++) {
    float theta = msg.angle_min + i*msg.angle_increment;
    float range = msg.ranges[i];
    if (std::isnan(range)) {
      continue;
    }
    Vector2d p;
    p[0] = range * cos(theta);
    p[1] = range * sin(theta);
    new_scan_list.push_back(p);
    if (theta < msg.angle_max / 2 &&
        theta > msg.angle_min / 2) {
      icp_scan_list.push_back(p);
    }
  }

  // Match centroids
  Vector2d TT = Vector2d::Zero();
  Rotation2D TR(0);
  if (icp_scan_list.size() && last_scan_list.size()) {
    double mmse = iterative_closest_point(icp_scan_list, TR, TT);
    std::normal_distribution<double> dist(0.0, mmse);
    for (int i = 0; i < NUM_PARTICLES; i++) {
      particle *p = &particle_set[i];
      Rotation2D r_yaw(p->x(2));
      Vector2d dx = r_yaw*TT;
      Vector3d Bu;
      Bu << dx, TR.smallestPositiveAngle();
      Vector3d delta(dist(e2), dist(e2), dist(e2));
      p->x += Bu + delta;
      if (p->x(2) > M_PI) {
        p->x(2) -= 2*M_PI;
      }
    }
  }
#ifdef USE_KD_TREE
  destroy_kd_tree(last_scan_tree);
  last_scan_tree = create_2d_tree(new_scan_list, 0);
#endif
  last_scan_list = new_scan_list;
}

/// Update with Motion Model
void motion_model_prediction_update(geometry_msgs::Twist odom_input, Matrix3d odom_cov, ros::Duration dt) {
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
  // ROS_INFO("MOTION MODEL");
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

void measurement_update(double ips_x, double ips_y, double ips_yaw) {
  static double varx = 1;
  static double vary = 1;
  static double varyaw = 2;

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
                      diffy*diffy/vary)/2);
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
  // ROS_INFO("RESAMPLE w = %f", running_sum);
}



int main(int argc, char **argv)
{
  particle_set = (particle*)malloc(sizeof(particle) * NUM_PARTICLES);
  for (int i = 0; i < NUM_PARTICLES; i++) {
    particle *p = &particle_set[i];
    p->x << 0, 0, 0;
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

  //Subscribe to the desired topics and assign callbacks
  ros::Subscriber pose_sub = n.subscribe("/gazebo/model_states", 1, pose_callback);
  ros::Subscriber odom_sub = n.subscribe("/odom", 1, odom_callback);
  ros::Subscriber map_sub = n.subscribe("/map", 1, map_callback);
  ros::Subscriber kinect_sub = n.subscribe("/scan", 1, image_callback);

  //Setup topics to Publish from this node
  ros::Publisher velocity_publisher = n.advertise<geometry_msgs::Twist>("/cmd_vel_mux/input/navi", 1);
  pose_publisher = n.advertise<geometry_msgs::PoseStamped>("/pose", 1, true);
  marker_pub = n.advertise<visualization_msgs::Marker>("/visualization_marker", 1, true);

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
