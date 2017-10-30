//  ///////////////////////////////////////////////////////////
//
// turtlebot_example.cpp
// This file contains example code for use with ME 597 lab 1
// It outlines the basic setup of a ros node and the various
// inputs and outputs.
//
// Author: James Servos
//
// //////////////////////////////////////////////////////////

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_datatypes.h>
#include <gazebo_msgs/ModelStates.h>
#include <visualization_msgs/Marker.h>
#include <random>

ros::Publisher pose_publisher;
ros::Publisher marker_pub;
geometry_msgs::Twist odom_input;


// Function defns
void prediction_update();
void measurement_update();

void pose_callback(const gazebo_msgs::ModelStates& message) {

	X = msg->pose.pose.position.x; // Robot X psotition
	Y = msg->pose.pose.position.y; // Robot Y psotition
 	Yaw = tf::getYaw(msg->pose.pose.orientation); // Robot Yaw
}


int main(int argc, char **argv)
{
		//Initialize the ROS framework
    ros::init(argc,argv,"main_control");
    ros::NodeHandle n;

    //Subscribe to the desired topics and assign callbacks
    ros::Subscriber odom_sub = n.subscribe("/odom", 1, odom_callback);
		ros::Subscriber ips_sub = n.subscrive("/gazebo/model_states", 1, )

    //Setup topics to Publish from this node
    ros::Publisher velocity_publisher = n.advertise<geometry_msgs::Twist>("/cmd_vel_mux/input/teleop", 1);

    //Velocity control variable
    geometry_msgs::Twist vel;

    //Set the loop rate
    ros::Rate loop_rate(20);    //20Hz update rate

    double x_last = 0;
    double y_last = 0;
    double yaw_last = 0;
    bool turning = false;

    while (ros::ok())
    {
    	loop_rate.sleep(); //Maintain the loop rate
    	ros::spinOnce();   //Check for new messages

    	//Main loop code goes here:

        if (turning) {
            vel.linear.x = 0;
            vel.angular.z = 0.2;
            double yaw_diff = Yaw - yaw_last;
            if (yaw_diff < 0) {
                yaw_diff += 2*M_PI;
            }
            if (yaw_diff > M_PI/2) {
                turning = false;
                x_last = X;
                y_last = Y;
            }


        } else {
            vel.linear.x = 0.2;
            vel.angular.z = 0;
            if (sqrt((x_last - X)*(x_last - X) +
                     (y_last - Y)*(y_last - Y)) > SQUARE_WIDTH) {
                turning = true;
                yaw_last = Yaw;
            }
        }


ROS_INFO("%f", vel.angular.z);
    	velocity_publisher.publish(vel); // Publish the command velocity
    }

    return 0;
}