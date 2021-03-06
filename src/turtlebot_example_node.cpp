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
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_datatypes.h>


//Callback function for the Position topic 
//void pose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
//{
	//This function is called when a new position message is received

//	double X = msg->pose.pose.position.x; // Robot X psotition
//	double Y = msg->pose.pose.position.y; // Robot Y psotition
//	double Yaw = tf::getYaw(msg->pose.pose.orientation); // Robot Yaw

//}


// Globals
#define SQUARE_WIDTH 0.3 
double X;
double Y;
double Yaw;


void pose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
{
	//This function is called when a new position message is received

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
    ros::Subscriber pose_sub = n.subscribe("/indoor_pos", 1, pose_callback);

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
