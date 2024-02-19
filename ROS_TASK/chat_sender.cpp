#include "ros/ros.h"
#include "std_msgs/String.h"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    ros::init(argc, argv, "chat_sender");
    ros::NodeHandle nh;
    ros::Publisher chat_pub = nh.advertise<std_msgs::String>("chat", 1000);
    ros::Rate loop_rate(10);

    while (ros::ok()) {
        std::string message;
        std::cout << "Enter message: ";
        std::getline(std::cin, message);

        std_msgs::String msg;
        msg.data = message;
        chat_pub.publish(msg);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
