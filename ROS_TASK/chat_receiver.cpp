#include "ros/ros.h"
#include "std_msgs/String.h"

void chatCallback(const std_msgs::String::ConstPtr& msg) {
    std::cout<< "I heard:"<<msg->data.c_str() << std::endl;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "chat_receiver");
    ros::NodeHandle nh;
    ros::Subscriber chat_sub = nh.subscribe("chat", 1000, chatCallback);
    ros::spin();

    return 0;
}
