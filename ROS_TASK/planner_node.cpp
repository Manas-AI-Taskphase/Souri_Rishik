#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <queue>
#include <vector>
#include <cmath>

struct Node {
    int x, y;
    int g, h;
    Node* parent;

    Node(int x, int y, int g, int h, Node* parent) : x(x), y(y), g(g), h(h), parent(parent) {}

    int getF() const {
        return g + h;
    }
};

class AStarPathPlanner {
public:
    AStarPathPlanner(ros::NodeHandle& nh) : nh_(nh) {
        map_sub_ = nh_.subscribe("/map", 1, &AStarPathPlanner::mapCallback, this);
        path_pub_ = nh_.advertise<nav_msgs::Path>("/path", 1);
    }

    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& map) {
        map_ = map;
        initializeGrid();
        runAStar();
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber map_sub_;
    ros::Publisher path_pub_;
    nav_msgs::OccupancyGrid::ConstPtr map_;
    std::vector<std::vector<int>> grid_;
    std::vector<Node*> path_;

    void initializeGrid() {
        grid_.clear();
        grid_.resize(map_->info.width, std::vector<int>(map_->info.height, 0));

        for (int x = 0; x < map_->info.width; ++x) {
            for (int y = 0; y < map_->info.height; ++y) {
                int index = y * map_->info.width + x;
                grid_[x][y] = map_->data[index];
            }
        }
    }

    void runAStar() {
        std::priority_queue<Node*, std::vector<Node*>, std::function<bool(Node*, Node*)>> openSet(
            [](Node* a, Node* b) { return a->getF() > b->getF(); });

        int startX = 0;
        int startY = 0;
        int goalX = map_->info.width - 1;
        int goalY = map_->info.height - 1;

        Node* startNode = new Node(startX, startY, 0, calculateHeuristic(startX, startY, goalX, goalY), nullptr);
        Node* goalNode = new Node(goalX, goalY, 0, 0, nullptr);

        openSet.push(startNode);

        while (!openSet.empty()) {
            Node* currentNode = openSet.top();
            openSet.pop();

            if (currentNode->x == goalNode->x && currentNode->y == goalNode->y) {
                reconstructPathAndPublish(currentNode);
                break;
            }

            exploreNeighbors(currentNode, goalNode, openSet);
        }
    }

    void exploreNeighbors(Node* currentNode, Node* goalNode, std::priority_queue<Node*, std::vector<Node*>, std::function<bool(Node*, Node*)>>& openSet) {
        const int moves[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

        for (const auto& move : moves) {
            int nextX = currentNode->x + move[0];
            int nextY = currentNode->y + move[1];

            if (isValidPosition(nextX, nextY)) {
                Node* nextNode = new Node(nextX, nextY, currentNode->g + 1, calculateHeuristic(nextX, nextY, goalNode->x, goalNode->y), currentNode);
                openSet.push(nextNode);
            }
        }
    }

    bool isValidPosition(int x, int y) {
        return x >= 0 && x < map_->info.width && y >= 0 && y < map_->info.height && grid_[x][y] == 0;
    }

    int calculateHeuristic(int x, int y, int goalX, int goalY) {
        return static_cast<int>(std::hypot(x - goalX, y - goalY));
    }

    void reconstructPathAndPublish(Node* goalNode) {
        path_.clear();

        while (goalNode != nullptr) {
            path_.push_back(goalNode);
            goalNode = goalNode->parent;
        }

        if (path_.empty()) {
            ROS_WARN("No valid path found.");
            return;
        }

        std::reverse(path_.begin(), path_.end());

        nav_msgs::Path path_msg;
        path_msg.header.stamp = ros::Time::now();
        path_msg.header.frame_id = "map";

        for (const auto& node : path_) {
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = ros::Time::now();
            pose_stamped.header.frame_id = "map";

            // Convert grid indices to real-world coordinates
            pose_stamped.pose.position.x = node->x * map_->info.resolution + map_->info.origin.position.x;
            pose_stamped.pose.position.y = node->y * map_->info.resolution + map_->info.origin.position.y;
            pose_stamped.pose.position.z = 0;
            pose_stamped.pose.orientation.w = 1.0;

            // Debug message to print path coordinates
            ROS_INFO("Path Point: (%.2f, %.2f)", pose_stamped.pose.position.x, pose_stamped.pose.position.y);

            path_msg.poses.push_back(pose_stamped);
        }

        path_pub_.publish(path_msg);

        ROS_INFO("Published path with %zu waypoints", path_msg.poses.size());
}

};

int main(int argc, char** argv) {
    ros::init(argc, argv, "a_star_path_planner");
    ros::NodeHandle nh;

    AStarPathPlanner pathPlanner(nh);

    ros::spin();

    return 0;
}
