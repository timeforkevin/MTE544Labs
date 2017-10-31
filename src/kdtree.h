#ifndef KDTREE_H
#define KDTREE_H

#include <random>
#include <limits>
#include <Eigen/Dense>

typedef Eigen::Matrix<double, 2, 1> Vector2d;

typedef struct kd_node_t {
  Vector2d point;
  kd_node_t *before;
  kd_node_t *after;
} kd_node;

kd_node* create_2d_tree(std::vector<Vector2d>& points, int depth) {
  if (!points.size()) {
    return NULL;
  }
  kd_node* node = (kd_node*) malloc(sizeof(kd_node));

  // Choose median randomly
  int median_idx = rand() % points.size();
  node->point = points[median_idx];
  std::vector<Vector2d> before_median;
  std::vector<Vector2d> after_median;

  for(int i = 0; i < points.size(); i++) {
    if (i != median_idx) {
      if (points[i][depth % 2] < node->point[depth % 2]) {
        before_median.push_back(points[i]);
      } else {
        after_median.push_back(points[i]);
      }
    }
  }
  node->before = create_2d_tree(before_median, depth + 1);
  node->after = create_2d_tree(after_median, depth + 1);

  return node;
}

double nearest_neighbour_squared_2d(kd_node* root, Vector2d& search_point, int depth) {
  double cur_best = std::numeric_limits<double>::infinity();
  // Recursively traverse
  if (search_point[depth % 2] < root->point[depth % 2]) {
    if (root->before != NULL) {
      cur_best = nearest_neighbour_squared_2d(root->before, search_point, depth + 1);
    }
  } else {
    if (root->after != NULL) {
      cur_best = nearest_neighbour_squared_2d(root->after, search_point, depth + 1);
    }
  }

  double cur_dist = (search_point[0] - root->point[0]) *
                   (search_point[0] - root->point[0]) *
                   (search_point[1] - root->point[1]) *
                   (search_point[1] - root->point[1]);
  if (cur_dist < cur_best) {
    cur_best = cur_dist;
  }

  double radius = sqrt(cur_best);
  if (search_point[depth % 2] < root->point[depth % 2] &&
     (search_point[depth % 2] + radius > root->point[depth % 2])) {
    if (root->after != NULL) {
      cur_dist = nearest_neighbour_squared_2d(root->after, search_point, depth + 1);
    }
  } else
  if (search_point[depth % 2] > root->point[depth % 2] &&
     (search_point[depth % 2] - radius < root->point[depth % 2])) {
    if (root->before != NULL) {
      cur_dist = nearest_neighbour_squared_2d(root->before, search_point, depth + 1);
    }
  }

  if (cur_dist < cur_best) {
    cur_best = cur_dist;
  }
  return cur_best;
}

void destroy_kd_tree(kd_node* root) {
  if (root == NULL) {
    return;
  }
  if (root->before != NULL) {
    destroy_kd_tree(root->before);
  }
  if (root->after != NULL) {
    destroy_kd_tree(root->after);
  }
  free(root);
}
#endif