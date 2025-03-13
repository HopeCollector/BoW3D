#include <fmt/format.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <CLI/CLI.hpp>
#include <fkYAML/node.hpp>
#include <fstream>

#include "BoW3D.h"
#include "LinK3D_Extractor.h"
#include "expr-utils/data_loader.hh"

namespace fs = std::filesystem;

namespace {
std::string config_path{""};

struct Config {
  std::string dataset_type{""};
  std::string lidar_path{""};
  std::string pose_path{""};
  std::string calib_path{""};
  std::string output_path{""};

  // Parameters of LinK3D
  int nScans = 64;  // Number of LiDAR scan lines
  float scanPeriod = 0.1;
  float minimumRange = 0.1;
  float distanceTh = 0.4;
  int matchTh = 6;

  // Parameters of BoW3D
  float thr = 3.5;
  int thf = 5;
  int num_add_retrieve_features = 5;
};

int parse_param(int argc, char** argv) {
  CLI::App app{"STDesc expriement"};
  app.add_option("-c,--config", config_path, "Path to the config file")
      ->required();
  CLI11_PARSE(app, argc, argv);
  return 0;
}

Config load_config(const std::string& config_path) {
  Config ret;
  fkyaml::node node;
  {
    std::ifstream file(config_path);
    node = fkyaml::node::deserialize(file);
  }

  // Parameters of LinK3D
  ret.nScans = node["nScans"].get_value<int>();
  ret.scanPeriod = node["scanPeriod"].get_value<float>();
  ret.minimumRange = node["minimumRange"].get_value<float>();
  ret.distanceTh = node["distanceTh"].get_value<float>();
  ret.matchTh = node["matchTh"].get_value<int>();

  // Parameters of BoW3D
  ret.thr = node["thr"].get_value<float>();
  ret.thf = node["thf"].get_value<int>();
  ret.num_add_retrieve_features =
      node["num_add_retrieve_features"].get_value<int>();

  // Parameters of dataset
  ret.dataset_type = node["dataset_type"].get_value<std::string>();
  ret.lidar_path = node["lidar_path"].get_value<std::string>();
  ret.pose_path = node["pose_path"].get_value<std::string>();
  ret.calib_path = node["calib_path"].get_value<std::string>();
  ret.output_path = node["output_path"].get_value<std::string>();
  return ret;
}

struct Result {
  size_t key_frame_id;
  size_t loop_frame_id;
  double score;
  double iou;
  Eigen::Vector3f center;

  friend std::ostream& operator<<(std::ostream& os, const Result& res) {
    os << res.key_frame_id << "," << res.loop_frame_id << "," << res.score
       << "," << res.iou << "," << res.center.x() << "," << res.center.y()
       << "," << res.center.z();
    return os;
  }
};

double iou(pcl::PointCloud<pcl::PointXYZI>::ConstPtr cld1,
           pcl::PointCloud<pcl::PointXYZI>::ConstPtr cld2) {
  pcl::KdTreeFLANN<pcl::PointXYZI> tree;
  tree.setInputCloud(cld1);
  std::vector<bool> marks1(cld1->size(), false);
  std::vector<bool> marks2(cld2->size(), false);

  // search the near points in 0.5m
  for (size_t i = 0; i < cld2->size(); i++) {
    const auto& p = cld2->at(i);
    std::vector<int> indices;
    std::vector<float> distances;
    tree.radiusSearch(p, 0.5, indices, distances);
    if (indices.empty()) {
      continue;
    }
    // mark the indices
    for (const auto& idx : indices) {
      marks1[idx] = true;
    }
    marks2[i] = true;
  }

  // calculate the iou, no need to minus the intersection, because this is point
  // num, not volume
  size_t inter = std::count(marks1.begin(), marks1.end(), true) +
                 std::count(marks2.begin(), marks2.end(), true);
  return double(inter) / double(cld1->size() + cld2->size());
}
}  // namespace

int main(int argc, char** argv) {
  // parse parameters
  parse_param(argc, argv);
  auto cfg = load_config(config_path);

  // data loader
  auto loader = utils::create_loader(cfg.dataset_type, cfg.lidar_path,
                                     cfg.pose_path, cfg.calib_path);

  // Link3D
  auto pLinK3dExtractor =
      new BoW3D::LinK3D_Extractor(cfg.nScans, cfg.scanPeriod, cfg.minimumRange,
                                  cfg.distanceTh, cfg.matchTh);

  // BoW3D
  auto pBoW3D = new BoW3D::BoW3D(pLinK3dExtractor, cfg.thr, cfg.thf,
                                 cfg.num_add_retrieve_features);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cur_cld;
  std::deque<Result> results;
  size_t frame_num = loader->size();
  while (true) {
    auto tmp = loader->next();
    if (tmp) {
      cur_cld.reset(new pcl::PointCloud<pcl::PointXYZ>);
      cur_cld->resize(tmp->size());
      std::transform(
          tmp->begin(), tmp->end(), cur_cld->begin(),
          [](const auto& p) { return pcl::PointXYZ(p.x, p.y, p.z); });
    } else {
      break;
    }

    // do the BoW3D things
    auto pCurrentFrame = new BoW3D::Frame(pLinK3dExtractor, cur_cld);
    size_t frameId = pCurrentFrame->mnId;
    if (pCurrentFrame->mnId < 2) {
      pBoW3D->update(pCurrentFrame);
      continue;
    }
    int loopFrameId = -1;
    Eigen::Matrix3d loopRelR;
    Eigen::Vector3d loopRelt;
    auto pairs =
        pBoW3D->retrieve(pCurrentFrame, loopFrameId, loopRelR, loopRelt);
    pBoW3D->update(pCurrentFrame);

    // record the result
    for (const auto& par : pairs) {
      Result res;
      res.key_frame_id = frameId;
      res.loop_frame_id = par.loop_frame_id;
      res.score = 1.0 / (par.loop_rel_t.norm() + 1.0);
      auto cld1 = loader->seq(res.key_frame_id, true);
      auto cld2 = loader->seq(res.loop_frame_id, true);
      res.iou = iou(cld1, cld2);
      // calculate the center
      Eigen::Vector4f center1;
      pcl::compute3DCentroid(*cld1, center1);
      Eigen::Vector4f center2;
      pcl::compute3DCentroid(*cld2, center2);
      res.center = ((center1 * cld1->size() + center2 * cld2->size()) /
                    (cld1->size() + cld2->size()))
                       .block<3, 1>(0, 0);
      results.push_back(res);
    }

    // print progress
    fmt::print("Processing({}/{}): ", frameId, frame_num);
    if (loopFrameId != -1) {
      fmt::print("{}, {}, {:.2f}\n", frameId, loopFrameId, loopRelt.norm());
    } else {
      fmt::print("no loop found\n");
    }
  }

  // save the results
  std::ofstream ofs(cfg.output_path);
  for (const auto& res : results) {
    ofs << res << std::endl;
  }

  return 0;
}